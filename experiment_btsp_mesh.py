"""This module is for testing a MESH network with BTSP layer.add(

The MESH network is proposed in the paper "Content Addressable Memory Without
Catastrophic Forgetting by Heteroassociation with a Fixed Scaffold"
"""

from dataclasses import dataclass
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from custom_networks import btsp_memory_scaffold, btsp_step_topk
import experiment_framework.utils.layers as layers
import experiment_framework.auto_experiment.auto_experiment as auto_experiment
import experiment_framework.utils.parameterization as parameterization
import experiment_framework.utils.logging as logging

REPEAT_NUM = 1


# print kbits(4, 3)
# Output: [1110, 1101, 1011, 0111]  array of shape (Ng,Npos)
def kbits(n, k):
    result = []
    for bits in itertools.combinations(range(n), k):
        s = [0.0] * n
        for bit in bits:
            s[bit] = 1.0
        result.append(s)
    return np.array(result).T

def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def pseudotrain(sbook, ca1book, Npatts):
    ca1inv = np.linalg.pinv(ca1book[:,:Npatts])
    return (1/Npatts)*np.einsum('ij, jl -> il', sbook[:,:Npatts], ca1inv[:Npatts,:]) 

def train_pcsc(pbook, sbook, Npatts): # hebbian learning
    return (1/Npatts)*np.einsum('ij, lj -> il', pbook[:,:Npatts], sbook[:,:Npatts])

@dataclass
class MetaParams:
    """Meta-parameters that generate other parameters."""

    # network structure
    feature_dim: int | list
    hidden_dim: int | list
    label_dim: int | list

    # memory scaffold
    label_topk: int | list
    btsp_fq: float | list
    btsp_fw: float | list
    btsp_topk: int | list

    # dataset
    pattern_num: int | list

    # device
    device: str = "cuda"


@dataclass
class MESHNetworkParams:
    """Parameters for MESH network."""

    heteroassociation: layers.PseudoinverseLayerParams
    memory_scaffold: btsp_memory_scaffold.BTSPMemoryScaffoldNetworkParams


@dataclass
class DatasetParams:
    """Parameters for the dataset."""

    # testing dataset
    pattern_num: int
    pattern_dim: int

    # predefined labels
    label_topk: int
    label_dim: int

    # debug
    hidden_dim: int
    
    # device
    device: str = "cuda"


@dataclass
class ExperimentParams:
    """Parameters for the experiment."""

    network_params: MESHNetworkParams
    dataset_params: DatasetParams


class BTSPMeshExperiment(auto_experiment.ExperimentInterface):
    """Test the memory capacity of a MESH network."""

    def __init__(self) -> None:
        super().__init__()

        self.meta_params = MetaParams(
            feature_dim=300,
            hidden_dim=300,
            label_dim=18,
            label_topk=3,
            btsp_fq=0.01,
            btsp_fw=1,
            btsp_topk=200,
            pattern_num=np.arange(1, 800, 15).tolist(),
            device="cuda",
        )
        
        self.experiment_name = "Heteroassociation_Memory_Capacity_Test"
        self.experiment_folder = logging.init_experiment_folder("data", self.experiment_name,False)
        
        # for debugging
        self.correct_accuracys = []
        self.accuracys = []
        self.pattern_nums = []

    def load_parameters(self):
        meta_combinations = parameterization.recursive_iterate_dataclass(
            self.meta_params
        )
        params = []
        for combination in meta_combinations:
            current_combination: MetaParams = combination
            network_params = MESHNetworkParams(
                layers.PseudoinverseLayerParams(
                    current_combination.feature_dim,
                    current_combination.hidden_dim,
                    current_combination.device,
                ),
                btsp_memory_scaffold.BTSPMemoryScaffoldNetworkParams(
                    layers.HebbianLayerParams(
                        current_combination.hidden_dim,
                        current_combination.label_dim,
                        current_combination.device,
                    ),
                    layers.TopKLayerParams(
                        current_combination.label_topk,
                    ),
                    btsp_step_topk.BTSPStepTopKNetworkParams(
                        layers.BTSPLayerParams(
                            current_combination.label_dim,
                            current_combination.hidden_dim,
                            current_combination.btsp_fq,
                            current_combination.btsp_fw,
                            current_combination.device,
                        ),
                        layers.TopKLayerParams(
                            current_combination.btsp_topk,
                        ),
                        layers.StepLayerParams(0),
                    ),
                ),
            )
            dataset_params = DatasetParams(
                current_combination.pattern_num,
                current_combination.feature_dim,
                current_combination.label_topk,
                current_combination.label_dim,
                current_combination.hidden_dim,
                current_combination.device,
            )
            params.append(ExperimentParams(network_params, dataset_params))

        return params

    def load_dataset(self):
        """Use random generated dataset, no need to load from file."""

    def summarize_results(self):
        """Not implemented."""
        
        # temp: plot accuracy vs pattern number
        plt.plot(self.pattern_nums, self.accuracys)
        # add correct accuracy in the same plot
        # plt.plot(self.pattern_nums, self.correct_accuracys)
        if self.meta_params.feature_dim != self.meta_params.hidden_dim:
            # plot a line x = feature_dim
            plt.axvline(x=self.meta_params.feature_dim, color='r', linestyle='--')
            # label this line as "Feature neuron number"
            plt.text(self.meta_params.feature_dim, 0.5, "Feature neuron number", rotation=90)
            # plot a line x = hidden_dim
            plt.axvline(x=self.meta_params.hidden_dim, color='g', linestyle='--')
            # label this line as "Hidden neuron number"
            plt.text(self.meta_params.hidden_dim, 0.5, "Hidden neuron number", rotation=90)
        else:
            # plot a line x = feature_dim
            plt.axvline(x=self.meta_params.feature_dim, color='r', linestyle='--')
            # label this line as "Feature/Hidden Neuron number"
            plt.text(self.meta_params.feature_dim, 0.5, "Feature/Hidden Neuron number", rotation=90)
        plt.xlabel("Pattern number")
        plt.ylabel("Accuracy")
        # add legend
        plt.legend(["Reconstructed accuracy"])
        plt.title("Accuracy vs Pattern number")
        # save figure
        figure_name = "Acc_vs_Pat_num_" + str(self.meta_params.feature_dim) + "_" + str(self.meta_params.hidden_dim) + ".svg"
        figure_path = self.experiment_folder + "/" + figure_name
        plt.savefig(figure_path)

    def execute_experiment_process(self, parameters, dataset):
        """Execute one memory capacity test."""
        parameters: ExperimentParams = parameters
        
        # generate dataset
        dataset = torch.sign(
            torch.randn(
                parameters.dataset_params.pattern_num,
                parameters.dataset_params.pattern_dim,
            )
        ).to(parameters.dataset_params.device)
        
        # generate predefined labels
        predefined_labels = (
            torch.from_numpy(
                kbits(
                    parameters.dataset_params.label_dim,
                    parameters.dataset_params.label_topk,
                )
            )
            .transpose(0, 1)
            .to(parameters.dataset_params.device)
            .float()
        )

        # create network
        heteroassociation = layers.PseudoinverseLayer(
            parameters.network_params.heteroassociation
        )
        format_conversion = layers.BinaryFormatConversionLayer()
        memory_scaffold = btsp_memory_scaffold.BTSPMemoryScaffoldNetwork(
            parameters.network_params.memory_scaffold
        )
        
        # debug: sample a random btsp weight from (0,1) normal distribution
        random_weights = torch.randn(parameters.dataset_params.label_dim, parameters.dataset_params.hidden_dim).to(parameters.dataset_params.device).float()

        # hidden_states = torch.sign(predefined_labels @ random_weights)
        hidden_states = torch.sign(torch.matmul(predefined_labels, random_weights))
        
        # # debug: implementation by MESH authors
        # m = parameters.dataset_params.label_topk
        # Ng = parameters.network_params.memory_scaffold.hebbian_forward.output_dim # num grid cells, label layer size
        # Npos = nCr(Ng,m)  # number of combinations of patterns that has dimension Ng and m non-zero elements
        # Np = parameters.dataset_params.hidden_dim    # num place cells, hidden layer size
        # Ns = parameters.dataset_params.pattern_dim   # num sensory cells, input layer size
        # sparsity = 1
        # gbook = predefined_labels.to("cpu").numpy().T   # predefined labels
        # sbook = dataset.to("cpu").numpy().T    # features, should have half 1s and half -1s

        # Wpg = random_weights.to("cpu").numpy().T # fixed random projection matrix
        # pbook = hidden_states.to("cpu").numpy().T   #(Np,Npos) predefined hidden states
        # Wgp = train_pcsc(gbook, pbook, Npos) # hebbs rule, (Np,Ng) weight matrix
        
        # Wsp = pseudotrain(sbook, pbook, parameters.dataset_params.pattern_num)
        # Wps = pseudotrain(pbook, sbook, parameters.dataset_params.pattern_num)
        # # print("Wsp shape:", np.shape(Wsp))
        # # print("Wps shape:", np.shape(Wps))
        # # print("sbook shape:", np.shape(sbook)) 
        # # print("pbook shape:", np.shape(pbook))
        # reconstructed_pbook = np.sign(Wps@sbook)
        # reconstructed_sbook = np.sign(Wsp@reconstructed_pbook)
        # # print("reconstructed_pbook shape:", np.shape(reconstructed_pbook))
        # # print("reconstructed_sbook shape:", np.shape(reconstructed_sbook))
        # error = np.sum(np.abs(reconstructed_sbook - sbook))
        # error_rate = error / (parameters.dataset_params.pattern_num * Ns)
        # accuracy = 1 - error_rate

        # pretrain BTSP & obtain predefined hidden states
        # hidden_states = memory_scaffold.pretrain_btsp_forward(predefined_labels)


        # random_weights = torch.sign(random_weights)
        # random_weights = format_conversion.dense_to_sparse(random_weights)
        
        # memory_scaffold.reset_weights(random_weights)
        # self.correct_accuracys.append(accuracy)
        
        # test memory capacity
        # train heteroassociation
        hidden_states_num = hidden_states.shape[0]
        if hidden_states_num < parameters.dataset_params.pattern_num:
            print("Not enough hidden states for training heteroassociation.")
            return
        heteroassociation_output = heteroassociation.learn_and_forward(
            [dataset, hidden_states[: parameters.dataset_params.pattern_num, :], parameters.dataset_params.pattern_num]
        )
        # reconstructed_dataset = heteroassociation.feedback(dense_recalled_hidden_states)
        reconstructed_dataset = heteroassociation.feedback(heteroassociation_output) # for debugging
        
        # calculate error
        error = torch.sum(torch.abs(dataset - reconstructed_dataset)).item()
        error_rate = float(error) / (
            float(parameters.dataset_params.pattern_num)
            * float(parameters.dataset_params.pattern_dim)
        )
        
        self.accuracys.append(1 - error_rate)
        self.pattern_nums.append(parameters.dataset_params.pattern_num)

        # print(f"Error rate: {error_rate}")


if __name__ == "__main__":
    experiment = auto_experiment.SimpleBatchExperiment(BTSPMeshExperiment(), REPEAT_NUM)
    experiment.run()
    experiment.evaluate()
    # print(nCr(18,3))
