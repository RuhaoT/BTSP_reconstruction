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
            feature_dim=816,
            hidden_dim=300,
            label_dim=18,
            label_topk=3,
            btsp_fq=0.01,
            btsp_fw=1,
            btsp_topk=200,
            pattern_num=np.arange(1, 816, 10).tolist(),
            device="cuda",
        )
        
        # for debugging
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
        plt.xlabel("Pattern number")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Pattern number")
        plt.show()

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

        # create network
        heteroassociation = layers.PseudoinverseLayer(
            parameters.network_params.heteroassociation
        )
        format_conversion = layers.BinaryFormatConversionLayer()
        memory_scaffold = btsp_memory_scaffold.BTSPMemoryScaffoldNetwork(
            parameters.network_params.memory_scaffold
        )

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

        # pretrain BTSP & obtain predefined hidden states
        # hidden_states = memory_scaffold.pretrain_btsp_forward(predefined_labels)

        # debug: sample a random btsp weight from (0,1) normal distribution
        random_weights = torch.randn(parameters.dataset_params.label_dim, parameters.dataset_params.hidden_dim).to(parameters.dataset_params.device).float()
        # random_weights = torch.sign(random_weights)
        # random_weights = format_conversion.dense_to_sparse(random_weights)
        
        # memory_scaffold.reset_weights(random_weights)
            
        # debug: create a random matrix as hidden states
        hidden_states = torch.sign(torch.randn(parameters.dataset_params.pattern_num, parameters.dataset_params.hidden_dim)).to(parameters.dataset_params.device)

        # test memory capacity
        # train heteroassociation
        hidden_states_num = hidden_states.shape[0]
        if hidden_states_num < parameters.dataset_params.pattern_num:
            print("Not enough hidden states for training heteroassociation.")
            return
        heteroassociation_output = heteroassociation.learn_and_forward(
            [dataset, hidden_states[: parameters.dataset_params.pattern_num, :]]
        )
        # reconstructed_dataset = heteroassociation.feedback(dense_recalled_hidden_states)
        reconstructed_dataset = heteroassociation.feedback(heteroassociation_output) # for debugging
        
        # calculate error
        error = torch.sum(torch.abs(dataset - reconstructed_dataset))
        error_rate = error / (
            parameters.dataset_params.pattern_num
            * parameters.dataset_params.pattern_dim
        )
        
        self.accuracys.append(float(1 - error_rate))
        self.pattern_nums.append(parameters.dataset_params.pattern_num)

        # print(f"Error rate: {error_rate}")


if __name__ == "__main__":
    experiment = auto_experiment.SimpleBatchExperiment(BTSPMeshExperiment(), REPEAT_NUM)
    experiment.run()
    experiment.evaluate()
    # print(nCr(18,3))
