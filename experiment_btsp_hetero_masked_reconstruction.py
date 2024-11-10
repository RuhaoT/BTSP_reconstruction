"""Masked reconstruction experiment of BTSP network with hetero-associative preprocessing

This this the first stage 3 experiment, aims to explore the performance of
BTSP-based networks on recalling a example dataset"""

import os
import dataclasses
import tqdm
import numpy as np
import torch
import torch.linalg
import pyarrow as pa
from pyarrow import parquet as pq
import matplotlib as mpl
import matplotlib.pyplot as plt
from experiment_framework.auto_experiment.auto_experiment import (
    SimpleBatchExperiment,
    ExperimentInterface,
)
from experiment_framework.utils import logging, parameterization, layers
from dataset import s3dataset
from custom_networks import b_h_network, hebbian_step, btsp_step_topk


REPEAT_NUM = 1
BATCH_SIZE = 35


@dataclasses.dataclass
class MetaParams:
    """Meta parameters for the experiment"""

    # logging parameters
    experiment_name: str | list
    data_folder: str | list
    timed: bool | list

    # network parameters
    feature_dim: int | list
    hidden_dim: int | list
    output_dim: int | list  # 300 * (39000/25000)
    btsp_topk: int | list  # 15 * (300/2500) = 1.8
    btsp_fq: float | list
    btsp_fw: float | list

    # dataset parameters
    use_random_dataset: bool | list
    dataset_size: int | list
    hidden_sparsity: float | list

    # experiment parameters
    flip_ratio: float | list  # noise on each pattern

    # device parameters
    device: str


@dataclasses.dataclass
class ExperimentParams:
    """Parameters for one experiment execution"""

    b_h_network: b_h_network.BHNetworkParams
    
    hetero_learning: layers.PseudoinverseLayerParams

    feature_dim: int
    hidden_dim: int
    use_random_dataset: bool
    dataset_size: int
    hidden_sparsity: float
    flip_ratio: float

    device: str

@dataclasses.dataclass
class ResultInfo:
    """Result information"""

    error_rate: pa.float64
    pattern_num: pa.int64
    flip_ratio: pa.float64


class BTSPMaskedReconstructionExperiment(ExperimentInterface):
    """BTSP masked reconstruction experiment"""

    def __init__(self) -> None:
        """Constructor."""
        self.meta_params = MetaParams(
            "BTSPMaskedReconstructionExperiment",
            "data",
            False,
            88,
            2500,
            3900,
            15,
            0.01,
            1.0,
            True,
            np.arange(10, 200, 10).tolist(),
            0.01,
            np.linspace(0.0, 0.4, 10).tolist(),
            "cuda",
        )

        self.experiment_folder = logging.init_experiment_folder(
            self.meta_params.data_folder,
            self.meta_params.experiment_name,
            self.meta_params.timed,
        )
        
        log_schema = logging.dataclass_to_pyarrow_schema(ResultInfo)
        
        self.logger = logging.ParquetTableRecorder(
            os.path.join(self.experiment_folder, "results.parquet"),
            log_schema,
            batch_size=2000,
        )

        self.candidate_hidden_states = None
    def load_parameters(self):
        """Load the parameters."""
        combinations: list[MetaParams] = parameterization.recursive_iterate_dataclass(
            self.meta_params
        )
        experiment_params = []
        for combination in combinations:
            experiment_param = ExperimentParams(
                b_h_network.BHNetworkParams(
                    btsp_step_topk.BTSPStepTopKNetworkParams(
                        layers.BTSPLayerParams(
                            combination.hidden_dim,
                            combination.output_dim,
                            combination.btsp_fq,
                            combination.btsp_fw,
                            combination.device,
                        ),
                        layers.TopKLayerParams(
                            combination.btsp_topk,
                        ),
                        layers.StepLayerParams(
                            1e-5,
                        ),
                    ),
                    hebbian_step.HebbianStepNetworkParams(
                        layers.HebbianLayerParams(
                            combination.output_dim, # feedback layer input dim
                            combination.hidden_dim,
                            combination.device,
                        ),
                        layers.StepLayerParams(
                            1e-5,
                        ),
                    ),
                ),
                layers.PseudoinverseLayerParams(
                    combination.feature_dim,
                    combination.hidden_dim,
                    combination.device,
                ),
                combination.feature_dim,
                combination.hidden_dim,
                combination.use_random_dataset,
                combination.dataset_size,
                combination.hidden_sparsity,
                combination.flip_ratio,
                combination.device,
            )
            experiment_params.append(experiment_param)
        return experiment_params

    def load_dataset(self):
        """Not implemented"""
        # find a list of hidden states that are full rank
        # hidden states should be full rank
        format_converter = layers.BinaryFormatConversionLayer()
        candidate_hidden_states = []
        progress = tqdm.tqdm(total=10)
        print("Finding candidate hidden states")
        while len(candidate_hidden_states) < 10:
            hidden_states = (torch.rand(
                size=[
                    self.meta_params.hidden_dim,
                    self.meta_params.hidden_dim,
                ],
                device=self.meta_params.device,
            ) < self.meta_params.hidden_sparsity).to(self.meta_params.device)
            dense_hidden_states = format_converter.sparse_to_dense(hidden_states.float())
            if torch.linalg.matrix_rank(dense_hidden_states.float()) > 0.5*self.meta_params.hidden_dim:
                candidate_hidden_states.append(hidden_states)
                progress.update(1)
        self.candidate_hidden_states = candidate_hidden_states

    def execute_experiment_process(self, parameters: ExperimentParams, dataset):
        """Execute the experiment process."""
        with torch.no_grad():
            # prepare network
            b_h_network_instance = b_h_network.BHNetwork(parameters.b_h_network)
            hetero_learning_instance = layers.PseudoinverseLayer(parameters.hetero_learning)
            format_converter = layers.BinaryFormatConversionLayer()

            # prepare data
            if parameters.use_random_dataset:
                dataset = torch.sign(
                    torch.randn(
                        size=[
                            parameters.dataset_size,
                            parameters.feature_dim,
                        ],
                        device=parameters.device,
                    )
                ).to(parameters.device)

                # flip mask
                flip_mask = (
                    torch.rand(
                        size=[
                            parameters.dataset_size,
                            parameters.feature_dim,
                        ],
                        device=parameters.device,
                    )
                    < parameters.flip_ratio
                )
            else:
                print("Not implemented")
                exit(1)
                
            # prepare hidden states
            # pick a random hidden state
            hidden_states = self.candidate_hidden_states[np.random.randint(0, 10)]
            
            # train scaffold on hidden states
            # batch learning
            batches = torch.split(hidden_states, BATCH_SIZE)
            for batch in batches:
                b_h_network_instance.learn_and_forward(batch)
            
            # train hetero-associative memory on clean dataset
            dense_hidden_states = format_converter.sparse_to_dense(hidden_states)
            hetero_learning_instance.learn([dataset, dense_hidden_states, parameters.dataset_size])

            # apply mask
            masked_dataset = dataset.clone()
            masked_dataset = torch.where(flip_mask, -masked_dataset, masked_dataset)
            actural_flip_ratio = torch.sum(flip_mask) / (
                parameters.feature_dim * parameters.dataset_size
            )

            # forward
            masked_hetero_output = hetero_learning_instance.forward(masked_dataset)
            
            # reconstruct hidden states
            sparse_masked_hetero_output = format_converter.dense_to_sparse(masked_hetero_output)
            sparse_btsp_output = b_h_network_instance.forward(sparse_masked_hetero_output)
            reconstructed_sparse_hidden_nobin = b_h_network_instance.reconstruct(sparse_btsp_output)
            # find max value for grid search
            max_value = torch.max(reconstructed_sparse_hidden_nobin)
            
            # grid search
            # use 5% granularity
            candidates = torch.linspace(0, max_value, 20, device=parameters.device)
            errors = torch.empty(20, device=parameters.device)
            for index, threshold_candidate in enumerate(candidates):
                reconstructed_sparse_hidden = reconstructed_sparse_hidden_nobin > threshold_candidate
                reconstructed_dense_hidden = format_converter.sparse_to_dense(reconstructed_sparse_hidden)
                reconstructed_output = hetero_learning_instance.feedback(reconstructed_dense_hidden)
                # calculate error
                error = torch.sum(torch.abs(reconstructed_output != dataset))
                errors[index] = error
            min_error = torch.min(errors)
            
            error_rate = min(
                min_error
                / (
                    parameters.feature_dim
                    * parameters.dataset_size
                    * parameters.hidden_sparsity
                ),
                1.0,
            )

            # record results
            result = ResultInfo(
                [float(error_rate)],
                [int(parameters.dataset_size)],
                [float(actural_flip_ratio)],
            )
            result = dataclasses.asdict(result)
            self.logger.record(result)

        return

    def summarize_results(self):
        """Summarize the results."""
        # stop recording
        if self.logger.recording:
            self.logger.close()
            
        # load results
        table = pq.read_table(os.path.join(self.experiment_folder, "results.parquet"))
        
        # use mean as the merge function
        table = table.group_by(["pattern_num", "flip_ratio"]).aggregate(
        [("error_rate", "mean")]
        )
        
        # plot error rate
        x = np.array(table["pattern_num"])
        y = np.array(table["flip_ratio"])
        z = np.array(table["error_rate_mean"])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(x, y, z, cmap="coolwarm", edgecolor="none")
        ax.invert_xaxis()
        ax.set_xlabel("Pattern number")
        ax.set_ylabel("Flip ratio")
        ax.set_zlabel("Error rate")
        ax.set_title("Error rate of BTSP masked reconstruction")
        # save the plot
        plt.savefig(os.path.join(self.experiment_folder, "error_rate.svg"))

        return


if __name__ == "__main__":
    experiment = SimpleBatchExperiment(BTSPMaskedReconstructionExperiment(), REPEAT_NUM)
    experiment.run()
    experiment.evaluate()
