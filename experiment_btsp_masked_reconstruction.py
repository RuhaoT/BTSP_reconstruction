"""Masked reconstruction experiment of BTSP network on simple dataset

This this the first stage 3 experiment, aims to explore the performance of
BTSP-based networks on recalling a example dataset"""

import os
import dataclasses
import numpy as np
import torch
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


REPEAT_NUM = 2
BATCH_SIZE = 35


@dataclasses.dataclass
class MetaParams:
    """Meta parameters for the experiment"""

    # logging parameters
    experiment_name: str | list
    data_folder: str | list
    timed: bool | list

    # network parameters
    input_dim: int | list
    output_dim: int | list  # 300 * (39000/25000)
    btsp_topk: int | list  # 15 * (300/2500) = 1.8
    btsp_fq: float | list
    btsp_fw: float | list

    # dataset parameters
    use_random_dataset: bool | list
    dataset_size: int | list
    data_sparsity: float | list

    # experiment parameters
    flip_ratio: float | list  # noise on each pattern

    # device parameters
    device: str


@dataclasses.dataclass
class ExperimentParams:
    """Parameters for one experiment execution"""

    b_h_network: b_h_network.BHNetworkParams

    input_dim: int
    use_random_dataset: bool
    dataset_size: int
    data_sparsity: float
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
            "BTSPBandMatrixMaskedReconstruction",
            "data",
            False,
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

        self.error_rates = []
        self.pattern_nums = []
        self.flip_ratios = []

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
                            combination.input_dim,
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
                            combination.input_dim,
                            combination.device,
                        ),
                        layers.StepLayerParams(
                            1e-5,
                        ),
                    ),
                ),
                combination.input_dim,
                combination.use_random_dataset,
                combination.dataset_size,
                combination.data_sparsity,
                combination.flip_ratio,
                combination.device,
            )
            experiment_params.append(experiment_param)
        return experiment_params

    def load_dataset(self):
        """Not implemented"""

    def execute_experiment_process(self, parameters: ExperimentParams, dataset):
        """Execute the experiment process."""
        with torch.no_grad():
            # prepare data
            if parameters.use_random_dataset:
                while True:
                    dataset = (
                        torch.rand(
                            size=[
                                parameters.dataset_size,
                                parameters.input_dim,
                            ],
                            device=parameters.device,
                        )
                        < parameters.data_sparsity
                    ).float().to(parameters.device)
                    if (torch.linalg.matrix_rank(dataset)) == parameters.dataset_size:
                        dataset = dataset.bool()
                        break
                
                # create a band matrix with dim max(input_dim, dataset_size)
                # bandwidth = int(parameters.input_dim * parameters.data_sparsity)
                # dataset = torch.zeros(
                #     size=[
                #         parameters.dataset_size,
                #         parameters.input_dim,
                #     ],
                #     device=parameters.device,
                # ).bool()
                # for i in range(parameters.dataset_size):
                #     dataset[i, i:min(i+bandwidth,parameters.input_dim)] = True

                # flip mask
                flip_mask = (
                    torch.rand(
                        size=[
                            parameters.dataset_size,
                            parameters.input_dim,
                        ],
                        device=parameters.device,
                    )
                    < parameters.flip_ratio
                )
            else:
                print("Not implemented")
                exit(1)

            # prepare network
            b_h_network_instance = b_h_network.BHNetwork(parameters.b_h_network)

            # train on clean data
            # batch learning
            batches = torch.split(dataset, BATCH_SIZE)
            for batch in batches:
                b_h_network_instance.learn_and_forward(batch)

            # apply mask
            masked_dataset = dataset.clone()
            masked_dataset = torch.where(flip_mask, ~masked_dataset, masked_dataset)
            actural_flip_ratio = torch.sum(flip_mask) / (
                parameters.input_dim * parameters.dataset_size
            )

            # recall
            masked_forward_output = b_h_network_instance.forward(masked_dataset)
            nobinarized_reconstructed_dataset = b_h_network_instance.hebbian_feedback_nobinarize(masked_forward_output)
            
            # use sparsity based thresholding
            activated_neuron = (
                parameters.dataset_size
                * parameters.input_dim
                * parameters.data_sparsity
            )
            firing_set = torch.topk(
                nobinarized_reconstructed_dataset.flatten(), int(activated_neuron) + 1, largest=True
            )
            min_firing_value = firing_set.values[-2]
            max_resting_value = firing_set.values[-1]
            sparsity_based_threshold = (min_firing_value + max_resting_value) / 2

            # calculate error rate
            sparsity_based_reconstructed_output = (
                nobinarized_reconstructed_dataset > sparsity_based_threshold
            )

            # calculate error
            error = torch.sum(torch.abs(sparsity_based_reconstructed_output != dataset)).item()
            error_rate = min(
                error
                / (
                    parameters.input_dim
                    * parameters.dataset_size
                    * parameters.data_sparsity
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
        plt.show()

        return


if __name__ == "__main__":
    experiment = SimpleBatchExperiment(BTSPMaskedReconstructionExperiment(), REPEAT_NUM)
    experiment.run()
    experiment.evaluate()
