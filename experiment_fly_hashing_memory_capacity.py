"""Initial test with new experiment framework.

Intended to test the new experiment framework with the BTSPOnly network.
"""

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
from experiment_framework.utils import parameterization, logging, layers
from custom_networks import f_h_network, fly_hashing_step_topk, hebbian_step

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
REPEAT_NUM = 2
BATCH_SIZE = 30


@dataclasses.dataclass
class DatasetParams:
    """Parameters for the dataset."""

    memory_item: int
    memory_dim: int
    fp: float


@dataclasses.dataclass
class ExtraParams:
    """Extra parameters for the experiment."""

    sample_per_kc: int
    weight_sparsity: float
    feedback_threshold: float


@dataclasses.dataclass
class ExperimentParams:
    """Parameters for one specific experiment process."""

    network_params: f_h_network.FHNetworkParams
    dataset_params: DatasetParams


class FlyHashingMemCapExperiment(ExperimentInterface):
    """BTSP only experiment.

    The only purpose of this experiment is to verify the new experiment framework.
    """

    def __init__(self):
        """Constructor."""

        self.memory_item = np.arange(
            500, 30000, 3000
        ).tolist()  # number of memory items
        # self.memory_item = 200
        self.input_dim = 88  # dimension of the memory item
        self.fp = 0.5
        self.output_dim = 2500
        self.output_sparsity = np.linspace(
            0.005, 0.3, 16
        )  # target sparsity of the output
        self.memory_topk = [
            int(x) for x in (self.output_dim * np.array(self.output_sparsity))
        ]
        self.device = "cuda"
        self.kc_sample = np.arange(2, 12, 1).tolist()  # input synapses for KC cells
        total_synapses = [x * self.output_dim for x in self.kc_sample]
        self.weight_sparsity = [
            (x / self.output_dim / self.input_dim) for x in total_synapses
        ]
        print(len(self.weight_sparsity))
        self.feedback_threshold = 0  # threshold for the memory neuron to fire
        self.params_network = f_h_network.FHNetworkParams(
            fly_hashing_forward=fly_hashing_step_topk.FlyHashingStepTopKNetworkParams(
                fly_hashing=layers.FlyHashingLayerParams(
                    input_dim=self.input_dim,
                    output_dim=self.output_dim,
                    sparsity=self.weight_sparsity,
                    device=self.device,
                ),
                fly_hashing_topk=layers.TopKLayerParams(
                    top_k=self.memory_topk,
                ),
                fly_hashing_topk_step=layers.StepLayerParams(
                    threshold=1e-5,
                ),
            ),
            hebbian_feedback=hebbian_step.HebbianStepNetworkParams(
                hebbian=layers.HebbianLayerParams(
                    input_dim=self.output_dim,  # for feedback
                    output_dim=self.input_dim,
                    device=self.device,
                ),
                hebbian_feedback_threshold=layers.StepLayerParams(
                    threshold=self.feedback_threshold,
                ),
            ),
        )
        self.params_dataset = DatasetParams(
            memory_item=self.memory_item,
            memory_dim=self.input_dim,
            fp=self.fp,
        )
        self.params = ExperimentParams(
            network_params=self.params_network,
            dataset_params=self.params_dataset,
        )

        # data recording
        self.experiment_name = "F-H_mem_cap_exp"
        self.data_folder = "data"
        self.result_name = "results.parquet"

        self.experiment_folder = logging.init_experiment_folder(
            self.data_folder, self.experiment_name
        )

        self.result_template = {
            "Memory Items": pa.int64(),
            "Output Sparsity": pa.float64(),
            "Sample per KC": pa.float64(),
            "Sparsity-based Threshold": pa.float64(),
            "Grid Search Threshold": pa.float64(),
            "Sparsity Error Rate": pa.float64(),
            "Grid Search Error Rate": pa.float64(),
        }
        self.result_schema = pa.schema(self.result_template)

        self.data_recorder = logging.ParquetTableRecorder(
            os.path.join(self.experiment_folder, self.result_name),
            self.result_schema,
            500,
        )

    def load_parameters(self):
        return parameterization.recursive_iterate_dataclass(self.params)

    def load_dataset(self):
        return None

    def execute_experiment_process(self, parameters: ExperimentParams, dataset):
        with torch.no_grad():
            dataset = (
                torch.rand(
                    size=[
                        parameters.dataset_params.memory_item,
                        parameters.dataset_params.memory_dim,
                    ],
                    device="cuda",
                )
                < parameters.dataset_params.fp
            ).float()

            network = f_h_network.FHNetwork(parameters.network_params)

            # batch learning
            for i in range(0, parameters.dataset_params.memory_item, BATCH_SIZE):
                memory_batch = dataset[
                    i : min(i + BATCH_SIZE, parameters.dataset_params.memory_item)
                ]
                network.learn_and_forward(memory_batch)

            # test
            data_output = network.forward(dataset)
            feedback_nobinarize = network.hebbian_feedback_nobinarize(data_output)
            max_feedback = torch.max(feedback_nobinarize)

            # find the optimal threshold, using 2 approaches
            # 1. sparsity-based topk filtering
            avg_activated_neuro = int(
                parameters.dataset_params.memory_dim
                * parameters.dataset_params.memory_item
                * parameters.dataset_params.fp
            )
            sparsity_based_threshold = (
                torch.topk(
                    feedback_nobinarize.flatten(), avg_activated_neuro, largest=True
                ).values[-1]
                - 1e-5
            )

            # test difference error
            sparsity_based_reconstruction = (
                feedback_nobinarize > sparsity_based_threshold
            )
            sparsity_error = torch.sum(sparsity_based_reconstruction != dataset)
            sparsity_error_rate = torch.min(sparsity_error / avg_activated_neuro, 1)

            # 2. grid search
            # use 5% granularity
            threshold_candidates = torch.linspace(0, max_feedback, 20)
            mse = torch.zeros_like(threshold_candidates)
            for i, threshold in enumerate(threshold_candidates):
                feedback = feedback_nobinarize > threshold
                mse[i] = torch.sum(feedback != dataset)
            grid_search_threshold = threshold_candidates[torch.argmin(mse)]
            grid_search_error = torch.min(mse)
            grid_search_error_rate = torch.min(grid_search_error / avg_activated_neuro, 1)

            result = {
                "Memory Items": [parameters.dataset_params.memory_item],
                "Output Sparsity": [
                    parameters.network_params.fly_hashing_forward.fly_hashing_topk.top_k
                    / parameters.network_params.fly_hashing_forward.fly_hashing.output_dim
                ],
                "Sample per KC": [
                    parameters.network_params.fly_hashing_forward.fly_hashing.sparsity
                    * parameters.network_params.fly_hashing_forward.fly_hashing.input_dim
                ],
                "Sparsity-based Threshold": [sparsity_based_threshold.item()],
                "Grid Search Threshold": [grid_search_threshold.item()],
                "Sparsity Error Rate": [sparsity_error_rate.item()],
                "Grid Search Error Rate": [grid_search_error_rate.item()],
            }

            self.data_recorder.record(result)
            return result

    def summarize_results(self):
        # if the data recorder is not closed, close it
        if self.data_recorder.recording:
            self.data_recorder.close()

        # plot the memory capacity
        plot_memory_capacity(self.experiment_folder, self.result_name)


def plot_memory_capacity(experiment_folder, result_name):
    """Plot the memory capacity of the F-H network."""
    # 3D plot the memory capacity
    # 1. load the data
    table = pq.read_table(
        os.path.join(experiment_folder, result_name),
        columns=[
            "Memory Items",
            "Output Sparsity",
            "Sample per KC",
            "Grid Search Error Rate",
        ],
    )

    # 2. merge batch data
    # if the memory items, fp, fq and topk are the same, merge the accuracy
    # use mean as the merge function
    table = table.group_by(
        ["Memory Items", "Output Sparsity", "Sample per KC"]
    ).aggregate([("Grid Search Error Rate", "mean")])

    # 3. plot the 3D graph
    fig = plt.figure()
    # for each different fq, add a subplot
    # find all different fq values
    output_sparsity_values = table.column("Output Sparsity").to_numpy()
    unique_output_sparsity_levels = np.unique(output_sparsity_values)
    subplot_num = len(unique_output_sparsity_levels)
    subplot_cols = np.ceil(np.sqrt(subplot_num)).astype(int)
    subplot_rows = np.ceil(subplot_num / subplot_cols).astype(int)
    # adjust figure size based on the subplot number
    fig.set_size_inches(6 * subplot_cols, 6 * subplot_rows)
    vmin = 0
    vmax = 1

    for index, output_sparsity_value in enumerate(unique_output_sparsity_levels):
        # filter the table by topk
        filtered_table = table.filter(
            table.column("Output Sparsity") == output_sparsity_value
        )
        # generate subplot position
        ax = fig.add_subplot(subplot_rows, subplot_cols, index + 1, projection="3d")
        error_rate = filtered_table.column("Grid Search Error Rate_mean").to_numpy()
        ax.plot_trisurf(
            filtered_table.column("Sample per KC").to_numpy(),
            filtered_table.column("Memory Items").to_numpy(),
            error_rate,
            cmap="coolwarm",
            edgecolor="none",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Output Sparsity = {output_sparsity_value}", y=1.0)
        ax.set_xlabel("Sample per KC")
        ax.set_ylabel("Memory Items")
        ax.set_zlabel("Error Rate")
        ax.invert_xaxis()

    plt.subplots_adjust(hspace=0.4)

    # Add a color bar at the right side of the whole image
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="coolwarm"), cax=cbar_ax)
    cbar.set_label("Error Rate")
    # plt.show()

    # Customize the axes labels
    # ax.set(xlabel="fp", ylabel="Memory Items", zlabel="Error Rate")

    # Customize the title and color bar
    # ax.set_title("Memory Capacity for B-H Network")
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # save the plot
    plt.savefig(
        os.path.join(experiment_folder, "memory_capacity_test.svg"),
        transparent=True,
    )


# main function
if __name__ == "__main__":
    experiment = SimpleBatchExperiment(FlyHashingMemCapExperiment(), REPEAT_NUM)
    experiment.run()
    experiment.evaluate()
    # plot_memory_capacity("data/B-H_network_exp_20241103-160609", "results.parquet")
    print("Experiment finished.")
