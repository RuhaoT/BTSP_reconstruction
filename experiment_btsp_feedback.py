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
import custom_networks.b_h_network
import custom_networks.btsp_step_topk
import custom_networks.hebbian_step
from experiment_framework.auto_experiment.auto_experiment import (
    SimpleBatchExperiment,
    ExperimentInterface,
)
from experiment_framework.utils import parameterization, logging, layers
import custom_networks

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
REPEAT_NUM = 1
BATCH_SIZE = 25

@dataclasses.dataclass
class DatasetParams:
    """Parameters for the dataset."""
    memory_item: int
    memory_dim: int
    fp: float

@dataclasses.dataclass
class ExperimentParams:
    """Parameters for one specific experiment process."""
    network_params: custom_networks.b_h_network.BHNetworkParams
    dataset_params: DatasetParams

class BTSPOnlyExperiment(ExperimentInterface):
    """BTSP only experiment.

    The only purpose of this experiment is to verify the new experiment framework.
    """

    def __init__(self):
        """Constructor."""

        self.memory_item = np.arange(10, 200, 50).tolist()  # number of memory items
        # self.memory_item = 200
        self.memory_dim = 2500
        self.fp = np.linspace(
            0.0005, 0.01, 4
        ).tolist()  # sparseness of the EACH memory item
        self.fw = 1  # sparseness of the weight matrix
        self.output_dim = 3900
        self.memory_topk = 15
        self.feedback_threshold = 0  # threshold for the memory neuron to fire
        self.params_network = custom_networks.b_h_network.BHNetworkParams(
            btsp_step_topk_forward=custom_networks.btsp_step_topk.BTSPStepTopKNetworkParams(
                btsp= layers.BTSPLayerParams(
                    input_dim=self.memory_dim,
                    memory_neurons=self.output_dim,
                    fq=np.linspace(0.001, 0.01, 10).tolist(),
                    fw=self.fw,
                    device="cuda",
                ),
                btsp_topk=layers.TopKLayerParams(
                    top_k=self.memory_topk,
                ),
                btsp_topk_step=layers.StepLayerParams(
                    threshold=self.feedback_threshold,
                ),
            ),
            hebbian_step_feedback=custom_networks.hebbian_step.HebbianStepFeedbackNetworkParams(
                hebbian=layers.HebbianFeedbackLayerParams(
                    input_dim=self.memory_dim,
                    output_dim=self.output_dim,
                    device="cuda",
                ),
                hebbian_feedback_threshold=layers.StepLayerParams(
                    threshold=self.feedback_threshold,
                ),
            )
        )
        self.params_dataset = DatasetParams(
            memory_item=self.memory_item,
            memory_dim=self.memory_dim,
            fp=self.fp,
        )
        self.params = ExperimentParams(
            network_params=self.params_network,
            dataset_params=self.params_dataset,
        )

        # data recording
        self.experiment_name = "B-H_network_exp"
        self.data_folder = "data"
        self.result_name = "results.parquet"

        self.experiment_folder = logging.init_experiment_folder(
            self.data_folder, self.experiment_name
        )

        self.result_template = {
            "fq": pa.float64(),
            "fp": pa.float64(),
            "Memory Items": pa.int64(),
            "MSE": pa.int64(),
            "Optimal threshold": pa.float64(),
            "Accuracy": pa.float64(),
            "Topk": pa.int64(),
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
            )

            network = custom_networks.b_h_network.BHNetwork(parameters.network_params)

            # learning phase
            # batch learning
            batch_size = min(BATCH_SIZE, parameters.dataset_params.memory_item)

            for batch in range(0, parameters.dataset_params.memory_item, batch_size):
                network.learn_and_forward(
                    dataset[
                        batch : min(
                            batch + batch_size, parameters.dataset_params.memory_item
                        )
                    ]
                )

            # testing phase
            forward_output = network.forward(dataset)
            reconstructed_output = network.hebbian_feedback_nobinarize(forward_output)

            # grid search for the best feedback threshold
            # find the max element in the reconstructed_output
            max_reconstructed_output = torch.max(reconstructed_output).item()
            max_reconstructed_output = max(max_reconstructed_output, 1)
            # use granularity 5% of the max element
            candidate_threshold = torch.arange(
                0, max_reconstructed_output, 0.05 * max_reconstructed_output
            )
            min_difference = float("inf")
            best_threshold = 0
            for threshold in candidate_threshold:
                network.hebbian_feedback.hebbian_feedback_threshold.threshold = threshold
                reconstructed_output = network.reconstruct(forward_output)
                current_difference = torch.sum(reconstructed_output != dataset).item()
                if current_difference < min_difference:
                    min_difference = current_difference
                    best_threshold = threshold.item()

            # calculate accuracy
            accuracy = 1 - min_difference / (
                parameters.dataset_params.memory_item
                * parameters.dataset_params.memory_dim
                * parameters.dataset_params.fp
            )

            result = {
                "Memory Items": [parameters.dataset_params.memory_item],
                "fq": [parameters.network_params.btsp_step_topk_forward.btsp.fq],
                "fp": [parameters.dataset_params.fp],
                "MSE": [min_difference],
                "Optimal threshold": [best_threshold],
                "Accuracy": [accuracy],
                "Topk": [parameters.network_params.btsp_step_topk_forward.btsp_topk.top_k],
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
    """Plot the memory capacity of the B-H network."""
    # 3D plot the memory capacity
    # 1. load the data
    table = pq.read_table(
        os.path.join(experiment_folder, result_name),
        columns=["Memory Items", "fq", "fp", "Topk", "Accuracy"],
    )

    # 2. merge batch data
    # if the memory items, fp, fq and topk are the same, merge the accuracy
    # use mean as the merge function
    table = table.group_by(["Memory Items", "fq", "fp", "Topk"]).aggregate(
        [("Accuracy", "mean")]
    )

    # find optimal accuracy for the fq dimension
    table = table.group_by(["Memory Items", "fp", "Topk"]).aggregate(
        [("Accuracy_mean", "max")]
    )

    # 3. plot the 3D graph
    fig = plt.figure()
    # for each different topk, add a subplot
    # find all different topk values
    topk_values = table.column("Topk").to_numpy()
    unique_topk_values = np.unique(topk_values)
    subplot_num = len(unique_topk_values)
    subplot_cols = np.ceil(np.sqrt(subplot_num)).astype(int)
    subplot_rows = np.ceil(subplot_num / subplot_cols).astype(int)
    # adjust figure size based on the subplot number
    fig.set_size_inches(6 * subplot_cols, 6 * subplot_rows)
    vmin = 0
    vmax = 1

    for index, topk in enumerate(unique_topk_values):
        # filter the table by topk
        filtered_table = table.filter(table.column("Topk") == topk)
        # generate subplot position
        ax = fig.add_subplot(subplot_rows, subplot_cols, index + 1, projection="3d")
        accuracy = filtered_table.column("Accuracy_mean_max").to_numpy()
        error_rate = 1 - accuracy
        ax.plot_trisurf(
            filtered_table.column("fp").to_numpy(),
            filtered_table.column("Memory Items").to_numpy(),
            error_rate,
            cmap="coolwarm",
            edgecolor="none",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Topk = {topk}", y=1.0)
        ax.set_xlabel("fp")
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
    experiment = SimpleBatchExperiment(BTSPOnlyExperiment(), REPEAT_NUM)
    experiment.run()
    experiment.evaluate()
    # plot_memory_capacity("data/B-H_network_exp_20241023-113402", "results.parquet")
    print("Experiment finished.")
