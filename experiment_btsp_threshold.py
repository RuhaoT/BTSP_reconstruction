"""Explore 2 different ways to set the threshold for the BTSP network.
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
from custom_networks import b_h_network, hebbian_step, btsp_step_topk

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
REPEAT_NUM = 2
BATCH_SIZE = 80


@dataclasses.dataclass
class DatasetParams:
    """Parameters for the dataset."""

    memory_item: int
    memory_dim: int
    fp: float


@dataclasses.dataclass
class ResultInfo:
    """Result information.

    Note: Must use pyarrow compatible types.
    """

    memory_items: pa.int64
    fp: pa.float64
    grid_search_optimal_threshold: pa.float64
    grid_search_error_rate: pa.float64
    sparsity_based_optimal_threshold: pa.float64
    sparsity_based_error_rate: pa.float64


@dataclasses.dataclass
class ExperimentParams:
    """Parameters for one specific experiment process."""

    network_params: b_h_network.BHNetworkParams
    dataset_params: DatasetParams


class BTSPOnlyExperiment(ExperimentInterface):
    """BTSP only experiment.

    The only purpose of this experiment is to verify the new experiment framework.
    """

    def __init__(self):
        """Constructor."""

        self.memory_item = np.arange(100, 25000, 500).tolist()  # number of memory items
        # self.memory_item = [5, 10] # for testing
        self.memory_dim = 2500
        self.fp = np.linspace(
            0.0005, 0.06, 10
        ).tolist()  # sparseness of the EACH memory item, fix to 0.01
        # self.fp = 0.01
        self.fw = 1  # sparseness of the weight matrix
        # self.fq = np.linspace(0.001, 0.01, 10).tolist()
        # 保证3-5neuro激活的基础上越小越好，0.0005/0.001 comparison
        self.fq = 0.01
        self.output_dim = 3900
        self.memory_topk = 15
        self.feedback_threshold = 0  # threshold for the memory neuron to fire
        self.params_network = b_h_network.BHNetworkParams(
            btsp_step_topk_forward=btsp_step_topk.BTSPStepTopKNetworkParams(
                btsp=layers.BTSPLayerParams(
                    input_dim=self.memory_dim,
                    memory_neurons=self.output_dim,
                    fq=self.fq,
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
            hebbian_step_feedback=hebbian_step.HebbianStepNetworkParams(
                hebbian=layers.HebbianLayerParams(
                    input_dim=self.output_dim,  # Note this is a feedback layer
                    output_dim=self.memory_dim,
                    device="cuda",
                ),
                hebbian_feedback_threshold=layers.StepLayerParams(
                    threshold=self.feedback_threshold,
                ),
            ),
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
        self.experiment_name = "B-H_threshold_exp"
        self.data_folder = "data"
        self.result_name = "results.parquet"
        self.params_name = "params.json"

        self.experiment_folder = logging.init_experiment_folder(
            self.data_folder, self.experiment_name
        )
        logging.save_dataclass_to_json(
            self.params, os.path.join(self.experiment_folder, self.params_name)
        )

        self.result_template = logging.dataclass_to_pyarrow_schema(ResultInfo)
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
            network = b_h_network.BHNetwork(parameters.network_params)
            # learning phase
            # batch learning
            batch_size = min(BATCH_SIZE, parameters.dataset_params.memory_item)
            for batch in range(0, parameters.dataset_params.memory_item, batch_size):
                network.learn_and_forward(
                    dataset[
                        batch : min(
                            batch + batch_size,
                            parameters.dataset_params.memory_item,
                        )
                    ]
                )
            # testing phase
            forward_output = network.forward(dataset)
            reconstructed_output = network.hebbian_feedback_nobinarize(forward_output)

            # find the max element in the reconstructed_output
            max_reconstructed_output = torch.max(reconstructed_output)

            # Method 1: grid search for the best feedback threshold
            # grid search for the best feedback threshold
            grid_search_ceiling = torch.maximum(
                max_reconstructed_output, torch.tensor(1.0, device="cuda")
            )
            # use granularity 5% of the max element
            candidate_threshold = torch.arange(
                0,
                grid_search_ceiling,
                0.05 * grid_search_ceiling,
                device="cuda",
            )
            min_difference = float("inf")
            grid_search_best_threshold = 0
            for threshold in candidate_threshold:
                grid_search_result = reconstructed_output > threshold
                current_difference = torch.sum(grid_search_result != dataset).item()
                if current_difference < min_difference:
                    min_difference = current_difference
                    grid_search_best_threshold = threshold.item()
            # calculate error rate
            grid_search_error_rate = min_difference / (
                parameters.dataset_params.memory_item
                * parameters.dataset_params.memory_dim
                * parameters.dataset_params.fp
            )
            grid_search_error_rate = min(grid_search_error_rate, 1.0)

            # Method 2: sparsity based topk filter
            activated_neuron = (
                parameters.dataset_params.memory_item
                * parameters.dataset_params.memory_dim
                * parameters.dataset_params.fp
            )
            firing_set = torch.topk(
                reconstructed_output.flatten(), int(activated_neuron) + 1, largest=True
            )
            min_firing_value = firing_set.values[-2]
            max_resting_value = firing_set.values[-1]
            sparsity_based_threshold = (min_firing_value + max_resting_value) / 2

            # calculate error rate
            sparsity_based_reconstructed_output = (
                reconstructed_output > sparsity_based_threshold
            )
            sparsity_based_error_rate = torch.sum(
                sparsity_based_reconstructed_output != dataset
            ).item() / (
                parameters.dataset_params.memory_item
                * parameters.dataset_params.memory_dim
                * parameters.dataset_params.fp
            )
            sparsity_based_error_rate = min(sparsity_based_error_rate, 1.0)

            result = ResultInfo(
                memory_items=parameters.dataset_params.memory_item,
                fp=parameters.dataset_params.fp,
                grid_search_optimal_threshold=float(grid_search_best_threshold),
                grid_search_error_rate=float(grid_search_error_rate),
                sparsity_based_optimal_threshold=float(sparsity_based_threshold),
                sparsity_based_error_rate=float(sparsity_based_error_rate),
            )
            result = dataclasses.asdict(result)
            result = {key: [value] for key, value in result.items()}
            self.data_recorder.record(result)
            return result

    def summarize_results(self):
        # if the data recorder is not closed, close it
        if self.data_recorder.recording:
            self.data_recorder.close()

        # plot the memory capacity
        plot_error_difference(self.experiment_folder, self.result_name)


def plot_error_difference(experiment_folder, result_name):
    """Plot the error difference between 2 approaches of setting the threshold."""
    # 1. load the data, load all columns
    table = pq.read_table(os.path.join(experiment_folder, result_name))

    # 2. merge batch data
    # use mean as the merge function
    table = table.group_by(["memory_items", "fp"]).aggregate(
        [("grid_search_error_rate", "mean"), ("sparsity_based_error_rate", "mean")]
    )
    # set error rate to 1 if it is larger than 1
    table = table.set_column(
        table.schema.get_field_index("grid_search_error_rate_mean"),
        "grid_search_error_rate_mean",
        pa.array(np.clip(table["grid_search_error_rate_mean"].to_numpy(), 0, 1)),
    )
    table = table.set_column(
        table.schema.get_field_index("sparsity_based_error_rate_mean"),
        "sparsity_based_error_rate_mean",
        pa.array(np.clip(table["sparsity_based_error_rate_mean"].to_numpy(), 0, 1)),
    )

    # 3. plot the 3D graph
    fig = plt.figure()
    # adjust figure size
    fig.set_size_inches(12, 12)
    gs = fig.add_gridspec(2,4)
    # set limits
    error_rate_min = 0
    error_rate_max = 1

    # 3.1 plot the grid search error rate
    ax_1 = fig.add_subplot(gs[0,:2], projection="3d")
    # get the data
    x = table["memory_items"].to_numpy()
    y = table["fp"].to_numpy()
    z = table["grid_search_error_rate_mean"].to_numpy()
    # plot the data
    ax_1.plot_trisurf(
        x,
        y,
        z,
        cmap="coolwarm",
        edgecolor="none",
        vmin=error_rate_min,
        vmax=error_rate_max,
    )
    ax_1.invert_xaxis()
    # set labels
    ax_1.set_xlabel("Memory Items")
    ax_1.set_ylabel("Sparsity")
    ax_1.set_zlabel("Error Rate")
    ax_1.set_title("Grid Search Error Rate")
    # set the color bar

    # 3.2 plot the sparsity based error rate
    ax_2 = fig.add_subplot(gs[0,2:], projection="3d")
    # get the data
    x = table["memory_items"].to_numpy()
    y = table["fp"].to_numpy()
    z = table["sparsity_based_error_rate_mean"].to_numpy()
    # plot the data
    ax_2.plot_trisurf(
        x,
        y,
        z,
        cmap="coolwarm",
        edgecolor="none",
        vmin=error_rate_min,
        vmax=error_rate_max,
    )
    ax_2.invert_xaxis()
    # set labels
    ax_2.set_xlabel("Memory Items")
    ax_2.set_ylabel("Sparsity")
    ax_2.set_zlabel("Error Rate")
    ax_2.set_title("Sparsity Based Error Rate")

    # 3.3 plot the error difference
    ax_3 = fig.add_subplot(gs[1,-3:-1], projection="3d")
    # get the data
    x = table["memory_items"].to_numpy()
    y = table["fp"].to_numpy()
    z = (
        table["sparsity_based_error_rate_mean"].to_numpy()
        - table["grid_search_error_rate_mean"].to_numpy()
    )
    # create a cmap, where PiYG with 0 as the center
    # use data min and max as boundaries
    data_min = np.min(z)
    data_max = np.max(z)
    norm = mpl.colors.TwoSlopeNorm(vmin=data_min, vcenter=0, vmax=data_max)
    # plot the data
    ax_3.plot_trisurf(x, y, z, cmap="PiYG", edgecolor="none", norm=norm)
    ax_3.invert_xaxis()
    # set labels
    ax_3.set_xlabel("Memory Items")
    ax_3.set_ylabel("Sparsity")
    ax_3.set_zlabel("Sparsity Based - Grid Search Error Rate")
    ax_3.set_title("Error Rate Difference")

    # add color bar, one for the difference and one for the error rate
    # create a scalar mappable
    sm = plt.cm.ScalarMappable(cmap="PiYG", norm=norm)
    # add color bar for the difference
    cbar = plt.colorbar(sm, ax=ax_3, pad=0.15)
    cbar.set_label("Error Rate Difference")
    # add color bar for the error rate
    error_rate_norm = mpl.colors.Normalize(vmin=error_rate_min, vmax=error_rate_max)
    error_rate_sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=error_rate_norm)
    cbar = plt.colorbar(error_rate_sm, ax=[ax_1, ax_2], pad=0.075)
    cbar.set_label("Error Rate")

    # save the plot
    plt.savefig(
        os.path.join(experiment_folder, "error_difference_test.svg"),
        transparent=True,
    )


# main function
if __name__ == "__main__":
    # experiment = SimpleBatchExperiment(BTSPOnlyExperiment(), REPEAT_NUM)
    # experiment.run()
    # experiment.evaluate()
    plot_error_difference("data/B-H_threshold_exp_20241106-213527", "results.parquet")
    print("Experiment finished.")
