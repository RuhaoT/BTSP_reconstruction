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
from custom_networks import b_h_network, hebbian_step, btsp_step_topk

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
REPEAT_NUM = 1
BATCH_SIZE = 30

@dataclasses.dataclass
class MetaParams:
    """Meta parameters for the experiment."""

    pattern_num: int | list
    pattern_dim: int | list
    pattern_fp: float | list
    btsp_fq: float | list
    btsp_fw: float | list
    output_dim: int | list
    btsp_topk: int | list
    feedback_threshold: float | list
    experiment_name: str | list
    data_folder: str | list
    result_name: str | list
    device: str | list

@dataclasses.dataclass
class DatasetParams:
    """Parameters for the dataset."""

    memory_item: int
    memory_dim: int
    fp: float


@dataclasses.dataclass
class ExperimentParams:
    """Parameters for one specific experiment process."""

    network_params: b_h_network.BHNetworkParams
    dataset_params: DatasetParams
    
@dataclasses.dataclass
class ResultInfo:
    """Result information for the experiment."""

    memory_items: pa.int64
    fq: pa.float64
    fp: pa.float64
    mse: pa.float64
    optimal_threshold: pa.float64
    accuracy: pa.float64
    topk: pa.int64


class BTSPOnlyExperiment(ExperimentInterface):
    """BTSP only experiment.

    The only purpose of this experiment is to verify the new experiment framework.
    """

    def __init__(self):
        """Constructor."""
        
        self.meta_params = MetaParams(
            pattern_num=[1,2,3],
            pattern_dim=2500,
            pattern_fp=[0.1,0.2,0.3],
            btsp_fq=[0.001,0.002,0.003],
            btsp_fw=1,
            output_dim=3900,
            btsp_topk=15,
            feedback_threshold=0,
            experiment_name="B-H_network_refactor_test",
            data_folder="data",
            result_name="results.parquet",
            device="cuda",
        )

        # data recording
        self.experiment_folder = logging.init_experiment_folder(
            self.meta_params.data_folder, self.meta_params.experiment_name, timed=False
        )
        
        self.result_schema = logging.dataclass_to_pyarrow_schema(ResultInfo)

        self.data_recorder = logging.ParquetTableRecorder(
            os.path.join(self.experiment_folder, self.meta_params.result_name),
            self.result_schema,
            500,
            True,
        )

    def load_parameters(self):
        """Load the parameters for the experiment."""
        meta_combinations: list[MetaParams] = parameterization.recursive_iterate_dataclass(self.meta_params)
        experiment_params: list[ExperimentParams] = []
        for meta_combination in meta_combinations:
            experiment_param = ExperimentParams(
                network_params=b_h_network.BHNetworkParams(
                    btsp_step_topk_forward=btsp_step_topk.BTSPStepTopKNetworkParams(
                        btsp=layers.BTSPLayerParams(
                            input_dim=meta_combination.pattern_dim,
                            memory_neurons=meta_combination.output_dim,
                            fq=meta_combination.btsp_fq,
                            fw=meta_combination.btsp_fw,
                            device=meta_combination.device,
                        ),
                        btsp_topk=layers.TopKLayerParams(
                            top_k=meta_combination.btsp_topk,
                        ),
                        btsp_topk_step=layers.StepLayerParams(
                            threshold=1e-6,
                        )
                    ),
                    hebbian_step_feedback=hebbian_step.HebbianStepNetworkParams(
                        hebbian=layers.HebbianLayerParams(
                            input_dim=meta_combination.output_dim, # Note this is a feedback layer
                            output_dim=meta_combination.pattern_dim,
                            device=meta_combination.device,
                        ),
                        hebbian_feedback_threshold=layers.StepLayerParams(
                            threshold=meta_combination.feedback_threshold,
                        )
                    ),
                ),
                dataset_params=DatasetParams(
                    memory_item=meta_combination.pattern_num,
                    memory_dim=meta_combination.pattern_dim,
                    fp=meta_combination.pattern_fp,
                ),
            )
            experiment_params.append(experiment_param)
        return experiment_params
    
    def load_dataset(self):
        """No need to load dataset for this experiment."""

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
            for batch in torch.split(dataset, BATCH_SIZE):
                network.learn_and_forward(batch)

            # testing phase
            forward_output = network.forward(dataset)
            reconstructed_output = network.hebbian_feedback_nobinarize(forward_output)
            # grid search for the best feedback threshold
            # find the max element in the reconstructed_output
            max_reconstructed_output = torch.max(reconstructed_output)
            max_reconstructed_output = torch.maximum(
                max_reconstructed_output, torch.tensor(1.0, device="cuda")
            )
            # use granularity 5% of the max element
            candidate_threshold = torch.arange(
                0,
                max_reconstructed_output,
                0.05 * max_reconstructed_output,
                device="cuda",
            )
            min_difference = float("inf")
            best_threshold = 0
            for threshold in candidate_threshold:
                network.hebbian_feedback.hebbian_feedback_threshold.threshold = (
                    threshold
                )
                reconstructed_output = network.reconstruct(forward_output)
                current_difference = torch.sum(reconstructed_output != dataset).item()
                if current_difference < min_difference:
                    min_difference = current_difference
                    best_threshold = threshold.item()
            # calculate accuracy
            accuracy = max(
                1
                - min_difference
                / (
                    parameters.dataset_params.memory_item
                    * parameters.dataset_params.memory_dim
                    * parameters.dataset_params.fp
                ),
                0,
            )
            result = ResultInfo(
                memory_items=[parameters.dataset_params.memory_item],
                fq=[parameters.network_params.btsp_step_topk_forward.btsp.fq],
                fp=[parameters.dataset_params.fp],
                mse=[min_difference],
                optimal_threshold=[best_threshold],
                accuracy=[accuracy],
                topk=[parameters.network_params.btsp_step_topk_forward.btsp_topk.top_k],
            )
            self.data_recorder.record(dataclasses.asdict(result))
            return result

    def summarize_results(self):
        # if the data recorder is not closed, close it
        if self.data_recorder.recording:
            self.data_recorder.close()

        # plot the memory capacity
        plot_memory_capacity(self.experiment_folder, self.meta_params.result_name)


def plot_memory_capacity(experiment_folder, result_name):
    """Plot the memory capacity of the B-H network."""
    # 3D plot the memory capacity
    # 1. load the data
    table = pq.read_table(
        os.path.join(experiment_folder, result_name),
        columns=["memory_items", "fq", "fp", "topk", "accuracy"],
    )

    # 2. merge batch data
    # if the memory items, fp, fq and topk are the same, merge the accuracy
    # use mean as the merge function
    table = table.group_by(["memory_items", "fq", "fp", "topk"]).aggregate(
        [("accuracy", "mean")]
    )

    # 3. plot the 3D graph
    fig = plt.figure()
    # for each different fq, add a subplot
    # find all different fq values
    fq_values = table.column("fq").to_numpy()
    unique_fq_values = np.unique(fq_values)
    subplot_num = len(unique_fq_values)
    subplot_cols = np.ceil(np.sqrt(subplot_num)).astype(int)
    subplot_rows = np.ceil(subplot_num / subplot_cols).astype(int)
    # adjust figure size based on the subplot number
    fig.set_size_inches(6 * subplot_cols, 6 * subplot_rows)
    vmin = 0
    vmax = 1

    for index, fq in enumerate(unique_fq_values):
        # filter the table by topk
        filtered_table = table.filter(table.column("fq") == fq)
        # generate subplot position
        ax = fig.add_subplot(subplot_rows, subplot_cols, index + 1, projection="3d")
        accuracy = filtered_table.column("accuracy_mean").to_numpy()
        error_rate = 1 - accuracy
        ax.plot_trisurf(
            filtered_table.column("fp").to_numpy(),
            filtered_table.column("memory_items").to_numpy(),
            error_rate,
            cmap="coolwarm",
            edgecolor="none",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"fq = {fq}", y=1.0)
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
    # plot_memory_capacity("data/B-H_network_exp_20241103-160609", "results.parquet")
    print("Experiment finished.")
