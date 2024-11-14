"""Initial test the mask reconstruction ability of the B-H network on our minimal dataset.
"""

import os
import dataclasses
import numpy as np
import torch
import pyarrow as pa
from pyarrow import parquet as pq
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from experiment_framework.auto_experiment.auto_experiment import (
    SimpleBatchExperiment,
    ExperimentInterface,
)
from experiment_framework.utils import parameterization, logging, layers
from custom_networks import (
    fb_h_network,
    hebbian_step,
    btsp_step_topk,
    fly_hashing_step_topk,
)
from dataset import s3dataset

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
REPEAT_NUM = 1
BATCH_SIZE = 30


@dataclasses.dataclass
class MetaParams:
    """Meta parameters for the experiment."""

    # data parameters
    pattern_num: int | list = 0
    pattern_fp: float | list = 0.0
    coordinate_precision: int | list = 0
    color_precision: int | list = 0
    dataset_file: str = "none"

    # network parameters
    sparse_representation_dim: int | list = 0
    btsp_fq: float | list = 0.0
    btsp_fw: float | list = 0.0
    output_dim: int | list = 0
    btsp_topk: int | list = 0
    feedback_threshold: float | list = 0.0

    fly_hashing_sparsity: float | list = 0.0
    fly_hashing_topk: int | list = 0
    fly_hashing_topk_step: float | list = 0.0

    # experiment parameters
    mask_type: str | list = "none"
    mask_ratio: float | list = 0.0
    experiment_name: str | list = "none"
    data_folder: str | list = "none"
    result_name: str | list = "none"
    device: str | list = "none"


@dataclasses.dataclass
class DatasetParams:
    """Parameters for the dataset."""

    memory_item: int
    memory_dim: int
    fp: float
    coordinate_precision: int
    color_precision: int
    device: str


@dataclasses.dataclass
class ExperimentParams:
    """Parameters for one specific experiment process."""

    # network_params: b_h_network.BHNetworkParams
    network_params: fb_h_network.FBHNetworkParams
    dataset_params: DatasetParams


@dataclasses.dataclass
class ResultInfo:
    """Result information for the experiment."""

    optimal_threshold: float = 0.0
    difference_ratio: float = 0.0



class FBHMinimalDatasetExp(ExperimentInterface):
    """BTSP only experiment.

    The only purpose of this experiment is to verify the new experiment framework.
    """

    def __init__(self):
        """Constructor."""

        self.meta_params = MetaParams(
            pattern_num=100,
            coordinate_precision=8,
            color_precision=8,
            dataset_file="dataset/dataset.csv",
            pattern_fp=0,
            sparse_representation_dim=12500,
            btsp_fq=np.linspace(0.001, 0.01, 10).tolist(),
            btsp_fw=1,
            output_dim=19500,
            btsp_topk=np.arange(1, 201, 10).tolist(),
            feedback_threshold=0,
            fly_hashing_sparsity=np.linspace(0.01, 0.1, 10).tolist(),
            fly_hashing_topk=np.arange(1, 21, 1).tolist(),
            fly_hashing_topk_step=1e-6,
            mask_type="none",
            mask_ratio=0,
            experiment_name="FBH_network_minimal_dataset_exp",
            data_folder="data",
            result_name="results.parquet",
            device="cuda",
        )

        # data recording
        self.experiment_folder = logging.init_experiment_folder(
            self.meta_params.data_folder, self.meta_params.experiment_name, timed=False
        )

        self.result_schema = pa.Table.from_pydict(
            logging.dict_elements_to_tuple(dataclasses.asdict(ResultInfo()))
        ).schema
        
        self.data_schema = pa.Table.from_pydict(
            logging.dict_elements_to_tuple(dataclasses.asdict(MetaParams()))
        ).schema
        
        # for debugging
        # print(pa.unify_schemas([self.result_schema, self.data_schema]))

        self.data_recorder = logging.ParquetTableRecorder(
            os.path.join(self.experiment_folder, self.meta_params.result_name),
            self.data_schema,
            5000,
            True,
        )
        
        self.result_recorder = logging.ParquetTableRecorder(
            os.path.join(self.experiment_folder, self.meta_params.result_name),
            self.result_schema,
            5000,
            False,
        )

        self.dataset = None

        # for debugging
        self.min_diff_ratio = 1
        self.best_reconstructed_dataset = None

    def load_parameters(self):
        """Load the parameters for the experiment."""
        meta_combinations: list[MetaParams] = (
            parameterization.recursive_iterate_dataclass(self.meta_params)
        )
        experiment_params: list[ExperimentParams] = []
        for meta_combination in meta_combinations:
            # network_params=b_h_network.BHNetworkParams(
            #     btsp_step_topk_forward=btsp_step_topk.BTSPStepTopKNetworkParams(
            #         btsp=layers.BTSPLayerParams(
            #             input_dim=meta_combination.pattern_dim,
            #             memory_neurons=meta_combination.output_dim,
            #             fq=meta_combination.btsp_fq,
            #             fw=meta_combination.btsp_fw,
            #             device=meta_combination.device,
            #         ),
            #         btsp_topk=layers.TopKLayerParams(
            #             top_k=meta_combination.btsp_topk,
            #         ),
            #         btsp_topk_step=layers.StepLayerParams(
            #             threshold=1e-6,
            #         )
            #     ),
            #     hebbian_step_feedback=hebbian_step.HebbianStepNetworkParams(
            #         hebbian=layers.HebbianLayerParams(
            #             input_dim=meta_combination.output_dim, # Note this is a feedback layer
            #             output_dim=meta_combination.pattern_dim,
            #             device=meta_combination.device,
            #         ),
            #         hebbian_feedback_threshold=layers.StepLayerParams(
            #             threshold=meta_combination.feedback_threshold,
            #         )
            #     ),
            # ),
            pattern_dim = (
                meta_combination.coordinate_precision * 2
                + meta_combination.color_precision * 3
            )
            experiment_network_params = fb_h_network.FBHNetworkParams(
                fly_hashing_step_topk.FlyHashingStepTopKNetworkParams(
                    layers.FlyHashingLayerParams(
                        input_dim=pattern_dim,
                        output_dim=meta_combination.sparse_representation_dim,
                        sparsity=meta_combination.fly_hashing_sparsity,
                        device=meta_combination.device,
                    ),
                    layers.TopKLayerParams(
                        top_k=meta_combination.fly_hashing_topk,
                    ),
                    layers.StepLayerParams(
                        threshold=meta_combination.fly_hashing_topk_step,
                    ),
                ),
                btsp_step_topk.BTSPStepTopKNetworkParams(
                    layers.BTSPLayerParams(
                        meta_combination.sparse_representation_dim,
                        meta_combination.output_dim,
                        meta_combination.btsp_fq,
                        meta_combination.btsp_fw,
                        meta_combination.device,
                    ),
                    layers.TopKLayerParams(
                        meta_combination.btsp_topk,
                    ),
                    layers.StepLayerParams(
                        threshold=1e-6,
                    ),
                ),
                hebbian_step.HebbianStepNetworkParams(
                    layers.HebbianLayerParams(
                        meta_combination.output_dim,  # Note this is a feedback layer
                        pattern_dim,
                        meta_combination.device,
                    ),
                    layers.StepLayerParams(
                        threshold=meta_combination.feedback_threshold,
                    ),
                ),
            )
            experiment_param = ExperimentParams(
                network_params=experiment_network_params,
                dataset_params=DatasetParams(
                    memory_item=meta_combination.pattern_num,
                    memory_dim=pattern_dim,
                    fp=meta_combination.pattern_fp,
                    coordinate_precision=meta_combination.coordinate_precision,
                    color_precision=meta_combination.color_precision,
                    device=meta_combination.device,
                ),
            )
            experiment_params.append(experiment_param)
            
            # save the meta parameters
            self.data_recorder.record(logging.dict_elements_to_tuple(dataclasses.asdict(meta_combination)))

        # for debugging
        # test = meta_combinations[0]
        # test_dict = dataclasses.asdict(test)
        # print(test_dict)
        # # try converting to schema
        # test_schema = logging.dataclass_to_pyarrow_schema(test)
        return experiment_params

    def load_dataset(self):
        """Load prepaired dataset CSV."""
        self.dataset = s3dataset.MinimalBTSPDataset()
        self.dataset.from_file(self.meta_params.dataset_file)

    def execute_experiment_process(self, parameters: ExperimentParams, dataset):
        with torch.no_grad():
            # initialize the network
            fb_h_network_instance = fb_h_network.FBHNetwork(parameters.network_params)

            # initialize the dataset
            self.dataset.coordinate_precision = (
                parameters.dataset_params.coordinate_precision
            )
            self.dataset.color_precision = parameters.dataset_params.color_precision
            # TODO(Ruhao Tian): convert to binary tensors, use as low precision as possible,
            # because each element is a bit
            dataset_tensors = torch.tensor(
                self.dataset.to_binary_tensors(),
                dtype=torch.float32,
                device=parameters.dataset_params.device,
            )
            dataset_sparsity = self.dataset.calculate_sparsity()

            # train the network
            batches = torch.split(dataset_tensors, BATCH_SIZE)
            for batch in batches:
                fb_h_network_instance.learn_and_forward(batch)

            # TODO(Ruhao Tian): add noise

            # test the network
            forward_output = fb_h_network_instance.forward(dataset_tensors)

            feedback_output = fb_h_network_instance.feedback_nobinarize(forward_output)
            # use sparsity-based thresholding
            # TODO(Ruhao Tian): change to grid search when noise is added
            activated_neurons = int(dataset_sparsity * dataset_tensors.numel())
            top_elements = torch.topk(
                feedback_output.flatten(), activated_neurons + 1, largest=True
            )
            min_firing_value = top_elements.values[-2]
            max_resting_value = top_elements.values[-1]
            threshold = (min_firing_value + max_resting_value) / 2

            reconstructed_output = torch.where(feedback_output > threshold, 1, 0)
            difference_ratio = (
                torch.sum(torch.abs(reconstructed_output - dataset_tensors))
                / dataset_tensors.numel()
            )
            print(f"Threshold: {threshold}, Difference ratio: {difference_ratio}")

            # save parameters and results
            result_info = ResultInfo(
                optimal_threshold=float(threshold),
                difference_ratio=float(difference_ratio),
            )
            result_info_dict = logging.dict_elements_to_tuple(dataclasses.asdict(result_info))
            
            self.result_recorder.record(result_info_dict)

            # for debugging
            if difference_ratio < self.min_diff_ratio:
                self.min_diff_ratio = difference_ratio
                self.best_reconstructed_dataset = s3dataset.MinimalBTSPDataset(
                    self.dataset.coordinate_precision,
                    self.dataset.color_precision,
                ).from_binary_tensors(reconstructed_output.cpu().numpy())
            return

    def summarize_results(self):
        # if the data recorder is not closed, close it
        if self.data_recorder.recording:
            self.data_recorder.close()
        if self.result_recorder.recording:
            self.result_recorder.close()

        # for debugging
        self.best_reconstructed_dataset.plot_dataset(
            display=False,
            save_as=self.experiment_folder + "/best_reconstructed_dataset.svg",
        )

        # plot the memory capacity
        # plot_memory_capacity(self.experiment_folder, self.meta_params.result_name)


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
    # set random seed
    experiment = SimpleBatchExperiment(FBHMinimalDatasetExp(), REPEAT_NUM)
    experiment.run()
    experiment.evaluate()
    # plot_memory_capacity("data/B-H_network_exp_20241103-160609", "results.parquet")
    print("Experiment finished.")
