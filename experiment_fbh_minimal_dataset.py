"""Initial test the mask reconstruction ability of the B-H network on our minimal dataset.
"""

import os
import dataclasses
import numpy as np
import torch
import pyarrow as pa
from pyarrow import parquet as pq
from tqdm import tqdm
import pandas as pd
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
BATCH_SIZE = 100


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
    data_folder: str | list = "none"
    device: str | list = "none"

    # metaparams id
    metaparams_id: int = -1


@dataclasses.dataclass
class DatasetParams:
    """Parameters for the dataset."""

    memory_item: int
    memory_dim: int
    fp: float
    coordinate_precision: int
    color_precision: int
    device: str
    mask_type: str
    mask_ratio: float


@dataclasses.dataclass
class ExperimentParams:
    """Parameters for one specific experiment process."""

    # network_params: b_h_network.BHNetworkParams
    network_params: fb_h_network.FBHNetworkParams
    dataset_params: DatasetParams
    experiment_index: int = -1


@dataclasses.dataclass
class ResultInfo:
    """Result information for the experiment."""

    optimal_threshold: pa.float64
    difference_ratio: pa.float64
    reconstructed_output: pa.binary
    experiment_index: pa.int64


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
            # btsp_fq=0.001,
            btsp_fw=1,
            output_dim=19500,
            btsp_topk=np.arange(1, 201, 10).tolist(),
            # btsp_topk=10,
            feedback_threshold=0,
            fly_hashing_sparsity=np.linspace(0.01, 0.1, 10).tolist(),
            # fly_hashing_sparsity=0.01,
            fly_hashing_topk=np.arange(1, 21, 1).tolist(),
            # fly_hashing_topk=1,
            fly_hashing_topk_step=1e-6,
            mask_type=["colors", "coordinates", "full", "none"],
            mask_ratio=np.linspace(0.0, 0.9, 10).tolist(),
            data_folder="data",
            device="cuda",
        )

        self.experiment_name = "FBH_network_minimal_dataset_exp"
        self.metadata_file_name = "metadatas.parquet"
        self.result_file_name = "results.parquet"

        # data recording
        self.experiment_folder = logging.init_experiment_folder(
            self.meta_params.data_folder, self.experiment_name, timed=False
        )

        self.result_schema = logging.pyarrow_dataclass_to_schema(ResultInfo)

        self.data_schema = pa.Table.from_pydict(
            logging.dict_elements_to_tuple(dataclasses.asdict(MetaParams()))
        ).schema

        # for debugging
        print(pa.unify_schemas([self.result_schema, self.data_schema]))

        self.data_recorder = logging.ParquetTableRecorder(
            os.path.join(self.experiment_folder, self.metadata_file_name),
            self.data_schema,
            5000,
            True,
            "snappy",
        )

        self.result_recorder = logging.ParquetTableRecorder(
            os.path.join(self.experiment_folder, self.result_file_name),
            self.result_schema,
            1000,
            True,
            "snappy",
        )

        self.dataset = None

        # for debugging
        # self.min_diff_ratio = 1
        # self.best_reconstructed_dataset = None

    def load_parameters(self):
        """Load the parameters for the experiment."""
        meta_combinations: list[MetaParams] = (
            parameterization.recursive_iterate_dataclass(self.meta_params)
        )
        experiment_params: list[ExperimentParams] = []
        progess_bar = tqdm(total=len(meta_combinations))
        for index, meta_combination in enumerate(meta_combinations):
            meta_combination.metaparams_id = index
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
                    mask_type=meta_combination.mask_type,
                    mask_ratio=meta_combination.mask_ratio,
                ),
                experiment_index=index,
            )
            experiment_params.append(experiment_param)

            # save the meta parameters
            self.data_recorder.record(
                logging.dict_elements_to_tuple(dataclasses.asdict(meta_combination))
            )
            progess_bar.update(index + 1)

        if self.data_recorder.recording:
            self.data_recorder.close()
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
            # dataset_sparsity = self.dataset.calculate_sparsity()

            # train the network
            batches = torch.split(dataset_tensors, BATCH_SIZE)
            for batch in batches:
                fb_h_network_instance.learn_and_forward(batch)

            # TODO(Ruhao Tian): add noise
            self.dataset.set_mask(
                parameters.dataset_params.mask_type,
                parameters.dataset_params.mask_ratio,
            )
            masked_tensors = torch.tensor(
                self.dataset.to_binary_tensors(masked=True),
                dtype=torch.float32,
                device=parameters.dataset_params.device,
            )

            # test the network
            forward_output = fb_h_network_instance.forward(masked_tensors)

            feedback_output = fb_h_network_instance.feedback_nobinarize(forward_output)
            # use sparsity-based thresholding
            # TODO(Ruhao Tian): change to grid search when noise is added
            max_feedback = torch.max(feedback_output)
            threshold_candidates = torch.linspace(0, max_feedback, 15)
            min_difference = torch.tensor(float("inf"))
            best_threshold = 0
            for threshold in threshold_candidates:
                reconstructed_output = torch.where(feedback_output > threshold, 1, 0)
                difference = torch.sum(
                    torch.abs(reconstructed_output - dataset_tensors)
                )
                if difference < min_difference:
                    min_difference = difference
                    best_threshold = threshold

            reconstructed_output = torch.where(
                feedback_output > best_threshold, True, False
            )
            difference_ratio = (
                torch.sum(torch.abs(reconstructed_output.float() - dataset_tensors))
                / dataset_tensors.numel()
            )
            reconstructed_output_bin = (
                reconstructed_output.cpu().numpy().flatten().tobytes()
            )

            # save parameters and results
            result_info = ResultInfo(
                optimal_threshold=float(best_threshold),
                difference_ratio=float(difference_ratio),
                reconstructed_output=reconstructed_output_bin,
                experiment_index=int(parameters.experiment_index),
            )
            result_info_dict = logging.dict_elements_to_tuple(
                dataclasses.asdict(result_info), ignore_iterable=True
            )

            self.result_recorder.record(result_info_dict)

            # for debugging
            # if difference_ratio < self.min_diff_ratio:
            #     self.min_diff_ratio = difference_ratio
            #     self.best_reconstructed_dataset = s3dataset.MinimalBTSPDataset(
            #         self.dataset.coordinate_precision,
            #         self.dataset.color_precision,
            #     ).from_binary_tensors(reconstructed_output.cpu().numpy())
            return

    def summarize_results(self):
        # if the data recorder is not closed, close it
        if self.data_recorder.recording:
            self.data_recorder.close()
        if self.result_recorder.recording:
            self.result_recorder.close()

        # for debugging
        # self.best_reconstructed_dataset.plot_dataset(
        #     display=False,
        #     save_as=self.experiment_folder + "/best_reconstructed_dataset.svg",
        # )

        # load metaparams and results
        metadatas = pq.read_table(
            os.path.join(self.experiment_folder, self.metadata_file_name)
        )
        results = pq.read_table(
            os.path.join(self.experiment_folder, self.result_file_name)
        )
        full_params = results.join(
            metadatas, keys="experiment_index", right_keys="metaparams_id"
        )
        full_params = full_params.to_pandas()

        # extract best result for each mask type and ratio
        best_results = []
        for mask_type in self.meta_params.mask_type:
            for mask_ratio in self.meta_params.mask_ratio:
                mask_results = full_params[
                    (full_params["mask_type"] == mask_type)
                    & (full_params["mask_ratio"] == mask_ratio)
                ]
                best_result = mask_results.loc[
                    mask_results["difference_ratio"].idxmin()
                ]
                best_results.append(best_result)

        # for each mask type and ratio, plot the best result
        for best_result in best_results:
            result_pattern_length = (
                best_result["coordinate_precision"] * 2
                + best_result["color_precision"] * 3
            )
            reconstructed_output = (
                np.frombuffer(best_result["reconstructed_output"], dtype=np.bool)
                .astype(int)
                .reshape(-1, result_pattern_length)
            )
            reconstructed_dataset = s3dataset.MinimalBTSPDataset(
                best_result["coordinate_precision"],
                best_result["color_precision"],
            ).from_binary_tensors(reconstructed_output)
            # start plotting
            fig = plt.figure(figsize=(10, 5))
            reconstructed_dataset.plot_dataset(figure=fig, subplot_index=121, display=False)
            
            ax = fig.add_subplot(122)
            # turn off the axis
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                (
                    f"Mask type: {best_result['mask_type']}\n"
                    f"Mask ratio: {best_result['mask_ratio']}\n"
                    f"Difference ratio: {best_result['difference_ratio']}\n"
                    f"Optimal threshold: {best_result['optimal_threshold']}\n"
                    f"btsp topk: {best_result['btsp_topk']}\n"
                    f"btsp fq: {best_result['btsp_fq']}\n"
                    f"fly hashing sparsity: {best_result['fly_hashing_sparsity']}\n"
                    f"fly hashing topk: {best_result['fly_hashing_topk']}\n"
                ),
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="center",
                horizontalalignment="center",
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # save the figure
            fig.savefig(
                os.path.join(
                    self.experiment_folder,
                    f"mask_{best_result['mask_type']}_ratio_{best_result['mask_ratio']}.svg",
                )
            )
            # close the figure
            plt.close(fig)


# main function
if __name__ == "__main__":
    # set random seed
    experiment = SimpleBatchExperiment(FBHMinimalDatasetExp(), REPEAT_NUM)
    experiment.run()
    experiment.evaluate()
    print("Experiment finished.")
