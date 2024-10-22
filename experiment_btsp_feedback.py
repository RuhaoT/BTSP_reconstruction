"""Initial test with new experiment framework.

Intended to test the new experiment framework with the BTSPOnly network.
"""

import os
import numpy as np
import torch
import pyarrow as pa
from pyarrow import parquet as pq
import matplotlib.pyplot as plt
from experiment_framework.auto_experiment.auto_experiment import (
    SimpleBatchExperiment,
    ExperimentInterface,
)
from experiment_framework.utils import parameterization, logging
import custom_networks.simple_btsp_feedback

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class BTSPOnlyExperiment(ExperimentInterface):
    """BTSP only experiment.

    The only purpose of this experiment is to verify the new experiment framework.
    """

    def __init__(self):
        """Constructor."""
        self.memory_item = np.arange(10, 200, 10).tolist()  # number of memory items
        self.memory_dim = 2500
        self.fp = np.linspace(
            0.0005, 0.01, 20
        ).tolist()  # sparseness of the EACH memory item
        self.fw = 1  # sparseness of the weight matrix
        self.output_dim = 3900
        self.memory_topk = int(
            self.output_dim * 0.5
        )  # topk for the memory neuron to fire
        self.feedback_threshold = 0  # threshold for the memory neuron to fire
        self.params_btsp = {
            "input_dim": self.memory_dim,
            "memory_neurons": self.output_dim,
            "fq": np.linspace(0.001, 0.01, 10).tolist(),
            "fw": self.fw,
            "device": "cuda",
        }
        self.params_hebbian = {
            "input_dim": self.memory_dim,
            "output_dim": self.output_dim,
            "device": "cuda",
        }
        self.params_dataset = {
            "fp": self.fp,
            "memory_item": self.memory_item,
            "memory_dim": self.memory_dim,
        }
        self.params = {
            "btsp": self.params_btsp,
            "btsp_topk": {"topk": self.memory_topk},
            "hebbian": self.params_hebbian,
            "hebbian_feedback_threshold": {"threshold": self.feedback_threshold},
            "dataset": self.params_dataset,
        }

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
        }
        self.result_schema = pa.schema(self.result_template)

        self.data_recorder = logging.ParquetTableRecorder(
            os.path.join(self.experiment_folder, self.result_name),
            self.result_schema,
            500,
        )

    def load_parameters(self):
        return parameterization.recursive_iterate_dict(self.params)

    def load_dataset(self):
        return None

    def execute_experiment_process(self, parameters, dataset):
        with torch.no_grad():
            dataset = (
                torch.rand(
                    size=[
                        parameters["dataset"]["memory_item"],
                        parameters["dataset"]["memory_dim"],
                    ],
                    device="cuda",
                )
                < parameters["dataset"]["fp"]
            )

            network = custom_networks.simple_btsp_feedback.SimpleBTSPFeedbackNetwork(
                parameters
            )

            # learning phase
            # batch learning
            batch_size = min(25, parameters["dataset"]["memory_item"])

            for batch in range(0, parameters["dataset"]["memory_item"], batch_size):
                network.learn_and_forward(
                    dataset[
                        batch : min(
                            batch + batch_size, parameters["dataset"]["memory_item"]
                        )
                    ]
                )

            # testing phase
            forward_output = network.forward(dataset)
            reconstructed_output = network.hebbian_feedback(forward_output)

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
                network.hebbian_feedback_threshold.threshold = threshold
                reconstructed_output = network.reconstruct(forward_output)
                current_difference = torch.sum(reconstructed_output != dataset).item()
                if current_difference < min_difference:
                    min_difference = current_difference
                    best_threshold = threshold.item()

            # calculate accuracy
            accuracy = 1 - min_difference / (
                parameters["dataset"]["memory_item"]
                * parameters["dataset"]["memory_dim"]
                * parameters["dataset"]["fp"]
            )

            result = {
                "Memory Items": [parameters["dataset"]["memory_item"]],
                "fq": [parameters["btsp"]["fq"]],
                "fp": [parameters["dataset"]["fp"]],
                "MSE": [min_difference],
                "Optimal threshold": [best_threshold],
                "Accuracy": [accuracy],
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
        columns=["Memory Items", "fq", "fp", "Accuracy"],
    )

    # 2. merge batch data
    # if the memory items, fp and fq are the same, merge the accuracy
    # use mean as the merge function
    table = table.group_by(["Memory Items", "fq", "fp"]).aggregate(
        [("Accuracy", "mean")]
    )

    # find optimal accuracy for the fq dimension
    table = table.group_by(["Memory Items", "fp"]).aggregate([("Accuracy_mean", "max")])

    # 3. convert to numpy array and calculate memory capacity
    memory_items = table["Memory Items"].to_numpy()  # Y axis
    fp = table["fp"].to_numpy()  # X axis
    accuracy = table["Accuracy_mean_max"].to_numpy()
    error_rate = 1 - accuracy

    # 4. plot the 3D graph
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        fp,
        memory_items,
        error_rate,
        cmap="coolwarm",
        edgecolor="none",
        linewidth=0,
        antialiased=False,
    )

    ax.invert_xaxis()

    # Customize the axes labels
    ax.set(xlabel="fp", ylabel="Memory Items", zlabel="Error Rate")

    # Customize the title and color bar
    ax.set_title("Memory Capacity for B-H Network")
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # save the plot
    plt.savefig(
        os.path.join(experiment_folder, "memory_capacity_test.svg"),
        transparent=True,
    )


# main function
if __name__ == "__main__":
    # experiment = SimpleBatchExperiment(BTSPOnlyExperiment(), 5)
    # experiment.run()
    # experiment.evaluate()
    plot_memory_capacity("data/B-H_network_exp_20241020-005928", "results.parquet")
    print("Experiment finished.")
