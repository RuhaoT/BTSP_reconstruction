"""Layer module for the experiment framework.

This module contains the layer classes for the experiment framework. Layers are
the basic building blocks of the neural network. Each layer are designed via
behavioral classes.

Example:

BTSP_layer = BTSPLayer(input_dim=10, memory_neurons=5, fq=0.5, fw=0.5)
"""

from abc import ABC, abstractmethod
from typing import List
import torch


# Layer Behavior
class LayerForward(ABC):
    """
    This is the abstract class for the layer forward behavior.
    """

    @abstractmethod
    def forward(self, input_data):
        """
        This is the method that performs the forward pass.
        """


class LayerFeedback(ABC):
    """
    This is the abstract class for the layer feedback behavior.
    """

    @abstractmethod
    def feedback(self, upper_feedback_data):
        """
        This is the method that performs the feedback pass.
        """


# TODO(Ruhao Tian): Better abstraction for the layer behavior
class LayerLearn(ABC):
    """
    This is the abstract class for the layer learn behavior.
    """

    @abstractmethod
    def learn(self, training_data):
        """
        This is the method that performs the learning pass.
        """


class LayerLearnForward(ABC):
    """
    This is the abstract class for the layer learn forward behavior.
    """

    @abstractmethod
    def learn_and_forward(self, training_data):
        """
        This is the method that performs the learning and forward pass.
        """


class LayerWeightReset(ABC):
    """
    This is the abstract class for the layer weight reset behavior.
    """

    @abstractmethod
    def weight_reset(self):
        """
        This is the method that reset the weights.
        """


# derived layer classes
class TopKLayer(LayerForward):
    """
    This is the class for the top-k layer.
    """

    def __init__(self, top_k: int) -> None:
        """
        This is the constructor of the class.
        """
        self.top_k = top_k

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        This is the method that performs the forward pass.
        """
        selected_values, selected_indices = torch.topk(
            input_data, self.top_k, dim=-1, sorted=False
        )
        output_data = torch.zeros_like(input_data)
        output_data.scatter_(-1, selected_indices, selected_values)
        return output_data


class StepLayer(LayerForward):
    """
    This is the class for the step layer.
    """

    def __init__(self, threshold: float) -> None:
        """
        This is the constructor of the class.
        """
        self.threshold = threshold

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        This is the method that performs the forward pass.
        """
        return input_data > self.threshold


class RectifierLayer(LayerForward):
    """
    This is the class for the rectifier layer.
    """

    def __init__(self, threshold: float) -> None:
        """
        This is the constructor of the class.
        """
        self.threshold = threshold

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        This is the method that performs the forward pass.
        """
        output_data = torch.zeros_like(input_data)
        output_data[input_data > self.threshold] = input_data[
            input_data > self.threshold
        ]
        return output_data


class FlyHashingLayer(LayerForward, LayerWeightReset):
    """
    This is the class for the fly hashing layer.
    """

    def __init__(
        self, input_dim: int, output_dim: int, sparsity: float, device="cpu"
    ) -> None:
        """
        This is the constructor of the class.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.device = device
        self.weights = None
        self.weight_reset()

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        This is the method that performs the forward pass.
        """
        output_data = torch.matmul(input_data, self.weights)
        return output_data

    def weight_reset(self) -> None:
        """
        This is the method that reset the weights.
        """
        self.weights = (
            torch.rand(self.input_dim, self.output_dim, device=self.device)
            < self.sparsity
        )


class HebbianFeedbackLayer(LayerFeedback, LayerLearn, LayerWeightReset):
    """
    This is the class for the Hebbian feedback layer.
    """

    def __init__(self, input_dim: int, output_dim: int, device="cpu") -> None:
        """
        This is the constructor of the class.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        # note the weights are stored in an transposed way
        self.weights = None
        self.weight_reset()

    def feedback(self, upper_feedback_data: torch.Tensor) -> torch.Tensor:
        """
        This is the method that performs the feedback pass.
        """
        output_data = torch.matmul(upper_feedback_data.float(), self.weights.float())
        return output_data

    def learn(self, training_data: List) -> None:
        """
        This is the method that performs the learning pass.

        Args:
            input_data (torch.Tensor): The input data, note this input data
                requires special format. The first element of the list is the
                presynaptic data, and the second element of the list is the
                postsynaptic data. For each tensor, the first dimension is the
                batch dimension, and data is store in the second dimension as
                1-d tensors.
        """

        # calculate hebbian weight change
        hebbian_weight_change = torch.bmm(
            training_data[0].unsqueeze(2).float(), training_data[1].unsqueeze(1).float()
        )

        # calculate final hebbian weight change
        hebbian_weight_change = hebbian_weight_change.sum(dim=0).bool()

        # update the weights
        self.weights = torch.logical_or(
            self.weights, torch.transpose(hebbian_weight_change, 0, 1)
        )

    def weight_reset(self) -> None:
        """
        This is the method that reset the weights.
        """
        self.weights = torch.zeros(
            self.output_dim, self.input_dim, device=self.device
        ).bool()


class BTSPLayer(LayerForward, LayerLearn, LayerLearnForward, LayerWeightReset):
    """This is the class for BTSP layer.

    Attributes:
        input_dim (int): The input dimension.
        memory_neurons (int): The number of memory neurons.
        fq: plateau potential possibility
        fw: connection ratio between neurons
        device: The device to deploy the layer.
        weights: The weights of the layer.
        connection_matrix: The matrix describing which neurons are connected.
    """

    def __init__(
        self,
        input_dim: int,
        memory_neurons: int,
        fq: float,
        fw: float,
        device="cpu",
    ) -> None:
        """Initialize the layer."""
        self.input_dim = input_dim
        self.memory_neurons = memory_neurons
        self.fq = fq
        self.fw = fw
        self.device = device
        self.weights = None
        self.connection_matrix = None
        self.weight_reset()

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass."""
        output_data = torch.matmul(input_data.float(), self.weights.float())
        return output_data

    def learn_and_forward(self, training_data: torch.Tensor) -> torch.Tensor:
        """One-shot learning while forward pass.

        Args:
            training_data (torch.Tensor): The training data, the same as normal
                 input data.
        """

        fq_half = self.fq / 2

        with torch.no_grad():
            # plateau weight change possibility document if each neuron has weight
            # change possibiltiy when receiving a memory item
            # shape: (batch_size, memory_neurons)
            plateau_weight_change_possibility = (
                torch.rand(
                    training_data.shape[0], self.memory_neurons, device=self.device
                )
                < fq_half
            )

            # plateau weight change synapse document if each synapse has plateau
            # potential when receiving memory item
            plateau_weight_change_synapse = plateau_weight_change_possibility.unsqueeze(
                1
            )

            # weight change allowance synapse document if each synapse
            # satisfies the plateau potential condition and the connection matrix
            # shape: (batch_size, input_dim, memory_neurons)
            weight_change_allowance_synapse = (
                plateau_weight_change_synapse * self.connection_matrix
            )

            # weight_change_sequence is a binary matrix, indicating the update of
            # each weight during the training process
            weight_change_sequence = (
                weight_change_allowance_synapse * training_data.unsqueeze(2)
            )

            # weight_change_sum is the number of total weight changes for each synapse
            # as weights are binary, the sum is the number of changes
            # shape: (batch_size, input_dim, memory_neurons)
            weight_change_sum = torch.cumsum(weight_change_sequence.int(), dim=0) % 2

            # weight sequence is the weight after each training data
            weight_sequence = torch.where(
                weight_change_sum > 0, ~self.weights, self.weights
            )

            # update the weights
            # final weight is stored in the last element of the weight_sequence
            self.weights = weight_sequence[-1]

            # calculate output DURING learning
            # shape: (batch_size, memory_neurons)
            output_data = torch.bmm(
                training_data.unsqueeze(1).float(), weight_sequence.float()
            )

            # remove the neuron dimension
            output_data = output_data.squeeze(1)
            return output_data

    def learn(self, training_data: torch.Tensor) -> None:
        """This is basically the same as learn_and_forward-.

        TODO(Ruhao Tian): refactor this to avoid code duplication.
        """
        self.learn_and_forward(training_data)

    def weight_reset(self) -> None:
        """Reset the weights."""
        self.weights = torch.zeros(
            (self.input_dim, self.memory_neurons), device=self.device
        ).bool()
        self.connection_matrix = (
            torch.rand((self.input_dim, self.memory_neurons), device=self.device)
            < self.fw
        ).bool()