"""Layer module for the experiment framework.

This module contains the layer classes for the experiment framework. Layers are
the basic building blocks of the neural network. Each layer are designed via
behavioral classes.

Example:

BTSP_layer = BTSPLayer(input_dim=10, memory_neurons=5, fq=0.5, fw=0.5)
"""

from dataclasses import dataclass
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
    def forward(self, upper_feedback_data):
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
    def weight_reset(self, *args, **kwargs):
        """
        This is the method that reset the weights.
        """


# derived layer classes
@dataclass
class TopKLayerParams:
    """Parameter Dataclass for Top-K layer"""

    top_k: int


class TopKLayer(LayerForward):
    """
    This is the class for the top-k layer.
    """

    def __init__(self, params: TopKLayerParams) -> None:
        """
        This is the constructor of the class.
        """
        self.top_k = params.top_k

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


@dataclass
class StepLayerParams:
    """Parameter Dataclass for Step layer"""

    threshold: float


class StepLayer(LayerForward):
    """
    This is the class for the step layer.
    
    TODO(Ruhao Tian): make this layer type independent.
    """

    def __init__(self, params: StepLayerParams) -> None:
        """
        This is the constructor of the class.
        """
        self.threshold = params.threshold

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        This is the method that performs the forward pass.
        """
        return input_data > self.threshold


@dataclass
class RectifierLayerParams:
    """Parameter Dataclass for Rectifier layer"""

    threshold: float


class RectifierLayer(LayerForward):
    """
    This is the class for the rectifier layer.
    """

    def __init__(self, params: RectifierLayerParams) -> None:
        """
        This is the constructor of the class.
        """
        self.threshold = params.threshold

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        This is the method that performs the forward pass.
        """
        output_data = torch.zeros_like(input_data)
        output_data[input_data > self.threshold] = input_data[
            input_data > self.threshold
        ]
        return output_data


@dataclass
class FlyHashingLayerParams:
    """Parameter Dataclass for Fly-hashing layer"""

    input_dim: int
    output_dim: int
    sparsity: float
    device: str


class FlyHashingLayer(LayerForward, LayerWeightReset):
    """
    This is the class for the fly hashing layer.
    """

    def __init__(self, params: FlyHashingLayerParams) -> None:
        """
        This is the constructor of the class.
        """
        self.input_dim = params.input_dim
        self.output_dim = params.output_dim
        self.sparsity = params.sparsity
        self.device = params.device
        self.weight_reset()

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        This is the method that performs the forward pass.
        """
        output_data = torch.matmul(input_data, self.weights.float())
        return output_data

    def weight_reset(self, *args, **kwargs) -> None:
        """
        This is the method that reset the weights.
        """
        self.weights = (
            torch.rand(self.input_dim, self.output_dim, device=self.device)
            < self.sparsity
        )


@dataclass
class HebbianLayerParams:
    """Parameter Dataclass for Hebbian feedback layer"""

    input_dim: int
    output_dim: int
    device: str
    binary_sparse: bool = False  # broken, keep false


class HebbianLayer(LayerForward, LayerLearn, LayerWeightReset):
    """
    This is the class for the Hebbian feedback layer.

    TODO(Ruhao Tian): Fix the weight saturation issue of binary sparse weights.
    ? Shall we add normalization to the weights?
    """

    def __init__(self, params: HebbianLayerParams) -> None:
        """
        This is the constructor of the class.
        """
        self.input_dim = params.input_dim
        self.output_dim = params.output_dim
        self.device = params.device
        self.binary_sparse = params.binary_sparse
        self.weights = None
        self.weight_reset()

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        This is the method that performs the feedback pass.
        """
        output_data = torch.matmul(input_data.float(), self.weights.float())
        return output_data

    def learn(self, training_data: List) -> None:
        """
        This is the method that performs the learning pass.

        Args:
            input_data (List): The input data, note this input data
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
        hebbian_weight_change = hebbian_weight_change.sum(dim=0)

        if self.binary_sparse:
            hebbian_weight_change = hebbian_weight_change.bool()

            # update the weights
            self.weights = torch.logical_or(self.weights, hebbian_weight_change)

            return
        else:
            # update the weights
            self.weights = self.weights + hebbian_weight_change
            return

    def weight_reset(self, *args, **kwargs) -> None:
        """
        This is the method that reset the weights.
        """
        self.weights = torch.zeros(
            self.input_dim, self.output_dim, device=self.device
        ).float()
        if self.binary_sparse:
            self.weights = self.weights.bool()


@dataclass
class BTSPLayerParams:
    """Parameter Dataclass for BTSP layer"""

    input_dim: int
    memory_neurons: int
    fq: float
    fw: float
    device: str


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

    def __init__(self, params: BTSPLayerParams) -> None:
        """Initialize the layer."""
        self.input_dim = params.input_dim
        self.memory_neurons = params.memory_neurons
        self.fq = params.fq
        self.fw = params.fw
        self.device = params.device
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
            weight_change_sum = (
                weight_change_allowance_synapse * training_data.unsqueeze(2)
            )

            # weight_change_sum is the number of total weight changes for each synapse
            # as weights are binary, the sum is the number of changes
            # shape: (batch_size, input_dim, memory_neurons)
            weight_change_sum = torch.cumsum(weight_change_sum.int(), dim=0) % 2

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

    def weight_reset(self, *args, **kwargs) -> None:
        """Reset the weights."""
        self.weights = (
            torch.rand(self.input_dim, self.memory_neurons, device=self.device)
            < self.fq
        ).bool()
        if "weight" in kwargs:
            if kwargs["weight"] is not None:
                self.weights = kwargs["weight"]
        self.connection_matrix = (
            torch.rand((self.input_dim, self.memory_neurons), device=self.device)
            < self.fw
        ).bool()
        return


@dataclass
class PseudoinverseLayerParams:
    """Parameter Dataclass for Pseudoinverse layer"""

    input_dim: int
    output_dim: int
    device: str


class PseudoinverseLayer(
    LayerForward, LayerFeedback, LayerLearn, LayerLearnForward, LayerWeightReset
):
    """This is the class for the Pseudoinverse layer.

    NOTE: This layer takes in {-1, 1} binary input data.

    Attributes:
        input_dim (int): The input dimension.
        output_dim (int): The output dimension.
        device: The device to deploy the layer.
        weight_forward: The weights of the layer.
        weight_feedback: The weights of the layer.
    """

    def __init__(self, params: PseudoinverseLayerParams) -> None:
        """Initialize the layer."""
        self.input_dim = params.input_dim
        self.output_dim = params.output_dim
        self.device = params.device
        self.weight_forward = None
        self.weight_feedback = None
        self.weight_reset()

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass.

        Simple matrix multiplication between input data and forward weights.
        A sign function is applied to the output data.
        """

        output_data = torch.matmul(input_data.float(), self.weight_forward.float())
        return torch.sign(output_data)

    def feedback(self, upper_feedback_data: torch.Tensor) -> torch.Tensor:
        """Perform the feedback pass.

        Simple matrix multiplication between feedback data and feedback weights.
        A sign function is applied to the output data.
        """

        output_data = torch.matmul(
            upper_feedback_data.float(), self.weight_feedback.float()
        )
        return torch.sign(output_data)

    def weight_reset(self, *args, **kwargs) -> None:
        """Reset the weights.

        Reset both forward and feedback weights.
        """

        if "weight_forward" in kwargs:
            if kwargs["weight_forward"] is not None:
                self.weight_forward = kwargs["weight_forward"]
        else:
            self.weight_forward = torch.rand(
                self.input_dim, self.output_dim, device=self.device
            )

        if "weight_feedback" in kwargs:
            if kwargs["weight_feedback"] is not None:
                self.weight_feedback = kwargs["weight_feedback"]
        else:
            self.weight_feedback = torch.rand(
                self.output_dim, self.input_dim, device=self.device
            )

    def learn(self, training_data: list[torch.Tensor, torch.Tensor, int]) -> None:
        """Perform the learning pass.

        Args:
            training_data (List): The presynaptic, postsynaptic data, and the
                number of patterns. The first element of the list is the
                presynaptic data, the second element of the list is the
                postsynaptic data, and the third element of the list is the
                number of patterns.
        """

        with torch.no_grad():
            # calculate pseudo-inverse matrix
            presynaptic_data = training_data[0].transpose(0, 1).float()
            postsynaptic_data = training_data[1].transpose(0, 1).float()
            pattern_num = training_data[2]

            presynaptic_data_pinv = torch.pinverse(presynaptic_data)
            postsynaptic_data_pinv = torch.pinverse(postsynaptic_data)

            # update the weights
            self.weight_forward = torch.matmul(
                postsynaptic_data[:, :pattern_num],
                presynaptic_data_pinv,
            ).transpose(0, 1)
            self.weight_feedback = torch.matmul(
                presynaptic_data,
                postsynaptic_data_pinv[:pattern_num, :],
            ).transpose(0, 1)
        # scale the weights
        self.weight_forward = self.weight_forward / pattern_num
        self.weight_feedback = self.weight_feedback / pattern_num

    def learn_and_forward(self, training_data):
        """Perform the learning and forward pass.

        Args:
            training_data (List): The presynaptic and postsynaptic data,
                both torch.Tensor. The first element of the list is the
                presynaptic data, and the second element of the list is the
                postsynaptic data.
        """

        self.learn(training_data)
        return self.forward(training_data[0])


class BinaryFormatConversionLayer:
    """This is a special layer for the binary format conversion.

    Provide converting options between {-1, 1} and {0, 1} binary format.
    """

    def __init__(self) -> None:
        """No need to initialize anything."""

    def dense_to_sparse(self, input_data: torch.Tensor) -> torch.Tensor:
        """Convert dense binary format to sparse binary format.

        Args:
            input_data (torch.Tensor): The input data in dense format.

        Returns:
            torch.Tensor: The output data in sparse format.
        """

        return torch.where(input_data > 0, torch.tensor(1), torch.tensor(0))

    def sparse_to_dense(self, input_data: torch.Tensor) -> torch.Tensor:
        """Convert sparse binary format to dense binary format.

        Args:
            input_data (torch.Tensor): The input data in sparse format.

        Returns:
            torch.Tensor: The output data in dense format.
        """

        return torch.where(input_data > 0.5, torch.tensor(1), torch.tensor(-1))
