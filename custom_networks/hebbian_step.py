"""This module is for the Hebbian step feedback network."""

from dataclasses import dataclass
import torch
from experiment_framework.utils import layers


@dataclass
class HebbianStepNetworkParams:
    """Parameters for Hebbian step feedback network."""

    hebbian: layers.HebbianLayerParams
    hebbian_feedback_threshold: layers.StepLayerParams


class HebbianStepNetwork:
    """Hebbian step feedback network."""

    def __init__(self, params: HebbianStepNetworkParams) -> None:
        """Constructor."""
        self.hebbian = layers.HebbianLayer(params.hebbian)
        self.hebbian_feedback_threshold = layers.StepLayer(
            params.hebbian_feedback_threshold
        )
        self.reset_weights()

    def forward_nobinarize(self, input_data: torch.Tensor):
        """Forward the Hebbian layer."""
        hebbian_output = self.hebbian.forward(input_data)
        return hebbian_output

    def forward(self, input_data: torch.Tensor):
        """Forward the Hebbian layer."""
        hebbian_output = self.forward_nobinarize(input_data)
        return self.hebbian_feedback_threshold.forward(hebbian_output)

    def learn(self, input_data: list):
        """Learn the input data."""
        self.hebbian.learn(input_data)

    def reset_weights(self):
        """Reset the weights."""
        self.hebbian.weight_reset()
