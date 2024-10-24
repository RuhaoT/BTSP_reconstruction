"""This module is for the Hebbian step feedback network."""

from dataclasses import dataclass
import torch
from experiment_framework.utils import layers


@dataclass
class HebbianStepFeedbackNetworkParams:
    """Parameters for Hebbian step feedback network."""

    hebbian: layers.HebbianFeedbackLayerParams
    hebbian_feedback_threshold: layers.StepLayerParams


class HebbianStepFeedbackNetwork:
    """Hebbian step feedback network."""

    def __init__(self, params: HebbianStepFeedbackNetworkParams) -> None:
        """Constructor."""
        self.hebbian = layers.HebbianFeedbackLayer(params.hebbian)
        self.hebbian_feedback_threshold = layers.StepLayer(
            params.hebbian_feedback_threshold
        )
        self.reset_weights()
        
    def feedback_nobinarize(self, input_data: torch.Tensor):
        """Feedback the Hebbian layer."""
        hebbian_output = self.hebbian.feedback(input_data)
        return hebbian_output

    def feedback(self, input_data: torch.Tensor):
        """Feedback the Hebbian layer."""
        hebbian_output = self.feedback_nobinarize(input_data)
        return self.hebbian_feedback_threshold.forward(hebbian_output)

    def learn(self, input_data: torch.Tensor):
        """Learn the input data."""
        self.hebbian.learn(input_data)

    def reset_weights(self):
        """Reset the weights."""
        self.hebbian.weight_reset()
