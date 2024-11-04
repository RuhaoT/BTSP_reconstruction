"""One-layer Fly-hashing network with Hebbian feedback."""

from dataclasses import dataclass
import torch
from custom_networks import fly_hashing_step_topk, hebbian_step

@dataclass
class FHNetworkParams:
    """Parameters for Fly-hashing feedback network."""
    fly_hashing_forward: fly_hashing_step_topk.FlyHashingStepTopKNetworkParams
    hebbian_feedback: hebbian_step.HebbianStepNetworkParams

class FHNetwork:
    """One-layer Fly-hashing network with Hebbian feedback."""

    def __init__(self, params: FHNetworkParams) -> None:
        """Constructor."""
        self.fly_hashing_forward = fly_hashing_step_topk.FlyHashingStepTopKNetwork(
            params.fly_hashing_forward
        )
        self.hebbian_feedback = hebbian_step.HebbianStepNetwork(
            params.hebbian_feedback
        )
        self.reset_weights()

    def reset_weights(self):
        """Reset the weights."""
        self.fly_hashing_forward.reset_weights()
        self.hebbian_feedback.reset_weights()

    def forward(self, input_data: torch.Tensor):
        """Forward pass."""
        forward_output = self.fly_hashing_forward.forward(input_data)
        return forward_output

    def learn_and_forward(self, input_data: torch.Tensor):
        """Forward pass and learning."""
        forward_output = self.forward(input_data)
        self.hebbian_feedback.learn([forward_output, input_data])
        return forward_output

    def hebbian_feedback_nobinarize(self, input_data: torch.Tensor):
        """Feedback the hebbian layer."""
        return self.hebbian_feedback.forward_nobinarize(input_data)

    def reconstruct(self, input_data: torch.Tensor):
        """Reconstruct the input data."""
        return self.hebbian_feedback.forward(input_data)
