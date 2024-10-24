"""Module for FB-2H network.

The FB-2H network is a two-module network with feedback connections. The first
module is a Fly-hashing feedback network, and the second layer is a BTSP feedback
network.
"""

from dataclasses import dataclass
import torch
from custom_networks import f_h_network, simple_btsp_feedback_deprecated

@dataclass
class FB2HNetworkParams:

class FB2HNetwork:
    """FB-2H network."""

    def __init__(self, params: dict) -> None:
        """Constructor."""
        self.fly_hashing_feedback = f_h_network.FHNetwork(
            params["F-H"]
        )
        self.simple_btsp_feedback = simple_btsp_feedback_deprecated.SimpleBTSPFeedbackNetwork(
            params["B-H"]
        )
        self.reset_weights()

    def reset_weights(self):
        """Reset the weights."""
        self.fly_hashing_feedback.reset_weights()
        self.simple_btsp_feedback.reset_weights()

    def forward(self, input_data: torch.Tensor):
        """Forward pass."""
        fly_hashing_output = self.fly_hashing_feedback.forward(input_data)
        simple_btsp_output = self.simple_btsp_feedback.forward(fly_hashing_output)
        return simple_btsp_output

    def learn_and_forward(self, input_data: torch.Tensor):
        """Forward pass and learning."""
        fly_hashing_output = self.fly_hashing_feedback.learn_and_forward(input_data)
        simple_btsp_output = self.simple_btsp_feedback.learn_and_forward(
            fly_hashing_output
        )
        return simple_btsp_output

    def reconstruct(self, input_data: torch.Tensor):
        """Reconstruct the input data."""
        simple_btsp_output = self.simple_btsp_feedback.reconstruct(input_data)
        fly_hashing_output = self.fly_hashing_feedback.reconstruct(simple_btsp_output)
        return fly_hashing_output

    def feedback_btsp(self, input_data: torch.Tensor):
        """Feedback the BTSP layer."""
        return self.simple_btsp_feedback.hebbian_feedback(input_data)

    def feedback_fh(self, input_data: torch.Tensor):
        """Feedback the Fly-hashing layer with the reconstructed data for B-H."""
        btsp_reconstruction = self.simple_btsp_feedback.reconstruct(input_data)
        return self.fly_hashing_feedback.hebbian_feedback(btsp_reconstruction)
