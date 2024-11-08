"""Simple BTSP network with Hebbian feedback layer.
"""

from dataclasses import dataclass
import torch
from custom_networks import btsp_step_topk, hebbian_step

@dataclass
class BHNetworkParams:
    """Parameters for simple BTSP feedback network."""
    btsp_step_topk_forward: btsp_step_topk.BTSPStepTopKNetworkParams
    hebbian_step_feedback: hebbian_step.HebbianStepNetworkParams

class BHNetwork:
    """simple one-layer BTSP network."""

    def __init__(self, params: BHNetworkParams) -> None:
        """Constructor."""
        self.btsp_forward = btsp_step_topk.BTSPStepTopKNetwork(
            params.btsp_step_topk_forward
        )
        self.hebbian_feedback = hebbian_step.HebbianStepNetwork(
            params.hebbian_step_feedback
        )
        self.reset_weights()

    def forward(self, input_data: torch.Tensor):
        """Forward pass."""
        btsp_output = self.btsp_forward.forward(input_data)
        return btsp_output

    def learn_and_forward(self, input_data: torch.Tensor):
        """Forward pass and learning."""
        btsp_output = self.btsp_forward.learn_and_forward(input_data)
        self.hebbian_feedback.learn([btsp_output, input_data])
        return btsp_output

    def reset_weights(self, new_btsp_weights=None):
        """Reset the weights."""
        self.btsp_forward.reset_weights(new_btsp_weights)
        self.hebbian_feedback.reset_weights()

    def hebbian_feedback_nobinarize(self, input_data: torch.Tensor):
        """Feedback the hebbian layer."""
        return self.hebbian_feedback.forward_nobinarize(input_data)

    def reconstruct(self, input_data: torch.Tensor):
        """Reconstruct the input data."""
        hebbian_output = self.hebbian_feedback.forward(input_data)
        return hebbian_output
