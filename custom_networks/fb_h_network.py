"""Module for FB-H network.

The FB-H network is a two-forward-module network with single feedback connections.
The first module is a Fly-hashing network, and the second layer is a BTSP network.
An Hebbian feedback layer is added with an end-to-end approach.
"""

import dataclasses
import torch
from custom_networks import fly_hashing_step_topk, btsp_step_topk, hebbian_step

@dataclasses.dataclass
class FBHNetworkParams:
    """Params for FB-H network"""

    fly_hashing_forward: fly_hashing_step_topk.FlyHashingStepTopKNetworkParams
    btsp_forward: btsp_step_topk.BTSPStepTopKNetworkParams
    hebbian_feedback: hebbian_step.HebbianStepNetworkParams


class FBHNetwork:
    """Fly-Hashing BTSP network with end-to-end Hebbian feedback"""

    def __init__(self, params: FBHNetworkParams) -> None:
        self.fly_hashing_forward = fly_hashing_step_topk.FlyHashingStepTopKNetwork(
            params.fly_hashing_forward
        )
        self.btsp_forward = btsp_step_topk.BTSPStepTopKNetwork(params.btsp_forward)
        self.hebbian_feedback = hebbian_step.HebbianStepNetwork(params.hebbian_feedback)
        self.reset_weights()

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Complete forward pass"""
        fly_hashing_output = self.fly_hashing_forward.forward(input_data)
        return self.btsp_forward.forward(fly_hashing_output)

    def feedback_nobinarize(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feedback pass without setting firing threshold"""
        return self.hebbian_feedback.forward_nobinarize(input_data)

    # TODO(Ruhao Tian): Refine the name 'reconstruct' as normally it involves both forward and feedback pass
    def reconstruct(self, input_data: torch.Tensor) -> torch.Tensor:
        """feedback pass"""
        return self.hebbian_feedback.forward(input_data)

    def learn_and_forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Forward pass and learning"""
        fly_hashing_output = self.fly_hashing_forward.forward(input_data)
        btsp_output = self.btsp_forward.forward(fly_hashing_output)
        self.hebbian_feedback.learn([btsp_output, input_data])
        return btsp_output

    def reset_weights(self, new_btsp_weights=None):
        """Reset all weights"""
        self.fly_hashing_forward.reset_weights()
        self.btsp_forward.reset_weights(new_btsp_weights)
        self.hebbian_feedback.reset_weights()
