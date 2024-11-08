"""A memory scaffold network that use BTSP layer instead of RP layer.

This network is inspired from "Content Addressable Memory Without Catastrophic
Forgetting by Heteroassociation with a Fixed Scaffold"
by Sugandha Sharma, Sarthak Chandra, and Ila R. Fiete
"""

from dataclasses import dataclass
import torch
import custom_networks
from experiment_framework.utils import layers

@dataclass
class BTSPMemoryScaffoldNetworkParams:
    """Parameters for BTSP memory scaffold network."""
    hebbian_forward: layers.HebbianLayerParams
    hebbian_topk: layers.TopKLayerParams
    btsp_feedback: custom_networks.btsp_step_topk.BTSPStepTopKNetworkParams
    
class BTSPMemoryScaffoldNetwork:
    """BTSP memory scaffold network."""
    
    def __init__(self, params: BTSPMemoryScaffoldNetworkParams) -> None:
        """Constructor."""
        self.hebbian_forward = layers.HebbianLayer(params.hebbian_forward)
        self.hebbian_topk = layers.TopKLayer(params.hebbian_topk)
        self.hebbian_step = layers.StepLayer(layers.StepLayerParams(
            threshold=0,
            )
        )
        self.btsp_feedback = custom_networks.btsp_step_topk.BTSPStepTopKNetwork(
            params.btsp_feedback
        )
        self.reset_weights()
        
    def forward(self, input_data: torch.Tensor):
        """Forward pass."""
        hebbian_output = self.hebbian_forward.forward(input_data)
        hebbian_topk_output = self.hebbian_topk.forward(hebbian_output)
        hebbian_step_output = self.hebbian_step.forward(hebbian_topk_output)
        return hebbian_step_output
    
    def feedback(self, input_data: torch.Tensor):
        """Feedback the BTSP layer."""
        return self.btsp_feedback.forward(input_data)
    
    def one_shot_recovery(self, input_data: torch.Tensor):
        """One-shot recovery."""
        hebbian_output = self.hebbian_forward.forward(input_data)
        hebbian_topk_output = self.hebbian_topk.forward(hebbian_output)
        hebbian_step_output = self.hebbian_step.forward(hebbian_topk_output)
        return self.btsp_feedback.forward(hebbian_step_output)
    
    def pretrain_hebbian(self, input_data: torch.Tensor):
        """Pretrain the Hebbian layer."""
        self.hebbian_forward.learn(input_data)
    
    def pretrain_btsp_forward(self, input_data: torch.Tensor):
        """Pretrain the BTSP layer."""
        return self.btsp_feedback.learn_and_forward(input_data)
        
    def reset_weights(self, new_btsp_weights=None):
        """Reset the weights."""
        self.hebbian_forward.weight_reset()
        self.btsp_feedback.reset_weights(new_btsp_weights)
