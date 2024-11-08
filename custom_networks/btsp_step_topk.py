"""This module is for the BTSP step top-k forward network."""

import torch
from experiment_framework.utils import layers
from dataclasses import dataclass


@dataclass
class BTSPStepTopKNetworkParams:
    """Parameters for BTSP step top-k network."""

    btsp: layers.BTSPLayerParams
    btsp_topk: layers.TopKLayerParams
    btsp_topk_step: layers.StepLayerParams


class BTSPStepTopKNetwork:
    """BTSP step top-k network."""

    def __init__(self, params: BTSPStepTopKNetworkParams) -> None:
        """Constructor."""
        self.btsp = layers.BTSPLayer(params.btsp)
        self.btsp_topk = layers.TopKLayer(params.btsp_topk)
        self.btsp_topk_step = layers.StepLayer(params.btsp_topk_step)
        self.reset_weights()

    def forward(self, input_data: torch.Tensor):
        """Forward pass."""
        btsp_output = self.btsp.forward(input_data)
        btsp_topk_output = self.btsp_topk.forward(btsp_output)
        btsp_topk_step = self.btsp_topk_step.forward(btsp_topk_output)
        return btsp_topk_step

    def learn_and_forward(self, input_data: torch.Tensor):
        """Forward pass and learning."""
        btsp_output = self.btsp.learn_and_forward(input_data)
        btsp_topk_output = self.btsp_topk.forward(btsp_output)
        btsp_topk_step = self.btsp_topk_step.forward(btsp_topk_output)
        return btsp_topk_step

    def reset_weights(self, new_btsp_weights=None):
        """Reset the weights."""
        self.btsp.weight_reset(weight=new_btsp_weights)
