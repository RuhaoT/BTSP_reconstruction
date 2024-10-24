"""This module is for the Fly-hashing step top-k forward network."""

from dataclasses import dataclass
import torch
from experiment_framework.utils import layers


@dataclass
class FlyHashingStepTopKNetworkParams:
    """Parameters for Fly-hashing step top-k network."""

    fly_hashing: layers.FlyHashingLayerParams
    fly_hashing_topk: layers.TopKLayerParams
    fly_hashing_topk_step: layers.StepLayerParams


class FlyHashingStepTopKNetwork:
    """Fly-hashing step top-k network."""

    def __init__(self, params: FlyHashingStepTopKNetworkParams) -> None:
        self.fly_hashing = layers.FlyHashingLayer(params.fly_hashing)
        self.fly_hashing_topk = layers.TopKLayer(params.fly_hashing_topk)
        self.fly_hashing_topk_step = layers.StepLayer(params.fly_hashing_topk_step)
        self.reset_weights()

    def forward(self, input_data: torch.Tensor):
        """Forward pass."""
        fly_hashing_output = self.fly_hashing.forward(input_data)
        fly_hashing_topk_output = self.fly_hashing_topk.forward(fly_hashing_output)
        fly_hashing_topk_step = self.fly_hashing_topk_step.forward(
            fly_hashing_topk_output
        )
        return fly_hashing_topk_step

    def reset_weights(self):
        """Reset the weights."""
        self.fly_hashing.weight_reset()
