"""Simple one-layer BTSP network.
"""

import torch
from experiment_framework.utils import layers


class SimpleBTSPFeedbackNetwork:
    """simple one-layer BTSP network."""

    def __init__(self, params: dict) -> None:
        """Constructor."""
        self.btsp = layers.BTSPLayer(
            params["btsp"]["input_dim"],
            params["btsp"]["memory_neurons"],
            params["btsp"]["fq"],
            params["btsp"]["fw"],
            params["btsp"]["device"],
        )
        self.btsp_topk = layers.TopKLayer(
            params["btsp_topk"]["topk"],
        )
        self.btsp_topk_step = layers.StepLayer(
            1e-5,
        )
        self.hebbian = layers.HebbianFeedbackLayer(
            params["hebbian"]["input_dim"],
            params["hebbian"]["output_dim"],
            params["hebbian"]["device"],
        )
        self.hebbian_feedback_threshold = layers.StepLayer(
            params["hebbian_feedback_threshold"]["threshold"],
        )
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
        self.hebbian.learn([input_data, btsp_topk_step])
        return btsp_output

    def reset_weights(self):
        """Reset the weights."""
        self.btsp.weight_reset()
        self.hebbian.weight_reset()
        
    def hebbian_feedback(self, input_data: torch.Tensor):
        """Feedback the hebbian layer."""
        return self.hebbian.feedback(input_data)

    def reconstruct(self, input_data: torch.Tensor):
        """Reconstruct the input data."""
        hebbian_output = self.hebbian_feedback(input_data)
        hebbian_threshold_output = self.hebbian_feedback_threshold.forward(
            hebbian_output
        )
        return hebbian_threshold_output
