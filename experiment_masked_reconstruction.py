"""Masked reconstruction experiment on simple dataset

This this the first stage 3 experiment, aims to explore the performance of
BTSP-based networks on recalling a example dataset"""

import os
import dataclasses
import numpy as np
import torch
import pyarrow as pa
from pyarrow import parquet as pq
import matplotlib as mpl
import matplotlib.pyplot as plt
from experiment_framework.auto_experiment.auto_experiment import (
    SimpleBatchExperiment,
    ExperimentInterface,
)
from dataset import s3dataset

class MaskedReconstructionExperiment(ExperimentInterface):
    """Masked reconstruction experiment on minimal dataset.
    
    Explore the performance of a series of networks on a minimal multi-modal dataset.
    """
    
    def __init__(self):
        """Constructor."""
        # device part
        self.device = "cuda"
        
        # dataset part
        self.coordinate_dim = 32
        self.color_dim = 8
        # TODO(Ruhao Tian): Refactor the dataset part.
        self.grid_dataset = s3dataset.grid_dataset("dataset/dataset.csv", self.coordinate_dim,self.color_dim)
        self.datatensors = self.grid_dataset.convert_to_binary().to_device(self.device)
        self.avg_data_sparsity = self.grid_dataset.sparseness()
        
        # network shape
        self.input_dim = self.coordinate_dim * 2 + self.color_dim * 3
        self.sparse_dim = 2500
        self.output_dim = 3900
        self.feedback_threshold = 0  # threshold for the memory neuron to fire

        # hashing properties
        self.sparse_representation_sparsity = 0.01        
        self.fly_hashing_topk = self.sparse_representation_sparsity
        self.fly_hashing_weight_sparsity = 6 / 50 * (self.fly_hashing_topk / 0.05)
        
        # btsp properties
        self.btsp_topk = 15
        
        # hebbian properties
        # IMPORTANT QUESTION: can we set hebbian firing threshold by data sparsity?