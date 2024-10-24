"""Module for FB-H network.

The FB-H network is a two-forward-module network with single feedback connections.
The first module is a Fly-hashing network, and the second layer is a BTSP network.
An Hebbian feedback layer is added with an end-to-end approach.
"""

import torch
from experiment_framework.utils import layers
from custom_networks import fly_hashing, simple_btsp

class FBHNetwork():
    
    def 