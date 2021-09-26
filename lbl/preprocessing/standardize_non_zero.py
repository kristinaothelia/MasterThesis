from typing import Union, Tuple, List
import numpy as np
import torch

# https://numpy.org/doc/stable/reference/generated/numpy.pad.html
# Padding class: pytorch tensors

# eeeeehh..

class StandardizeNonZero(object):
    """
    Pad an aurora image to 480x480 px.
    Designed for 469x469 and 471x471 px images
    """


    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon

    def __call__(self, tensor: torch.Tensor):

        mask = tensor != 0
        mean = tensor[mask].mean()
        std = tensor[mask].std()
        tensor[mask] = (tensor[mask] - mean)/(std + self.epsilon)

        return tensor
