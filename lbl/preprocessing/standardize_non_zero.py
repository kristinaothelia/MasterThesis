from typing import Union, Tuple, List
import numpy as np
import torch

class StandardizeNonZero(object):

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon

    def __call__(self, tensor: torch.Tensor):

        mask = tensor != 0
        mean = tensor[mask].mean()
        std = tensor[mask].std()
        tensor[mask] = (tensor[mask] - mean)/(std + self.epsilon)

        return tensor
