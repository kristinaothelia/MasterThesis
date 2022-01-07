from typing import Union, Tuple, List
import numpy as np
import torch, sys

import torch.nn.functional as F

class PadImage(object):

    def __init__(self, size: Union[Tuple[int], int]):
        self.size = size if isinstance(size, tuple) else (size, size)

    def __call__(self, tensor: torch.Tensor):

        h, w = tensor.shape[-2:]

        target_h, target_w = self.size
        assert target_h > h or target_w > w, "image must be smaller than desired size"

        pad_h = target_h - h
        pad_w = target_w - w

        tensor = F.pad(tensor, pad=((pad_w + 1) // 2, pad_w // 2, (pad_h + 1) // 2, pad_h // 2))

        return tensor
