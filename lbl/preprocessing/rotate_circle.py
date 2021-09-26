import numpy as np

import torch
import torchvision

def RotateCircle(tensor: torch.Tensor):

    size = tensor.shape[-2:]
    angle = np.random.uniform(0, 360)

    tensor = torchvision.transforms.functional.rotate(
        img=tensor,
        angle=angle,
        expand=True,
        fill=0,
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
    tensor = torchvision.transforms.functional.center_crop(tensor, size)

    return tensor

