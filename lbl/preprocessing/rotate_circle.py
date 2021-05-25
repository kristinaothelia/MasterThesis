from typing import Union, List

import numpy as np

class RotateCircle(object):
    """
    ***torchvision.Transforms compatible***

    Rotates circle in the middle of image...
    """

    def __init__(self, rotations: int):
        """
        Args:
            mask (torch.Tensor): The mask used in under-sampling the given k-space data,
                                 assumes shape: (number_of_columns_in_kspace)
        """
        self.rotations = rotations

    def __call__(self, tensor: np.ndarray):
        """
        Args:
            tensor (torch.Tensor): Tensor of the k-space data with
                                   shape (coil, rows, columns) or (rows, columns)
        Returns:
            torch.Tensor: K-space tensor with same shape and applied mask on columns

        """
        shape = tensor.shape





        # https://stackoverflow.com/questions/39188198/only-rotate-part-of-image-python






        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(rotations={0})'.format(self.rotations)


if __name__=='__main__':
    import matplotlib.pyplot as plt

    x = np.zeros(shape=(2, 469, 469))

    x[:, :200] = 1
    plt.imshow(x[0])
    plt.show()
    rotate = RotateCircle(rotations=2)

    y = rotate(tensor=x)

    plt.imshow(y[0])
    plt.show()

    plt.imshow(y[1])
    plt.show()
