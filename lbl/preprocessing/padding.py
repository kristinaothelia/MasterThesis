from typing import Union, Tuple, List
import numpy as np
import torch, sys

# https://numpy.org/doc/stable/reference/generated/numpy.pad.html
# Padding class: pytorch tensors

# eeeeehh..

class PadImage(object):
    """
    Pad an aurora image to 480x480 px.
    Designed for 469x469 and 471x471 px images
    """


    def __init__(self):
        pass

    '''
    def __init__(self, size: Union[int, tuple, list, torch.Tensor]=(480, 480)):

        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, torch.Tensor):
            self.size = size.shape
        else:
            self.size = size
    '''

    def __call__(self, tensor: torch.Tensor):

        shape = tensor.shape

        if shape != (469, 469) and shape != (471, 471):
            print("Wrong input shape")
            sys.exit()

        if shape == (469, 469):

            padded = np.pad(tensor, ((6, 5), (5, 6)), 'constant', constant_values=0)

        elif shape == (471, 471):

            padded = np.pad(tensor, ((5, 4), (4, 5)), 'constant', constant_values=0)

        '''
        if self.size == (469, 469):

            padded = np.pad(tensor, ((6, 5), (5, 6)), 'constant', constant_values=0)

        elif self.size == (471, 471):

            padded = np.pad(tensor, ((5, 4), (4, 5)), 'constant', constant_values=0)
        '''

        #return padded
        return torch.from_numpy(padded)


if __name__=='__main__':

    input = torch.Tensor(469,469)
    input = torch.Tensor(471,471)

    #padded = PadImage(input)
    padded = PadImage()
    print(padded(input).shape)
