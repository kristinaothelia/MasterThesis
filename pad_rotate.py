import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import sys
import imageio
import matplotlib.pyplot as plt

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.model    import Model
from lbl.trainer.trainer import Trainer
#from lbl.preprocessing.padding import PadImage

from typing import Union, Tuple, List


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


    container = DatasetContainer.from_json('datasets/Full_aurora.json')
    print(container[0])
    print(container[0].label)
    print(container[0].shape)

    print(container[0].image_path)

    im = imageio.imread(container[0].image_path)
    print(im.shape)
    plt.imshow(im)
    plt.show()

    '''
    for entry in container:
        #if entry.label == label:
            #if entry.human_prediction == pred_level:

        #counter += 1
        img = entry.open()
        #plt.title('Label: {0}, image: {1}/{2}'.format(str(entry.label), counter, tot))

        #plt.imshow(img, cmap='gray')
        plt.imshow(img)
        plt.show()
        sys.exit()
    '''


    '''
    input = torch.Tensor(469,469)
    input = torch.Tensor(471,471)

    #padded = PadImage(input)
    padded = PadImage()
    print(padded(input).shape)
    '''
