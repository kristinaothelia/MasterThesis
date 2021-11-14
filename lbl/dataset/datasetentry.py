from typing import Union, Dict

import imageio
import matplotlib.image as mpimg

from pathlib import Path

import sys


class DatasetEntry(object):
    """
    A class used to store information about the different training objects
    This class works like a regular dict
    """

    def __init__(self,
                 image_path: Union[str, Path] = None,
                 datasetname: str = None,
                 dataset_type: str = None,
                 label: str = None,
                 wavelength: str = None,
                 timepoint: str = None,
                 human_prediction: bool = None,
                 shape: tuple = None):
        """
        Args:
            image_path (str, Path): The path where the data is stored
            datasetname (str): The name of the dataset the data is from
            dataset_type (str): What kind of data the data is
            shape (tuple): The shape of the data
        """
        if isinstance(image_path, (Path, str)):
            self.image_path = str(image_path)
            if not Path(image_path).is_file():
                print('The path: ' + str(image_path))
                print('Is not an existing file, are you sure this is the correct path?')
        else:
            self.image_path = image_path

        self.datasetname = datasetname
        self.dataset_type = dataset_type
        self.label = label
        self.wavelength = wavelength
        self.timepoint = timepoint
        self.human_prediction = human_prediction
        self.shape = shape
        self.score = dict()
        self.solarwind = dict()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return self.__str__()

    def open(self, open_func=None):
        """
        Open the file
        Args:
            open_func (the function to open the file)
        returns:
            the opened file
        """
        if open_func is not None:
            image = open_func(self.image_path)
        else:
            suffix = Path(self.image_path).suffix
            if suffix == '.png':
                image = self.open_png(self.image_path)
            elif suffix == '.jpeg':
                image = self.open_jpeg(self.image_path)
            else:
                raise TypeError('cannot open file: ', self.image_path)

        return image

    def open_png(self, image_path):
        return imageio.imread(image_path)

    def open_jpeg(self, image_path):
        return mpimg.imread(image_path)

    def add_score(self, score: Dict[str, float]):
        """
        Add reconstruction score to entry for a given slice in volume
        Args:
            img_slice (int): The slice the score is for (-1 is the entire volume)
            score (Dict[str, float]): Dict of metrics with score
        """
        self.score = score

    def add_solarwind(self, solarwind: Dict[str, float]):
        """
        ?
        Args:
            ?
        """
        self.solarwind = solarwind

    def add_shape(self, open_func=None, shape=None):
        """
        Add shape to entry
        Args:
            open_func (callable): function for opening file
            shape (tuple): shape of file
            keyword (str): potential keyword for opening file
        """
        if isinstance(shape, tuple):
            self.shape = shape
        else:
            img = self.open(open_func=open_func)
            self.shape = img.shape

    def keys(self):
        """
        dict keys of class
        """
        return self.to_dict().keys()

    def to_dict(self) -> dict:
        """
        returns:
            dict format of this class
        """
        return {'image_path': self.image_path,
                #'datasetname': self.datasetname,
                #'dataset_type': self.dataset_type,
                'wavelength': self.wavelength,
                'timepoint': self.timepoint,
                'shape': self.shape,
                'label': self.label,
                'human_prediction': self.human_prediction,
                'score': self.score,
                'solarwind': self.solarwind
                }

    def from_dict(self, in_dict: dict):
        """
        Args:
            in_dict: dict, dict format of this class
        Fills in the variables from the dict
        """
        if isinstance(in_dict, dict):
            self.image_path = in_dict['image_path']
            #self.datasetname = in_dict['datasetname']
            #self.dataset_type = in_dict['dataset_type']
            self.wavelength = in_dict['wavelength']
            self.timepoint = in_dict['timepoint']
            self.label = in_dict['label']
            self.human_prediction = in_dict['human_prediction']
            self.shape = in_dict['shape']
            self.score = in_dict['score']
            self.solarwind = in_dict['solarwind']

        return self
