import torch
import torchvision

from .datasetcontainer import DatasetContainer


class DatasetLoader(torch.utils.data.Dataset):
    """
    An iterable datasetloader for the dataset container to make my life easier
    """
    LABELS = {
        "aurora-less": torch.tensor([1, 0, 0, 0]),
        "arc": torch.tensor([0, 1, 0, 0]),
        "diffuse": torch.tensor([0, 0, 1, 0]),
        "discrete": torch.tensor([0, 0, 0, 1]),
    }

    def __init__(self,
                 container: DatasetContainer,
                 transforms: torchvision.transforms = None,
                 ):
        """
        Args:
            container: The container that is to be loaded
            transforms: Transforms the data is gone through before model input
        """
        self.container = container
        self.transforms = transforms

    def __len__(self):
        return len(self.container)

    def __getitem__(self, index):

        entry = self.container[index]
        image = entry.open()

        if entry.label is None:
            raise TypeError("label is None")

        # For reconstruction where the train image is masked and thus have a different transform
        if self.transforms is not None:
            train = self.transforms(image)
        else:
            train = image

        valid = self.LABELS[entry.label]

        return train, valid

    def __iter__(self):
        self.current_index = 0
        self.max_length = len(self)
        return self

    def __next__(self):

        if not self.current_index < self.max_length:
            raise StopIteration

        item = self[self.current_index]
        self.current_index += 1
        return item
