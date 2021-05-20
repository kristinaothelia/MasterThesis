from typing import Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .dataset import DatasetContainer

import matplotlib


class ClassCorrector(object):

    LABELS = {
        "0": "aurora-less",
        "1": "arc",
        "2": "diffuse",
        "3": "discrete",
    }

    def __init__(self, container: DatasetContainer):
        self.container = container


    def correct_class(self, label: str, prediction_level: int = 0, save_path: Union[str, Path] = None):
        """
        Args:
            label (str): label to be checked
            prediction_level (int): the confidence level of the label to be checked
                0 - checks human_prediction = False
                1 - checks human_prediction = None
                2 - checks human_prediction = True
        returns:
            save a new container over the old one for every time a label is corrected
                and it will flip human_prediction = True
        """
        pred_level = [False, None, True][prediction_level]
        plt.ion()
        plt.show(block=False)
        for entry in self.container:
            if entry.label == label:
                if entry.human_prediction == pred_level:

                    img = entry.open()
                    if entry.label is not None:
                        plt.title(entry.label)

                    plt.imshow(img, cmap='gray')
                    print("Filename: ", Path(entry.image_path).stem)
                    text = input("Correct class [hit enter], if not type in correct class integer:")

                    if text == "":
                        if text == "" and entry.label is None:
                            print('Label cannot be None, skipping correction')
                            print('----------------------------------')
                            continue
                        entry.human_prediction = True
                    else:
                        correct_label = self.LABELS[text]
                        entry.label = correct_label
                        entry.human_prediction = True

                    print('----------------------------------')
                    plt.gca().clear()

                self.container.to_json(save_path)

        plt.close()

