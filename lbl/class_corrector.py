from typing import Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import sys

from .dataset import DatasetContainer


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

        tot = 0
        counter = 0
        for entry in self.container:
            if entry.label == label:
                if entry.human_prediction == pred_level:
                    tot += 1

        for entry in self.container:
            if entry.label == label:
                if entry.human_prediction == pred_level:

                    counter += 1
                    img = entry.open()
                    plt.title('Label: {0}, image: {1}/{2}'.format(str(entry.label), counter, tot))

                    #plt.imshow(img, cmap='gray')
                    plt.imshow(img)
                    # Add image count?
                    print("Filename: ", Path(entry.image_path).stem)
                    print("Score: ", entry.score)

                    text = input("Correct class [hit enter], if not type in correct class integer [0,1,2,3]:")

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
