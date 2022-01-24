from typing import Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import sys

from .dataset import DatasetContainer

import pandas as pd
# -----------------------------------------------------------------------------

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
        print(pred_level)
        plt.ion()
        plt.show(block=False)
        #plt.show() # If Linux

        tot = 0
        counter = 0
        correct = 0
        correct_from = []
        correct_to = []

        for entry in self.container:
            if entry.label == label:
                if entry.human_prediction == pred_level:
                    tot += 1

        for entry in self.container:
            if entry.label == label:
                if entry.human_prediction == pred_level:

                    '''
                    def keywithmaxval(d):
                        v=list(d.values())
                        k=list(d.keys())
                        #return k[v.index(max(v))]
                        return max(v), k[v.index(max(v))]
                    pred, pred_label = keywithmaxval(entry.score)
                    #if pred > 0.87:
                    if pred_label == entry.label:
                        pass
                    else:
                    '''

                    counter += 1
                    img = entry.open()
                    plt.title('Label: {0}, image: {1}/{2}'.format(str(entry.label), counter, tot))

                    plt.imshow(img, cmap='gray') #, cmap='gray'
                    #plt.pause(0.001) # If Linux

                    # Add image count?
                    print("Filename: ", Path(entry.image_path).stem)
                    #print("Score: ", entry.score)
                    for key, value in entry.score.items():
                        print("%11s : %.4f" %(key, value))

                    text = input("Correct class [hit enter], if not type in correct class integer [0,1,2,3]:")

                    if text == "":
                        if text == "" and entry.label is None:
                            print('Label cannot be None, skipping correction')
                            print('----------------------------------')
                            continue
                        entry.human_prediction = True
                        correct += 1
                    else:
                        correct_label = self.LABELS[text]
                        correct_from.append(entry.label)
                        correct_to.append(correct_label)
                        entry.label = correct_label
                        entry.human_prediction = True

                    print("Current accuracy: ", correct/counter)

                    print('----------------------------------')
                    plt.gca().clear()

                    self.container.to_json(save_path)


        plt.close()

        print("Correct predicted: ", correct)
        print("Counter:           ", counter)
        print("Accuracy:          ", correct/counter)
        data = {'Predicted from b0': correct_from, 'Corrected to': correct_to}
        df = pd.DataFrame(data, columns = ["Predicted with b0", "Corrected to"])
        df.to_excel("test.xlsx", index = False)
