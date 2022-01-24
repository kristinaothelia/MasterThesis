# Move images (from json file) to new folders folders

from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import glob
import shutil
import os

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']
aurora_less = []; arc = []; diff = []; disc = []

container_file = 'datasets/Full_aurora_new_rt_predicted_efficientnet-b3_TESTNEW_.json'
container = DatasetContainer.from_json(container_file)
print("len container: ", len(container))

for entry in container:
    if entry.label == LABELS[0]:
        aurora_less.append(entry.image_path)
    if entry.label == LABELS[1]:
        arc.append(entry.image_path)
    if entry.label == LABELS[2]:
        diff.append(entry.image_path)
    if entry.label == LABELS[3]:
        disc.append(entry.image_path)

print(len(aurora_less))
print(len(arc))
print(len(diff))
print(len(disc))
print(len(aurora_less)+len(arc)+len(diff)+len(disc))


def move(container, label, toFolder):
    print("Moving {} images".format(label))
    for imageName in container:
        shutil.copy(imageName, toFolder)
        #shutil.copy(imageName, '/itf-fi-ml/home/koolsen/Aurora/Data_subfolders/arc/')

move(container=aurora_less, label=LABELS[0], toFolder=r'C:\Users\Krist\Documents\dataset_subfolders\aurora_less')
#move(container=arc, label=LABELS[1], toFolder=r'C:\Users\Krist\Documents\dataset_subfolders\arc')
#move(container=diff, label=LABELS[2], toFolder=r'C:\Users\Krist\Documents\dataset_subfolders\diffuse')
#move(container=disc, label=LABELS[3], toFolder=r'C:\Users\Krist\Documents\dataset_subfolders\discrete')
