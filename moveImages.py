# Move images (from json file) to new folders folders

from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import glob
import shutil
import os
import random

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']
# Lists containing image paths
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

def move(container, label, toFolder, frac=0.1, testFolder=None):

    print("\nMaking test set for {}".format(label))
    test_list = random.sample(container, int(len(container)*frac))
    for imageName in test_list:
        shutil.copy(imageName, testFolder)

    list = [value for value in container if value not in test_list]
    print("{} images moved to test folder".format(len(test_list)))
    print("{} images left for train folder".format(len(list)))

    print("Moving {} images".format(label))
    for imageName in list:
        shutil.copy(imageName, toFolder)
        #shutil.copy(imageName, '/itf-fi-ml/home/koolsen/Aurora/Data_subfolders/arc/')

path = r'C:\Users\Krist\Documents\dataset_subfolders'
move(container=aurora_less, label=LABELS[0], toFolder=path+r'\no_aurora', testFolder=path+r'\no_aurora_test')
move(container=arc, label=LABELS[1], toFolder=path+r'\arc', testFolder=path+r'\arc_test')
move(container=diff, label=LABELS[2], toFolder=path+r'\diffuse', testFolder=path+r'\diffuse_test')
move(container=disc, label=LABELS[3], toFolder=path+r'\discrete', testFolder=path+r'\discrete_test')

#move(container=arc, label=LABELS[1], toFolder=r'C:\Users\Krist\Documents\dataset_subfolders\arc')
#move(container=diff, label=LABELS[2], toFolder=r'C:\Users\Krist\Documents\dataset_subfolders\diffuse')
#move(container=disc, label=LABELS[3], toFolder=r'C:\Users\Krist\Documents\dataset_subfolders\discrete')
