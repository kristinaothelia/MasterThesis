# Move images to subfolders

from lbl.dataset import DatasetEntry, DatasetInfo, DatasetContainer
import numpy as np
import glob
import shutil
import os

LABELS = ['aurora-less', 'arc', 'diffuse', 'discrete']

aurora_less = []
arc = []
diff = []
disc = []

container_file = 'datasets/NEW_TEST.json'
#container_file = 'datasets/NEW_TEST_ml.json' # NB! remember to make folders!

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


for imageName in aurora_less:
    shutil.copy(imageName, r'C:\Users\Krist\Documents\dataset_subfolders\aurora_less')
    #shutil.copy(imageName, '/itf-fi-ml/home/koolsen/Aurora/Data_subfolders/aurora_less/')
for imageName in arc:
    shutil.copy(imageName, r'C:\Users\Krist\Documents\dataset_subfolders\arc')
    #shutil.copy(imageName, '/itf-fi-ml/home/koolsen/Aurora/Data_subfolders/arc/')
for imageName in diff:
    shutil.copy(imageName, r'C:\Users\Krist\Documents\dataset_subfolders\diffuse')
    #shutil.copy(imageName, '/itf-fi-ml/home/koolsen/Aurora/Data_subfolders/diffuse/')
for imageName in disc:
    shutil.copy(imageName, r'C:\Users\Krist\Documents\dataset_subfolders\discrete')
    #shutil.copy(imageName, '/itf-fi-ml/home/koolsen/Aurora/Data_subfolders/discrete/')
