import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import sys

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.model    import Model
#from lbl.trainer.trainer import Trainer
#from lbl.preprocessing.padding import PadImage


is_cuda = torch.cuda.is_available()
device  = torch.device('cuda' if is_cuda else 'cpu')

container = DatasetContainer.from_json('datasets/Full_aurora.json')
length    = len(container) # 7980
counter   = 0
# Remove images with label
for i in range(length):
    i -= counter
    if container[i].label != None:
        del container[i]
        counter += 1



# Load a saved model
path  = r"C:\Users\Krist\Documents\MasterThesis\checkpoint-best.pth"
model = Model()

#https://pytorch.org/tutorials/beginner/saving_loading_models.html
checkpoint = torch.load(path)   # checkpoint = torch.load(model, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
optimizer = checkpoint['optimizer'] # or optimizer.load_state_dict(checkpoint['optimizer'])

model.eval()
