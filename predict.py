import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import sys

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.model import Model

from lbl.preprocessing import (
    PadImage,
    RotateCircle,
    StandardizeNonZero,
    )

import matplotlib.pyplot as plt


LABELS = {
    0: "aurora-less",
    1: "arc",
    2: "diffuse",
    3: "discrete",
}

container = DatasetContainer.from_json('datasets/Full_aurora_jon.json')
padder = PadImage(size=480)

for entry in container:
    print(entry.label)
    img_og = entry.open()
    img = torch.from_numpy(np.float32(img_og))
    rotate = RotateCircle(img.unsqueeze(0))
    rotate = padder(rotate)

    plt.subplot(1, 2, 1)
    plt.imshow(img_og, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(rotate.squeeze(0).numpy(), cmap='gray')
    plt.show()





exit()

transforms = torchvision.transforms.Compose([
    lambda x: np.float32(x),
    lambda x: torch.from_numpy(x),
    lambda x: x.unsqueeze(0),
    #PadImage(torch.Tensor(input))
    #torchvision.transforms.Pad(padding=[5, 5, 4, 4], fill=0),
    lambda tensor: F.normalize(tensor=tensor,
                           mean=tensor.mean(axis=(1, 2)),
                           std=tensor.std(axis=(1, 2)),
                           ),
    ])



# Load a saved model
path  = r"C:\Users\Krist\Documents\MasterThesis\checkpoint-best.pth"
model = Model(1, 4, 96)

checkpoint = torch.load(model, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

model.eval()

with torch.no_grad():
    for entry in container:

        if entry.label is None:
            score = dict()
            img = entry.open()
            x = transforms(img)  # pad an stuff
            pred = model(x)

            pred = torch.softmax(pred, dim=-1)
            prediction = torch.argmax(pred, dim=-1)

            for i, label_pred in enumerate(prediction[0]):
                score[LABELS[i]] = label_pred

            entry.lable = LABELS[prediction]
            entry.human_prediction = False



