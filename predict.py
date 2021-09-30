import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import sys

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.model import Model
from lbl.models.efficientnet.efficientnet import EfficientNet

from lbl.preprocessing import (
    PadImage,
    RotateCircle,
    StandardizeNonZero,
    )

import matplotlib.pyplot as plt


from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")


LABELS = {
    0: "aurora-less",
    1: "arc",
    2: "diffuse",
    3: "discrete",
}

container = DatasetContainer.from_json('datasets/Full_aurora_ml.json')

#img_size = 224  # EfficientNet-b0
#img_size = 240  # EfficientNet-b1
img_size = 260  # EfficientNet-b2
img_size = 300  # EfficientNet-b3

transforms = torchvision.transforms.Compose([
    lambda x: np.float32(x),
    lambda x: torch.from_numpy(x),
    lambda x: x.unsqueeze(0),
    lambda x: torch.nn.functional.interpolate(
            input=x.unsqueeze(0),
            size=img_size,
            mode='bicubic',
            align_corners=True,
            ).squeeze(0),
    StandardizeNonZero(),
    # PadImage(size=480),
    ])



# Load a saved model
#path  = "models/2021-09-26/best_validation/checkpoint-best.pth"
#path  = "models/2021-09-29/best_validation/checkpoint-best.pth"
path  = "./2021-09-30/best_validation/checkpoint-best.pth"

#model = Model(1, 4, 128)
model = EfficientNet.from_name(model_name='efficientnet-b3', num_classes=4, in_channels=1)

checkpoint = torch.load(path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

model = model.to('cuda:0')

model.eval()

with torch.no_grad():
    for entry in tqdm(container):

        if entry.label is None:
            score = dict()
            img = entry.open()
            x = transforms(img)  # pad an stuff
            x = x.unsqueeze(0)
            x = x.to('cuda:0')

            pred = model(x).to('cpu')
            pred = torch.softmax(pred, dim=-1)
            prediction = torch.argmax(pred, dim=-1)

            for i, label_pred in enumerate(pred[0]):
                score[LABELS[i]] = float(label_pred)

            entry.label = LABELS[int(prediction[0])]
            entry.human_prediction = False
            entry.add_score(score)

container.to_json(path='./datasets/Full_aurora_predicted.json')
