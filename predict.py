import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys

from tqdm import tqdm

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.model import Model
from lbl.models.efficientnet.efficientnet import EfficientNet
from lbl.models.efficientnet.config import efficientnet_params

from lbl.preprocessing import (
    PadImage,
    RotateCircle,
    StandardizeNonZero,
    )

import warnings
warnings.filterwarnings("ignore")
# -----------------------------------------------------------------------------

# with only 2 classes (aurora/no aurora):
'''
path = "models/2class/b2/2021-10-05/best_validation/checkpoint-best.pth"
model = EfficientNet.from_name(model_name=model_name, num_classes=2, in_channels=1)

LABELS = {
    0: "aurora-less",
    1: "aurora"
}
'''

LABELS = {
    0: "aurora-less",
    1: "arc",
    2: "diffuse",
    3: "discrete",
}

model_names = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4']

def predict(model_name, model_path, container, LABELS, save_file):

    img_size = efficientnet_params(model_name)['resolution']

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

    #model = Model(1, 4, 128)
    model = EfficientNet.from_name(model_name=model_name, num_classes=4, in_channels=1)

    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    model = model.to('cuda:3')
    model.eval()

    with torch.no_grad():
        for entry in tqdm(container):

            if entry.label is None:

                score = dict()
                img = entry.open()
                x = transforms(img)
                x = x.unsqueeze(0)
                x = x.to('cuda:3')

                pred = model(x).to('cpu')
                pred = torch.softmax(pred, dim=-1)
                prediction = torch.argmax(pred, dim=-1)

                for i, label_pred in enumerate(pred[0]):
                    score[LABELS[i]] = float(label_pred)

                entry.label = LABELS[int(prediction[0])]
                entry.human_prediction = False
                entry.add_score(score)

            else:
                score = dict()
                img = entry.open()
                x = transforms(img)
                x = x.unsqueeze(0)
                x = x.to('cuda:3')

                pred = model(x).to('cpu')
                pred = torch.softmax(pred, dim=-1)
                prediction = torch.argmax(pred, dim=-1)

                for i, label_pred in enumerate(pred[0]):
                    score[LABELS[i]] = float(label_pred)

                entry.human_prediction = False
                entry.add_score(score)

    #container.to_json(path='./datasets/Full_aurora_predicted.json')
    container.to_json(path=save_file)


# make predictions with chosen model and data set

mlnodes_path = '/itf-fi-ml/home/koolsen/Master/'

# Load a saved model. UPDATE
model_name = model_names[3]
#model_path = "models/b2/2021-10-02/best_validation/checkpoint-best.pth"
model_path = "models/report/b3_16/best_validation/checkpoint-best.pth"

json_file = 'datasets/Full_aurora_new_rt_ml.json'
#container = DatasetContainer.from_json(mlnodes_path+json_file)
container = DatasetContainer.from_json(json_file)
#save_file = mlnodes_path+json_file[:-5]+'_predicted_'+model_name+'.json'
save_file = json_file[:-5]+'_predicted_'+model_name+'_TESTNEW_.json'

predict(model_name, model_path, container, LABELS, save_file)

"""
# Load json file to add predictions
json_file = 'Aurora_G_omni_mean.json'
container = DatasetContainer.from_json(mlnodes_path+json_file)
save_file = mlnodes_path+json_file[:-5]+'_predicted_'+model_name+'.json'

predict(model_name, model_path, container, LABELS, save_file)


# Load json file to add predictions
#json_file = 'Aurora_G.json'
json_file = 'Aurora_4yr_G_omni_mean.json'
container = DatasetContainer.from_json(mlnodes_path+json_file)
save_file = mlnodes_path+json_file[:-5]+'_predicted_'+model_name+'.json'

predict(model_name, model_path, container, LABELS, save_file)
"""
