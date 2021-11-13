import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import sys

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.efficientnet.efficientnet import EfficientNet
from lbl.models.efficientnet.config import efficientnet_params
from lbl.trainer.trainer import Trainer
from lbl.preprocessing import (
    PadImage,
    RotateCircle,
    StandardizeNonZero,
    )
# -----------------------------------------------------------------------------
container = DatasetContainer.from_json('datasets/Full_aurora_ml.json')
#container = DatasetContainer.from_json('datasets/Full_aurora_predicted.json')
#container = DatasetContainer.from_json('files_new.json')

# Remove images with no label
length = len(container)
print(length)
counter = 0
for i in range(length):
    i -= counter
    if container[i].label == None:
        del container[i]
        counter += 1
print(counter)

train, valid = container.split(seed=42, split=0.8)

#model_name = 'efficientnet-b0'
#model_name = 'efficientnet-b1'
model_name = 'efficientnet-b2'
#model_name = 'efficientnet-b3'
#model_name = 'efficientnet-b4'

img_size = efficientnet_params(model_name)['resolution']

# rotation class: numpy arrays. Padding class: pytorch tensors
train_transforms = torchvision.transforms.Compose([
    lambda x: np.float32(x),
    lambda x: torch.from_numpy(x),
    lambda x: x.unsqueeze(0),
    RotateCircle,
    lambda x: torch.nn.functional.interpolate(
            input=x.unsqueeze(0),
            size=img_size,
            mode='bicubic',
            align_corners=True,
            ).squeeze(0),
    StandardizeNonZero(),
    # PadImage(size=480),
    ])

# No need to rotate validation images
valid_transforms = torchvision.transforms.Compose([
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


train_loader = DatasetLoader(container=train, transforms=train_transforms)
valid_loader = DatasetLoader(container=valid, transforms=valid_transforms)

train_loader = torch.utils.data.DataLoader(dataset      = train_loader,
                                           num_workers  = 4,
                                           batch_size   = 16,
                                           shuffle      = True,
                                           )
valid_loader = torch.utils.data.DataLoader(dataset      = valid_loader,
                                           num_workers  = 4,
                                           batch_size   = 1,
                                           shuffle      = False,
                                           )

model = EfficientNet.from_name(model_name=model_name, num_classes=4, in_channels=1)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)

loss         = torch.nn.CrossEntropyLoss()
optimizer    = torch.optim.Adam(params=model.parameters(), lr=2e-3, amsgrad=True)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=75, gamma=0.1)


trainer = Trainer(model             = model,
                  loss_function     = loss,
                  optimizer         = optimizer,
                  data_loader       = train_loader,
                  valid_data_loader = valid_loader,
                  lr_scheduler      = lr_scheduler,
                  epochs            = 100,
                  save_period       = 50,
                  savedir           = './models/{}_newtest'.format(model_name[-2:]),
                  #savedir           = '/itf-fi-ml/home/koolsen/Master/',
                  device            = 'cuda:2',
                  )

trainer.train()
