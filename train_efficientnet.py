import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import sys

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.efficientnet.efficientnet import EfficientNet
from lbl.trainer.trainer import Trainer
from lbl.preprocessing import (
    PadImage,
    RotateCircle,
    StandardizeNonZero,
    )

container = DatasetContainer.from_json('datasets/Full_aurora_ml.json')
#container = DatasetContainer.from_json('datasets/Full_aurora_predicted.json')

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

#img_size = 224  # EfficientNet-b0
#img_size = 240  # EfficientNet-b1
img_size = 260  # EfficientNet-b2
img_size = 300  # EfficientNet-b3

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
                                           batch_size   = 8,
                                           shuffle      = True,
                                           )
valid_loader = torch.utils.data.DataLoader(dataset      = valid_loader,
                                           num_workers  = 4,
                                           batch_size   = 1,
                                           shuffle      = False,
                                           )


model = EfficientNet.from_name(model_name='efficientnet-b2', num_classes=4, in_channels=1)

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
                  epochs            = 150,
                  save_period       = 50,
                  savedir           = './models/b2',
                  #savedir           = '/itf-fi-ml/home/koolsen/Master/',
                  device            = 'cuda:0',
                  )

trainer.train()
