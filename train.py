import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import sys

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.model    import Model
from lbl.trainer.trainer import Trainer
from lbl.preprocessing import (
    PadImage,
    RotateCircle,
    StandardizeNonZero,
    )

container = DatasetContainer.from_json('datasets/Full_aurora_ml.json')


# Remove images with no label
length = len(container)
counter = 0
for i in range(length):
    i -= counter
    if container[i].label == None:
        del container[i]
        counter += 1

train, valid = container.split(seed=42, split=0.8)


# rotation class: numpy arrays. Padding class: pytorch tensors
train_transforms = torchvision.transforms.Compose([
    lambda x: np.float32(x),
    lambda x: torch.from_numpy(x),
    lambda x: x.unsqueeze(0),
    RotateCircle,
    StandardizeNonZero(),
    PadImage(size=480),
    ])

valid_transforms = torchvision.transforms.Compose([
    lambda x: np.float32(x),
    lambda x: torch.from_numpy(x),
    lambda x: x.unsqueeze(0),
    StandardizeNonZero(),
    PadImage(size=480),
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


model = Model(1, 4, 128)

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
                  valid_data_loader = train_loader,
                  lr_scheduler      = lr_scheduler,
                  epochs            = 100,
                  save_period       = 25,
                  savedir           = './models',
                  device            = 'cuda:0',
                  )

trainer.train()
