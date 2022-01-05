import torch
import torchvision
import torchvision.transforms.functional as F
import torchvision.models as models
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
# -----------------------------------------------------------------------------

container = DatasetContainer.from_json('datasets/Full_aurora_ml_corr_NEW.json')
img_size = 240
batch_size_train = 16
learningRate = 2e-3
stepSize = 75
g = 0.1

def remove_noLabel_img(container):
    """ Remove images with no label """

    length = len(container)
    print('original container length:   ', length)
    counter = 0
    for i in range(length):
        i -= counter
        if container[i].label == None:
            del container[i]
            counter += 1
    print('removed images with no label: ', counter)
    return container

container = remove_noLabel_img(container)
train, valid = container.split(seed=42, split=0.8)

def count(container):

    clear = 0; arc = 0; diff = 0; disc = 0

    for i in range(len(container)):
        if train[i].label == LABELS[0]:
            clear += 1
        if train[i].label == LABELS[1]:
            arc += 1
        if train[i].label == LABELS[2]:
            diff += 1
        if train[i].label == LABELS[3]:
            disc += 1

    return clear, arc, diff, disc

clear, arc, diff, disc = count(train)
print("class count, train: ", [clear, arc, diff, disc])

class_weights = [clear/clear, clear/arc, clear/diff, clear/disc]
print(class_weights)

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
                                           batch_size   = batch_size_train,
                                           shuffle      = True,
                                           )
valid_loader = torch.utils.data.DataLoader(dataset      = valid_loader,
                                           num_workers  = 4,
                                           batch_size   = 1,
                                           shuffle      = False,
                                           )


#model = Model(1, 4, 128)
model = models.resnet50().to(device)         # Resnet network with 50 hidden layers.
model_name = 'resnet50'

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)

loss         = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))  #torch.nn.CrossEntropyLoss()
optimizer    = torch.optim.Adam(params=model.parameters(), lr=learningRate, amsgrad=True)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=stepSize, gamma=g)


trainer = Trainer(model             = model,
                  loss_function     = loss,
                  optimizer         = optimizer,
                  data_loader       = train_loader,
                  valid_data_loader = valid_loader,
                  lr_scheduler      = lr_scheduler,
                  epochs            = 200,
                  model_info        = [batch_size_train, learningRate, stepSize, g, params/1e6, model_name],
                  save_period       = 100,
                  savedir           = './models',
                  device            = 'cuda:2',
                  )

trainer.train()
