import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import sys

import torchvision.models as models

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

LABELS = {
    0: "aurora-less",
    1: "arc",
    2: "diffuse",
    3: "discrete",
}

def train(json_file, model_name, ep=100, batch_size_train=8, learningRate=2e-3, stepSize=75, g=0.1):

    container = DatasetContainer.from_json(json_file)

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

    '''
    new_length = len(container)
    clear = 0
    arc = 0
    diff = 0
    disc = 0

    for i in range(new_length):
        if container[i].label == LABELS[0]:
            clear += 1
        if container[i].label == LABELS[1]:
            arc += 1
        if container[i].label == LABELS[2]:
            diff += 1
        if container[i].label == LABELS[3]:
            disc += 1

    class_count = [clear, arc, diff, disc]
    class_weights = [clear/clear, clear/arc, clear/diff, clear/disc]
    print(class_weights)
    #sample_weights = [0] * len(container)
    '''

    train, valid = container.split(seed=42, split=0.8)

    length = len(train)
    print(length)

    clear = 0
    arc = 0
    diff = 0
    disc = 0

    for i in range(length):
        if train[i].label == LABELS[0]:
            clear += 1
        if train[i].label == LABELS[1]:
            arc += 1
        if train[i].label == LABELS[2]:
            diff += 1
        if train[i].label == LABELS[3]:
            disc += 1

    class_count = [clear, arc, diff, disc]
    class_weights = [clear/clear, clear/arc, clear/diff, clear/disc]
    #print(class_weights)

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

    # Test with new image folders!
    #for idx, (data, label) in enumerate(train_loader):
    #    print(label)
    #    class_weight = class_weights[label]
    #    sample_weights[idx] = class_weight


    #sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

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

    #model = models.resnet50().to(device)         # Resnet network with 50 hidden layers.
    #model.fc = nn.Linear(512, 4).to(device)      # Alter output layer for current dataset.

    model = EfficientNet.from_name(model_name=model_name, num_classes=4, in_channels=1)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('The number of params in Million: ', params/1e6)

    loss         = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
    #loss         = torch.nn.CrossEntropyLoss()
    optimizer    = torch.optim.Adam(params=model.parameters(), lr=learningRate, amsgrad=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=stepSize, gamma=g)

    trainer = Trainer(model             = model,
                      loss_function     = loss,
                      optimizer         = optimizer,
                      data_loader       = train_loader,
                      valid_data_loader = valid_loader,
                      lr_scheduler      = lr_scheduler,
                      epochs            = ep,
                      model_info        = [batch_size_train, learningRate, stepSize, g, params/1e6, model_name[-1:]],
                      save_period       = 100,
                      savedir           = './models/{}/batch_size_{}/lr_{}'.format(model_name[-2:], batch_size_train, learningRate),
                      #savedir           = '/itf-fi-ml/home/koolsen/Master/',
                      device            = 'cuda:3',
                      )

    trainer.train()


#json_file = 'datasets/Full_aurora_predicted.json'
json_file = 'datasets/NEW_TEST_ml.json'
#json_file = 'datasets/NEW_TEST.json'

model_name = ['efficientnet-b0',
              'efficientnet-b1',
              'efficientnet-b2',
              'efficientnet-b3',
              'efficientnet-b4',
              'efficientnet-b6']

#train(json_file, model_name[0], ep=100, batch_size_train=8, learningRate=2e-3, stepSize=75, g=0.1)
#train(json_file, model_name[0], ep=100, batch_size_train=16, learningRate=2e-3, stepSize=75, g=0.1)
train(json_file, model_name[0], ep=200, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
#train(json_file, model_name[0], ep=200, batch_size_train=8, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[0], ep=200, batch_size_train=32, learningRate=1e-3, stepSize=75, g=0.1)

#train(json_file, model_name[2], ep=200, batch_size_train=8, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[2], ep=200, batch_size_train=16, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[2], ep=200, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[2], ep=200, batch_size_train=32, learningRate=1e-3, stepSize=75, g=0.1)

train(json_file, model_name[2], ep=200, batch_size_train=24, learningRate=1e-3, stepSize=50, g=0.1)
train(json_file, model_name[2], ep=200, batch_size_train=24, learningRate=1e-3, stepSize=25, g=0.1)
#train(json_file, model_name[2], ep=200, batch_size_train=24, learningRate=5e-4, stepSize=75, g=0.1)

#train(json_file, model_name[4], ep=100, batch_size_train=8, learningRate=2e-3, stepSize=75, g=0.1)
#train(json_file, model_name[4], ep=100, batch_size_train=16, learningRate=2e-3, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=200, batch_size_train=8, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=200, batch_size_train=16, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=200, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[4], ep=100, batch_size_train=8, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[5], ep=100, batch_size_train=8, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[4], ep=100, batch_size_train=16, learningRate=1e-3, stepSize=75, g=0.1)
exit()
train(json_file, model_name[4], ep=200, batch_size_train=16, learningRate=1e-3, stepSize=75, g=0.1)
#train(json_file, model_name[4], ep=200, batch_size_train=16, learningRate=5e-4, stepSize=75, g=0.1)
train(json_file, model_name[4], ep=200, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
