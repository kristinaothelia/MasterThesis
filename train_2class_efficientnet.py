import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import sys

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader_2class

from lbl.models.efficientnet.efficientnet import EfficientNet
from lbl.models.efficientnet.config import efficientnet_params
from lbl.trainer.trainer import Trainer
from lbl.preprocessing import (
    PadImage,
    RotateCircle,
    StandardizeNonZero,
    )
# -----------------------------------------------------------------------------
#container = DatasetContainer.from_json('datasets/Full_aurora_ml_corr.json')

def train(json_file, model_name, ep=100, batch_size_train=8, learningRate=2e-3, stepSize=75, g=0.1):

    container = DatasetContainer.from_json(json_file)

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
    img_size = efficientnet_params(model_name)['resolution']

    def count(container):

        aurora_less = 0; aurora = 0

        for i in range(len(container)):
            if train[i].label == 'aurora-less':
                aurora_less += 1
            if train[i].label == 'aurora':
                aurora += 1

        return aurora_less, aurora

    aurora_less, aurora = count(train)
    print("class count, train: ", [aurora_less, aurora])

    class_weights = [aurora_less/aurora_less, aurora_less/aurora]

    # rotation class: numpy arrays. Padding class: pytorch tensors
    train_transforms = torchvision.transforms.Compose([
        lambda x: np.float32(x),
        lambda x: torch.from_numpy(x),
        lambda x: x.unsqueeze(0),
        RotateCircle,
        lambda x: torch.nn.functional.interpolate(
                input=x.unsqueeze(0),
                size=img_size,
                mode='bilinear', # 'bicubic'
                align_corners=True,
                ).squeeze(0),
        StandardizeNonZero(),
        ])

    # No need to rotate validation images
    valid_transforms = torchvision.transforms.Compose([
        lambda x: np.float32(x),
        lambda x: torch.from_numpy(x),
        lambda x: x.unsqueeze(0),
        lambda x: torch.nn.functional.interpolate(
                input=x.unsqueeze(0),
                size=img_size,
                mode='bilinear',
                align_corners=True,
                ).squeeze(0),
        StandardizeNonZero(),
        ])


    train_loader = DatasetLoader_2class(container=train, transforms=train_transforms)
    valid_loader = DatasetLoader_2class(container=valid, transforms=valid_transforms)

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

    model = EfficientNet.from_name(model_name=model_name, num_classes=2, in_channels=1)
    #model = models.efficientnet_b3(pretrained=False)

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
                      epochs            = 200,
                      model_info        = [batch_size_train, learningRate, stepSize, g, params/1e6, model_name[-1:]],
                      save_period       = 200,
                      savedir           = './models/2class/{}/{}'.format(model_name[-2:], batch_size_train),
                      #savedir           = '/itf-fi-ml/home/koolsen/Master/',
                      device            = 'cuda:2',
                      )

    trainer.train()

model_name = ['efficientnet-b0',
              'efficientnet-b1',
              'efficientnet-b2',
              'efficientnet-b3',
              'efficientnet-b4',
              'efficientnet-b6']

# NB! Right file?
json_file = 'datasets/Full_aurora_ml_corr_NEW_2class.json'
train(json_file, model_name[3], ep=300, batch_size_train=16, learningRate=0.01, stepSize=280, g=0.1)
