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

# I chose B3 because it provides a nice balance between accuracy and training time.

LABELS = {
    0: "aurora-less",
    1: "arc",
    2: "diffuse",
    3: "discrete",
}

model_name = ['efficientnet-b0',
              'efficientnet-b1',
              'efficientnet-b2',
              'efficientnet-b3',
              'efficientnet-b4',
              'efficientnet-b6']

def train(model, json_file, model_name, ep=100, batch_size_train=8, learningRate=2e-3, stepSize=75, g=0.1):

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

    # Change seed?
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
    class_weights = [clear/clear, clear/arc, clear/diff, clear/disc]
    class_weights = [0.9, clear/arc, clear/diff, clear/disc]
    print("class count, train: ", [clear, arc, diff, disc])
    print("weights, train:     ", class_weights)

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
                mode='bicubic', # 'nearest' | 'linear' | 'bilinear' | 'bicubic
                align_corners=True,
                ).squeeze(0),
        StandardizeNonZero(),
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225]),
        # PadImage(size=480),
        ])

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

    sample_weights = [0] * len(train_loader)
    key_list = list(LABELS.keys())
    val_list = list(LABELS.values())

    for idx, entry in enumerate(train):
        entry_ = val_list.index(entry.label)
        class_weight = class_weights[entry_]
        sample_weights[idx] = class_weight

    sampler = torch.utils.data.WeightedRandomSampler(sample_weights,
                                                     num_samples=len(sample_weights),
                                                     replacement=True,
                                                     )

    train_loader = torch.utils.data.DataLoader(dataset      = train_loader,
                                               num_workers  = 4,
                                               batch_size   = batch_size_train,
                                               sampler      = sampler,
                                               shuffle      = False,
                                               #shuffle      = True,
                                               )

    valid_loader = torch.utils.data.DataLoader(dataset      = valid_loader,
                                               num_workers  = 4,
                                               batch_size   = 1,
                                               shuffle      = False,
                                               )

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('The number of params in Million: ', params/1e6)

    #loss         = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
    loss         = torch.nn.CrossEntropyLoss()
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
                      save_period       = 250,
                      savedir           = './models/{}/batch_size_{}/lr_{}/st_{}/g_{}'.format(model_name[-2:], batch_size_train, learningRate, stepSize, g),
                      device            = 'cuda:2',
                      )

    trainer.train()


#json_file = 'datasets/Full_aurora_corr.json'    # local laptop path
json_file = 'datasets/Full_aurora_ml_corr_NEW.json'

#model = EfficientNet.from_name(model_name=model_name[2], num_classes=4, in_channels=1)
#train(model, json_file, model_name[2], ep=350, batch_size_train=16, learningRate=0.01, stepSize=300, g=1.1)

# With weights in loss
#model = EfficientNet.from_name(model_name=model_name[3], num_classes=4, in_channels=1)
#train(model, json_file, model_name[3], ep=200, batch_size_train=16, learningRate=0.1, stepSize=200, g=0.5)
#train(model, json_file, model_name[3], ep=400, batch_size_train=16, learningRate=0.01, stepSize=350, g=0.5)
#Res:


# With weights in sampler
model = EfficientNet.from_name(model_name=model_name[3], num_classes=4, in_channels=1)
train(model, json_file, model_name[3], ep=300, batch_size_train=16, learningRate=0.1, stepSize=250, g=0.5)
train(model, json_file, model_name[3], ep=400, batch_size_train=16, learningRate=0.01, stepSize=350, g=0.5)
train(model, json_file, model_name[2], ep=400, batch_size_train=16, learningRate=0.01, stepSize=350, g=0.5)
train(model, json_file, model_name[4], ep=400, batch_size_train=8, learningRate=0.01, stepSize=350, g=0.5)
# Res:

# Without weights
#model = EfficientNet.from_name(model_name=model_name[3], num_classes=4, in_channels=1)
#train(model, json_file, model_name[3], ep=200, batch_size_train=16, learningRate=0.1, stepSize=200, g=0.5)
#train(model, json_file, model_name[3], ep=400, batch_size_train=16, learningRate=0.01, stepSize=350, g=0.5)
# Res:

# B2, ep:32, lr:0.001, st:75, g:0.1 - acc: 0.85

'''
train(json_file, model_name[3], ep=200, batch_size_train=16, learningRate=0.1, stepSize=150, g=0.1)
train(json_file, model_name[3], ep=200, batch_size_train=16, learningRate=0.01, stepSize=150, g=0.1)
train(json_file, model_name[3], ep=200, batch_size_train=16, learningRate=0.001, stepSize=150, g=0.1)
'''
#train(json_file, model_name[4], ep=200, batch_size_train=8, learningRate=0.05, stepSize=80, g=0.1)
#train(json_file, model_name[4], ep=100, batch_size_train=16, learningRate=1e-2, stepSize=75, g=0.1)

#train(json_file, model_name[3], ep=150, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.05)
#train(json_file, model_name[3], ep=150, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
#train(json_file, model_name[3], ep=150, batch_size_train=24, learningRate=1e-2, stepSize=75, g=0.1)
#train(json_file, model_name[3], ep=150, batch_size_train=24, learningRate=1e-4, stepSize=75, g=0.1)
#train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=80, g=0.1)
#train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=100, g=0.1)

#train(json_file, model_name[3], ep=100, batch_size_train=32, learningRate=1e-3, stepSize=90, g=0.1)

'''
# Test with more metrics
train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.01)
train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=75, g=1.0)

train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=60, g=0.1)
train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=90, g=0.1)
'''

# Test:
'''
train(json_file, model_name[2], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[2], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=100, g=0.1)
train(json_file, model_name[2], ep=100, batch_size_train=24, learningRate=1e-4, stepSize=100, g=0.1)
train(json_file, model_name[2], ep=100, batch_size_train=24, learningRate=1e-2, stepSize=100, g=0.1)

train(json_file, model_name[3], ep=100, batch_size_train=16, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=25, g=0.1)
train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=100, g=0.1)

train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-2, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=100, batch_size_train=24, learningRate=1e-4, stepSize=75, g=0.1)
'''

'''
train(json_file, model_name[0], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=1, g=0.1)
train(json_file, model_name[0], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=50, g=0.1)
train(json_file, model_name[0], ep=100, batch_size_train=32, learningRate=1e-3, stepSize=100, g=0.1)

train(json_file, model_name[0], ep=100, batch_size_train=16, learningRate=1e-3, stepSize=100, g=0.1)
train(json_file, model_name[0], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=100, g=0.1)
train(json_file, model_name[0], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=100, g=0.01)

#train(json_file, model_name[2], ep=200, batch_size_train=8, learningRate=1e-3, stepSize=75, g=0.1)
#train(json_file, model_name[2], ep=200, batch_size_train=16, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[2], ep=100, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
#train(json_file, model_name[2], ep=200, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[2], ep=200, batch_size_train=32, learningRate=1e-3, stepSize=75, g=0.1)
'''

'''
train(json_file, model_name[2], ep=200, batch_size_train=24, learningRate=1e-2, stepSize=75, g=0.1)
train(json_file, model_name[2], ep=200, batch_size_train=32, learningRate=1e-3, stepSize=75, g=0.01)
train(json_file, model_name[2], ep=200, batch_size_train=24, learningRate=1e-2, stepSize=75, g=0.01)
train(json_file, model_name[2], ep=200, batch_size_train=24, learningRate=1e-3, stepSize=100, g=0.1)
#train(json_file, model_name[2], ep=200, batch_size_train=24, learningRate=5e-4, stepSize=75, g=0.1)
'''
#train(json_file, model_name[4], ep=100, batch_size_train=8, learningRate=2e-3, stepSize=75, g=0.1)
#train(json_file, model_name[4], ep=100, batch_size_train=16, learningRate=2e-3, stepSize=75, g=0.1)
'''
train(json_file, model_name[3], ep=150, batch_size_train=8, learningRate=1e-4, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=150, batch_size_train=8, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=150, batch_size_train=8, learningRate=1e-2, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=150, batch_size_train=8, learningRate=1e-3, stepSize=100, g=0.1)
train(json_file, model_name[3], ep=150, batch_size_train=8, learningRate=1e-3, stepSize=25, g=0.1)
#train(json_file, model_name[3], ep=200, batch_size_train=16, learningRate=1e-3, stepSize=75, g=0.1)
#train(json_file, model_name[3], ep=200, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[3], ep=150, batch_size_train=32, learningRate=1e-3, stepSize=75, g=0.1) # Failed, cuda memory
'''



'''
train(json_file, model_name[4], ep=150, batch_size_train=16, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[4], ep=150, batch_size_train=24, learningRate=1e-3, stepSize=75, g=0.1)
train(json_file, model_name[4], ep=100, batch_size_train=32, learningRate=1e-3, stepSize=75, g=0.1)
'''
