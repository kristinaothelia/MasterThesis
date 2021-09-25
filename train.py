import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import sys

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.model    import Model
from lbl.trainer.trainer import Trainer
from lbl.preprocessing.padding import PadImage


container = DatasetContainer.from_json('datasets/Full_aurora.json')

print(container[0].label)
print(container[0].shape)
'''
print(torch.Tensor(container[0].shape))
print(torch.Tensor(471,471))

# Need to pad images
#for i in range(length):

#input = torch.Tensor(471,471)
input = container[0].shape

padded = PadImage(torch.Tensor(input))

print(padded(input).shape)
'''
# Need to rotate and make larger dataset


# Remove images with no label
length = len(container)
counter = 0
for i in range(length):
    i -= counter
    if container[i].label == None:
        del container[i]
        counter += 1

'''
print(container[0])
{'image_path': 'C:\\Users\\Krist\\Documents\\dataset\\5577\\nya6_20141018_174307_5577_cal.png',
 'datasetname': 'green',
 'dataset_type': 'png',
 'wavelength': None,
 'timepoint': '2014-10-18 17:17:07',
 'shape': [469, 469],
 'label': 'aurora-less',
 'human_prediction': True,
 'score': {}}
'''


# rotation class: numpy arrays. Padding class: pytorch tensors
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


loader = DatasetLoader(container=container, transforms=transforms)

train_loader = torch.utils.data.DataLoader(dataset      = loader,
                                           num_workers  = 0,
                                           batch_size   = 2,
                                           shuffle      = True)


model = Model(1, 4, 96)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('The number of params in Million: ', params/1e6)

loss         = torch.nn.CrossEntropyLoss()
optimizer    = torch.optim.Adam(params=model.parameters(), lr=2e-3, amsgrad=True)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.1)


trainer = Trainer(model             = model,
                  loss_function     = loss,
                  optimizer         = optimizer,
                  data_loader       = train_loader,
                  valid_data_loader = train_loader,
                  lr_scheduler      = lr_scheduler,
                  epochs            = 200,
                  save_period       = 50,
                  savedir           = './models',
                  device            = 'cpu',
                  )
# device            = 'cpu'
# device            = 'cuda:0',
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ??

trainer.train()
