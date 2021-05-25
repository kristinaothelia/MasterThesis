import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np

from lbl.dataset import DatasetContainer
from lbl.dataset import DatasetLoader

from lbl.models.model    import Model
from lbl.trainer.trainer import Trainer

container = DatasetContainer.from_json('datasets/6300.json')
#container = DatasetContainer.from_json('datasets/6300_k.json')
#container2 = DatasetContainer.from_json('datasets/5577_k_aurora.json')

# Remove images with no label
length = len(container)
counter = 0
for i in range(length):
    i -= counter
    if container[i].label == None:
        del container[i]
        counter += 1

# rotation class: numpy arrays. Padding class: pytorch tensors
transforms = torchvision.transforms.Compose([
    lambda x: np.float32(x),
    lambda x: torch.from_numpy(x),
    lambda x: x.unsqueeze(0),
    torchvision.transforms.Pad(padding=[5, 5, 4, 4], fill=0),
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
                  device            = 'cuda:0',
                  )
# device            = 'cpu'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ??

trainer.train()
