import torch
import torch.nn.functional as F
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 ):
        super().__init__()


        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)

        self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)

        self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = nn.ReLU(inplace=True)(x)

        return x



class Model(nn.Module):

    def __init__(self,
                 n_channels: int = 1,
                 n_classes: int = 4,
                 n: int = 64,
                 ):

        super(Model, self).__init__()

        self.n_channels = n_channels
        self.n_classes  = n_classes

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(n)
        self.act   = nn.ReLU(inplace=True)

        self.double1 = DoubleConv(in_channels=n, out_channels=n)

        # Half the resolution
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double1 = DoubleConv(in_channels=n, out_channels=n)  # Double conv layer

        # Half the resolution
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double2 = DoubleConv(in_channels=n, out_channels=n)  # Double conv layer

        # Half the resolution
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double3 = DoubleConv(in_channels=n, out_channels=n)  # Double conv layer

        # Half the resolution
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double4 = DoubleConv(in_channels=n, out_channels=n)  # Double conv layer

        # Half the resolution
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double5 = DoubleConv(in_channels=n, out_channels=n)  # Double conv layer

        self.conv2 = nn.Conv2d(in_channels=n, out_channels=n, kernel_size=2, padding=0, bias=False)
        self.norm2 = nn.BatchNorm2d(n)

        # Half the resolution
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double6 = DoubleConv(in_channels=n, out_channels=n)  # Double conv layer

        self.linear1 = nn.Linear(in_features=n*3*3, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=n_classes)


    def forward(self, x):

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.double1(x)

        # Half the resolution
        x = self.pool1(x)
        x = self.double1(x)  # Double conv layer

        # Half the resolution
        x = self.pool2(x)
        x = self.double2(x)  # Double conv layer

        # Half the resolution
        x = self.pool3(x)
        x = self.double3(x)  # Double conv layer

        # Half the resolution
        x = self.pool4(x)
        x = self.double4(x)  # Double conv layer

        # Half the resolution
        x = self.pool5(x)
        x = self.double5(x)  # Double conv layer

        x = self.conv2(x)
        x = self.norm2(x)

        # Half the resolution
        x = self.pool6(x)
        x = self.double6(x)  # Double conv layer

        x = torch.flatten(input=x, start_dim=1, end_dim=-1)

        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)

        return x
