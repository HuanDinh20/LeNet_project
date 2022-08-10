import torch
from torch import nn


class LeNet(nn.Module):
    """this architecture is based on [LeCun et al., 1998b], keep all the original network"""

    def __init__(self, in_channels, num_classes):
        super(LeNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(400, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, self.num_classes)
        )

    def forward(self, X):
        X = self.net(X)
        return X

    def layer_summary(self, X_shape: tuple):
        """
        X_shape: shape of input
        print the shape of each layer
        """
        X = torch.rand(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape: \t", X.shape)



