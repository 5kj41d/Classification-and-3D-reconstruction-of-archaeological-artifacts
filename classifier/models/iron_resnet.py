import torch
import torch.nn as nn


class IronResNet(nn.Module):
    """
    A ResNet-like CNN for image classification of iron vs others
    """

    def __init__(self, num_classes):
        super(IronResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_block(64, 64, 2)
        self.layer2 = self._make_block(64, 128, 2)
        self.layer3 = self._make_block(128, 256, 2)
        self.fc = nn.Linear(256 * 256, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def _make_block(self, in_planes, out_planes, num_repeats):
        """
        A helper function to create a residual block
        """
        layers = []
        layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        for i in range(num_repeats - 1):
            layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x
