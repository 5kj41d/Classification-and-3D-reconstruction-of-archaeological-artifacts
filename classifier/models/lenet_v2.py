import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),  # 3 input channels (RGB), 6 output channels, 5x5 kernel
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5), # 6 input channels, 16 output channels, 5x5 kernel
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 13 * 13, 120),  # 16*13*13 is the size of the input to the first fully connected layer
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),  # 2 output classes: Coin and Other Artifacts
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Apply convolutional layers
        x = self.conv_layers(x)
        # Flatten the output for fully connected layers
        x = x.view(-1, self.num_flat_features(x))
        # Apply fully connected layers
        x = self.fc_layers(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # Exclude batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features