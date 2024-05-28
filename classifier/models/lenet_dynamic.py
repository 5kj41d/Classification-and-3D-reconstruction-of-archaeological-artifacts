import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5_Dynamic(nn.Module):
    def __init__(self, conv_kernel, conv_stride, pool_kernel, pool_stride, fc1_size, fc2_size):
        super(LeNet5_Dynamic, self).__init__()
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=conv_kernel, stride=conv_stride),  # 3 input channels (RGB), 6 output channels, 5x5 kernel
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride),
            nn.Conv2d(6, 16, kernel_size=conv_kernel, stride=conv_stride), # 6 input channels, 16 output channels, 5x5 kernel
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)
        )
        # Calculate output size of convolutional layers
        self._calculate_conv_output_size(torch.randn(1, 3, 64, 64))
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, fc1_size),  
            nn.ReLU(),
            nn.Linear(fc1_size, fc2_size),
            nn.ReLU(),
            nn.Linear(fc2_size, 2),  # 2 output classes: Coin and Other Artifacts
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

    def _calculate_conv_output_size(self, x):
        # Forward pass through convolutional layers to get output size
        with torch.no_grad():
            conv_output = self.conv_layers(x)
        self.conv_output_size = conv_output.view(conv_output.size(0), -1).size(1)