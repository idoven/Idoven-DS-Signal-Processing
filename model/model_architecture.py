import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGNet(nn.Module):
    def __init__(self, input_channels, signal_length, num_classes):
        super(ECGNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv1d(input_channels, 6, 20)
        self.conv2 = nn.Conv1d(6, 2, 20)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(1924, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = self.conv1(x)
        x = self.conv2(x)
        # If the size is a square, you can specify with a single number
        # x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x