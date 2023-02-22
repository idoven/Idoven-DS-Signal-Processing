import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGNet(nn.Module):
    def __init__(self, input_channels: int, signal_length: int, num_classes: int, config: dict):
        super(ECGNet, self).__init__()
        conv_kernels = int(config['conv_kernels'])
        conv_layers = int(config['conv_layers'])
        conv_net = [nn.Conv1d(input_channels, conv_kernels, 10, padding='same', dilation=2)]
        for _ in range(conv_layers-2):
            conv_net.append(nn.Conv1d(conv_kernels, conv_kernels, 10, padding='same', dilation=2))
        conv_net.append(nn.Conv1d(conv_kernels, 2, 10, padding='same', dilation=2))
        self.conv_net = nn.Sequential(*conv_net)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2000, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = F.max_pool1d(F.relu(self.conv1(x)), 2)
        x = self.conv_net(x)
        # If the size is a square, you can specify with a single number
        # x = F.max_pool1d(F.relu(self.conv2(x)), 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x