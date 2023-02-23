import torch
import torch.nn as nn
import torch.nn.functional as F


class ECGNet(nn.Module):
    """ Simple convolutional network to classify ECG signals."""
    def __init__(self, input_channels: int, signal_length: int, num_classes: int, config: dict):
        super(ECGNet, self).__init__()
        conv_kernels = int(config['conv_kernels'])
        conv_layers = int(config['conv_layers'])
        self.relu = nn.ReLU()
        conv_net = [nn.Conv1d(input_channels, conv_kernels, 10, padding='same', dilation=2)]
        conv_net.append(self.relu)
        for _ in range(conv_layers-2):
            conv_net.append(nn.Conv1d(conv_kernels, conv_kernels, 10, padding='same', dilation=2))
            conv_net.append(self.relu)
        conv_net.append(nn.Conv1d(conv_kernels, 2, 10, padding='same', dilation=2))
        conv_net.append(self.relu)
        self.conv_net = nn.Sequential(*conv_net)

        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2000, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_net(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.activation(x)
        return x