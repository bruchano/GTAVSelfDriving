import torch
import torchvision
from torchvision import utils, transforms
import cv2
import numpy as np


class SelfDriving(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(in_features=3, out_features=6, kernel_size=3)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvLayer(in_features=6, out_features=6, kernel_size=3)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = ConvLayer(in_features=6, out_features=12, kernel_size=3)
        self.conv3_2 = ConvLayer(in_features=12, out_features=16, kernel_size=3)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear = torch.nn.Linear(in_features=2560, out_features=4)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.maxpool1(self.conv1(x)))
        x = self.relu(self.maxpool2(self.conv2(x)))
        x = self.maxpool3(self.conv3_2(self.conv3_1(x)))
        return self.linear(x.view(1, -1))


class ConvLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, kernel_size=1, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.refpadding = torch.nn.ReflectionPad2d(padding)
        self.conv = torch.nn.Conv2d(in_features, out_features, kernel_size, stride)

    def forward(self, x):
        x = self.refpadding(x)
        return self.conv(x)
