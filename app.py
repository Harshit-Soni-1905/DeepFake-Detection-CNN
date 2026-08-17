import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class DeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolution Layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()

        # Fully Connected Layer
        self.fc1 = nn.Linear(128 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # First convolution block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Second convolution block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Third convolution block
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully Connected Layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output Layer
        x = self.fc2(x)

        return x
