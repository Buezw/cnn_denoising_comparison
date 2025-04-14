# models/alexnet.py

import torch.nn as nn
from models.shared_decoder import SharedDecoder

class AlexNetDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),  # -> 64x6x6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> 64x3x3

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> 192x1x1

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.norm = nn.BatchNorm2d(128)
        self.decoder = SharedDecoder(input_channels=128)

    def forward(self, x):
        features = self.encoder(x)
        features = self.norm(features)
        out = self.decoder(features)
        return out, features
