# models/vgg.py
import torch.nn as nn
from models.shared_decoder import SharedDecoder

class VGGDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x14x14
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 128x7x7
        )
        self.norm = nn.BatchNorm2d(128)  # ✅ 加这一行
        self.decoder = SharedDecoder(input_channels=128)

    def forward(self, x):
        features = self.encoder(x)
        features = self.norm(features)  # ✅ 加这一行
        out = self.decoder(features)
        return out, features
