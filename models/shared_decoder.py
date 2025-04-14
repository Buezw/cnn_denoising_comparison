# models/shared_decoder.py（稳定版）
import torch.nn as nn

class SharedDecoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decode(x)
