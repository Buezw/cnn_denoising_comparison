# data/mnist_noisy_dataset.py
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import torch

class NoisyMNIST(Dataset):
    def __init__(self, train=True, noise_level=0.5):
        self.data = torchvision.datasets.MNIST(
            root='./data', train=train, download=True,
            transform=transforms.ToTensor()
        )
        self.noise_level = noise_level

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, _ = self.data[idx]
        noisy_img = img + self.noise_level * torch.randn_like(img)
        noisy_img = torch.clamp(noisy_img, 0., 1.)
        return noisy_img, img  # 输入是噪声图像，输出是干净图像（监督目标）
