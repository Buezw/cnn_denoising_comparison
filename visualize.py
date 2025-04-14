# visualize.py
import torch
import argparse
import os
from torchvision.utils import save_image
from models import alexnet, vgg, nin
from data.mnist_noisy_dataset import NoisyMNIST
from torch.utils.data import DataLoader

def get_model(name):
    if name == "alexnet":
        return alexnet.AlexNetDenoiser()
    elif name == "vgg":
        return vgg.VGGDenoiser()
    elif name == "nin":
        return nin.NiNDenoiser()
    else:
        raise ValueError(f"Unknown model: {name}")

def visualize_feature_maps(model_name="alexnet", max_channels=8, batch_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device)
    model.eval()

    os.makedirs(f"results/feature_maps/{model_name}", exist_ok=True)

    dataset = NoisyMNIST(train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 取第 batch_idx 张图像
    for idx, (noisy, _) in enumerate(loader):
        if idx == batch_idx:
            noisy = noisy.to(device)
            _, features = model(noisy)  # encoder 输出
            break

    # features: (1, C, H, W)
    for i in range(min(max_channels, features.shape[1])):
        channel_map = features[0, i].unsqueeze(0)  # shape: (1, H, W)
        save_image(channel_map, f"results/feature_maps/{model_name}/map_{batch_idx:03d}_ch{i}.png")

    print(f"[{model_name}] Feature maps (up to {max_channels}) saved for sample #{batch_idx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alexnet")
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    visualize_feature_maps(model_name=args.model, max_channels=args.channels, batch_idx=args.index)
