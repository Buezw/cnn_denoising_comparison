# train.py

import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import argparse

from models import alexnet, vgg  # 继续添加其他模型时可扩展
from data.mnist_noisy_dataset import NoisyMNIST
from utils.plot_tools import save_loss_curve, save_denoised_output, save_feature_map

def get_model(name):
    if name == "alexnet":
        return alexnet.AlexNetDenoiser()
    elif name == "vgg":
        return vgg.VGGDenoiser()
    else:
        raise ValueError(f"Unknown model: {name}")

def train(model_name="alexnet", epochs=10, batch_size=64, lr=1e-3, log_fn=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device)

    # 权重初始化
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    # 数据加载
    train_set = NoisyMNIST(train=True, noise_level=0.5)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 优化器 & 损失
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    os.makedirs(f"results/output_images/{model_name}/train", exist_ok=True)
    os.makedirs(f"results/feature_maps/{model_name}/train", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output, features = model(noisy)

            loss = criterion(output, clean)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        # 日志输出：可 Streamlit 或终端打印
        msg = f"[{epoch+1}/{epochs}] Loss: {avg_loss:.4f}"
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

        # 保存输出图 & 特征图
        save_denoised_output(output, epoch, model_name, noisy=noisy, clean=clean)
        save_feature_map(features, epoch, model_name)

    save_loss_curve(losses, model_name)
    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alexnet")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train(model_name=args.model, epochs=args.epochs, lr=args.lr)
