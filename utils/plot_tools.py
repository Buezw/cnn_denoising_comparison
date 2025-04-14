# utils/plot_tools.py
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
import torch

def save_loss_curve(losses, model_name):
    os.makedirs("results/loss_curves", exist_ok=True)
    plt.figure()
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{model_name.upper()} Training Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/loss_curves/{model_name}_loss_curve.png")
    plt.close()

def save_denoised_output(output, epoch, model_name, noisy=None, clean=None, max_images=5):
    """
    保存对比图（clean | noisy | output），每行一种图像类型
    """
    import torchvision.utils as vutils

    save_dir = f"results/output_images/{model_name}/train"
    os.makedirs(save_dir, exist_ok=True)

    imgs = [output[:max_images].detach().cpu()]
    if noisy is not None:
        imgs.insert(0, noisy[:max_images].detach().cpu())
    if clean is not None:
        imgs.insert(0, clean[:max_images].detach().cpu())

    # 纵向堆叠
    stacked = torch.cat(imgs, dim=0)
    grid = vutils.make_grid(stacked, nrow=max_images, normalize=True, value_range=(0, 1))

    plt.figure(figsize=(max_images * 2, len(imgs) * 2))
    plt.axis('off')
    plt.imshow(grid.permute(1, 2, 0))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch:02d}.png")
    plt.close()

def save_feature_map(feature_tensor, epoch, model_name, max_channels=8):
    os.makedirs(f"results/feature_maps/{model_name}/train", exist_ok=True)
    fmap = feature_tensor[0]  # (C, H, W)
    for i in range(min(max_channels, fmap.shape[0])):
        ch = fmap[i].unsqueeze(0)
        save_image(ch, f"results/feature_maps/{model_name}/train/epoch{epoch:02d}_ch{i}.png")
