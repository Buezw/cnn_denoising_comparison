# evaluate.py
import torch
from torch.utils.data import DataLoader
import argparse
import os
import csv
from torchvision.utils import save_image
from models import alexnet, vgg, nin  # 其他模型后续添加
from data.mnist_noisy_dataset import NoisyMNIST
from utils.metrics import calculate_psnr, calculate_ssim

def get_model(name):
    if name == "alexnet":
        return alexnet.AlexNetDenoiser()
    elif name == "vgg":
        return vgg.VGGDenoiser()
    elif name == "nin":
        return nin.NiNDenoiser()
    else:
        raise ValueError(f"Unknown model: {name}")

def evaluate(model_name="alexnet", noise_level=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name).to(device)
    model.eval()

    os.makedirs(f"results/output_images/{model_name}/eval", exist_ok=True)
    os.makedirs(f"results/metrics", exist_ok=True)

    test_set = NoisyMNIST(train=False, noise_level=noise_level)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    psnr_list, ssim_list = [], []
    csv_path = f"results/metrics/{model_name}_eval.csv"

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Index', 'PSNR', 'SSIM'])

        with torch.no_grad():
            for idx, (noisy, clean) in enumerate(test_loader):
                noisy, clean = noisy.to(device), clean.to(device)
                output, _ = model(noisy)

                psnr = calculate_psnr(clean, output)
                ssim = calculate_ssim(clean, output)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                # 保存图像
                save_image(output, f"results/output_images/{model_name}/eval/{idx:03d}_denoised.png")
                writer.writerow([idx, f"{psnr:.4f}", f"{ssim:.4f}"])

    print(f"[{model_name}] Avg PSNR: {sum(psnr_list)/len(psnr_list):.2f}, SSIM: {sum(ssim_list)/len(ssim_list):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alexnet")
    args = parser.parse_args()
    evaluate(model_name=args.model)
