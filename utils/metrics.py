# utils/metrics.py
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

def calculate_psnr(gt, pred):
    return peak_signal_noise_ratio(gt.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy(), data_range=1.0)

def calculate_ssim(gt, pred):
    return structural_similarity(gt.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy(), data_range=1.0)
