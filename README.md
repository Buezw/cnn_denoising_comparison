# 🧠 CNN Denoising Comparison

This project provides a unified framework to compare classic CNN architectures (AlexNet, VGG, NiN, GoogLeNet, ResNet, DenseNet) on an image denoising task using MNIST. It includes model definition, training, evaluation (PSNR/SSIM), and feature map visualization.

---

## 📌 Project Objectives

- Compare different CNN architectures in an **image-to-image** task (denoising)
- Analyze:
  - Reconstruction quality (visually and quantitatively)
  - Structure-level behavior (feature reuse, convergence, etc.)
  - Efficiency metrics (parameters, inference time)
- Provide clean visualization and evaluation outputs

---

## 🏗 Supported Architectures

| Model      | Special Structure         | Denoising Strategy         |
|------------|---------------------------|----------------------------|
| AlexNet    | Large conv kernels        | Shallow, basic             |
| VGG        | Deep 3x3 stacks           | Powerful encoder           |
| NiN        | 1x1 conv (MLPConv)        | Rich local nonlinearity    |
| GoogLeNet  | Inception multi-path      | Multi-scale filtering      |
| ResNet     | Residual connections      | Better gradient flow       |
| DenseNet   | Dense skip connections    | High feature reuse         |

---

## 🚀 Quick Start

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🏋️‍♂️ Train a Model

```bash
python train.py --model vgg --epochs 10
```

Denoised outputs, loss curve and feature maps will be saved to `results/`.

### 📈 Evaluate PSNR / SSIM

```bash
python evaluate.py --model vgg
```

Results will be saved in `results/metrics/vgg_eval.csv`.

### 🧠 Visualize Feature Maps

```bash
python visualize.py --model vgg --index 0 --channels 8
```

Saves selected encoder feature maps to `results/feature_maps/vgg/`.

---

## 🗂 Project Structure

```bash
cnn_denoising_comparison/
├── models/
│   ├── alexnet.py         # AlexNet encoder + decoder
│   ├── vgg.py             # VGG-style CNN
│   ├── ...                # Other architectures
│   └── shared_decoder.py  # Common decoder module
├── data/
│   └── mnist_noisy_dataset.py  # Noisy input generator
├── utils/
│   ├── metrics.py         # PSNR / SSIM calculation
│   └── plot_tools.py      # Loss curve, image, fmap saving
├── train.py               # Training entry
├── evaluate.py            # Evaluation entry
├── visualize.py           # Feature map visualizer
├── requirements.txt
└── README.md
```

---

## 📊 Output Examples

- 🖼 Denoised output images: `results/output_images/{model}/`
- 📈 Loss curves: `results/loss_curves/{model}_loss_curve.png`
- 🧠 Feature maps: `results/feature_maps/{model}/epochXX_chY.png`
- 📋 Metrics CSV: `results/metrics/{model}_eval.csv`

*(See `examples/` folder for visual comparisons)*

---

## 🔧 TODO / Extensions

- [ ] Add GoogLeNet / ResNet / DenseNet models
- [ ] Add Gradio interface for interactive inference
- [ ] Add CIFAR-10 support
- [ ] Add visual embedding t-SNE

---

## 📎 License

MIT License. This project is intended for educational and research comparison purposes.
```

---
---

## ⏱ Training Performance Profiling

`profile_train.py` profiles a training step with `torch.profiler` (to find the bottleneck), then times steps **without** the profiler (median of several 100-step runs) and can train + evaluate PSNR/SSIM:

```bash
python profile_train.py --model vgg --tag baseline
python profile_train.py --model vgg --tag cl --channels-last
python profile_train.py --model vgg --tag cl_acc --channels-last --train-epochs 3 --eval
```

VGG, batch 64, CPU only (4 vCPU Intel Xeon @ 2.8 GHz, Cascade Lake, KVM), PyTorch 2.15:

| Config | Step time | Throughput | Test PSNR / SSIM (3 epochs) |
|---|---|---|---|
| Baseline (NCHW) | ~240 ms | ~265 img/s | 20.63 dB / 0.874 |
| `channels_last` | ~200 ms | ~320 img/s | 20.65 dB / 0.876 |

The profiler showed backward ≈ 56 % and forward ≈ 35 % of the step, with data loading only ≈ 6 %. `aten::convolution_backward` alone took ≈ 48 % of self CPU time, so the conv kernels were the bottleneck. Switching to `channels_last` (NHWC) lets oneDNN skip layout reorders. Trace files go to `prof/` (open with TensorBoard).
