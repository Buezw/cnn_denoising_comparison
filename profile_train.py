# profile_train.py
# Profile one training step of a denoiser, time steps without the profiler,
# and (optionally) train + evaluate PSNR/SSIM to check accuracy is unchanged.
#
#   python profile_train.py --model vgg --tag baseline
#   python profile_train.py --model vgg --tag workers4 --num-workers 4 --pin-memory
#   python profile_train.py --model vgg --tag baseline --train-epochs 2 --eval
import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader

from data.mnist_noisy_dataset import NoisyMNIST
from train import get_model
from utils.metrics import calculate_psnr, calculate_ssim


def make_loader(train, args, shuffle):
    ds = NoisyMNIST(train=train, noise_level=0.5)
    kw = dict(batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
              pin_memory=args.pin_memory and torch.cuda.is_available())
    if args.num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return DataLoader(ds, **kw)


def train_step(model, noisy, clean, criterion, opt, device, args):
    noisy = noisy.to(device, non_blocking=True)
    clean = clean.to(device, non_blocking=True)
    if args.channels_last:
        noisy = noisy.contiguous(memory_format=torch.channels_last)
    with record_function("forward"):
        out, _ = model(noisy)
        loss = criterion(out, clean)
    with record_function("backward"):
        opt.zero_grad(set_to_none=True)
        loss.backward()
    with record_function("optimizer_step"):
        opt.step()
    return loss


def batches(loader):
    """Endless iterator over a loader; wraps the next() call so DataLoader wait shows in the profile."""
    while True:
        it = iter(loader)
        while True:
            with record_function("data_loading"):
                try:
                    b = next(it)
                except StopIteration:
                    break
            yield b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vgg")
    p.add_argument("--tag", default="baseline")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--threads", type=int, default=0, help="torch.set_num_threads; 0 = default")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--timed-steps", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--train-epochs", type=int, default=0)
    p.add_argument("--eval", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = make_loader(True, args, shuffle=True)
    model.train()
    out_dir = f"results/profiling/{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. profiler (finds the bottleneck; its overhead skews absolute times) ----
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    gen = batches(loader)
    with profile(activities=acts, schedule=schedule(wait=2, warmup=3, active=10),
                 on_trace_ready=tensorboard_trace_handler(f"./prof/{args.model}_{args.tag}"),
                 record_shapes=True, profile_memory=True) as prof:
        for step in range(15):
            noisy, clean = next(gen)
            train_step(model, noisy, clean, criterion, opt, device, args)
            prof.step()
    ka = prof.key_averages()
    table = ka.table(sort_by="self_cpu_time_total", row_limit=15)
    print(table)
    with open(f"{out_dir}/{args.tag}_profile_table.txt", "w") as f:
        f.write(table)
    total_self = sum(e.self_cpu_time_total for e in ka)
    top_ops = [{"op": e.key, "self_cpu_ms_per_step": e.self_cpu_time_total / 1e3 / 10,
                "pct": 100 * e.self_cpu_time_total / total_self}
               for e in sorted(ka, key=lambda e: -e.self_cpu_time_total)[:10]]
    phases = {}
    for e in ka:
        if e.key in ("data_loading", "forward", "backward", "optimizer_step"):
            phases[e.key] = e.cpu_time_total / 1e3 / 10  # ms per step (inclusive)

    # ---- 2. clean timing, no profiler ----
    gen = batches(loader)
    for _ in range(args.warmup_steps):
        noisy, clean = next(gen)
        train_step(model, noisy, clean, criterion, opt, device, args)
    runs = []
    for _ in range(args.repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.timed_steps):
            noisy, clean = next(gen)
            train_step(model, noisy, clean, criterion, opt, device, args)
        if device.type == "cuda":
            torch.cuda.synchronize()
        runs.append((time.perf_counter() - t0) / args.timed_steps * 1e3)
    step_ms = float(np.median(runs))
    result = {
        "model": args.model, "tag": args.tag, "device": str(device), "batch_size": args.batch_size,
        "num_workers": args.num_workers, "threads": torch.get_num_threads(),
        "channels_last": args.channels_last,
        "step_ms_runs": runs, "step_ms": step_ms, "throughput_img_s": args.batch_size / step_ms * 1e3,
        "profiler_phase_ms_per_step": phases, "profiler_top_ops": top_ops,
    }
    print(json.dumps({k: result[k] for k in ("step_ms_runs", "step_ms", "throughput_img_s",
                                             "profiler_phase_ms_per_step")}, indent=2))

    # ---- 3. optional: train from scratch + evaluate, to confirm accuracy ----
    if args.train_epochs:
        torch.manual_seed(args.seed)
        model = get_model(args.model).to(device)
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        t0 = time.perf_counter()
        for ep in range(args.train_epochs):
            model.train()
            tot = 0.0
            for noisy, clean in loader:
                tot += train_step(model, noisy, clean, criterion, opt, device, args).item()
            print(f"[{ep+1}/{args.train_epochs}] loss {tot/len(loader):.5f}")
        result["train_wall_s"] = time.perf_counter() - t0
        result["train_epochs"] = args.train_epochs
        torch.save(model.state_dict(), f"{out_dir}/{args.tag}.pt")
    if args.eval:
        torch.manual_seed(1234)  # same noise for every run
        model.eval()
        test = DataLoader(NoisyMNIST(train=False, noise_level=0.5), batch_size=256, shuffle=False)
        psnr, ssim = [], []
        with torch.no_grad():
            for noisy, clean in test:
                x = noisy.to(device)
                if args.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                out, _ = model(x)
                for c, o in zip(clean, out.cpu()):
                    psnr.append(calculate_psnr(c, o))
                    ssim.append(calculate_ssim(c, o))
        result["test_psnr"] = float(np.mean(psnr))
        result["test_ssim"] = float(np.mean(ssim))
        print(f"test PSNR {result['test_psnr']:.2f}  SSIM {result['test_ssim']:.4f}")

    with open(f"{out_dir}/{args.tag}.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
