"""
Server learning-rate sweep at a fixed heavy-tailed channel.

The thesis §5.2 (Limitations) flags the absence of a per-method
learning-rate sweep as a weakness of the alpha-stable conclusion: the
non-monotonic alpha trend in Table 4.5 could come from a single global
lr that is not equally well matched to every (method, α) pair.

This script runs each method at a fixed (α, γ) channel and a list of
server learning rates, reporting final accuracy mean ± std across seeds.
The output JSON is consumable by the existing src/experiments/report.py
table writer (extension method `write_lr_sweep_table` if added) or by
plot_mechanism.py via diagnostics.

Usage:
    uv run -m src.experiments.lr_sweep \
        --dataset cifar10 --model resnet18 \
        --mac_alpha 1.3 --noise_scale 0.05 \
        --server_lrs 1e-3,3e-3,1e-2,3e-2,1e-1 \
        --seeds 42,43,44 \
        --rounds 100 --num_clients 100
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .accelerate import (
    add_accelerator_args,
    accelerator_summary,
    resolve_accelerator_args,
)
from .heavy_tail import (
    DEFAULT_METHODS,
    DEFAULT_SEEDS,
    _summary,
    run_trial,
)


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_lr_sweep(args, device: torch.device) -> dict:
    print(f"\n=== LR sweep at α={args.mac_alpha}, γ={args.noise_scale} ===")
    results: dict[str, dict[str, dict]] = {}
    for lr in args.server_lrs:
        lr_key = f"{lr:g}"
        results[lr_key] = {}
        for method in args.methods:
            print(f"  lr={lr_key:<8} method={method}", flush=True)
            trial_args = argparse.Namespace(**vars(args))
            trial_args.server_lr = lr
            seeds = args.seeds
            if len(seeds) > 1:
                with ThreadPoolExecutor(max_workers=len(seeds)) as executor:
                    futures = [
                        executor.submit(
                            run_trial, method, args.mac_alpha, seed,
                            args.use_mac, trial_args, device,
                        )
                        for seed in seeds
                    ]
                    runs = [f.result() for f in futures]
            else:
                runs = [run_trial(method, args.mac_alpha, seeds[0],
                                  args.use_mac, trial_args, device)]
            results[lr_key][method] = {
                "runs": runs,
                "summary": _summary(runs),
            }
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Per-method server learning-rate sweep at fixed (α, γ)")
    p.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    p.add_argument("--model", default="resnet18",
                   choices=["mlp", "convnet", "resnet18", "resnet34"])
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--num_clients", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--server_lrs", type=_parse_float_list,
                   default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--noise_scale", type=float, default=0.05)
    p.add_argument("--mac_alpha", type=float, default=1.3,
                   help="Channel tail index for the sweep (single value).")
    p.add_argument("--mac_clip", type=float, default=3.0)
    p.add_argument("--use_mac", action="store_true", default=False)
    p.add_argument("--methods", type=_parse_str_list, default=DEFAULT_METHODS)
    p.add_argument("--seeds", type=_parse_int_list, default=DEFAULT_SEEDS)
    p.add_argument("--non_iid", action="store_true", default=True)
    p.add_argument("--dir_conc", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--save_diagnostics", action="store_true", default=False)
    p.add_argument("--out_dir", type=str, default="results/heavytail")
    add_accelerator_args(p)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolve_accelerator_args(args, device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    print(f"Device: {device} | Config: {vars(args)}")
    print(accelerator_summary(args))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_lr_sweep(args, device)
    payload = {
        "study": "lr_sweep",
        "dataset": args.dataset,
        "model": args.model,
        "mac_alpha": args.mac_alpha,
        "noise_scale": args.noise_scale,
        "args": vars(args),
        "results": results,
    }
    path = out_dir / f"lr_sweep_{args.dataset}_{args.model}_alpha{args.mac_alpha:g}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")
