"""
Result plotting for comparative analysis and ablation studies.

Generates publication-quality figures matching the proposal's expected outcomes:
  - Fig 1: Test accuracy curves per method (comparison)
  - Fig 2: Training loss curves per method (comparison)
  - Fig 3: Final accuracy vs α (ablation A — validates theoretical link)
  - Fig 4: Final accuracy vs N (ablation B — validates scalability)

Usage:
    uv run src/experiments/plot.py --result results/comparison/cifar10_resnet18_alpha1.5_N10.json
    uv run src/experiments/plot.py --ablation_alpha results/ablation/ablation_alpha_cifar10_resnet18.json
    uv run src/experiments/plot.py --ablation_clients results/ablation/ablation_clients_cifar10_resnet18.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


METHOD_STYLE = {
    "fedavg":      {"label": "FedAvg-OTA",    "color": "#d62728", "ls": ":",  "lw": 1.5},
    "fedavgm":     {"label": "FedAvgM-OTA",   "color": "#ff7f0e", "ls": "--", "lw": 1.8},
    "adagrad_ota": {"label": "AdaGrad-OTA",   "color": "#1f77b4", "ls": "-.", "lw": 2.0},
    "adam_ota":    {"label": "Adam-OTA",       "color": "#2ca02c", "ls": "-",  "lw": 2.2},
}

OPT_STYLE = {
    "fedavgm":     {"label": "FedAvgM-OTA",   "color": "#ff7f0e", "marker": "s"},
    "adagrad_ota": {"label": "AdaGrad-OTA",   "color": "#1f77b4", "marker": "^"},
    "adam_ota":    {"label": "Adam-OTA",       "color": "#2ca02c", "marker": "o"},
}


def _set_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "lines.markersize": 6,
    })


def plot_comparison(result_path: str, out_dir: Path):
    with open(result_path) as f:
        data = json.load(f)

    args_d = data["args"]
    results = data["results"]
    rounds = list(range(1, args_d["rounds"] + 1))

    _set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for method, hist in results.items():
        s = METHOD_STYLE.get(method, {"label": method, "color": "gray", "ls": "-", "lw": 1.5})
        ax1.plot(rounds, hist["acc"], label=s["label"], color=s["color"],
                 ls=s["ls"], lw=s["lw"])
        ax2.plot(rounds, hist["loss"], label=s["label"], color=s["color"],
                 ls=s["ls"], lw=s["lw"])

    ax1.set_xlabel("Communication Rounds")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title(f"Test Accuracy — {args_d['dataset'].upper()}, α={args_d['alpha']}, N={args_d['num_clients']}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Communication Rounds")
    ax2.set_ylabel("Training Loss")
    ax2.set_title(f"Training Loss — {args_d['dataset'].upper()}, α={args_d['alpha']}, N={args_d['num_clients']}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    stem = Path(result_path).stem
    out = out_dir / f"{stem}_curves.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_ablation_alpha(result_path: str, out_dir: Path):
    """
    Fig 3: Final test accuracy vs tail index α for each optimizer.
    Validates: smaller α → worse convergence; Adam-OTA most resilient.
    """
    with open(result_path) as f:
        data = json.load(f)

    results = data["results"]  # {str(alpha): {opt: {acc, loss}}}
    alphas = sorted(float(k) for k in results)

    _set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for opt_name, style in OPT_STYLE.items():
        final_acc = [results[str(a)][opt_name]["acc"][-1] for a in alphas]
        final_loss = [results[str(a)][opt_name]["loss"][-1] for a in alphas]
        ax1.plot(alphas, final_acc, label=style["label"], color=style["color"],
                 marker=style["marker"], lw=2)
        ax2.plot(alphas, final_loss, label=style["label"], color=style["color"],
                 marker=style["marker"], lw=2)

    ax1.set_xlabel("Tail Index α")
    ax1.set_ylabel("Final Test Accuracy")
    ax1.set_title("Effect of Noise Tail Index α on Final Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.invert_xaxis()  # heavier tail (smaller α) on left

    ax2.set_xlabel("Tail Index α")
    ax2.set_ylabel("Final Training Loss")
    ax2.set_title("Effect of Noise Tail Index α on Final Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.invert_xaxis()

    plt.tight_layout()
    stem = Path(result_path).stem
    out = out_dir / f"{stem}_plot.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_ablation_clients(result_path: str, out_dir: Path):
    """
    Fig 4: Test accuracy vs number of clients N.
    Validates: more clients → better accuracy (OTA scalability benefit).
    """
    with open(result_path) as f:
        data = json.load(f)

    results = data["results"]  # {str(N): {acc, loss}}
    Ns = sorted(int(k) for k in results)

    _set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Full training curves colored by N
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(Ns)))
    rounds_count = len(next(iter(results.values()))["acc"])
    rounds = list(range(1, rounds_count + 1))

    for N, color in zip(Ns, cmap):
        r = results[str(N)]
        ax1.plot(rounds, r["acc"], label=f"N={N}", color=color, lw=2)
        ax2.plot(rounds, r["loss"], label=f"N={N}", color=color, lw=2)

    ax1.set_xlabel("Communication Rounds")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Scalability: Test Accuracy vs Rounds (Adam-OTA)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Communication Rounds")
    ax2.set_ylabel("Training Loss")
    ax2.set_title("Scalability: Training Loss vs Rounds (Adam-OTA)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    stem = Path(result_path).stem
    out = out_dir / f"{stem}_plot.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Plot ADOTA-FL experiment results")
    p.add_argument("--result", type=str, default=None,
                   help="Path to comparison JSON")
    p.add_argument("--ablation_alpha", type=str, default=None,
                   help="Path to alpha ablation JSON")
    p.add_argument("--ablation_clients", type=str, default=None,
                   help="Path to client-count ablation JSON")
    p.add_argument("--out_dir", type=str, default="results/figures")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.result:
        plot_comparison(args.result, out_dir)
    if args.ablation_alpha:
        plot_ablation_alpha(args.ablation_alpha, out_dir)
    if args.ablation_clients:
        plot_ablation_clients(args.ablation_clients, out_dir)

    if not any([args.result, args.ablation_alpha, args.ablation_clients]):
        print("No input specified. Use --result, --ablation_alpha, or --ablation_clients.")
