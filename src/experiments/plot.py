"""
Plotting utilities for the comparison and ablation experiments.

Figures:
  - Test accuracy / training loss curves per method (comparison)
  - Final accuracy / loss vs noise scale γ (ablation A — implicit noise filter)
  - Test accuracy / training loss vs rounds, coloured by N (ablation B — averaging gain)

Usage:
    uv run -m src.experiments.plot --result results/comparison/<file>.json
    uv run -m src.experiments.plot --ablation_noise   results/ablation/ablation_noise_<...>.json
    uv run -m src.experiments.plot --ablation_clients results/ablation/ablation_clients_<...>.json
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
    "fedavg":      {"label": "FedAvg-OTA",    "color": "#d62728", "marker": "x"},
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
    _set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for method, hist in results.items():
        s = METHOD_STYLE.get(method, {"label": method, "color": "gray", "ls": "-", "lw": 1.5})
        rounds = hist.get("eval_rounds") or list(range(1, len(hist["acc"]) + 1))
        ax1.plot(rounds, hist["acc"], label=s["label"], color=s["color"],
                 ls=s["ls"], lw=s["lw"])
        ax2.plot(rounds, hist["loss"], label=s["label"], color=s["color"],
                 ls=s["ls"], lw=s["lw"])

    finite_losses = [
        value
        for hist in results.values()
        for value in hist.get("loss", [])
        if np.isfinite(value) and value > 0
    ]
    if finite_losses and max(finite_losses) / max(min(finite_losses), 1e-12) > 100:
        ax2.set_yscale("log")
        ax2.text(
            0.02,
            0.96,
            "log scale",
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    title_suffix = (f"{args_d['dataset'].upper()}, "
                    f"γ={args_d['noise_scale']}, N={args_d['num_clients']}")

    ax1.set_xlabel("Communication Rounds")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title(f"Test Accuracy — {title_suffix}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Communication Rounds")
    ax2.set_ylabel("Test Loss")
    ax2.set_title(f"Test Loss — {title_suffix}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    stem = Path(result_path).stem
    out = out_dir / f"{stem}_curves.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_ablation_noise(result_path: str, out_dir: Path):
    """
    Final test accuracy / loss vs AWGN noise scale γ for each optimizer.
    Validates: as γ grows, FedAvg/FedAvgM degrade while AdaGrad/Adam stay stable
    (the second-moment accumulator acts as an implicit learning-rate filter).
    """
    with open(result_path) as f:
        data = json.load(f)

    results = data["results"]  # {str(gamma): {opt: {acc, loss}}}
    gammas = sorted(float(k) for k in results)

    _set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for opt_name, style in OPT_STYLE.items():
        if not all(opt_name in results[str(g)] for g in gammas):
            continue
        final_acc = [results[str(g)][opt_name]["acc"][-1] for g in gammas]
        final_loss = [results[str(g)][opt_name]["loss"][-1] for g in gammas]
        ax1.plot(gammas, final_acc, label=style["label"], color=style["color"],
                 marker=style["marker"], lw=2)
        ax2.plot(gammas, final_loss, label=style["label"], color=style["color"],
                 marker=style["marker"], lw=2)

    ax1.set_xscale("log")
    ax1.set_xlabel("Noise scale γ (log)")
    ax1.set_ylabel("Final Test Accuracy")
    ax1.set_title("Effect of AWGN Noise Scale on Final Accuracy")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    ax2.set_xscale("log")
    ax2.set_xlabel("Noise scale γ (log)")
    ax2.set_ylabel("Final Test Loss")
    ax2.set_title("Effect of AWGN Noise Scale on Final Loss")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

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
    ax2.set_ylabel("Test Loss")
    ax2.set_title("Scalability: Test Loss vs Rounds (Adam-OTA)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    stem = Path(result_path).stem
    out = out_dir / f"{stem}_plot.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def _summary_mean_std(block: dict, metric: str) -> tuple[float | None, float | None]:
    summary = block.get("summary", {})
    stats = summary.get(metric, {})
    return stats.get("mean"), stats.get("std")


def plot_alpha_ablation(result_path: str, out_dir: Path):
    """
    Final accuracy vs alpha-stable tail index alpha.
    Smaller alpha means heavier-tailed channel interference.
    """
    with open(result_path) as f:
        data = json.load(f)

    results = data["results"]
    alphas = sorted(float(k) for k in results)

    _set_style()
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.8))

    for opt_name, style in OPT_STYLE.items():
        means, stds, xs = [], [], []
        for alpha in alphas:
            block = results[f"{alpha:g}"].get(opt_name)
            if not block:
                continue
            mean, std = _summary_mean_std(block, "final_acc")
            if mean is None:
                continue
            xs.append(alpha)
            means.append(mean)
            stds.append(std or 0.0)
        if xs:
            ax.errorbar(xs, means, yerr=stds, label=style["label"],
                        color=style["color"], marker=style["marker"],
                        lw=2, capsize=3)

    ax.set_xlabel(r"Tail index $\alpha$ (2.0 = AWGN)")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("Effect of Alpha-Stable Interference")
    ax.xaxis.set_major_locator(mticker.FixedLocator(alphas))
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    stem = Path(result_path).stem
    out = out_dir / f"{stem}_plot.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_mac_compare(result_path: str, out_dir: Path):
    """Bar chart comparing no-MAC and MAC final accuracy."""
    with open(result_path) as f:
        data = json.load(f)

    results = data["results"]
    methods = [method for method in OPT_STYLE if method in results.get("no_mac", {})]
    labels = [OPT_STYLE[m]["label"] for m in methods]
    x = np.arange(len(methods))
    width = 0.36

    def series(key: str):
        means, stds = [], []
        for method in methods:
            mean, std = _summary_mean_std(results[key][method], "final_acc")
            means.append(mean if mean is not None else 0.0)
            stds.append(std if std is not None else 0.0)
        return means, stds

    no_mac_mean, no_mac_std = series("no_mac")
    mac_mean, mac_std = series("mac")

    _set_style()
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
    ax.bar(x - width / 2, no_mac_mean, width, yerr=no_mac_std,
           label="No MAC", color="#9e9e9e", capsize=3)
    ax.bar(x + width / 2, mac_mean, width, yerr=mac_std,
           label="MAC", color="#4c78a8", capsize=3)

    alpha = data["args"].get("mac_alpha", "?")
    gamma = data["args"].get("noise_scale", "?")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title(rf"MAC vs No-MAC ($\alpha$={alpha}, $\gamma$={gamma})")
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.legend()
    ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    stem = Path(result_path).stem
    out = out_dir / f"{stem}_plot.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_lr_sweep(result_path: str, out_dir: Path):
    """Final accuracy vs server learning rate, one curve per method (mean ± std)."""
    with open(result_path) as f:
        data = json.load(f)

    results = data["results"]
    lrs = sorted(float(k) for k in results)

    _set_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.8))

    for opt_name, style in OPT_STYLE.items():
        means, stds, xs = [], [], []
        for lr in lrs:
            block = results[f"{lr:g}"].get(opt_name)
            if not block:
                continue
            mean, std = _summary_mean_std(block, "final_acc")
            if mean is None:
                continue
            xs.append(lr)
            means.append(mean)
            stds.append(std or 0.0)
        if xs:
            ax.errorbar(xs, means, yerr=stds, label=style["label"],
                        color=style["color"], marker=style["marker"],
                        lw=2, capsize=3)

    ax.set_xscale("log")
    alpha = data.get("mac_alpha", data.get("args", {}).get("mac_alpha", "?"))
    gamma = data.get("noise_scale", data.get("args", {}).get("noise_scale", "?"))
    ax.set_xlabel(r"Server learning rate $\eta$ (log)")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title(rf"Server LR sweep ($\alpha$={alpha}, $\gamma$={gamma})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    stem = Path(result_path).stem
    out = out_dir / f"{stem}_plot.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_mac_sweep(result_path: str, out_dir: Path):
    """Heatmap of final accuracy across (clip factor k, noise scale γ) for each method."""
    with open(result_path) as f:
        data = json.load(f)

    results = data["results"]
    gammas = sorted(float(g) for g in results)
    ks = sorted({float(k) for g_key in results for k in results[g_key].keys()})
    methods = [m for m in OPT_STYLE if any(
        m in results[f"{g:g}"].get(f"{k:g}", {})
        for g in gammas for k in ks
    )]
    if not methods:
        print(f"No methods found in {result_path}")
        return

    _set_style()
    fig, axes = plt.subplots(1, len(methods), figsize=(4.2 * len(methods), 4.2),
                             squeeze=False)
    for idx, method in enumerate(methods):
        ax = axes[0, idx]
        grid = np.full((len(gammas), len(ks)), np.nan)
        for i, g in enumerate(gammas):
            for j, k in enumerate(ks):
                block = results[f"{g:g}"].get(f"{k:g}", {}).get(method)
                mean, _ = _summary_mean_std(block or {}, "final_acc")
                if mean is not None:
                    grid[i, j] = mean
        im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(ks)), [f"{k:g}" for k in ks])
        ax.set_yticks(range(len(gammas)), [f"{g:g}" for g in gammas])
        ax.set_xlabel("Clip factor k  (0 = no MAC)")
        ax.set_ylabel(r"Noise scale $\gamma$")
        ax.set_title(OPT_STYLE[method]["label"])
        for i in range(len(gammas)):
            for j in range(len(ks)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j] * 100:.1f}",
                            ha="center", va="center",
                            color="white" if grid[i, j] < 0.4 else "black",
                            fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    alpha = data.get("args", {}).get("mac_alpha", "?")
    fig.suptitle(rf"MAC sweep: final accuracy at $\alpha$={alpha}")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    stem = Path(result_path).stem
    out = out_dir / f"{stem}_plot.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="Plot ADOTA-FL experiment results")
    p.add_argument("--result", type=str, default=None,
                   help="Path to comparison JSON")
    p.add_argument("--ablation_noise", type=str, default=None,
                   help="Path to noise-scale ablation JSON")
    p.add_argument("--ablation_clients", type=str, default=None,
                   help="Path to client-count ablation JSON")
    p.add_argument("--alpha_ablation", type=str, default=None,
                   help="Path to alpha-stable ablation JSON")
    p.add_argument("--mac_compare", type=str, default=None,
                   help="Path to MAC comparison JSON")
    p.add_argument("--lr_sweep", type=str, default=None,
                   help="Path to LR sweep JSON")
    p.add_argument("--mac_sweep", type=str, default=None,
                   help="Path to MAC k×γ sweep JSON")
    p.add_argument("--out_dir", type=str, default="results/figures")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.result:
        plot_comparison(args.result, out_dir)
    if args.ablation_noise:
        plot_ablation_noise(args.ablation_noise, out_dir)
    if args.ablation_clients:
        plot_ablation_clients(args.ablation_clients, out_dir)
    if args.alpha_ablation:
        plot_alpha_ablation(args.alpha_ablation, out_dir)
    if args.mac_compare:
        plot_mac_compare(args.mac_compare, out_dir)
    if args.lr_sweep:
        plot_lr_sweep(args.lr_sweep, out_dir)
    if args.mac_sweep:
        plot_mac_sweep(args.mac_sweep, out_dir)

    if not any([
        args.result,
        args.ablation_noise,
        args.ablation_clients,
        args.alpha_ablation,
        args.mac_compare,
        args.lr_sweep,
        args.mac_sweep,
    ]):
        print("No input specified. Use --result, --ablation_noise, --ablation_clients, "
              "--alpha_ablation, --mac_compare, --lr_sweep, or --mac_sweep.")
