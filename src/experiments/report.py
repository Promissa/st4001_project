"""
Generate LaTeX result tables from experiment JSON files.

The thesis can include these generated tables after running the heavy-tail
experiments, avoiding hand-copied single-run values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_LABELS = {
    "fedavg": "FedAvg-OTA",
    "fedavgm": "FedAvgM-OTA",
    "adagrad_ota": "AdaGrad-OTA",
    "adam_ota": "Adam-OTA",
}


def _mean_std(block: dict, metric: str = "final_acc") -> tuple[float | None, float | None]:
    stats = block.get("summary", {}).get(metric, {})
    return stats.get("mean"), stats.get("std")


def _fmt_acc(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    std = 0.0 if std is None else std
    return f"{mean * 100:.2f}\\% $\\pm$ {std * 100:.2f}\\%"


def write_alpha_table(path: str, out_dir: Path) -> Path:
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    methods = [m for m in METHOD_LABELS if any(m in results[a] for a in results)]
    alphas = sorted(float(a) for a in results)

    out = out_dir / "alpha_ablation_table.tex"
    with open(out, "w") as f:
        f.write("\\begin{table}[!htbp]\n")
        f.write("    \\centering\n")
        f.write("    \\small\n")
        f.write("    \\caption{CIFAR-10 alpha-stable tail-index ablation: final accuracy, mean $\\pm$ std over seeds}\n")
        f.write("    \\label{tab:alpha-ablation}\n")
        f.write("    \\begin{tabular}{c|" + "p{0.18\\textwidth}" * len(methods) + "}\n")
        f.write("        \\hline\n")
        f.write("        $\\alpha$ & " + " & ".join(METHOD_LABELS[m] for m in methods) + " \\\\\n")
        f.write("        \\hline\n")
        for alpha in alphas:
            row = [f"{alpha:g}"]
            for method in methods:
                block = results[f"{alpha:g}"].get(method)
                row.append(_fmt_acc(*_mean_std(block)) if block else "--")
            f.write("        " + " & ".join(row) + " \\\\\n")
        f.write("        \\hline\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")
    return out


def write_mac_table(path: str, out_dir: Path) -> Path:
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    methods = [m for m in METHOD_LABELS if m in results.get("no_mac", {})]

    out = out_dir / "mac_compare_table.tex"
    with open(out, "w") as f:
        f.write("\\begin{table}[!htbp]\n")
        f.write("    \\centering\n")
        f.write("    \\small\n")
        f.write("    \\caption{CIFAR-10 MAC comparison under heavy-tailed interference: final accuracy, mean $\\pm$ std over seeds}\n")
        f.write("    \\label{tab:mac-compare}\n")
        f.write("    \\begin{tabular}{l|p{0.25\\textwidth}|p{0.25\\textwidth}}\n")
        f.write("        \\hline\n")
        f.write("        Method & No MAC & MAC \\\\\n")
        f.write("        \\hline\n")
        for method in methods:
            no_mac = _fmt_acc(*_mean_std(results["no_mac"][method]))
            mac = _fmt_acc(*_mean_std(results["mac"][method]))
            f.write(f"        {METHOD_LABELS[method]} & {no_mac} & {mac} \\\\\n")
        f.write("        \\hline\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")
    return out


def write_lr_sweep_table(path: str, out_dir: Path) -> Path:
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    lrs = sorted(float(lr) for lr in results)
    methods = [m for m in METHOD_LABELS if any(m in results[f"{lr:g}"] for lr in lrs)]

    out = out_dir / "lr_sweep_table.tex"
    alpha = data.get("mac_alpha", data.get("args", {}).get("mac_alpha", "?"))
    gamma = data.get("noise_scale", data.get("args", {}).get("noise_scale", "?"))
    with open(out, "w") as f:
        f.write("\\begin{table}[!htbp]\n")
        f.write("    \\centering\n")
        f.write("    \\small\n")
        f.write("    \\caption{Per-method server learning-rate sweep on CIFAR-10 "
                f"at $\\alpha={alpha}$, $\\gamma={gamma}$: final accuracy, mean $\\pm$ std over seeds}}\n")
        f.write("    \\label{tab:lr-sweep}\n")
        f.write("    \\begin{tabular}{c|" + "p{0.18\\textwidth}" * len(methods) + "}\n")
        f.write("        \\hline\n")
        f.write("        $\\eta$ & " + " & ".join(METHOD_LABELS[m] for m in methods) + " \\\\\n")
        f.write("        \\hline\n")
        for lr in lrs:
            row = [f"{lr:g}"]
            for method in methods:
                block = results[f"{lr:g}"].get(method)
                row.append(_fmt_acc(*_mean_std(block)) if block else "--")
            f.write("        " + " & ".join(row) + " \\\\\n")
        f.write("        \\hline\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")
    return out


def write_mac_sweep_table(path: str, out_dir: Path) -> Path:
    with open(path) as f:
        data = json.load(f)

    results = data["results"]
    gammas = sorted(float(g) for g in results)
    ks = sorted({float(k) for gkey in results for k in results[gkey].keys()})
    methods = [m for m in METHOD_LABELS if any(
        m in results[f"{g:g}"].get(f"{k:g}", {}) for g in gammas for k in ks
    )]

    out = out_dir / "mac_sweep_table.tex"
    alpha = data.get("args", {}).get("mac_alpha", "?")
    with open(out, "w") as f:
        f.write("\\begin{table}[!htbp]\n")
        f.write("    \\centering\n")
        f.write("    \\small\n")
        f.write(f"    \\caption{{CIFAR-10 MAC sweep at $\\alpha={alpha}$: "
                "final accuracy mean ($\\%$) across noise scale $\\gamma$ and clip factor $k$ "
                "($k{=}0$ denotes no MAC). Higher is better.}}\n")
        f.write("    \\label{tab:mac-sweep}\n")
        f.write("    \\begin{tabular}{c|c|" + "c" * len(ks) + "}\n")
        f.write("        \\hline\n")
        f.write("        Method & $\\gamma$ \\textbackslash{} $k$ & "
                + " & ".join(f"{k:g}" for k in ks) + " \\\\\n")
        f.write("        \\hline\n")
        for method in methods:
            for i, g in enumerate(gammas):
                row = [METHOD_LABELS[method] if i == 0 else "", f"{g:g}"]
                for k in ks:
                    block = results[f"{g:g}"].get(f"{k:g}", {}).get(method)
                    mean, _ = _mean_std(block or {})
                    row.append(f"{mean * 100:.2f}" if mean is not None else "--")
                f.write("        " + " & ".join(row) + " \\\\\n")
            f.write("        \\hline\n")
        f.write("    \\end{tabular}\n")
        f.write("\\end{table}\n")
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Generate thesis tables from ADOTA-FL JSON")
    p.add_argument("--alpha_ablation", type=str, default=None)
    p.add_argument("--mac_compare", type=str, default=None)
    p.add_argument("--lr_sweep", type=str, default=None)
    p.add_argument("--mac_sweep", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="template/data/generated")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wrote = []
    if args.alpha_ablation:
        wrote.append(write_alpha_table(args.alpha_ablation, out_dir))
    if args.mac_compare:
        wrote.append(write_mac_table(args.mac_compare, out_dir))
    if args.lr_sweep:
        wrote.append(write_lr_sweep_table(args.lr_sweep, out_dir))
    if args.mac_sweep:
        wrote.append(write_mac_sweep_table(args.mac_sweep, out_dir))

    if wrote:
        for path in wrote:
            print(f"Saved: {path}")
    else:
        print("No report input specified.")
