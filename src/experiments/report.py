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


def parse_args():
    p = argparse.ArgumentParser(description="Generate thesis tables from ADOTA-FL JSON")
    p.add_argument("--alpha_ablation", type=str, default=None)
    p.add_argument("--mac_compare", type=str, default=None)
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

    if wrote:
        for path in wrote:
            print(f"Saved: {path}")
    else:
        print("No report input specified.")
