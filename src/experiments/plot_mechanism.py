"""
Mechanism plots for adaptive OTA-FL.

The thesis §5.2 calls out the missing direct evidence for the central
mechanism: when the OTA aggregate is corrupted by an impulsive noise sample,
the second-moment accumulator v_t grows, so the effective per-coordinate
learning rate contracts.

This script reads diagnostic logs that NoisyOracle and run_round emit when
`--save_diagnostics` is enabled, then plots the following per round:

  - noise_norm, agg_norm           — the channel's impulsive signature
  - v_norm                         — second-moment accumulator growth
  - update_norm / agg_norm         — proxy for the effective server step
  - update_norm                    — actual server update magnitude

The expected mechanism shows up as: spikes in agg_norm at impulsive rounds,
coincident growth in v_norm, and a contracted update_to_agg ratio for the
adaptive methods, with FedAvg / FedAvgM giving the un-contracted ratio.

Usage:
    uv run -m src.experiments.plot_mechanism \
        --diagnostics results/heavytail/mechanism_<dataset>_<model>.json \
        --out_dir results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLE = {
    "fedavg":      {"label": "FedAvg-OTA",  "color": "#d62728", "ls": ":",  "lw": 1.5},
    "fedavgm":     {"label": "FedAvgM-OTA", "color": "#ff7f0e", "ls": "--", "lw": 1.8},
    "adagrad_ota": {"label": "AdaGrad-OTA", "color": "#1f77b4", "ls": "-.", "lw": 2.0},
    "adam_ota":    {"label": "Adam-OTA",     "color": "#2ca02c", "ls": "-",  "lw": 2.2},
}


def _set_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })


def _iter_method_runs(payload: dict, alpha_key: str | None):
    """Yield (alpha_key, method, run_dict) tuples for every diagnostic-bearing run."""
    study = payload.get("study")
    results = payload.get("results", {})

    if study == "alpha":
        alpha_keys = [alpha_key] if alpha_key else list(results.keys())
        for ak in alpha_keys:
            block = results.get(ak)
            if not block:
                continue
            for method, method_block in block.items():
                for run in method_block.get("runs", []):
                    if run.get("diagnostics"):
                        yield ak, method, run

    elif study == "mac":
        for mac_key in ("no_mac", "mac"):
            block = results.get(mac_key, {})
            for method, method_block in block.items():
                for run in method_block.get("runs", []):
                    if run.get("diagnostics"):
                        yield mac_key, method, run

    elif study == "mac_sweep":
        for outer_key, outer_block in results.items():
            for method, method_block in outer_block.items():
                for run in method_block.get("runs", []):
                    if run.get("diagnostics"):
                        yield outer_key, method, run

    else:
        for outer_key, outer_block in results.items():
            if not isinstance(outer_block, dict):
                continue
            for method, method_block in outer_block.items():
                if not isinstance(method_block, dict):
                    continue
                for run in method_block.get("runs", []):
                    if run.get("diagnostics"):
                        yield outer_key, method, run


def _aggregate_method_traces(
    payload: dict,
    alpha_key: str | None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Average diagnostic series across seeds for each method.

    Returns {method_name: {metric_name: np.array of length max_round}}.
    """
    raw: dict[str, list[dict[str, list[float]]]] = {}
    for _, method, run in _iter_method_runs(payload, alpha_key):
        diag_rows = run.get("diagnostics") or []
        if not diag_rows:
            continue
        rounds = [d.get("round") for d in diag_rows]
        max_round = max(rounds) if rounds else 0
        if max_round == 0:
            continue
        series: dict[str, list[float]] = {}
        for d in diag_rows:
            for key, value in d.items():
                if key == "round" or not isinstance(value, (int, float)):
                    continue
                if not np.isfinite(value):
                    continue
                series.setdefault(key, [np.nan] * max_round)
                idx = int(d["round"]) - 1
                if 0 <= idx < max_round:
                    series[key][idx] = float(value)
        if series:
            raw.setdefault(method, []).append(series)

    averaged: dict[str, dict[str, np.ndarray]] = {}
    for method, runs in raw.items():
        keys = set().union(*(set(r.keys()) for r in runs))
        method_traces: dict[str, np.ndarray] = {}
        for key in keys:
            lengths = [len(r[key]) for r in runs if key in r]
            if not lengths:
                continue
            length = max(lengths)
            stack = np.full((len(runs), length), np.nan)
            for i, r in enumerate(runs):
                series = r.get(key)
                if series is None:
                    continue
                stack[i, : len(series)] = series
            with np.errstate(all="ignore"):
                method_traces[key] = np.nanmean(stack, axis=0)
        averaged[method] = method_traces
    return averaged


def _safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) > eps, num / den, np.nan)


def _plot_panel(
    ax,
    method_traces: dict[str, dict[str, np.ndarray]],
    metric: str,
    title: str,
    ylabel: str,
    log: bool,
    rounds_axis: bool,
    transform=None,
) -> None:
    for method, style in METHOD_STYLE.items():
        traces = method_traces.get(method)
        if not traces or metric not in traces:
            continue
        series = traces[metric]
        if transform is not None:
            series = transform(traces)
            if series is None:
                continue
        x = np.arange(1, len(series) + 1) if rounds_axis else np.arange(len(series))
        finite = np.isfinite(series)
        if not finite.any():
            continue
        ax.plot(
            x[finite],
            series[finite],
            label=style["label"],
            color=style["color"],
            ls=style["ls"],
            lw=style["lw"],
        )
    if log:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Communication round")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both")
    ax.legend()


def plot_mechanism(diag_path: str, out_dir: Path, alpha_key: str | None = None) -> Path | None:
    with open(diag_path) as f:
        payload = json.load(f)

    traces = _aggregate_method_traces(payload, alpha_key)
    if not traces:
        print(f"No diagnostics found in {diag_path}")
        return None

    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))

    _plot_panel(
        axes[0, 0],
        traces,
        "agg_norm",
        "Aggregate update norm  ‖ḡ_t‖₂",
        "‖ḡ_t‖₂",
        log=True,
        rounds_axis=True,
    )
    _plot_panel(
        axes[0, 1],
        traces,
        "v_norm",
        "Second-moment accumulator  ‖v_t‖₂",
        "‖v_t‖₂",
        log=True,
        rounds_axis=True,
    )

    def effective_step(method_trace: dict[str, np.ndarray]) -> np.ndarray | None:
        if "update_norm" not in method_trace or "agg_norm" not in method_trace:
            return None
        return _safe_ratio(method_trace["update_norm"], method_trace["agg_norm"])

    _plot_panel(
        axes[1, 0],
        traces,
        "update_norm",
        "Effective server step  ‖Δw_t‖₂ / ‖ḡ_t‖₂",
        "‖Δw_t‖ / ‖ḡ_t‖",
        log=True,
        rounds_axis=True,
        transform=effective_step,
    )

    _plot_panel(
        axes[1, 1],
        traces,
        "noise_norm",
        "Channel noise norm  ‖ξ_t‖₂",
        "‖ξ_t‖₂",
        log=True,
        rounds_axis=True,
    )

    title_bits: list[str] = []
    args = payload.get("args", {})
    if "dataset" in args:
        title_bits.append(str(args["dataset"]).upper())
    if "model" in args:
        title_bits.append(str(args["model"]))
    if alpha_key:
        title_bits.append(rf"$\alpha$={alpha_key}")
    elif "alphas" in args and isinstance(args["alphas"], list) and len(args["alphas"]) == 1:
        title_bits.append(rf"$\alpha$={args['alphas'][0]}")
    if "noise_scale" in args:
        title_bits.append(rf"$\gamma$={args['noise_scale']}")
    suptitle = "Mechanism diagnostics" + (": " + ", ".join(title_bits) if title_bits else "")
    fig.suptitle(suptitle)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    stem = Path(diag_path).stem
    suffix = f"_alpha{alpha_key}" if alpha_key else ""
    out = out_dir / f"{stem}_mechanism{suffix}.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Mechanism plots for adaptive OTA-FL diagnostics")
    p.add_argument("--diagnostics", type=str, required=True,
                   help="Path to a heavy_tail.py JSON produced with --save_diagnostics.")
    p.add_argument("--alpha", type=str, default=None,
                   help="Optional alpha key to filter (e.g. 1.3). Defaults to all alphas in the file.")
    p.add_argument("--out_dir", type=str, default="results/figures")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_mechanism(args.diagnostics, out_dir, alpha_key=args.alpha)
