"""
Heavy-tail alpha-stable experiments for ADOTA-FL.

This runner makes alpha<2 a first-class experiment setting. It supports:
  - alpha ablation over symmetric alpha-stable interference
  - MAC vs no-MAC comparisons under a fixed heavy-tailed alpha
  - multi-seed summaries with final/best accuracy mean and std

Usage:
    uv run -m src.experiments.heavy_tail --study alpha
    uv run -m src.experiments.heavy_tail --study mac --mac_alpha 1.3
    uv run -m src.experiments.heavy_tail --study both
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..channel.ota import NoisyOracle
from ..data.datasets import (
    get_dataset,
    dirichlet_partition,
    iid_partition,
    make_loader,
    make_fast_cifar10_loaders,
    make_fast_cifar10_eval_loader,
)
from ..models.nets import get_model
from ..optimizers.adagrad_ota import AdaGradOTA
from ..optimizers.adam_ota import AdamOTA
from ..optimizers.baselines import FedAvgOTA, FedAvgMOTA
from .accelerate import (
    add_accelerator_args,
    accelerator_summary,
    configure_model,
    resolve_accelerator_args,
    should_collect_diagnostics,
    should_evaluate,
)
from .federated import run_round, evaluate


DEFAULT_ALPHAS = [1.1, 1.3, 1.5, 1.7, 1.9, 2.0]
DEFAULT_METHODS = ["fedavg", "fedavgm", "adagrad_ota", "adam_ota"]
DEFAULT_SEEDS = [42, 43, 44]


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(name: str, model: torch.nn.Module, args, alpha: float) -> object:
    params = list(model.parameters())
    if name == "fedavg":
        return FedAvgOTA(params, lr=args.server_lr)
    if name == "fedavgm":
        return FedAvgMOTA(params, lr=args.server_lr, momentum=args.momentum)
    if name == "adagrad_ota":
        return AdaGradOTA(params, lr=args.server_lr, alpha=alpha, beta1=args.momentum)
    if name == "adam_ota":
        return AdamOTA(
            params,
            lr=args.server_lr,
            alpha=alpha,
            beta1=args.momentum,
            beta2=args.beta2,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def _make_loaders(args, seed: int, device: torch.device | None = None):
    train_ds, test_ds = get_dataset(args.dataset)
    if args.non_iid:
        subsets = dirichlet_partition(
            train_ds,
            args.num_clients,
            concentration=args.dir_conc,
            seed=seed,
        )
    else:
        subsets = iid_partition(train_ds, args.num_clients, seed=seed)

    if args.use_fast_data:
        client_loaders = make_fast_cifar10_loaders(subsets, args.batch_size, device=device)
        test_loader = make_fast_cifar10_eval_loader(test_ds, args.eval_batch_size, device=device)
    else:
        client_loaders = [make_loader(subset, args.batch_size) for subset in subsets]
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    return client_loaders, test_loader


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _summary(runs: list[dict]) -> dict:
    ok_runs = [run for run in runs if run["status"] == "ok" and _finite(run["final_acc"])]
    final_acc = np.array([run["final_acc"] for run in ok_runs], dtype=float)
    best_acc = np.array([run["best_acc"] for run in ok_runs], dtype=float)
    final_loss = np.array([run["final_loss"] for run in ok_runs], dtype=float)

    def stats(values: np.ndarray) -> dict:
        if len(values) == 0:
            return {"mean": None, "std": None}
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    return {
        "num_runs": len(runs),
        "ok_runs": len(ok_runs),
        "failed_runs": len(runs) - len(ok_runs),
        "final_acc": stats(final_acc),
        "best_acc": stats(best_acc),
        "final_loss": stats(final_loss),
    }


def _same_float(left: float | None, right: float | None, tol: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


def _alpha_ablation_paths(args) -> list[Path]:
    out_dir = Path(args.out_dir)
    preferred = out_dir / f"alpha_ablation_{args.dataset}_{args.model}.json"
    paths = []
    if preferred.exists():
        paths.append(preferred)
    if out_dir.exists():
        for path in sorted(out_dir.rglob(f"alpha_ablation_{args.dataset}_{args.model}.json")):
            if path not in paths:
                paths.append(path)
    return paths


def _payload_matches_current_config(payload: dict, args) -> bool:
    if payload.get("study") != "alpha":
        return False
    if payload.get("dataset") != args.dataset or payload.get("model") != args.model:
        return False

    payload_args = payload.get("args", {})
    exact_keys = [
        "rounds",
        "num_clients",
        "local_epochs",
        "batch_size",
        "non_iid",
        "eval_every",
        "eval_batch_size",
        "fast_data",
        "use_fast_data",
        "amp",
        "amp_dtype",
        "channels_last",
        "use_amp",
        "use_channels_last",
    ]
    float_keys = [
        "server_lr",
        "local_lr",
        "momentum",
        "beta2",
        "noise_scale",
        "dir_conc",
    ]

    for key in exact_keys:
        if payload_args.get(key) != getattr(args, key):
            return False
    for key in float_keys:
        if not _same_float(payload_args.get(key), getattr(args, key)):
            return False
    return True


def _run_matches_no_mac_ablation(run: dict, method: str, seed: int, alpha: float, args) -> bool:
    return (
        run.get("method") == method
        and run.get("seed") == seed
        and run.get("use_mac") is False
        and _same_float(run.get("alpha"), alpha)
        and _same_float(run.get("noise_scale"), args.noise_scale)
    )


def _find_no_mac_ablation_runs(args) -> tuple[dict[str, dict[int, dict]], set[str]]:
    """Find matching no-MAC runs already produced by the alpha ablation study."""
    alpha_key = f"{args.mac_alpha:g}"
    wanted_seeds = set(args.seeds)
    cached_runs = {method: {} for method in args.methods}
    sources: set[str] = set()

    for path in _alpha_ablation_paths(args):
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping unreadable alpha ablation file {path}: {exc}", flush=True)
            continue

        if not _payload_matches_current_config(payload, args):
            continue

        alpha_results = payload.get("results", {}).get(alpha_key)
        if not isinstance(alpha_results, dict):
            continue

        for method in args.methods:
            method_block = alpha_results.get(method)
            if not isinstance(method_block, dict):
                continue
            method_runs = method_block.get("runs", [])
            if not isinstance(method_runs, list):
                continue
            for run in method_runs:
                seed = run.get("seed")
                if seed not in wanted_seeds or seed in cached_runs[method]:
                    continue
                if _run_matches_no_mac_ablation(run, method, seed, args.mac_alpha, args):
                    cached_runs[method][seed] = copy.deepcopy(run)
                    sources.add(str(path))

    return cached_runs, sources


def run_trial(
    method: str,
    alpha: float,
    seed: int,
    use_mac: bool,
    args,
    device: torch.device,
) -> dict:
    set_seed(seed)
    client_loaders, test_loader = _make_loaders(args, seed, device)

    set_seed(seed)
    model = configure_model(get_model(args.model, num_classes=10), args, device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    oracle = NoisyOracle(alpha=alpha, noise_scale=args.noise_scale, device=device)
    opt = build_optimizer(method, model, args, alpha)

    loss_hist, acc_hist = [], []
    eval_rounds = []
    diagnostics = []
    status = "ok"
    failed_round = None
    completed_round = 0

    for rnd in range(1, args.rounds + 1):
        need_diag = should_collect_diagnostics(
            rnd,
            args,
            save_diagnostics=args.save_diagnostics,
        )
        diag = run_round(
            model,
            client_loaders,
            oracle,
            opt,
            local_epochs=args.local_epochs,
            local_lr=args.local_lr,
            use_mac=use_mac,
            mac_clip=args.mac_clip,
            device=device,
            return_diagnostics=need_diag,
            use_amp=args.use_amp,
            amp_dtype=args.amp_dtype,
            channels_last=args.use_channels_last,
        )

        completed_round = rnd
        if args.save_diagnostics and diag is not None:
            diagnostics.append({"round": rnd, **diag})

        if diag is not None and diag["status"] != "ok":
            status = diag["status"]
            failed_round = rnd
            break

        evaluated = should_evaluate(rnd, args.rounds, args)
        if evaluated:
            loss, acc = evaluate(
                model,
                test_loader,
                device,
                use_amp=args.use_amp,
                amp_dtype=args.amp_dtype,
                channels_last=args.use_channels_last,
            )
            if not math.isfinite(loss) or not math.isfinite(acc):
                status = "nonfinite_eval"
                failed_round = rnd
                break

            loss_hist.append(float(loss))
            acc_hist.append(float(acc))
            eval_rounds.append(rnd)

        if rnd % args.log_every == 0 or rnd == 1:
            mac_label = "MAC" if use_mac else "no-MAC"
            if evaluated:
                print(
                    f"  alpha={alpha:<3} seed={seed:<3} {mac_label:<6} "
                    f"{method:<12} round {rnd:4d}/{args.rounds} "
                    f"loss={loss:.4f} acc={acc:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"  alpha={alpha:<3} seed={seed:<3} {mac_label:<6} "
                    f"{method:<12} round {rnd:4d}/{args.rounds} eval=skipped",
                    flush=True,
                )

    result = {
        "seed": seed,
        "method": method,
        "alpha": alpha,
        "noise_scale": args.noise_scale,
        "use_mac": use_mac,
        "mac_clip": args.mac_clip if use_mac else None,
        "rounds_requested": args.rounds,
        "rounds_completed": completed_round,
        "eval_rounds": eval_rounds,
        "status": status,
        "nonfinite": status != "ok",
        "failed_round": failed_round,
        "loss": loss_hist,
        "acc": acc_hist,
        "final_loss": loss_hist[-1] if loss_hist else None,
        "final_acc": acc_hist[-1] if acc_hist else None,
        "best_acc": max(acc_hist) if acc_hist else None,
    }
    if args.save_diagnostics:
        result["diagnostics"] = diagnostics
    return result


def run_alpha_ablation(args, device: torch.device) -> dict:
    print("\n=== Heavy-tail ablation: alpha-stable tail index ===")
    results = {}
    for alpha in args.alphas:
        alpha_key = f"{alpha:g}"
        results[alpha_key] = {}
        for method in args.methods:
            seeds = args.seeds
            if len(seeds) > 1:
                with ThreadPoolExecutor(max_workers=len(seeds)) as executor:
                    futures = [
                        executor.submit(run_trial, method, alpha, seed, False, args, device)
                        for seed in seeds
                    ]
                    runs = [f.result() for f in futures]
            else:
                runs = [run_trial(method, alpha, seeds[0], False, args, device)]
            results[alpha_key][method] = {
                "runs": runs,
                "summary": _summary(runs),
            }
    return results


def run_mac_compare(args, device: torch.device) -> dict:
    print("\n=== Robust pre-processing: MAC vs no-MAC ===")
    cached_no_mac, sources = _find_no_mac_ablation_runs(args)
    if sources:
        source_list = ", ".join(sorted(sources))
        reused = sum(len(seed_runs) for seed_runs in cached_no_mac.values())
        print(
            f"Reusing {reused} no-MAC run(s) from alpha ablation data: {source_list}",
            flush=True,
        )

    results = {"no_mac": {}, "mac": {}}
    for use_mac, key in [(False, "no_mac"), (True, "mac")]:
        for method in args.methods:
            runs = []
            reused_seeds = []
            seeds_to_run = []
            for seed in args.seeds:
                cached_run = None if use_mac else cached_no_mac.get(method, {}).get(seed)
                if cached_run is not None:
                    runs.append(cached_run)
                    reused_seeds.append(seed)
                else:
                    seeds_to_run.append(seed)

            if seeds_to_run:
                if len(seeds_to_run) > 1:
                    with ThreadPoolExecutor(max_workers=len(seeds_to_run)) as executor:
                        futures = [
                            executor.submit(run_trial, method, args.mac_alpha, seed, use_mac, args, device)
                            for seed in seeds_to_run
                        ]
                        trial_results = [f.result() for f in futures]
                else:
                    trial_results = [run_trial(method, args.mac_alpha, seeds_to_run[0], use_mac, args, device)]
                runs.extend(trial_results)

            method_result = {
                "runs": runs,
                "summary": _summary(runs),
            }
            if reused_seeds:
                method_result["reused_from_alpha_ablation"] = {
                    "seeds": reused_seeds,
                    "sources": sorted(sources),
                }
            results[key][method] = method_result
    return results


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Heavy-tail alpha-stable ADOTA-FL experiments")
    p.add_argument("--study", default="both", choices=["alpha", "mac", "both"])
    p.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    p.add_argument("--model", default="resnet18",
                   choices=["mlp", "convnet", "resnet18", "resnet34"])
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--num_clients", type=int, default=100)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--server_lr", type=float, default=0.01)
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--noise_scale", type=float, default=0.05)
    p.add_argument("--alphas", type=_parse_float_list,
                   default=DEFAULT_ALPHAS,
                   help="Comma-separated alpha values, e.g. 1.1,1.3,1.5,1.7,1.9,2.0")
    p.add_argument("--mac_alpha", type=float, default=1.3)
    p.add_argument("--mac_clip", type=float, default=3.0)
    p.add_argument("--methods", type=_parse_str_list,
                   default=DEFAULT_METHODS,
                   help="Comma-separated methods")
    p.add_argument("--seeds", type=_parse_int_list,
                   default=DEFAULT_SEEDS,
                   help="Comma-separated seeds")
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
    common = {
        "args": vars(args),
        "dataset": args.dataset,
        "model": args.model,
        "noise_scale": args.noise_scale,
    }

    if args.study in ("alpha", "both"):
        results = run_alpha_ablation(args, device)
        path = out_dir / f"alpha_ablation_{args.dataset}_{args.model}.json"
        _write_json(path, {**common, "study": "alpha", "results": results})

    if args.study in ("mac", "both"):
        results = run_mac_compare(args, device)
        path = out_dir / f"mac_compare_{args.dataset}_{args.model}_alpha{args.mac_alpha:g}.json"
        _write_json(path, {**common, "study": "mac", "results": results})
