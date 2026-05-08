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
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..channel.ota import NoisyOracle
from ..data.datasets import get_dataset, dirichlet_partition, iid_partition, make_loader
from ..models.nets import get_model
from ..optimizers.adagrad_ota import AdaGradOTA
from ..optimizers.adam_ota import AdamOTA
from ..optimizers.baselines import FedAvgOTA, FedAvgMOTA
from .federated import run_round, evaluate, resolve_devices


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


def _make_loaders(args, seed: int):
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


def run_trial(
    method: str,
    alpha: float,
    seed: int,
    use_mac: bool,
    args,
    device: torch.device,
    worker_devices: list[torch.device],
) -> dict:
    set_seed(seed)
    client_loaders, test_loader = _make_loaders(args, seed)

    set_seed(seed)
    model = get_model(args.model, num_classes=10).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile_model:
        if args.client_parallel == "off":
            model = torch.compile(model)
        else:
            print("compile_model ignored with client_parallel=auto; use --client_parallel off to enable it.")
    oracle = NoisyOracle(alpha=alpha, noise_scale=args.noise_scale, device=device)
    opt = build_optimizer(method, model, args, alpha)

    loss_hist, acc_hist = [], []
    diagnostics = []
    status = "ok"
    failed_round = None

    for rnd in range(1, args.rounds + 1):
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
            return_diagnostics=True,
            worker_devices=worker_devices,
            client_parallel=args.client_parallel,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            channels_last=args.channels_last,
        )

        if args.save_diagnostics:
            diagnostics.append({"round": rnd, **diag})

        if rnd == 1:
            worker_stats = diag.get("worker_stats", [])
            shard_text = ", ".join(
                f"{stat.get('device')}={stat.get('clients')} clients"
                for stat in worker_stats
            )
            time_text = ", ".join(
                f"{stat.get('device')} {stat.get('local_seconds', 0.0):.2f}s"
                for stat in worker_stats
                if "local_seconds" in stat
            )
            print(f"  execution={diag.get('execution_path', 'unknown')} | {shard_text}", flush=True)
            if time_text:
                print(f"  worker_time={time_text}", flush=True)

        if diag["status"] != "ok":
            status = diag["status"]
            failed_round = rnd
            break

        loss, acc = evaluate(
            model,
            test_loader,
            device,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            channels_last=args.channels_last,
        )
        if not math.isfinite(loss) or not math.isfinite(acc):
            status = "nonfinite_eval"
            failed_round = rnd
            break

        loss_hist.append(float(loss))
        acc_hist.append(float(acc))

        if rnd % args.log_every == 0 or rnd == 1:
            mac_label = "MAC" if use_mac else "no-MAC"
            print(
                f"  alpha={alpha:<3} seed={seed:<3} {mac_label:<6} "
                f"{method:<12} round {rnd:4d}/{args.rounds} "
                f"loss={loss:.4f} acc={acc:.4f}",
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
        "rounds_completed": len(acc_hist),
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


def run_alpha_ablation(args, device: torch.device, worker_devices: list[torch.device]) -> dict:
    print("\n=== Heavy-tail ablation: alpha-stable tail index ===")
    results = {}
    for alpha in args.alphas:
        alpha_key = f"{alpha:g}"
        results[alpha_key] = {}
        for method in args.methods:
            runs = []
            for seed in args.seeds:
                print(f"\nalpha={alpha_key} method={method} seed={seed}", flush=True)
                runs.append(run_trial(method, alpha, seed, False, args, device, worker_devices))
            results[alpha_key][method] = {
                "runs": runs,
                "summary": _summary(runs),
            }
    return results


def run_mac_compare(args, device: torch.device, worker_devices: list[torch.device]) -> dict:
    print("\n=== Robust pre-processing: MAC vs no-MAC ===")
    results = {"no_mac": {}, "mac": {}}
    for use_mac, key in [(False, "no_mac"), (True, "mac")]:
        for method in args.methods:
            runs = []
            for seed in args.seeds:
                print(f"\n{key} alpha={args.mac_alpha:g} method={method} seed={seed}", flush=True)
                runs.append(run_trial(method, args.mac_alpha, seed, use_mac, args, device, worker_devices))
            results[key][method] = {
                "runs": runs,
                "summary": _summary(runs),
            }
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
    p.add_argument("--devices", type=str, default="0,1",
                   help="Comma-separated visible CUDA device ids")
    p.add_argument("--client_parallel", choices=["auto", "off"], default="auto")
    p.add_argument("--amp", dest="amp", action="store_true", default=None)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--channels_last", dest="channels_last", action="store_true",
                   default=None)
    p.add_argument("--no_channels_last", dest="channels_last", action="store_false")
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--compile_model", action="store_true", default=False)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    worker_devices = resolve_devices(args.devices)
    device = worker_devices[0]
    if args.amp is None:
        args.amp = device.type == "cuda"
    if args.channels_last is None:
        args.channels_last = (
            device.type == "cuda"
            and args.dataset == "cifar10"
            and args.model in ("resnet18", "resnet34", "convnet")
        )
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")

    print(f"Device: {device} | Workers: {[str(d) for d in worker_devices]} | Config: {vars(args)}")

    out_dir = Path(args.out_dir)
    common = {
        "args": vars(args),
        "dataset": args.dataset,
        "model": args.model,
        "noise_scale": args.noise_scale,
    }

    if args.study in ("alpha", "both"):
        results = run_alpha_ablation(args, device, worker_devices)
        path = out_dir / f"alpha_ablation_{args.dataset}_{args.model}.json"
        _write_json(path, {**common, "study": "alpha", "results": results})

    if args.study in ("mac", "both"):
        results = run_mac_compare(args, device, worker_devices)
        path = out_dir / f"mac_compare_{args.dataset}_{args.model}_alpha{args.mac_alpha:g}.json"
        _write_json(path, {**common, "study": "mac", "results": results})
