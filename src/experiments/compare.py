"""
Comparative analysis: run all four server-side optimizers under identical
OTA channel conditions and record per-round test accuracy / training loss.

  1. FedAvg-OTA    (plain SGD server, no momentum)
  2. FedAvgM-OTA   (momentum SGD server) — primary baseline
  3. AdaGrad-OTA   (server-side AdaGrad on the OTA-aggregated gradient)
  4. Adam-OTA      (server-side Adam on the OTA-aggregated gradient)

Each method shares the same seed, data partition, and per-round noise
realisation for a fair comparison.

Usage:
    uv run -m src.experiments.compare
    uv run -m src.experiments.compare --dataset cifar10 --model resnet18 --rounds 200
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..channel.ota import NoisyOracle
from ..data.datasets import (
    get_dataset,
    iid_partition,
    dirichlet_partition,
    make_loader,
    make_fast_cifar10_loaders,
    make_fast_cifar10_eval_loader,
)
from ..models.nets import get_model
from ..optimizers.adagrad_ota import AdaGradOTA
from ..optimizers.adam_ota import AdamOTA
from ..optimizers.baselines import FedAvgOTA, FedAvgMOTA
from .federated import ClientParallelExecutor, run_round, evaluate, resolve_devices


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(name: str, model: torch.nn.Module, args) -> object:
    params = list(model.parameters())
    if name == "fedavg":
        return FedAvgOTA(params, lr=args.server_lr)
    elif name == "fedavgm":
        return FedAvgMOTA(params, lr=args.server_lr, momentum=args.momentum)
    elif name == "adagrad_ota":
        return AdaGradOTA(params, lr=args.server_lr, alpha=args.alpha,
                          beta1=args.momentum)
    elif name == "adam_ota":
        return AdamOTA(params, lr=args.server_lr, alpha=args.alpha,
                       beta1=args.momentum, beta2=args.beta2)
    raise ValueError(name)


def run_comparison(args) -> dict:
    """
    Run all methods and return results dict:
      { method_name: {"loss": [...], "acc": [...]} }
    """
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
    args.use_fast_data = (
        args.dataset == "cifar10"
        and (
            args.fast_data == "on"
            or (args.fast_data == "auto" and device.type == "cuda")
        )
    )
    if args.gpu_cache_data is None:
        args.gpu_cache_data = bool(args.use_fast_data and device.type == "cuda")
    args.eval_every = max(1, args.eval_every)
    args.profile_every = max(0, args.profile_every)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device} | Workers: {[str(d) for d in worker_devices]}")
    print(
        f"Accelerator: amp={args.amp} ({args.amp_dtype}) | "
        f"channels_last={args.channels_last} | fast_data={args.use_fast_data} | "
        f"gpu_cache_data={args.gpu_cache_data} | eval_every={args.eval_every}"
    )
    set_seed(args.seed)

    # --- Data (shared across all methods) ---
    train_ds, test_ds = get_dataset(args.dataset)
    if args.non_iid:
        client_subsets = dirichlet_partition(train_ds, args.num_clients,
                                             concentration=args.dir_conc,
                                             seed=args.seed)
    else:
        client_subsets = iid_partition(train_ds, args.num_clients, seed=args.seed)

    if args.use_fast_data:
        client_loaders = make_fast_cifar10_loaders(
            client_subsets,
            args.batch_size,
            gpu_cache=args.gpu_cache_data,
        )
        test_loader = make_fast_cifar10_eval_loader(
            test_ds,
            args.eval_batch_size,
            gpu_cache=args.gpu_cache_data,
        )
    else:
        client_loaders = [make_loader(s, args.batch_size) for s in client_subsets]
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    oracle = NoisyOracle(
        alpha=args.alpha,
        noise_scale=args.noise_scale,
        device=device,
    )

    methods = ["fedavg", "fedavgm", "adagrad_ota", "adam_ota"]
    results = {m: {"loss": [], "acc": [], "eval_rounds": []} for m in methods}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Method: {method.upper()}")
        print(f"{'='*50}")

        # Fresh model (same init for all methods via fixed seed)
        set_seed(args.seed)
        model = get_model(args.model, num_classes=10).to(device)
        if args.channels_last:
            model = model.to(memory_format=torch.channels_last)
        opt = build_optimizer(method, model, args)
        parallel_executor = None
        can_persist_parallel = (
            args.client_parallel == "auto"
            and len(worker_devices) > 1
            and all(worker_device.type == "cuda" for worker_device in worker_devices)
        )
        if can_persist_parallel:
            parallel_executor = ClientParallelExecutor(
                model,
                client_loaders,
                worker_devices,
                local_lr=args.local_lr,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                channels_last=args.channels_last,
            )

        try:
            for rnd in range(1, args.rounds + 1):
                want_diagnostics = (
                    rnd == 1
                    or (args.profile_every > 0 and rnd % args.profile_every == 0)
                )
                diagnostics = run_round(
                    model, client_loaders, oracle, opt,
                    local_epochs=args.local_epochs,
                    local_lr=args.local_lr,
                    use_mac=args.use_mac,
                    mac_clip=args.mac_clip,
                    device=device,
                    return_diagnostics=want_diagnostics,
                    worker_devices=worker_devices,
                    client_parallel=args.client_parallel,
                    use_amp=args.amp,
                    amp_dtype=args.amp_dtype,
                    channels_last=args.channels_last,
                    parallel_executor=parallel_executor,
                )
                if want_diagnostics and diagnostics is not None:
                    worker_stats = diagnostics.get("worker_stats", [])
                    shard_text = ", ".join(
                        f"{stat.get('device')}={stat.get('clients')} clients/{stat.get('samples')} samples"
                        for stat in worker_stats
                    )
                    print(f"  Execution: {diagnostics.get('execution_path', 'unknown')} | {shard_text}")

                should_eval = rnd == 1 or rnd == args.rounds or rnd % args.eval_every == 0
                if should_eval:
                    loss, acc = evaluate(
                        model,
                        test_loader,
                        device,
                        use_amp=args.amp,
                        amp_dtype=args.amp_dtype,
                        channels_last=args.channels_last,
                    )
                    results[method]["loss"].append(loss)
                    results[method]["acc"].append(acc)
                    results[method]["eval_rounds"].append(rnd)

                if rnd % args.log_every == 0 or rnd == 1:
                    if should_eval:
                        print(f"  Round {rnd:4d}/{args.rounds} | Loss: {loss:.4f} | Acc: {acc:.4f}")
                    else:
                        print(f"  Round {rnd:4d}/{args.rounds} | Eval: skipped")
        finally:
            if parallel_executor is not None:
                parallel_executor.close()

    return results


def parse_args():
    p = argparse.ArgumentParser(description="ADOTA-FL comparative analysis")
    p.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    p.add_argument("--model", default="resnet18",
                   choices=["mlp", "convnet", "resnet18", "resnet34"])
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--server_lr", type=float, default=0.1)
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9, help="β₁ for momentum / 1st moment")
    p.add_argument("--beta2", type=float, default=0.3, help="Adam-OTA β₂")
    p.add_argument("--alpha", type=float, default=2.0,
                   help="Stability index of the noise (2.0 = AWGN, default)")
    p.add_argument("--noise_scale", type=float, default=0.05,
                   help="Noise scale γ (std ≈ γ·√2 for AWGN)")
    p.add_argument("--non_iid", action="store_true", default=True)
    p.add_argument("--dir_conc", type=float, default=0.1,
                   help="Dirichlet concentration for non-IID partition")
    p.add_argument("--use_mac", action="store_true", default=False,
                   help="Apply Median Anchored Clipping pre-processing")
    p.add_argument("--mac_clip", type=float, default=3.0)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="results/comparison")
    p.add_argument("--devices", type=str, default="0,1")
    p.add_argument("--client_parallel", choices=["auto", "off"], default="auto")
    p.add_argument("--amp", dest="amp", action="store_true", default=None)
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--amp_dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--channels_last", dest="channels_last", action="store_true",
                   default=None)
    p.add_argument("--no_channels_last", dest="channels_last", action="store_false")
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--fast_data", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--gpu_cache_data", dest="gpu_cache_data", action="store_true", default=None)
    p.add_argument("--no_gpu_cache_data", dest="gpu_cache_data", action="store_false")
    p.add_argument("--diagnostics_every", type=int, default=0)
    p.add_argument("--profile_every", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_comparison(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.dataset}_{args.model}_alpha{args.alpha}_N{args.num_clients}.json"
    with open(out_file, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nResults saved to {out_file}")
