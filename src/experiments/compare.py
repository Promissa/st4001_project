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
from .accelerate import (
    add_accelerator_args,
    accelerator_summary,
    configure_model,
    resolve_accelerator_args,
    should_evaluate,
)
from .federated import run_round, evaluate


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolve_accelerator_args(args, device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    print(f"Device: {device}")
    print(accelerator_summary(args))
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
        client_loaders = make_fast_cifar10_loaders(client_subsets, args.batch_size)
        test_loader = make_fast_cifar10_eval_loader(test_ds, args.eval_batch_size)
    else:
        client_loaders = [make_loader(s, args.batch_size) for s in client_subsets]
        test_loader = DataLoader(
            test_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=2,
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
        model = configure_model(get_model(args.model, num_classes=10), args, device)
        opt = build_optimizer(method, model, args)

        for rnd in range(1, args.rounds + 1):
            run_round(
                model, client_loaders, oracle, opt,
                local_epochs=args.local_epochs,
                local_lr=args.local_lr,
                use_mac=args.use_mac,
                mac_clip=args.mac_clip,
                device=device,
                use_amp=args.use_amp,
                amp_dtype=args.amp_dtype,
                channels_last=args.use_channels_last,
            )
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
                results[method]["loss"].append(loss)
                results[method]["acc"].append(acc)
                results[method]["eval_rounds"].append(rnd)

            if rnd % args.log_every == 0 or rnd == 1:
                if evaluated:
                    print(f"  Round {rnd:4d}/{args.rounds} | Loss: {loss:.4f} | Acc: {acc:.4f}")
                else:
                    print(f"  Round {rnd:4d}/{args.rounds} | Eval: skipped")

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
    add_accelerator_args(p)
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
