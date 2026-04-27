"""
Ablation studies (AWGN setting, α = 2).

Two ablation axes:
  (A) Vary noise scale γ ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5}
      → Validates the central claim: as channel noise grows, plain
        FedAvg/FedAvgM degrade or diverge while the adaptive optimizers
        (AdaGrad-OTA, Adam-OTA) implicitly compress their effective step
        size and stay stable — i.e. the second-moment accumulator behaves
        as a noise-aware learning-rate filter.

  (B) Vary number of clients N ∈ {5, 10, 20, 50, 100}
      → Validates the OTA averaging gain (noise variance contracts as 1/N).

Results are saved as JSON, plotted by src/experiments/plot.py.

Usage:
    uv run -m src.experiments.ablation --study noise
    uv run -m src.experiments.ablation --study clients
    uv run -m src.experiments.ablation --study both
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..channel.ota import NoisyOracle
from ..data.datasets import get_dataset, dirichlet_partition, make_loader
from ..models.nets import get_model
from ..optimizers.adagrad_ota import AdaGradOTA
from ..optimizers.adam_ota import AdamOTA
from ..optimizers.baselines import FedAvgOTA, FedAvgMOTA
from .federated import run_round, evaluate


NOISE_SCALES = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
CLIENT_VALUES = [5, 10, 20, 50, 100]


def _build_optimizer(name: str, model: torch.nn.Module, args) -> object:
    params = list(model.parameters())
    if name == "fedavg":
        return FedAvgOTA(params, lr=args.server_lr)
    if name == "fedavgm":
        return FedAvgMOTA(params, lr=args.server_lr, momentum=args.momentum)
    if name == "adagrad_ota":
        return AdaGradOTA(params, lr=args.server_lr, alpha=args.alpha,
                          beta1=args.momentum)
    if name == "adam_ota":
        return AdamOTA(params, lr=args.server_lr, alpha=args.alpha,
                       beta1=args.momentum, beta2=args.beta2)
    raise ValueError(name)


def _run_single(
    model_name: str,
    dataset_name: str,
    num_clients: int,
    noise_scale: float,
    rounds: int,
    args,
    device: torch.device,
    optimizer_name: str = "adam_ota",
) -> dict:
    """Run one configuration, return {"loss": [...], "acc": [...]}."""
    train_ds, test_ds = get_dataset(dataset_name)
    client_subsets = dirichlet_partition(train_ds, num_clients,
                                         concentration=args.dir_conc)
    client_loaders = [make_loader(s, args.batch_size) for s in client_subsets]
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2)

    oracle = NoisyOracle(alpha=args.alpha, noise_scale=noise_scale, device=device)

    torch.manual_seed(args.seed)
    model = get_model(model_name, num_classes=10).to(device)
    opt = _build_optimizer(optimizer_name, model, args)

    loss_hist, acc_hist = [], []
    for _ in range(1, rounds + 1):
        run_round(model, client_loaders, oracle, opt,
                  local_epochs=args.local_epochs,
                  local_lr=args.local_lr,
                  use_mac=args.use_mac,
                  device=device)
        loss, acc = evaluate(model, test_loader, device)
        loss_hist.append(loss)
        acc_hist.append(acc)

    return {"loss": loss_hist, "acc": acc_hist}


def ablation_noise(args, device: torch.device) -> dict:
    """
    Ablation A: vary noise scale γ, fix N.
    Runs all four optimizers at each γ to expose the noise-resilience gap.
    """
    print("\n=== Ablation: AWGN noise scale γ ===")
    results = {}
    for gamma in NOISE_SCALES:
        results[gamma] = {}
        for opt_name in ["fedavg", "fedavgm", "adagrad_ota", "adam_ota"]:
            print(f"  γ={gamma:<6}  {opt_name} ...", flush=True)
            results[gamma][opt_name] = _run_single(
                args.model, args.dataset, args.num_clients,
                gamma, args.rounds, args, device, opt_name,
            )
            final_acc = results[gamma][opt_name]["acc"][-1]
            print(f"    → final acc={final_acc:.4f}")
    return results


def ablation_clients(args, device: torch.device) -> dict:
    """
    Ablation B: vary N, fix noise scale.
    Runs Adam-OTA at each N value.
    """
    print("\n=== Ablation: Number of clients N ===")
    results = {}
    for N in CLIENT_VALUES:
        print(f"  N={N} ...", flush=True)
        results[N] = _run_single(
            args.model, args.dataset, N,
            args.noise_scale, args.rounds, args, device, "adam_ota",
        )
        final_acc = results[N]["acc"][-1]
        print(f"    → final acc={final_acc:.4f}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="ADOTA-FL ablation studies (AWGN)")
    p.add_argument("--study", default="both", choices=["noise", "clients", "both"])
    p.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    p.add_argument("--model", default="resnet18",
                   choices=["mlp", "convnet", "resnet18", "resnet34"])
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--num_clients", type=int, default=10,
                   help="Fixed N for noise-scale ablation")
    p.add_argument("--alpha", type=float, default=2.0,
                   help="Stability index of the noise (2.0 = AWGN)")
    p.add_argument("--noise_scale", type=float, default=0.05,
                   help="Fixed γ for client-count ablation")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--server_lr", type=float, default=1e-4)
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.3)
    p.add_argument("--dir_conc", type=float, default=0.1)
    p.add_argument("--use_mac", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="results/ablation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Config: {vars(args)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.study in ("noise", "both"):
        res_noise = ablation_noise(args, device)
        path = out_dir / f"ablation_noise_{args.dataset}_{args.model}.json"
        with open(path, "w") as f:
            json.dump({"args": vars(args),
                       "results": {str(k): v for k, v in res_noise.items()}},
                      f, indent=2)
        print(f"Saved: {path}")

    if args.study in ("clients", "both"):
        res_clients = ablation_clients(args, device)
        path = out_dir / f"ablation_clients_{args.dataset}_{args.model}.json"
        with open(path, "w") as f:
            json.dump({"args": vars(args),
                       "results": {str(k): v for k, v in res_clients.items()}},
                      f, indent=2)
        print(f"Saved: {path}")
