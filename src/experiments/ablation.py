"""
Weeks 15-16: Ablation Studies

Two ablation axes from the proposal:
  (A) Vary noise tail index α ∈ {1.1, 1.3, 1.5, 1.7, 1.9, 2.0}
      → Validates the theoretical link: smaller α → slower convergence.
      → Confirms hypothesis: Adam-OTA more resilient than AdaGrad-OTA.

  (B) Vary number of clients N ∈ {5, 10, 20, 50, 100}
      → Validates scalability: more clients → better accuracy (OTA benefit).

Results are saved as JSON and plotted automatically.

Usage:
    uv run src/experiments/ablation.py --study alpha
    uv run src/experiments/ablation.py --study clients
    uv run src/experiments/ablation.py --study both
"""

import argparse
import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..channel.ota import NoisyOracle
from ..data.datasets import get_dataset, dirichlet_partition, make_loader
from ..models.nets import get_model
from ..optimizers.adagrad_ota import AdaGradOTA
from ..optimizers.adam_ota import AdamOTA
from ..optimizers.baselines import FedAvgMOTA
from .federated import run_round, evaluate


ALPHA_VALUES = [1.1, 1.3, 1.5, 1.7, 1.9, 2.0]
CLIENT_VALUES = [5, 10, 20, 50, 100]


def _run_single(
    model_name: str,
    dataset_name: str,
    num_clients: int,
    alpha: float,
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

    oracle = NoisyOracle(alpha=alpha, noise_scale=args.noise_scale, device=device)

    torch.manual_seed(args.seed)
    model = get_model(model_name, num_classes=10).to(device)
    params = list(model.parameters())

    if optimizer_name == "adagrad_ota":
        opt = AdaGradOTA(params, lr=args.server_lr, alpha=alpha, beta1=args.momentum)
    elif optimizer_name == "adam_ota":
        opt = AdamOTA(params, lr=args.server_lr, alpha=alpha,
                      beta1=args.momentum, beta2=args.beta2)
    else:
        opt = FedAvgMOTA(params, lr=args.server_lr, momentum=args.momentum)

    loss_hist, acc_hist = [], []
    for rnd in range(1, rounds + 1):
        run_round(model, client_loaders, oracle, opt,
                  local_epochs=args.local_epochs,
                  local_lr=args.local_lr,
                  use_mac=args.use_mac,
                  device=device)
        loss, acc = evaluate(model, test_loader, device)
        loss_hist.append(loss)
        acc_hist.append(acc)

    return {"loss": loss_hist, "acc": acc_hist}


def ablation_alpha(args, device: torch.device) -> dict:
    """
    Ablation A: vary α, fix N.
    Runs AdaGrad-OTA, Adam-OTA, FedAvgM-OTA for each α value.
    """
    print("\n=== Ablation: Noise tail index α ===")
    results = {}
    for alpha in ALPHA_VALUES:
        results[alpha] = {}
        for opt_name in ["fedavgm", "adagrad_ota", "adam_ota"]:
            print(f"  α={alpha:.1f}  {opt_name} ...", flush=True)
            results[alpha][opt_name] = _run_single(
                args.model, args.dataset, args.num_clients,
                alpha, args.rounds, args, device, opt_name,
            )
            final_acc = results[alpha][opt_name]["acc"][-1]
            print(f"    → final acc={final_acc:.4f}")
    return results


def ablation_clients(args, device: torch.device) -> dict:
    """
    Ablation B: vary N, fix α.
    Runs Adam-OTA for each N value (best method per proposal hypothesis).
    """
    print("\n=== Ablation: Number of clients N ===")
    results = {}
    for N in CLIENT_VALUES:
        print(f"  N={N} ...", flush=True)
        results[N] = _run_single(
            args.model, args.dataset, N,
            args.alpha, args.rounds, args, device, "adam_ota",
        )
        final_acc = results[N]["acc"][-1]
        print(f"    → final acc={final_acc:.4f}")
    return results


def parse_args():
    p = argparse.ArgumentParser(description="ADOTA-FL ablation studies")
    p.add_argument("--study", default="both", choices=["alpha", "clients", "both"])
    p.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10"])
    p.add_argument("--model", default="resnet18",
                   choices=["mlp", "convnet", "resnet18", "resnet34"])
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--num_clients", type=int, default=10,
                   help="Fixed N for alpha ablation")
    p.add_argument("--alpha", type=float, default=1.5,
                   help="Fixed α for client-count ablation")
    p.add_argument("--noise_scale", type=float, default=0.01)
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

    if args.study in ("alpha", "both"):
        res_alpha = ablation_alpha(args, device)
        path = out_dir / f"ablation_alpha_{args.dataset}_{args.model}.json"
        with open(path, "w") as f:
            json.dump({"args": vars(args), "results": {str(k): v for k, v in res_alpha.items()}}, f, indent=2)
        print(f"Saved: {path}")

    if args.study in ("clients", "both"):
        res_clients = ablation_clients(args, device)
        path = out_dir / f"ablation_clients_{args.dataset}_{args.model}.json"
        with open(path, "w") as f:
            json.dump({"args": vars(args), "results": {str(k): v for k, v in res_clients.items()}}, f, indent=2)
        print(f"Saved: {path}")
