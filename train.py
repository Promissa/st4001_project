"""
ADOTA-FL — single-run training entry point.

Trains a global model under simulated Analog-OTA aggregation with AWGN.
For comparative analysis and ablation studies use the dedicated runners:

    uv run -m src.experiments.compare
    uv run -m src.experiments.ablation
    uv run -m src.experiments.plot

Usage:
    uv run train.py
    uv run train.py --dataset cifar10 --model resnet18 --optimizer adam_ota \\
                    --noise_scale 0.05 --rounds 200 --non_iid
"""

import argparse
import json
from pathlib import Path

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from src.channel.ota import NoisyOracle
from src.data.datasets import get_dataset, iid_partition, dirichlet_partition, make_loader
from src.experiments.federated import run_round, evaluate
from src.models.nets import get_model
from src.optimizers.adagrad_ota import AdaGradOTA
from src.optimizers.adam_ota import AdamOTA
from src.optimizers.baselines import FedAvgOTA, FedAvgMOTA


def build_optimizer(name: str, model, args):
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
    raise ValueError(f"Unknown optimizer: {name}")


def parse_args():
    p = argparse.ArgumentParser(description="ADOTA-FL single-run trainer")
    p.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10"])
    p.add_argument("--model", default="mlp",
                   choices=["mlp", "convnet", "resnet18", "resnet34"])
    p.add_argument("--optimizer", default="adam_ota",
                   choices=["fedavg", "fedavgm", "adagrad_ota", "adam_ota"])
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--server_lr", type=float, default=0.1)
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.3)
    # Channel
    p.add_argument("--alpha", type=float, default=2.0,
                   help="Stability index of the noise (2.0 = AWGN, default)")
    p.add_argument("--noise_scale", type=float, default=0.05,
                   help="Noise scale γ (std ≈ γ·√2 for AWGN)")
    # Data
    p.add_argument("--non_iid", action="store_true",
                   help="Use Dirichlet non-IID partition (Dir=0.1)")
    p.add_argument("--dir_conc", type=float, default=0.1)
    # Optional robust pre-processing (off by default; AWGN setting)
    p.add_argument("--use_mac", action="store_true", default=False,
                   help="Apply Median Anchored Clipping before the optimizer")
    p.add_argument("--mac_clip", type=float, default=3.0)
    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_results", action="store_true")
    p.add_argument("--out_dir", type=str, default="results/single")
    # TensorBoard
    p.add_argument("--use_tensorboard", action="store_true", default=True,
                   help="Enable TensorBoard logging")
    p.add_argument("--log_dir", type=str, default="runs",
                   help="TensorBoard log directory")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable cuDNN autotuner for better convolution performance
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # TensorBoard writer
    writer = None
    if args.use_tensorboard:
        log_dir = Path(args.log_dir) / f"{args.dataset}_{args.model}_{args.optimizer}_alpha{args.alpha}_N{args.num_clients}"
        log_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(log_dir))
        print(f"TensorBoard: {log_dir}")
        print(f"  Run: tensorboard --logdir={args.log_dir}")

    # Log hyperparameters
    if writer is not None:
        hparams = {
            "dataset": args.dataset,
            "model": args.model,
            "optimizer": args.optimizer,
            "rounds": args.rounds,
            "num_clients": args.num_clients,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "server_lr": args.server_lr,
            "local_lr": args.local_lr,
            "alpha": args.alpha,
            "noise_scale": args.noise_scale,
            "use_mac": args.use_mac,
            "mac_clip": args.mac_clip,
        }
        writer.add_hparams(hparams, {"hparam/best_accuracy": 0.0})

    print(f"Device:    {device}")
    print(f"Optimizer: {args.optimizer.upper()}")
    print(f"Dataset:   {args.dataset} | Model: {args.model}")
    print(f"Noise:     α={args.alpha}, scale={args.noise_scale}")
    print(f"Clients:   N={args.num_clients} | Non-IID: {args.non_iid} (Dir={args.dir_conc})")
    print(f"MAC:       {args.use_mac}")
    print()

    # Data
    train_ds, test_ds = get_dataset(args.dataset)
    if args.non_iid:
        subsets = dirichlet_partition(train_ds, args.num_clients, args.dir_conc)
    else:
        subsets = iid_partition(train_ds, args.num_clients)
    client_loaders = [make_loader(s, args.batch_size) for s in subsets]
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # Model, oracle, optimizer
    model = get_model(args.model).to(device)
    oracle = NoisyOracle(alpha=args.alpha, noise_scale=args.noise_scale, device=device)
    opt = build_optimizer(args.optimizer, model, args)

    loss_hist, acc_hist = [], []

    for rnd in range(1, args.rounds + 1):
        run_round(
            model, client_loaders, oracle, opt,
            local_epochs=args.local_epochs,
            local_lr=args.local_lr,
            use_mac=args.use_mac,
            mac_clip=args.mac_clip,
            device=device,
        )
        loss, acc = evaluate(model, test_loader, device)
        loss_hist.append(loss)
        acc_hist.append(acc)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("train/loss", loss, rnd)
            writer.add_scalar("train/accuracy", acc, rnd)

        if rnd % args.log_every == 0 or rnd == 1:
            print(f"Round {rnd:4d}/{args.rounds} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    print(f"\nFinal  | Loss: {loss_hist[-1]:.4f} | Acc: {acc_hist[-1]:.4f}")

    if writer is not None:
        writer.close()

    if args.save_results:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = (f"{args.dataset}_{args.model}_{args.optimizer}"
                 f"_alpha{args.alpha}_N{args.num_clients}.json")
        with open(out_dir / fname, "w") as f:
            json.dump({"args": vars(args), "loss": loss_hist, "acc": acc_hist}, f, indent=2)
        print(f"Results saved to {out_dir / fname}")


if __name__ == "__main__":
    main()
