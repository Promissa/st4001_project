"""Shared runtime acceleration options for experiment entry points."""

from __future__ import annotations

import argparse

import torch


def add_accelerator_args(parser: argparse.ArgumentParser) -> None:
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp.add_argument("--no_amp", dest="amp", action="store_false")
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])

    channels_last = parser.add_mutually_exclusive_group()
    channels_last.add_argument("--channels_last", dest="channels_last", action="store_true", default=None)
    channels_last.add_argument("--no_channels_last", dest="channels_last", action="store_false")

    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--diagnostics_every", type=int, default=0)
    parser.add_argument("--fast_data", default="auto", choices=["auto", "on", "off"])


def resolve_accelerator_args(args, device: torch.device) -> None:
    cuda = device.type == "cuda"
    dataset = getattr(args, "dataset", "")
    model = getattr(args, "model", "")
    supports_channels_last = dataset == "cifar10" and model in {"convnet", "resnet18", "resnet34"}

    args.use_amp = cuda if args.amp is None else bool(args.amp)
    args.use_channels_last = (
        cuda and supports_channels_last
        if args.channels_last is None
        else bool(args.channels_last) and supports_channels_last
    )
    args.use_fast_data = (
        cuda and dataset == "cifar10"
        if args.fast_data == "auto"
        else args.fast_data == "on"
    )
    args.eval_every = max(1, int(args.eval_every))
    args.diagnostics_every = max(0, int(args.diagnostics_every))


def configure_model(model: torch.nn.Module, args, device: torch.device) -> torch.nn.Module:
    model = model.to(device)
    if getattr(args, "use_channels_last", False):
        model = model.to(memory_format=torch.channels_last)
    return model


def should_evaluate(round_idx: int, total_rounds: int, args) -> bool:
    return (
        round_idx == 1
        or round_idx == total_rounds
        or round_idx % args.eval_every == 0
    )


def should_collect_diagnostics(round_idx: int, args, save_diagnostics: bool = False) -> bool:
    return (
        save_diagnostics
        or round_idx == 1
        or (
            args.diagnostics_every > 0
            and round_idx % args.diagnostics_every == 0
        )
    )


def accelerator_summary(args) -> str:
    return (
        f"AMP:       {args.use_amp} ({args.amp_dtype}) | "
        f"channels_last={args.use_channels_last}\n"
        f"Fast data: {args.use_fast_data} ({args.fast_data}) | "
        f"eval_every={args.eval_every} | diagnostics_every={args.diagnostics_every}"
    )
