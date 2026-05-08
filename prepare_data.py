"""
Prepare and verify the datasets used by the ADOTA-FL experiments.

This script collects the training/evaluation data required by the current
codebase: MNIST and CIFAR-10. Wireless channel samples are not downloaded;
they are generated synthetically by the OTA noise simulator during training.

Usage:
    uv run prepare_data.py
    uv run prepare_data.py --dataset mnist --partition non_iid --num_clients 100
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.data.datasets import get_dataset, iid_partition, dirichlet_partition


SUPPORTED_DATASETS = ("mnist", "cifar10")

EXPECTED_FILES = {
    "mnist": [
        "MNIST/raw/train-images-idx3-ubyte",
        "MNIST/raw/train-labels-idx1-ubyte",
        "MNIST/raw/t10k-images-idx3-ubyte",
        "MNIST/raw/t10k-labels-idx1-ubyte",
    ],
    "cifar10": [
        "cifar-10-batches-py/data_batch_1",
        "cifar-10-batches-py/data_batch_2",
        "cifar-10-batches-py/data_batch_3",
        "cifar-10-batches-py/data_batch_4",
        "cifar-10-batches-py/data_batch_5",
        "cifar-10-batches-py/test_batch",
        "cifar-10-batches-py/batches.meta",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/verify MNIST and CIFAR-10 for ADOTA-FL."
    )
    parser.add_argument(
        "--dataset",
        choices=[*SUPPORTED_DATASETS, "all"],
        default="all",
        help="Dataset to prepare. Defaults to all supported datasets.",
    )
    parser.add_argument(
        "--data_root",
        default="./data",
        help="Directory where torchvision stores datasets.",
    )
    parser.add_argument(
        "--partition",
        choices=["none", "iid", "non_iid"],
        default="non_iid",
        help="Partition summary to record in the manifest.",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=100,
        help="Number of clients N for the recorded federated partition.",
    )
    parser.add_argument(
        "--dir_conc",
        type=float,
        default=0.1,
        help="Dirichlet concentration for non-IID partitioning.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for partitioning.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Noise tail index recorded for the experiment setup.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.05,
        help="Noise scale gamma recorded for the experiment setup.",
    )
    parser.add_argument(
        "--manifest",
        default="./data/training_data_manifest.json",
        help="JSON manifest path for the collected data and setup.",
    )
    return parser.parse_args()


def datasets_to_prepare(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(SUPPORTED_DATASETS)
    return [dataset_arg]


def verify_expected_files(dataset_name: str, data_root: Path) -> list[dict[str, Any]]:
    verified = []
    for relative_path in EXPECTED_FILES[dataset_name]:
        path = data_root / relative_path
        verified.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return verified


def build_partition_summary(train_ds, args: argparse.Namespace) -> dict[str, Any]:
    if args.partition == "none":
        return {"type": "none"}

    if args.partition == "iid":
        subsets = iid_partition(train_ds, args.num_clients, seed=args.seed)
        partition_type = "iid"
    else:
        subsets = dirichlet_partition(
            train_ds,
            args.num_clients,
            concentration=args.dir_conc,
            seed=args.seed,
        )
        partition_type = "non_iid_dirichlet"

    client_sizes = [len(subset) for subset in subsets]
    return {
        "type": partition_type,
        "num_clients": args.num_clients,
        "dirichlet_concentration": args.dir_conc if args.partition == "non_iid" else None,
        "seed": args.seed,
        "min_client_samples": min(client_sizes),
        "max_client_samples": max(client_sizes),
        "total_client_samples": sum(client_sizes),
    }


def prepare_dataset(dataset_name: str, args: argparse.Namespace, data_root: Path) -> dict[str, Any]:
    train_ds, test_ds = get_dataset(dataset_name, root=str(data_root))
    files = verify_expected_files(dataset_name, data_root)
    missing = [item["path"] for item in files if not item["exists"]]
    if missing:
        missing_list = "\n  - ".join(missing)
        raise FileNotFoundError(f"{dataset_name} is missing expected files:\n  - {missing_list}")

    return {
        "name": dataset_name,
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "expected_files": files,
        "partition": build_partition_summary(train_ds, args),
    }


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest)
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    datasets = [
        prepare_dataset(dataset_name, args, data_root)
        for dataset_name in datasets_to_prepare(args.dataset)
    ]

    manifest = {
        "data_root": str(data_root),
        "datasets": datasets,
        "channel_data": {
            "physical_wireless_dataset_required": False,
            "noise_source": "synthetic alpha-stable/AWGN samples generated during training",
            "alpha": args.alpha,
            "noise_scale_gamma": args.noise_scale,
        },
        "unsupported_reference_datasets": ["cifar100", "emnist"],
    }

    with open(manifest_path, "w") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)

    print(f"Prepared datasets: {', '.join(item['name'] for item in datasets)}")
    for item in datasets:
        partition = item["partition"]
        print(
            f"  {item['name']}: {item['train_samples']} train, "
            f"{item['test_samples']} test, partition={partition['type']}"
        )
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
