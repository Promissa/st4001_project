"""
Dataset loading and federated partitioning for OTA-FL experiments.

Partitioning strategies:
  - iid_partition      : Uniform random split — each client sees all classes equally.
  - dirichlet_partition: Non-IID via Dirichlet(Dir) distribution over class labels.
                         Dir=0.1 → highly heterogeneous (proposal default).
                         Dir=1.0 → mildly heterogeneous.
                         Dir→∞  → approaches IID.
"""

import warnings

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


def get_dataset(name: str, root: str = "./data") -> tuple:
    """
    Load a dataset by name.

    Args:
        name: One of 'mnist' or 'cifar10'.
        root: Directory to cache downloaded data.

    Returns:
        (train_dataset, test_dataset)
    """
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train = datasets.MNIST(root, train=True, download=True, transform=transform)
        test = datasets.MNIST(root, train=False, download=True, transform=transform)
    elif name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        # Torchvision can emit a NumPy deprecation warning while unpickling
        # valid CIFAR-10 files. Keep long experiment logs readable.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed.*",
                category=Warning,
                module=r"torchvision\.datasets\.cifar",
            )
            train = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
            test = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose 'mnist' or 'cifar10'.")
    return train, test


def iid_partition(dataset, num_clients: int, seed: int = 42) -> list[Subset]:
    """Partition dataset into equal IID shards, one per client."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    shards = np.array_split(indices, num_clients)
    return [Subset(dataset, shard.tolist()) for shard in shards]


def dirichlet_partition(
    dataset,
    num_clients: int,
    concentration: float = 0.1,
    seed: int = 42,
    min_samples: int = 10,
) -> list[Subset]:
    """
    Non-IID partition via symmetric Dirichlet distribution over class labels.

    For each class c, samples a proportion vector p ~ Dir(concentration) of
    length num_clients, then assigns that fraction of class-c samples to each
    client. This is the standard heterogeneous data setup used in the proposal.

    Args:
        dataset:       PyTorch dataset with integer labels.
        num_clients:   Number of federated clients N.
        concentration: Dirichlet concentration parameter (Dir).
                       Lower → more heterogeneous. Proposal default: 0.1.
        seed:          RNG seed for reproducibility.
        min_samples:   Minimum samples per client (clients below this are merged
                       with their neighbour to avoid empty loaders).

    Returns:
        List of Subset objects, one per client.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = int(labels.max()) + 1

    # Group indices by class
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c_idx in class_indices:
        rng.shuffle(c_idx)
        # Sample Dirichlet proportions for this class
        proportions = rng.dirichlet(np.repeat(concentration, num_clients))
        # Convert to split points
        splits = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(c_idx, splits)):
            client_indices[client_id].extend(chunk.tolist())

    # Ensure no client is empty
    for i, idx in enumerate(client_indices):
        if len(idx) < min_samples:
            # Borrow from the largest client
            donor = max(range(num_clients), key=lambda j: len(client_indices[j]))
            borrow = client_indices[donor][:min_samples]
            client_indices[i].extend(borrow)
            client_indices[donor] = client_indices[donor][min_samples:]

    return [Subset(dataset, idx) for idx in client_indices]


def make_loader(subset: Subset, batch_size: int = 32, shuffle: bool = True, pin_memory: bool = True) -> DataLoader:
    # num_workers=0: avoid spawning 200 processes for 100 clients
    # pin_memory=True: faster CPU->GPU transfer with non_blocking=True
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=pin_memory, drop_last=False)
