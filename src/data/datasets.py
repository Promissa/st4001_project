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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


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


class FastCIFAR10Loader:
    """
    Lightweight tensor loader for CIFAR-10.

    It avoids per-sample PIL/torchvision transforms in the hot path. Batches are
    moved to the target GPU first, then crop/flip/normalization are applied as
    vectorized torch operations. This is much faster on AutoDL-style machines
    where 5090 GPUs can otherwise sit idle waiting for CPU transforms.
    """

    def __init__(
        self,
        dataset,
        indices: list[int] | np.ndarray | None,
        batch_size: int,
        shuffle: bool,
        augment: bool,
        gpu_cache: bool = False,
    ):
        if not hasattr(dataset, "data") or not hasattr(dataset, "targets"):
            raise TypeError("FastCIFAR10Loader requires a torchvision CIFAR10 dataset.")

        if indices is None:
            indices_np = np.arange(len(dataset.data))
        else:
            indices_np = np.asarray(indices, dtype=np.int64)

        data = np.ascontiguousarray(dataset.data[indices_np])
        targets = np.asarray(dataset.targets, dtype=np.int64)[indices_np]

        self.data = torch.from_numpy(data).permute(0, 3, 1, 2).contiguous()
        self.targets = torch.from_numpy(targets).long()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.gpu_cache = gpu_cache
        self.mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self._norm_cache: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
        self._gpu_cache: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def num_samples(self) -> int:
        return len(self.targets)

    def __len__(self) -> int:
        return (len(self.targets) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.targets)
        order = torch.randperm(n) if self.shuffle else torch.arange(n)
        for start in range(0, n, self.batch_size):
            batch_idx = order[start:start + self.batch_size]
            if self.gpu_cache:
                yield batch_idx, None
                continue
            yield self.data[batch_idx], self.targets[batch_idx]

    def _cached_data(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cache_key = (device.type, device.index)
        if cache_key not in self._gpu_cache:
            self._gpu_cache[cache_key] = (
                self.data.to(device=device, non_blocking=False),
                self.targets.to(device=device, non_blocking=False),
            )
        return self._gpu_cache[cache_key]

    def prepare_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None,
        device: torch.device,
        channels_last: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if y is None:
            data, targets = self._cached_data(device)
            idx = x.to(device=device, non_blocking=True)
            x = data.index_select(0, idx).float().div_(255.0)
            y = targets.index_select(0, idx)
        else:
            x = x.to(device, non_blocking=True).float().div_(255.0)
            y = y.to(device, non_blocking=True)

        if self.augment:
            x = _random_crop_flip_batch(x)

        cache_key = (device.type, device.index)
        if cache_key not in self._norm_cache:
            self._norm_cache[cache_key] = (
                self.mean.to(device=device),
                self.std.to(device=device),
            )
        mean, std = self._norm_cache[cache_key]
        x = (x - mean) / std
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x, y


def _random_crop_flip_batch(x: torch.Tensor, padding: int = 4) -> torch.Tensor:
    """Apply CIFAR-style random crop and horizontal flip to a GPU batch."""
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))

    batch_size = x.shape[0]
    crop_size = 32
    max_offset = padding * 2
    top = torch.randint(0, max_offset + 1, (batch_size,), device=x.device)
    left = torch.randint(0, max_offset + 1, (batch_size,), device=x.device)

    patches = x.unfold(2, crop_size, 1).unfold(3, crop_size, 1)
    batch_idx = torch.arange(batch_size, device=x.device)
    x = patches[batch_idx, :, top, left, :, :]

    flip_mask = (torch.rand(batch_size, device=x.device) < 0.5).view(batch_size, 1, 1, 1)
    return torch.where(flip_mask, x.flip(-1), x)


def make_fast_cifar10_loaders(
    subsets: list[Subset],
    batch_size: int,
    shuffle: bool = True,
    gpu_cache: bool = False,
) -> list[FastCIFAR10Loader]:
    loaders = []
    for subset in subsets:
        if not isinstance(subset, Subset):
            raise TypeError("Expected Subset objects produced by the partition helpers.")
        loaders.append(
            FastCIFAR10Loader(
                subset.dataset,
                subset.indices,
                batch_size=batch_size,
                shuffle=shuffle,
                augment=True,
                gpu_cache=gpu_cache,
            )
        )
    return loaders


def make_fast_cifar10_eval_loader(
    dataset,
    batch_size: int,
    gpu_cache: bool = False,
) -> FastCIFAR10Loader:
    return FastCIFAR10Loader(
        dataset,
        indices=None,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        gpu_cache=gpu_cache,
    )
