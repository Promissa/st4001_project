"""
Dataset loading and federated partitioning for OTA-FL experiments.

Partitioning strategies:
  - iid_partition      : Uniform random split — each client sees all classes equally.
  - dirichlet_partition: Non-IID via Dirichlet(Dir) distribution over class labels.
                         Dir=0.1 → highly heterogeneous (proposal default).
                         Dir=1.0 → mildly heterogeneous.
                         Dir→∞  → approaches IID.
"""

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
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        train = datasets.CIFAR10(root, train=True, download=True, transform=transform_train)
        test = datasets.CIFAR10(root, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose 'mnist' or 'cifar10'.")
    return train, test


def _dataset_labels(dataset) -> np.ndarray:
    """Return labels without invoking expensive image transforms when possible."""
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    return np.array([dataset[i][1] for i in range(len(dataset))])


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
    labels = _dataset_labels(dataset)
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


class FastCIFAR10Store:
    """Shared tensor cache for CIFAR-10, avoiding PIL transforms in every client."""

    def __init__(self, dataset):
        if not hasattr(dataset, "data") or not hasattr(dataset, "targets"):
            raise TypeError("Fast CIFAR-10 loader requires a torchvision CIFAR10 dataset.")
        images = torch.as_tensor(dataset.data, dtype=torch.uint8)
        self.images = images.permute(0, 3, 1, 2).contiguous()
        self.labels = torch.as_tensor(dataset.targets, dtype=torch.long)


def _random_crop_flip_batch(x: torch.Tensor, padding: int = 4) -> torch.Tensor:
    """Apply CIFAR-style random crop and flip to a whole GPU batch."""
    n, _, h, w = x.shape
    padded = F.pad(x, (padding, padding, padding, padding))
    crops = padded.unfold(2, h, 1).unfold(3, w, 1)
    top = torch.randint(0, 2 * padding + 1, (n,), device=x.device)
    left = torch.randint(0, 2 * padding + 1, (n,), device=x.device)
    batch = torch.arange(n, device=x.device)
    x = crops[batch, :, top, left].contiguous()

    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip.view(n, 1, 1, 1), x.flip(-1), x)


class FastCIFAR10Loader:
    """
    Lightweight CIFAR-10 client loader.

    It yields uint8 CPU batches and exposes prepare_batch(), which performs
    transfer, augmentation, normalization, and optional channels-last layout on
    the target device. This keeps the GPU fed without spawning one worker pool
    per federated client.
    """

    def __init__(
        self,
        store: FastCIFAR10Store,
        indices,
        batch_size: int,
        shuffle: bool = True,
        train: bool = True,
        pin_memory: bool = True,
        device: torch.device | None = None,
    ):
        self.store = store
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train
        self.pin_memory = pin_memory and torch.cuda.is_available() and (device is None or device.type != "cuda")
        self.device = device
        self._norm_cache: dict[tuple[str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

        # Pre-load client data to GPU to eliminate per-batch CPU→GPU transfer
        if device is not None and device.type == "cuda":
            self._images = store.images.index_select(0, self.indices).to(device, non_blocking=True)
            self._labels = store.labels.index_select(0, self.indices).to(device, non_blocking=True)
        else:
            self._images = None
            self._labels = None

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __iter__(self):
        if self._images is not None:
            # GPU-resident path: shuffle and slice directly on GPU
            order = (
                torch.randperm(len(self.indices), device=self.device)
                if self.shuffle
                else torch.arange(len(self.indices), device=self.device)
            )
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                yield self._images.index_select(0, idx), self._labels.index_select(0, idx)
        else:
            if self.shuffle:
                order = self.indices[torch.randperm(len(self.indices))]
            else:
                order = self.indices
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start + self.batch_size]
                x = self.store.images.index_select(0, idx)
                y = self.store.labels.index_select(0, idx)
                if self.pin_memory:
                    x = x.pin_memory()
                    y = y.pin_memory()
                yield x, y

    def _norm_tensors(self, device: torch.device, dtype: torch.dtype):
        key = (str(device), dtype)
        if key not in self._norm_cache:
            mean = torch.tensor(CIFAR10_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
            std = torch.tensor(CIFAR10_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
            self._norm_cache[key] = (mean, std)
        return self._norm_cache[key]

    def prepare_batch(
        self,
        batch,
        device: torch.device,
        channels_last: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        x = x.to(device, non_blocking=True).float().div_(255.0)
        y = y.to(device, non_blocking=True)
        if self.train:
            x = _random_crop_flip_batch(x)
        mean, std = self._norm_tensors(device, x.dtype)
        x = x.sub_(mean).div_(std)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x, y


def _subset_indices(subset: Subset):
    return subset.indices if isinstance(subset, Subset) else range(len(subset))


def make_fast_cifar10_loaders(
    subsets: list[Subset],
    batch_size: int,
    shuffle: bool = True,
    pin_memory: bool = True,
    device: torch.device | None = None,
) -> list[FastCIFAR10Loader]:
    if not subsets:
        return []
    store = FastCIFAR10Store(subsets[0].dataset)
    return [
        FastCIFAR10Loader(
            store,
            _subset_indices(subset),
            batch_size=batch_size,
            shuffle=shuffle,
            train=True,
            pin_memory=pin_memory,
            device=device,
        )
        for subset in subsets
    ]


def make_fast_cifar10_eval_loader(
    dataset,
    batch_size: int,
    pin_memory: bool = True,
    device: torch.device | None = None,
) -> FastCIFAR10Loader:
    store = FastCIFAR10Store(dataset)
    return FastCIFAR10Loader(
        store,
        range(len(dataset)),
        batch_size=batch_size,
        shuffle=False,
        train=False,
        pin_memory=pin_memory,
        device=device,
    )
