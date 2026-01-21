"""
Main training script for Adaptive OTA-FL experiments.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse

from ..ota_fl import NoisyOracle, AdaGradOTA, AdamOTA
from ..ota_fl.models import get_model
from .config import ExperimentConfig


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(
    dataset_name: str,
    data_dir: str = "data",
) -> Tuple[Dataset, Dataset]:
    """Load dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)
    
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(
            root=str(data_dir),
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            root=str(data_dir),
            train=False,
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            )
        ])
        train_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.CIFAR10(
            root=str(data_dir),
            train=False,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset


def split_data_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> List[Dataset]:
    """Split dataset in IID fashion."""
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    splits = np.array_split(indices, num_clients)
    
    client_datasets = []
    for split in splits:
        client_datasets.append(torch.utils.data.Subset(dataset, split))
    
    return client_datasets


def split_data_non_iid(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[Dataset]:
    """Split dataset in non-IID fashion using Dirichlet distribution."""
    np.random.seed(seed)
    num_classes = len(set([dataset[i][1] for i in range(len(dataset))]))
    
    # Get class indices
    class_indices = {i: [] for i in range(num_classes)}
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Sample from Dirichlet distribution
    proportions = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # Assign samples to clients
    client_indices = [[] for _ in range(num_clients)]
    for class_idx in range(num_classes):
        class_samples = class_indices[class_idx]
        np.random.shuffle(class_samples)
        
        # Split class samples according to proportions
        cumsum = np.cumsum(proportions[class_idx])
        cumsum = (cumsum * len(class_samples)).astype(int)
        cumsum = np.concatenate([[0], cumsum])
        
        for client_idx in range(num_clients):
            start = cumsum[client_idx]
            end = cumsum[client_idx + 1]
            client_indices[client_idx].extend(class_samples[start:end])
    
    # Create client datasets
    client_datasets = []
    for indices in client_indices:
        client_datasets.append(torch.utils.data.Subset(dataset, indices))
    
    return client_datasets


def train_local(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_epochs: int = 1,
) -> Dict[str, torch.Tensor]:
    """Train model locally and return gradients."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Store gradients
    gradients = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients[name] = torch.zeros_like(param.data)
    
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            model.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradients[name] += param.grad.data
    
    # Average gradients
    num_batches = len(dataloader) * num_epochs
    for name in gradients:
        gradients[name] /= num_batches
    
    return gradients


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model and return accuracy and loss."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss


def run_experiment(config: ExperimentConfig):
    """Run a complete federated learning experiment."""
    # Set seed
    set_seed(config.training.seed)
    
    # Setup device
    device = torch.device(
        config.training.device
        if torch.cuda.is_available() and config.training.device == "cuda"
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset: {config.data.name}")
    train_dataset, test_dataset = load_dataset(config.data.name)
    
    # Split data
    if config.data.data_split == "iid":
        client_datasets = split_data_iid(
            train_dataset,
            config.data.num_clients,
            seed=config.training.seed,
        )
    else:
        client_datasets = split_data_non_iid(
            train_dataset,
            config.data.num_clients,
            alpha=config.data.non_iid_alpha,
            seed=config.training.seed,
        )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
    )
    
    # Initialize model
    model = get_model(config.model.name, config.model.num_classes)
    model = model.to(device)
    
    # Initialize Noisy Oracle
    noisy_oracle = NoisyOracle(
        alpha=config.noise.alpha,
        noise_scale=config.noise.noise_scale,
        fading_variance=config.noise.fading_variance,
        use_clipping=config.noise.use_clipping,
        clip_threshold=config.noise.clip_threshold,
        device=device,
    )
    
    # Initialize optimizer
    if config.optimizer.name == "adagradota":
        optimizer = AdaGradOTA(
            model.parameters(),
            lr=config.optimizer.lr,
            alpha=config.optimizer.alpha,
            eps=config.optimizer.eps,
        )
    elif config.optimizer.name == "adamota":
        optimizer = AdamOTA(
            model.parameters(),
            lr=config.optimizer.lr,
            betas=tuple(config.optimizer.betas),
            alpha=config.optimizer.alpha,
            eps=config.optimizer.eps,
            weight_decay=config.optimizer.weight_decay,
            amsgrad=config.optimizer.amsgrad,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer.name}")
    
    # Training loop
    results = {
        "train_loss": [],
        "test_accuracy": [],
        "test_loss": [],
    }
    
    print("\nStarting training...")
    for round_num in tqdm(range(config.training.num_rounds), desc="Rounds"):
        # Sample clients (for simplicity, use all clients)
        selected_clients = list(range(config.data.num_clients))
        
        # Collect local gradients
        local_gradients = []
        for client_idx in selected_clients:
            client_loader = DataLoader(
                client_datasets[client_idx],
                batch_size=config.data.batch_size,
                shuffle=True,
            )
            
            # Train locally
            local_grads = train_local(
                model,
                client_loader,
                device,
                num_epochs=config.training.local_epochs,
            )
            local_gradients.append(local_grads)
        
        # Stack gradients for aggregation
        aggregated_gradients = {}
        for param_name in local_gradients[0].keys():
            # Stack gradients from all clients [num_clients, ...]
            stacked = torch.stack([grads[param_name] for grads in local_gradients])
            
            # Aggregate through Noisy Oracle
            aggregated = noisy_oracle.aggregate(stacked)
            aggregated_gradients[param_name] = aggregated
        
        # Update model using aggregated gradients
        # Set gradients manually
        for name, param in model.named_parameters():
            if name in aggregated_gradients:
                param.grad = aggregated_gradients[name]
        
        # Optimizer step
        optimizer.step()
        
        # Evaluate
        if (round_num + 1) % config.training.log_interval == 0:
            test_acc, test_loss = evaluate(model, test_loader, device)
            results["test_accuracy"].append(test_acc)
            results["test_loss"].append(test_loss)
            
            if (round_num + 1) % (config.training.log_interval * 5) == 0:
                print(
                    f"Round {round_num + 1}: "
                    f"Test Accuracy: {test_acc:.2f}%, "
                    f"Test Loss: {test_loss:.4f}"
                )
    
    # Save results
    save_dir = Path(config.training.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    results_path = save_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Adaptive OTA-FL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()
    
    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    
    # Run experiment
    run_experiment(config)


if __name__ == "__main__":
    main()
