"""
Configuration management for experiments.
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class NoiseConfig:
    """Configuration for noise generation."""
    alpha: float = 1.5  # Tail index (1 < alpha <= 2)
    noise_scale: float = 0.1  # Scale parameter
    fading_variance: float = 0.1  # Channel fading variance
    use_clipping: bool = True  # Whether to use MAC
    clip_threshold: Optional[float] = None  # MAC threshold (None for auto)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    name: str = "adamota"  # "adagradota" or "adamota"
    lr: float = 1e-3  # Learning rate
    alpha: float = 1.5  # α-norm parameter (should match noise alpha)
    betas: tuple = (0.9, 0.999)  # For Adam-OTA only
    eps: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False  # For Adam-OTA only


@dataclass
class ModelConfig:
    """Configuration for model."""
    name: str = "mlp"  # "mlp", "cnn", "resnet18", etc.
    num_classes: int = 10


@dataclass
class DataConfig:
    """Configuration for dataset."""
    name: str = "mnist"  # "mnist" or "cifar10"
    batch_size: int = 64
    num_clients: int = 100
    data_split: str = "iid"  # "iid" or "non-iid"
    non_iid_alpha: float = 0.5  # For non-IID split (Dirichlet parameter)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_rounds: int = 100
    local_epochs: int = 1
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42
    save_dir: str = "results"
    log_interval: int = 10


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create configuration from dictionary."""
        return cls(
            noise=NoiseConfig(**config_dict.get("noise", {})),
            optimizer=OptimizerConfig(**config_dict.get("optimizer", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "noise": self.noise.__dict__,
            "optimizer": self.optimizer.__dict__,
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
        }
    
    def save_yaml(self, config_path: str):
        """Save configuration to YAML file."""
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
