"""
Model architectures for MNIST and CIFAR-10 experiments.

Models:
  - MLP       : 2-layer MLP for MNIST (fast baseline)
  - ConvNet   : Small CNN for CIFAR-10
  - ResNet18  : Standard ResNet-18 for CIFAR-10 (proposal primary model)
  - ResNet34  : Standard ResNet-34 for CIFAR-10 (larger scale)
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


class MLP(nn.Module):
    """Simple 2-layer MLP for MNIST."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 200, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ConvNet(nn.Module):
    """Small CNN for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _make_resnet(base_fn, num_classes: int) -> nn.Module:
    """Build a ResNet adapted for CIFAR-10 (32x32 inputs)."""
    model = base_fn(weights=None, num_classes=num_classes)
    # Replace first conv (7x7 stride-2) with 3x3 stride-1 suitable for 32x32
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_model(name: str, num_classes: int = 10) -> nn.Module:
    models = {
        "mlp":      lambda: MLP(num_classes=num_classes),
        "convnet":  lambda: ConvNet(num_classes=num_classes),
        "resnet18": lambda: _make_resnet(resnet18, num_classes),
        "resnet34": lambda: _make_resnet(resnet34, num_classes),
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models)}")
    return models[name]()
