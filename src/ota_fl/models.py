"""
Model architectures for federated learning experiments.
"""

import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Get a model architecture by name.
    
    Args:
        model_name: Name of the model ('mlp', 'cnn', 'resnet18', etc.)
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    if model_name == "mlp":
        return MLP(num_classes=num_classes)
    elif model_name == "cnn":
        return CNN(num_classes=num_classes)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}")


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron for MNIST."""
    
    def __init__(self, num_classes: int = 10, hidden_dim: int = 128):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    """Simple CNN for CIFAR-10."""
    
    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
