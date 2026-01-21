"""Tests for α-stable noise generation."""

import torch
import pytest
from src.ota_fl.noise import generate_alpha_stable_noise, AlphaStableNoise


def test_alpha_stable_noise_shape():
    """Test that noise has correct shape."""
    noise = generate_alpha_stable_noise(alpha=1.5, size=(10, 5))
    assert noise.shape == (10, 5)


def test_alpha_stable_noise_alpha_range():
    """Test that alpha must be in (1, 2]."""
    with pytest.raises(ValueError):
        generate_alpha_stable_noise(alpha=0.5, size=(10,))
    with pytest.raises(ValueError):
        generate_alpha_stable_noise(alpha=3.0, size=(10,))


def test_alpha_stable_noise_gaussian_case():
    """Test that alpha=2 gives approximately Gaussian distribution."""
    noise = generate_alpha_stable_noise(alpha=2.0, size=(10000,))
    # Check that mean is approximately 0 and std is approximately sqrt(2)
    assert torch.abs(noise.mean()) < 0.1
    assert torch.abs(noise.std() - torch.sqrt(torch.tensor(2.0))) < 0.2


def test_alpha_stable_noise_class():
    """Test AlphaStableNoise class."""
    generator = AlphaStableNoise(alpha=1.5, scale=0.1)
    noise = generator((10, 5))
    assert noise.shape == (10, 5)


if __name__ == "__main__":
    pytest.main([__file__])
