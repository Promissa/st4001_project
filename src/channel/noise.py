"""
Alpha-stable noise generation using the Chambers-Mallows-Stuck method.

Supports AWGN (alpha=2) and heavy-tailed impulsive noise (1 < alpha < 2).
"""

import numpy as np
import torch


def sample_alpha_stable(
    alpha: float,
    size: tuple | int,
    scale: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Generate symmetric alpha-stable (SaS) noise using the Chambers-Mallows-Stuck method.

    Args:
        alpha: Tail index (0 < alpha <= 2). alpha=2 gives Gaussian, alpha<2 gives heavy tails.
        size: Output shape.
        scale: Scale parameter (gamma), controls noise power.
        device: Target torch device.

    Returns:
        Tensor of SaS noise samples.
    """
    if not (0 < alpha <= 2):
        raise ValueError(f"alpha must be in (0, 2], got {alpha}")

    if scale == 0:
        return torch.zeros(size, device=device, dtype=torch.float32)

    if alpha == 2:
        # Gaussian special case: scale maps to std = sqrt(2) * gamma
        return torch.randn(size, device=device, dtype=torch.float32) * (scale * (2 ** 0.5))

    # CMS method - generate on CPU, move to GPU asynchronously
    size_np = size if isinstance(size, tuple) else (size,)
    U = np.random.uniform(-np.pi / 2, np.pi / 2, size_np)
    W = np.random.exponential(1.0, size_np)

    # Symmetric case (beta=0)
    B = 0.0
    S = (np.cos(B * np.arctan(np.tan(np.pi * alpha / 2))) ** (1 / alpha))
    term1 = np.sin(alpha * (U + B * np.pi / (2 * alpha)))
    term2 = (np.cos(U - alpha * (U + B * np.pi / (2 * alpha))) / W) ** ((1 - alpha) / alpha)
    X = S * term1 * term2

    # Convert to tensor and move to device with non_blocking if possible
    return torch.as_tensor(X * scale, dtype=torch.float32, device=device)


def awgn(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add AWGN to a signal at a given SNR (dB)."""
    signal_power = signal.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(signal) * noise_power.sqrt()
    return signal + noise
