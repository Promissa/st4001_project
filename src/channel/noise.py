"""
Alpha-stable noise generation using the Chambers-Mallows-Stuck method.

Supports AWGN (alpha=2) and heavy-tailed impulsive noise (1 < alpha < 2).
"""

import math
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

    # CMS method, generated directly on the target device. This avoids the
    # CPU NumPy allocation + CPU->GPU copy that dominates heavy-tail runs.
    size_t = size if isinstance(size, tuple) else (size,)
    U = (torch.rand(size_t, device=device, dtype=torch.float32) - 0.5) * math.pi
    W = torch.empty(size_t, device=device, dtype=torch.float32).exponential_(1.0)
    W = W.clamp_min(torch.finfo(torch.float32).tiny)

    # Symmetric case (beta=0)
    B = 0.0
    S = math.cos(B * math.atan(math.tan(math.pi * alpha / 2))) ** (1 / alpha)
    term1 = torch.sin(alpha * (U + B * math.pi / (2 * alpha)))
    term2_base = torch.cos(U - alpha * (U + B * math.pi / (2 * alpha))) / W
    term2 = term2_base.clamp_min(torch.finfo(torch.float32).tiny).pow((1 - alpha) / alpha)
    X = S * term1 * term2

    return X * scale


def awgn(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add AWGN to a signal at a given SNR (dB)."""
    signal_power = signal.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(signal) * noise_power.sqrt()
    return signal + noise
