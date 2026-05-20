"""
Alpha-stable noise generation using the Chambers-Mallows-Stuck method.

The implementation uses the symmetric S_alpha(scale) convention with
characteristic function exp(-(scale * |t|)^alpha). Under this convention the
Gaussian endpoint alpha=2 is N(0, 2 * scale^2), i.e. std=sqrt(2)*scale.
"""

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
    if scale < 0:
        raise ValueError(f"scale must be non-negative, got {scale}")

    if scale == 0:
        return torch.zeros(size, device=device, dtype=torch.float32)

    if alpha == 2:
        # S_2(scale) has characteristic function exp(-scale^2 t^2), matching
        # N(0, 2*scale^2). This keeps gamma consistent with S_alpha(gamma).
        return torch.randn(size, device=device, dtype=torch.float32) * (2.0 ** 0.5 * scale)

    # CMS method directly in torch. On CUDA this avoids NumPy CPU generation
    # and a large host->device transfer for every parameter tensor.
    size_t = size if isinstance(size, tuple) else (size,)
    dtype = torch.float32
    pi = torch.pi
    U = (torch.rand(size_t, device=device, dtype=dtype) - 0.5) * pi
    W = torch.empty(size_t, device=device, dtype=dtype).exponential_(1.0)

    if alpha == 1:
        return torch.tan(U) * scale

    # Symmetric CMS formula (beta=0):
    # X = sin(alpha U) / cos(U)^(1/alpha)
    #     * (cos((1-alpha)U) / W)^((1-alpha)/alpha)
    cos_u = torch.cos(U).clamp_min(torch.finfo(dtype).tiny)
    term1 = torch.sin(alpha * U) / cos_u.pow(1.0 / alpha)
    term2 = (torch.cos((1.0 - alpha) * U) / W).pow((1.0 - alpha) / alpha)
    X = term1 * term2

    return X * scale


def awgn(signal: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Add AWGN to a signal at a given SNR (dB)."""
    signal_power = signal.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(signal) * noise_power.sqrt()
    return signal + noise
