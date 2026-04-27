"""
Analog Over-the-Air (A-OTA) aggregation — the "Noisy Oracle".

Models gradient aggregation with symmetric α-stable noise:
    g_t = (1/N) * sum_n ∇f_n(w_t)  +  ξ_t

where:
  - ξ_t ~ symmetric α-stable interference (i.i.d. entries)
"""

import torch
from .noise import sample_alpha_stable


class NoisyOracle:
    """
    Simulates the A-OTA gradient aggregation channel.

    Clients transmit gradients simultaneously; the server receives
    their noise-corrupted superposition.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        noise_scale: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            alpha:         Noise tail index (1 < alpha <= 2). alpha=2 → AWGN.
            noise_scale:   Scale parameter γ of the α-stable interference ξ_t.
            device:        Compute device.
        """
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.device = device

    def aggregate(self, gradients: list[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate client gradients over the air.

        Args:
            gradients: List of raw local gradients ∇f_n(w_t), one per client.
                       All tensors must have the same shape.

        Returns:
            Noisy aggregated gradient g_t (not yet divided by N — caller normalizes).
        """
        N = len(gradients)
        stacked = torch.stack(gradients)  # (N, *param_shape)
        aggregated = stacked.sum(dim=0)

        xi = sample_alpha_stable(
            alpha=self.alpha,
            size=aggregated.shape,
            scale=self.noise_scale,
            device=self.device,
        )
        return aggregated + xi

    def __repr__(self) -> str:
        return f"NoisyOracle(alpha={self.alpha}, noise_scale={self.noise_scale})"
