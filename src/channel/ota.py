"""
Analog Over-the-Air (A-OTA) aggregation - the "Noisy Oracle".

Models over-the-air aggregation with additive channel noise. The caller decides
whether the transmitted tensor is a raw gradient, pseudo-gradient, or model
delta, and applies the experiment-specific normalization after aggregation.

alpha=2 gives AWGN. alpha<2 gives symmetric alpha-stable heavy-tailed
interference, which is the core channel setting for the heavy-tail
experiments.
"""

import torch
from .noise import sample_alpha_stable


class NoisyOracle:
    """
    Simulates the A-OTA gradient aggregation channel.

    Clients transmit tensors simultaneously; the server receives their
    noise-corrupted superposition.
    """

    def __init__(
        self,
        alpha: float = 2.0,
        noise_scale: float = 0.05,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            alpha:         Stability index of ξ_t (default 2.0 → AWGN).
            noise_scale:   Scale parameter γ of the additive interference.
            device:        Compute device.
        """
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.device = device
        self.begin_round()

    def begin_round(self, collect_diagnostics: bool = False) -> None:
        """Reset per-round diagnostic statistics."""
        self.collect_diagnostics = collect_diagnostics
        self.round_noise_sq = 0.0
        self.round_noise_max_abs = 0.0
        self.round_noise_finite = True

    def aggregate_sum(self, aggregated: torch.Tensor) -> torch.Tensor:
        """Add OTA channel noise to an already summed client update tensor."""
        if self.noise_scale == 0:
            if self.collect_diagnostics:
                self.round_noise_sq += 0.0
                self.round_noise_max_abs = max(self.round_noise_max_abs, 0.0)
            return aggregated

        xi = sample_alpha_stable(
            alpha=self.alpha,
            size=aggregated.shape,
            scale=self.noise_scale,
            device=aggregated.device,
        ).to(dtype=aggregated.dtype)

        if self.collect_diagnostics:
            xi_finite = torch.isfinite(xi).all().item()
            self.round_noise_finite = self.round_noise_finite and bool(xi_finite)
            if xi_finite:
                xi_float = xi.detach().float()
                self.round_noise_sq += float(xi_float.pow(2).sum().item())
                self.round_noise_max_abs = max(
                    self.round_noise_max_abs,
                    float(xi_float.abs().max().item()) if xi_float.numel() else 0.0,
                )
            else:
                self.round_noise_sq = float("inf")
                self.round_noise_max_abs = float("inf")

        return aggregated + xi

    def aggregate(self, gradients: list[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate client tensors over the air.

        Args:
            gradients: List of same-shaped client tensors.

        Returns:
            Noisy aggregate sum. The caller applies any required normalization.
        """
        stacked = torch.stack(gradients)  # (N, *param_shape)
        aggregated = stacked.sum(dim=0)
        return self.aggregate_sum(aggregated)

    def diagnostics(self) -> dict[str, float | bool]:
        """Return per-round noise diagnostics collected during aggregation."""
        return {
            "noise_norm": self.round_noise_sq ** 0.5,
            "noise_max_abs": self.round_noise_max_abs,
            "noise_finite": self.round_noise_finite,
        }

    def __repr__(self) -> str:
        return f"NoisyOracle(alpha={self.alpha}, noise_scale={self.noise_scale})"
