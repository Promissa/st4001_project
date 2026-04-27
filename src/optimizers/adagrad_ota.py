"""
AdaGrad-OTA: server-side AdaGrad update for Over-the-Air FL.

Implements Algorithm 1 (AdaGrad branch) from Wang et al. (2024):

    Δ_t  = β₁ · Δ_{t-1}  +  (1 − β₁) · g_t          [momentum smoothing]
    v_t  = v_{t-1}  +  |Δ_t|^α                        [α-norm accumulation]
    w_{t+1} = w_t  −  η · Δ_t / (ᵅ√v_t + ε)          [α-root step]

The accumulated |Δ_t|^α grows faster under heavy-tailed noise (large |Δ_t|
from impulses), automatically compressing the effective learning rate —
acting as an implicit noise filter without requiring channel state information.

Convergence rate: O(ln T / T^(1−1/α))  (Theorem 1)
"""

import torch


class AdaGradOTA:
    """
    Server-side AdaGrad-OTA optimizer.

    Unlike standard torch.optim.Optimizer (which runs on a single model),
    this operates on the server's global model using the OTA-aggregated
    gradient g_t passed explicitly each round.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 0.01,
        alpha: float = 1.5,
        beta1: float = 0.0,
        eps: float = 1e-4,
    ):
        """
        Args:
            params: List of global model parameter tensors (model.parameters()).
            lr:     Step size η.
            alpha:  Tail index of interference distribution (matches noise alpha).
            beta1:  Momentum coefficient for Δ_t. beta1=0 → no momentum (pure AdaGrad).
            eps:    Numerical stability constant ε.
        """
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.beta1 = beta1
        self.eps = eps

        self.Delta = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, agg_grads: list[torch.Tensor]):
        """
        One server-side update step (one FL round).

        Args:
            agg_grads: OTA-aggregated gradients [g_t], one tensor per parameter.
                       These are already normalized by N (i.e., g_t = oracle_output / N).
        """
        for p, g, delta, v in zip(self.params, agg_grads, self.Delta, self.v):
            g = g.to(p.device)

            # Eq. (8): momentum smoothing of the aggregated gradient
            delta.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)

            # Eq. (9): accumulate α-th power of |Δ_t|  (entry-wise)
            v.add_(delta.abs().pow(self.alpha))

            # Eq. (11): update — divide by α-th root of v_t
            # α√v  is computed as  v^(1/α)
            denom = v.pow(1.0 / self.alpha).add_(self.eps)
            p.addcdiv_(delta, denom, value=-self.lr)

    def zero(self):
        """Reset momentum and accumulator (call between experiments)."""
        for d, v in zip(self.Delta, self.v):
            d.zero_()
            v.zero_()
