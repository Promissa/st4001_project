"""
Adam-OTA: server-side Adam update for Over-the-Air FL.

Implements Algorithm 1 (Adam branch) from Wang et al. (2024):

    Δ_t = β₁ · Δ_{t-1}  +  (1 − β₁) · g_t                [1st moment / momentum]
    v_t = β₂ · v_{t-1}  +  (1 − β₂) · |Δ_t|^α            [EMA of α-norm]
    w_{t+1} = w_t  −  η · Δ_t / (ᵅ√v_t + ε)              [α-root step]

Key difference from AdaGrad-OTA: v_t uses an exponential moving average
rather than a cumulative sum, making the step size more responsive to
recent gradient history and less sensitive to early noise impulses.

Convergence rate: O(1/T)  (Theorem 2) — faster than AdaGrad-OTA.
"""

import torch


class AdamOTA:
    """
    Server-side Adam-OTA optimizer.

    Operates on the server's global model using the OTA-aggregated
    gradient g_t passed explicitly each round.
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        alpha: float = 1.5,
        beta1: float = 0.9,
        beta2: float = 0.3,
        eps: float = 1e-4,
    ):
        """
        Args:
            params: List of global model parameter tensors.
            lr:     Step size η.
            alpha:  Tail index of interference (should match NoisyOracle.alpha).
            beta1:  1st-moment decay β₁ ∈ [0, 1).
            beta2:  2nd-moment decay β₂ ∈ (0, 1). Paper finds β₂=0.3 optimal (Fig. 4).
            eps:    Numerical stability ε.
        """
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.Delta = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, agg_grads: list[torch.Tensor]):
        """
        One server-side update step (one FL round).

        Args:
            agg_grads: OTA-aggregated gradients [g_t], one tensor per parameter,
                       already normalized by N.
        """
        for p, g, delta, v in zip(self.params, agg_grads, self.Delta, self.v):
            g = g.to(p.device)

            # Eq. (8): 1st moment — momentum smoothing
            delta.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)

            # Eq. (10): 2nd moment — EMA of α-th power (Adam branch)
            v.mul_(self.beta2).add_(delta.abs().pow(self.alpha), alpha=1.0 - self.beta2)

            # Eq. (11): update
            denom = v.pow(1.0 / self.alpha).add_(self.eps)
            p.addcdiv_(delta, denom, value=-self.lr)

    def zero(self):
        for d, v in zip(self.Delta, self.v):
            d.zero_()
            v.zero_()
