"""
Adam-OTA: server-side Adam update applied to the OTA-aggregated gradient.

The server receives a noisy aggregated gradient g_t each round and applies
a momentum-smoothed, second-moment-normalised step:

    Δ_t = β₁ · Δ_{t-1}  +  (1 − β₁) · g_t                [1st moment]
    v_t = β₂ · v_{t-1}  +  (1 − β₂) · |Δ_t|^α            [EMA of |·|^α]
    w_{t+1} = w_t  −  η · Δ_t / (v_t^{1/α} + ε)

α = 2 (default) recovers classical Adam (squared moments + square root) and
is the AWGN setting used throughout this project. The α parameter is kept
for generality but is not varied in the experiments.

The EMA second moment lets the effective per-coordinate learning rate
contract whenever channel noise inflates |Δ_t|, providing implicit
noise filtering without explicit channel-state information.
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
        alpha: float = 2.0,
        beta1: float = 0.9,
        beta2: float = 0.3,
        eps: float = 1e-4,
    ):
        """
        Args:
            params: List of global model parameter tensors.
            lr:     Step size η.
            alpha:  Exponent of the second-moment accumulator (default 2.0
                    → classical Adam). Should match NoisyOracle.alpha if a
                    non-Gaussian channel is used.
            beta1:  1st-moment decay β₁ ∈ [0, 1).
            beta2:  2nd-moment decay β₂ ∈ (0, 1).
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

            # 1st moment — momentum smoothing of the noisy aggregated gradient
            delta.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)

            # 2nd moment — EMA of |Δ|^α  (α = 2 → classical Adam)
            v.mul_(self.beta2).add_(delta.abs().pow(self.alpha), alpha=1.0 - self.beta2)

            # Parameter update — α-th-root normalisation
            denom = v.pow(1.0 / self.alpha).add_(self.eps)
            p.addcdiv_(delta, denom, value=-self.lr)

    def zero(self):
        for d, v in zip(self.Delta, self.v):
            d.zero_()
            v.zero_()
