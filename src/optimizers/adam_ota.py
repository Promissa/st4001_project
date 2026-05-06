"""
Adam-OTA: server-side Adam update applied to the OTA-aggregated gradient.

The server receives a noisy aggregated gradient g_t each round and applies
a momentum-smoothed, second-moment-normalised step with bias correction:

    Δ_t = β₁ · Δ_{t-1}  +  (1 − β₁) · g_t                [1st moment]
    v_t = β₂ · v_{t-1}  +  (1 − β₂) · |Δ_t|^α            [EMA of |·|^α]
    Δ̂_t = Δ_t / (1 − β₁^t)                              [bias-corrected]
    v̂_t = v_t / (1 − β₂^t)                              [bias-corrected]
    w_{t+1} = w_t  −  η · Δ̂_t / (v̂_t^{1/α} + ε)

α = 2 (default) with β₂ = 0.999, ε = 1e-8 recovers classical Adam exactly
(Kingma & Ba, 2015) — this is the AWGN setting used throughout this project.
The α parameter is kept for generality but is not varied in the experiments.

The bias-corrected EMA second moment lets the effective per-coordinate
learning rate contract whenever channel noise inflates |Δ_t|, providing
implicit noise filtering without explicit channel-state information.
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
        beta2: float = 0.999,
        eps: float = 1e-8,
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

        self.t = 0
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
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        step_size = self.lr / bc1

        for p, g, delta, v in zip(self.params, agg_grads, self.Delta, self.v):
            g = g.to(p.device)

            # 1st moment — momentum smoothing of the noisy aggregated gradient
            delta.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)

            # 2nd moment — EMA of |Δ|^α  (α = 2 → classical Adam)
            v.mul_(self.beta2).add_(delta.abs().pow(self.alpha), alpha=1.0 - self.beta2)

            # Bias-corrected α-th-root denominator
            denom = (v / bc2).pow_(1.0 / self.alpha).add_(self.eps)

            # Parameter update — bias correction folded into step_size = lr / bc1
            p.addcdiv_(delta, denom, value=-step_size)

    def zero(self):
        self.t = 0
        for d, v in zip(self.Delta, self.v):
            d.zero_()
            v.zero_()
