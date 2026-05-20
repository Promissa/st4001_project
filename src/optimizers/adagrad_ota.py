"""
AdaGrad-OTA: server-side AdaGrad update applied to the OTA-aggregated gradient.

The server receives a noisy aggregated gradient g_t each round and applies a
per-coordinate AdaGrad-style update:

    Δ_t  = β₁ · Δ_{t-1}  +  (1 − β₁) · g_t          [momentum smoothing]
    v_t  = v_{t-1}  +  |Δ_t|^α                        [accumulated |·|^α]
    w_{t+1} = w_t  −  η · Δ_t / (v_t^{1/α} + ε)

α = 2 (default) recovers classical AdaGrad at the AWGN endpoint. α < 2 uses
the fractional accumulator for alpha-stable heavy-tailed interference. β₁ = 0
turns off momentum smoothing.

The accumulator v_t grows whenever channel noise inflates |Δ_t|, so the
effective per-coordinate learning rate contracts — providing implicit
noise filtering without explicit channel-state information.
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
        alpha: float = 2.0,
        beta1: float = 0.0,
        eps: float = 1e-4,
    ):
        """
        Args:
            params: List of global model parameter tensors (model.parameters()).
            lr:     Step size η.
            alpha:  Exponent of the second-moment accumulator (default 2.0
                    → classical AdaGrad).
            beta1:  Momentum coefficient for Δ_t. beta1=0 → pure AdaGrad.
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
                       The federated loop has already applied the required
                       experiment-specific normalization.
        """
        for p, g, delta, v in zip(self.params, agg_grads, self.Delta, self.v):
            g = g.to(p.device)

            # Momentum smoothing of the noisy aggregated gradient
            delta.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)

            # Accumulate |Δ|^α entry-wise (α = 2 → classical AdaGrad)
            v.add_(delta.abs().pow(self.alpha))

            # Parameter update — α-th-root normalisation
            denom = v.pow(1.0 / self.alpha).add_(self.eps)
            p.addcdiv_(delta, denom, value=-self.lr)

    def zero(self):
        """Reset momentum and accumulator (call between experiments)."""
        for d, v in zip(self.Delta, self.v):
            d.zero_()
            v.zero_()
