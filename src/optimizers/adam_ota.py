"""
Adam-OTA: stable server-side Adam for OTA-aggregated gradients.

This project is an empirical validation framework, not a strict line-by-line
paper reproduction. For the heavy-tail experiments, we use a numerically
stable Adam-style server update:

    m_t = β₁ · m_{t-1}  +  (1 − β₁) · g_t
    v_t = β₂ · v_{t-1}  +  (1 − β₂) · |g_t|^α
    m_hat = m_t / (1 − β₁^t)
    v_hat = v_t / (1 − β₂^t)
    w_{t+1} = w_t  −  η · m_hat / (v_hat^{1/α} + ε)

When α = 2, this recovers classical Adam applied at the server. When α < 2,
the denominator follows the fractional alpha-stable accumulator used by the
heavy-tail experiments.
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
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, agg_grads: list[torch.Tensor]):
        """
        One server-side update step (one FL round).

        Args:
            agg_grads: OTA-aggregated gradients [g_t], one tensor per parameter,
                       already normalized by the federated loop.
        """
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for p, g, m, v in zip(self.params, agg_grads, self.m, self.v):
            g = g.to(p.device)

            # 1st moment — momentum smoothing of the noisy aggregated gradient
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)

            # 2nd moment — EMA of |g|^α  (α = 2 → classical Adam)
            v.mul_(self.beta2).add_(g.abs().pow(self.alpha), alpha=1.0 - self.beta2)

            # Bias correction keeps the first rounds numerically stable.
            m_hat = m / bc1
            denom = (v / bc2).pow(1.0 / self.alpha).add_(self.eps)

            p.addcdiv_(m_hat, denom, value=-self.lr)

    def zero(self):
        self.t = 0
        for m, v in zip(self.m, self.v):
            m.zero_()
            v.zero_()
