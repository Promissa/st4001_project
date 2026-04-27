"""
Baseline server-side optimizers for comparative analysis.

  - FedAvgOTA  : Plain gradient descent on the OTA-aggregated gradient.
                 Equivalent to FedAvg with OTA aggregation — no momentum,
                 no adaptive step. Most fragile under heavy-tailed noise.

  - FedAvgMOTA : FedAvg with Momentum (FedAvgM) applied server-side.
                 Standard baseline in OTA-FL literature.
                 Static learning rate — does not adapt to channel noise.

Both share the same interface as AdaGradOTA / AdamOTA: a .step(agg_grads) method.
"""

import torch


class FedAvgOTA:
    """
    Server-side SGD (no momentum). Baseline: FedAvg + OTA.

    Update: w_{t+1} = w_t - lr * g_t
    """

    def __init__(self, params: list[torch.Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr

    @torch.no_grad()
    def step(self, agg_grads: list[torch.Tensor]):
        for p, g in zip(self.params, agg_grads):
            p.sub_(g.to(p.device), alpha=self.lr)

    def zero(self):
        pass  # stateless


class FedAvgMOTA:
    """
    Server-side SGD with momentum. Baseline: FedAvgM + OTA.

    Update:
        buf_{t+1} = momentum * buf_t + g_t
        w_{t+1}  = w_t - lr * buf_{t+1}
    """

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.buf = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self, agg_grads: list[torch.Tensor]):
        for p, g, buf in zip(self.params, agg_grads, self.buf):
            g = g.to(p.device)
            buf.mul_(self.momentum).add_(g)
            p.sub_(buf, alpha=self.lr)

    def zero(self):
        for b in self.buf:
            b.zero_()
