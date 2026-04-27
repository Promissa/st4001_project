"""
Median Anchored Clipping (MAC) — optional robust pre-processor.

Off by default in the AWGN experiments; included for exploratory runs
where one wants a robust-statistics safety net against rare outliers.

Algorithm (server-side, applied to g_t before the optimizer step):
  1. m_i ← median of g_t coordinates.
  2. d_i ← |g_{t,i} − m_i|.
  3. g_clipped_i ← m_i + clip(g_{t,i} − m_i, −τ, +τ), τ = c · median(d_i).
"""

import torch


def median_anchored_clipping(
    grad: torch.Tensor,
    clip_factor: float = 3.0,
) -> torch.Tensor:
    """
    Apply MAC to a flat gradient tensor.

    Args:
        grad:        Aggregated gradient vector g_t (1-D tensor).
        clip_factor: Multiplier c for the clipping threshold τ = c · median(|g - median(g)|).

    Returns:
        Clipped gradient tensor of the same shape.
    """
    if grad.numel() == 1:
        return grad

    m = grad.median()
    deviations = (grad - m).abs()
    tau = clip_factor * deviations.median()
    return m + (grad - m).clamp(-tau, tau)


def apply_mac(
    agg_grads: list[torch.Tensor],
    clip_factor: float = 3.0,
) -> list[torch.Tensor]:
    """
    Apply MAC to a list of per-parameter aggregated gradients.

    Args:
        agg_grads:   List of aggregated gradient tensors (one per model parameter).
        clip_factor: Clipping multiplier c.

    Returns:
        List of MAC-clipped gradient tensors.
    """
    return [median_anchored_clipping(g.flatten(), clip_factor).reshape(g.shape)
            for g in agg_grads]
