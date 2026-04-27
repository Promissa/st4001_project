"""
Median Anchored Clipping (MAC) — robust pre-processing before the optimizer.

Reference: Li et al., "Robust federated learning over the air: Combating
heavy-tailed noise with median anchored clipping," WiOpt 2025. [Ref 2 in proposal]

MAC truncates extreme outliers in the OTA-aggregated gradient before they
reach the adaptive optimizer. This is a pre-processing step applied server-side
to g_t after OTA aggregation, before the AdaGrad/Adam update.

Algorithm:
  1. Compute coordinate-wise median m_i of the aggregated gradient g_t.
  2. Compute coordinate-wise deviation d_i = |g_{t,i} - m_i|.
  3. Clip: g_clipped_i = m_i + clip(g_{t,i} - m_i, -tau, +tau)
     where tau = c * median(d_i)  for a constant c (default 3.0).

This keeps the gradient anchored to its robust median estimate and removes
impulsive outliers beyond c standard-deviation-equivalents.
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
