"""
Median Anchored Clipping (MAC) — optional robust pre-processor.

Off by default, and enabled explicitly in the MAC comparison. It provides a
robust-statistics safety net against rare heavy-tailed outliers.

Algorithm (server-side, applied to g_t before the optimizer step):
  1. m_i ← median of g_t coordinates.
  2. d_i ← |g_{t,i} − m_i|.
  3. g_clipped_i ← m_i + clip(g_{t,i} − m_i, −τ, +τ), τ = c · median(d_i).
"""

import torch


def _empty_stats() -> dict[str, float | int]:
    return {
        "mac_total_coords": 0,
        "mac_clipped_coords": 0,
        "mac_clipped_fraction": 0.0,
    }


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
    if grad.numel() <= 1:
        return grad

    m = grad.median()
    deviations = (grad - m).abs()
    tau = clip_factor * deviations.median()
    return m + (grad - m).clamp(-tau, tau)


def median_anchored_clipping_with_stats(
    grad: torch.Tensor,
    clip_factor: float = 3.0,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Apply MAC and report how many coordinates were actually clipped."""
    if grad.numel() == 0:
        return grad, _empty_stats()
    if grad.numel() == 1:
        stats = _empty_stats()
        stats["mac_total_coords"] = int(grad.numel())
        return grad, stats

    m = grad.median()
    centered = grad - m
    deviations = centered.abs()
    tau = clip_factor * deviations.median()
    clipped = centered.clamp(-tau, tau)
    clipped_mask = clipped.ne(centered)
    clipped_count = int(clipped_mask.sum().item())
    total = int(grad.numel())
    return m + clipped, {
        "mac_total_coords": total,
        "mac_clipped_coords": clipped_count,
        "mac_clipped_fraction": clipped_count / total if total else 0.0,
    }


def _unflatten_like(flat: torch.Tensor, templates: list[torch.Tensor]) -> list[torch.Tensor]:
    outputs = []
    offset = 0
    for template in templates:
        numel = template.numel()
        outputs.append(flat[offset:offset + numel].reshape(template.shape))
        offset += numel
    return outputs


def apply_mac(
    agg_grads: list[torch.Tensor],
    clip_factor: float = 3.0,
    return_stats: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], dict[str, float | int]]:
    """
    Apply MAC to the full aggregated gradient vector.

    Args:
        agg_grads:   Per-parameter tensors making up the aggregated gradient.
        clip_factor: Clipping multiplier c.
        return_stats: If true, also return aggregate clipping diagnostics.

    Returns:
        List of MAC-clipped gradient tensors, or (list, stats) when
        return_stats=True.
    """
    if not agg_grads:
        if return_stats:
            return [], _empty_stats()
        return []

    flat = torch.cat([grad.reshape(-1) for grad in agg_grads])
    if not return_stats:
        clipped = median_anchored_clipping(flat, clip_factor)
        return _unflatten_like(clipped, agg_grads)

    clipped, stats = median_anchored_clipping_with_stats(flat, clip_factor)
    return _unflatten_like(clipped, agg_grads), stats
