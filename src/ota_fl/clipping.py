"""
Median Anchored Clipping (MAC) for robust gradient preprocessing.

This module implements MAC as a preprocessing step to truncate extreme outliers
before they reach the optimizer, providing robustness against heavy-tailed noise.
"""

import torch
from typing import Optional


def median_anchored_clipping(
    gradients: torch.Tensor,
    clip_threshold: Optional[float] = None,
    dim: int = 0,
) -> torch.Tensor:
    """
    Apply Median Anchored Clipping to gradients.
    
    MAC clips gradients based on their deviation from the median, which is more
    robust to outliers than mean-based clipping.
    
    Args:
        gradients: Tensor of gradients to clip (shape: [num_clients, ...])
        clip_threshold: Threshold for clipping. If None, uses median absolute deviation
        dim: Dimension along which to compute median (typically 0 for client dimension)
        
    Returns:
        Clipped gradients tensor
        
    References:
        Li, J., et al. (2025). Robust federated learning over the air: Combating 
        heavy-tailed noise with median anchored clipping.
    """
    if gradients.numel() == 0:
        return gradients
    
    # Compute median along the specified dimension
    median = torch.median(gradients, dim=dim, keepdim=True)[0]
    
    # Compute median absolute deviation (MAD)
    deviations = torch.abs(gradients - median)
    mad = torch.median(deviations, dim=dim, keepdim=True)[0]
    
    # Set default threshold if not provided
    if clip_threshold is None:
        # Use a multiple of MAD (e.g., 3 * MAD for robust clipping)
        clip_threshold = 3.0 * mad
        # Avoid zero threshold
        clip_threshold = torch.clamp(clip_threshold, min=1e-8)
    
    # Clip gradients
    clipped = torch.clamp(
        gradients,
        min=median - clip_threshold,
        max=median + clip_threshold,
    )
    
    return clipped
