"""
α-stable noise generation using Chambers-Mallows-Stuck method.

This module implements the generation of symmetric α-stable random variables
for simulating heavy-tailed electromagnetic interference in industrial IoT environments.
"""

import torch
import numpy as np
from typing import Optional, Tuple


def generate_alpha_stable_noise(
    alpha: float,
    size: Tuple[int, ...],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate symmetric α-stable random variables using Chambers-Mallows-Stuck method.
    
    The method generates stable random variables with characteristic exponent α
    where 1 < α ≤ 2. For α = 2, this reduces to Gaussian distribution.
    
    Args:
        alpha: Tail index (characteristic exponent), must satisfy 1 < alpha <= 2
        size: Shape of the output tensor
        device: Device to place the tensor on
        dtype: Data type of the tensor
        
    Returns:
        Tensor of α-stable random variables
        
    References:
        Chambers, J., Mallows, C., & Stuck, B. (1976). A Method for Simulating 
        Stable Random Variables. Journal of the American Statistical Association.
    """
    if alpha <= 1 or alpha > 2:
        raise ValueError(f"alpha must be in (1, 2], got {alpha}")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate uniform random variables
    # U ~ Uniform(-pi/2, pi/2)
    U = torch.rand(size, device=device, dtype=dtype) * np.pi - np.pi / 2
    # W ~ Exponential(1)
    W = -torch.log(torch.rand(size, device=device, dtype=dtype) + 1e-10)
    
    # Chambers-Mallows-Stuck transformation
    if alpha == 2.0:
        # Special case: Gaussian (variance = 2)
        return torch.sqrt(2 * W) * torch.sin(U)
    else:
        # General α-stable case (symmetric, beta=0)
        # For symmetric stable, we use the simplified formula
        phi = U
        B_alpha = torch.tan(np.pi * alpha / 2)
        
        # Compute the stable random variable
        # X = (sin(alpha * phi) / (cos(phi)^(1/alpha))) * 
        #     (cos((alpha-1) * phi) / W)^((1-alpha)/alpha)
        cos_phi = torch.cos(phi)
        # Avoid division by zero and negative values
        cos_phi = torch.clamp(torch.abs(cos_phi), min=1e-10)
        
        term1 = torch.sin(alpha * phi) / (cos_phi ** (1.0 / alpha))
        term2 = (torch.cos((alpha - 1) * phi) / (W + 1e-10)) ** ((1 - alpha) / alpha)
        
        X = term1 * term2
        
        return X


class AlphaStableNoise:
    """
    Generator for α-stable noise with configurable parameters.
    
    This class provides a convenient interface for generating α-stable noise
    with different tail indices and scales, suitable for simulating various
    levels of heavy-tailed interference.
    """
    
    def __init__(
        self,
        alpha: float = 1.5,
        scale: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize α-stable noise generator.
        
        Args:
            alpha: Tail index (1 < alpha <= 2). Lower values indicate heavier tails.
            scale: Scale parameter (dispersion) of the distribution
            device: Device to generate noise on
        """
        if alpha <= 1 or alpha > 2:
            raise ValueError(f"alpha must be in (1, 2], got {alpha}")
        
        self.alpha = alpha
        self.scale = scale
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def __call__(self, size: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate α-stable noise.
        
        Args:
            size: Shape of the output tensor
            
        Returns:
            Tensor of scaled α-stable random variables
        """
        noise = generate_alpha_stable_noise(
            alpha=self.alpha,
            size=size,
            device=self.device,
        )
        return self.scale * noise
    
    def sample(self, size: Tuple[int, ...]) -> torch.Tensor:
        """Alias for __call__."""
        return self(size)
