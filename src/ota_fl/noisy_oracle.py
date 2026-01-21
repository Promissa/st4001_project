"""
Noisy Oracle for simulating Over-the-Air aggregation.

This module implements the system model that simulates analog aggregation
over wireless channels with fading and heavy-tailed interference.
"""

import torch
from typing import Optional, Dict, Any
from .noise import AlphaStableNoise


class NoisyOracle:
    """
    Simulates Over-the-Air Federated Learning aggregation with channel impairments.
    
    The Noisy Oracle models the physical layer effects of OTA-FL, including:
    - Channel fading (multiplicative noise)
    - Heavy-tailed interference (additive α-stable noise)
    - Analog superposition of model updates
    """
    
    def __init__(
        self,
        alpha: float = 1.5,
        noise_scale: float = 0.1,
        fading_variance: float = 0.1,
        use_clipping: bool = True,
        clip_threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Noisy Oracle.
        
        Args:
            alpha: Tail index for α-stable noise (1 < alpha <= 2)
            noise_scale: Scale parameter for interference noise
            fading_variance: Variance of channel fading (multiplicative noise)
            use_clipping: Whether to apply MAC preprocessing
            clip_threshold: Threshold for MAC clipping (None for auto)
            device: Device to run computations on
        """
        self.alpha = alpha
        self.noise_scale = noise_scale
        self.fading_variance = fading_variance
        self.use_clipping = use_clipping
        self.clip_threshold = clip_threshold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize noise generator
        self.noise_generator = AlphaStableNoise(
            alpha=alpha,
            scale=noise_scale,
            device=self.device,
        )
    
    def aggregate(
        self,
        local_updates: torch.Tensor,
        apply_fading: bool = True,
        apply_interference: bool = True,
    ) -> torch.Tensor:
        """
        Simulate OTA aggregation of local model updates.
        
        The aggregation process:
        1. Optionally applies channel fading (multiplicative noise)
        2. Sums updates (analog superposition)
        3. Adds heavy-tailed interference (α-stable noise)
        
        Args:
            local_updates: Tensor of local updates [num_clients, ...]
            apply_fading: Whether to apply channel fading
            apply_interference: Whether to add interference noise
            
        Returns:
            Aggregated (and corrupted) global update
        """
        if local_updates.numel() == 0:
            return local_updates
        
        # Apply MAC clipping if enabled
        if self.use_clipping:
            from .clipping import median_anchored_clipping
            local_updates = median_anchored_clipping(
                local_updates,
                clip_threshold=self.clip_threshold,
            )
        
        # Apply channel fading (multiplicative noise)
        if apply_fading:
            # Simulate channel gains (Rayleigh fading)
            num_clients = local_updates.shape[0]
            channel_gains = torch.normal(
                mean=1.0,
                std=torch.sqrt(torch.tensor(self.fading_variance, device=self.device)),
                size=(num_clients,),
                device=self.device,
            )
            # Reshape to broadcast over update dimensions
            while channel_gains.dim() < local_updates.dim():
                channel_gains = channel_gains.unsqueeze(-1)
            local_updates = local_updates * channel_gains
        
        # Analog superposition (sum over clients)
        aggregated = torch.sum(local_updates, dim=0)
        
        # Add heavy-tailed interference
        if apply_interference:
            noise_shape = aggregated.shape
            interference = self.noise_generator(noise_shape)
            aggregated = aggregated + interference
        
        return aggregated
    
    def update_noise_parameters(
        self,
        alpha: Optional[float] = None,
        noise_scale: Optional[float] = None,
    ):
        """Update noise generation parameters."""
        if alpha is not None:
            self.alpha = alpha
            self.noise_generator.alpha = alpha
        if noise_scale is not None:
            self.noise_scale = noise_scale
            self.noise_generator.scale = noise_scale
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current noise statistics."""
        return {
            "alpha": self.alpha,
            "noise_scale": self.noise_scale,
            "fading_variance": self.fading_variance,
            "use_clipping": self.use_clipping,
        }
