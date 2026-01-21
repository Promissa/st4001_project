"""
Adaptive optimizers for Over-the-Air Federated Learning.

This module implements AdaGrad-OTA and Adam-OTA, which use fractional moments
(α-norms) for second-moment estimation to handle heavy-tailed noise.
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Dict, Any
import math


class AdaGradOTA(Optimizer):
    """
    AdaGrad optimizer adapted for Over-the-Air Federated Learning.
    
    Uses α-norms (fractional moments) instead of Euclidean norms for
    second-moment estimation to prevent gradient explosion under heavy-tailed noise.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-2,
        alpha: float = 1.5,
        eps: float = 1e-8,
        initial_accumulator_value: float = 0.0,
    ):
        """
        Initialize AdaGrad-OTA optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Initial learning rate
            alpha: Tail index for α-norm computation (should match noise α)
            eps: Small constant for numerical stability
            initial_accumulator_value: Initial value for gradient accumulator
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha <= 1 or alpha > 2:
            raise ValueError(f"alpha must be in (1, 2], got {alpha}")
        
        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            initial_accumulator_value=initial_accumulator_value,
        )
        super(AdaGradOTA, self).__init__(params, defaults)
        
        # Initialize state
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["sum_grad_alpha"] = torch.full_like(
                    p.data,
                    initial_accumulator_value,
                )
    
    def _compute_alpha_norm(self, tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Compute α-norm (fractional moment) of a tensor.
        
        For α = 2, this reduces to squared L2 norm.
        For α < 2, this is a fractional moment that is more robust to outliers.
        """
        if alpha == 2.0:
            return tensor ** 2
        else:
            # Use absolute value raised to alpha power
            return torch.abs(tensor) ** alpha
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model and returns loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                state["step"] += 1
                
                # Accumulate α-norm of gradients
                grad_alpha_norm = self._compute_alpha_norm(grad, alpha)
                state["sum_grad_alpha"].add_(grad_alpha_norm)
                
                # Compute adaptive learning rate
                # Use (sum_grad_alpha)^(1/alpha) for scaling
                if alpha == 2.0:
                    denominator = torch.sqrt(state["sum_grad_alpha"]) + eps
                else:
                    denominator = (state["sum_grad_alpha"] + eps) ** (1.0 / alpha)
                
                # Update parameters
                p.data.addcdiv_(grad, denominator, value=-lr)
        
        return loss


class AdamOTA(Optimizer):
    """
    Adam optimizer adapted for Over-the-Air Federated Learning.
    
    Extends AdaGrad-OTA with momentum and uses α-norms for second-moment
    estimation. Expected to converge faster than AdaGrad-OTA due to momentum.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        alpha: float = 1.5,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
    ):
        """
        Initialize Adam-OTA optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages (momentum, second moment)
            alpha: Tail index for α-norm computation
            eps: Small constant for numerical stability
            weight_decay: Weight decay (L2 penalty)
            amsgrad: Whether to use AMSGrad variant
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if alpha <= 1 or alpha > 2:
            raise ValueError(f"alpha must be in (1, 2], got {alpha}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super(AdamOTA, self).__init__(params, defaults)
        
        # Initialize state
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_alpha"] = torch.zeros_like(p.data)
                if amsgrad:
                    state["max_exp_avg_alpha"] = torch.zeros_like(p.data)
    
    def _compute_alpha_norm(self, tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        """Compute α-norm (fractional moment) of a tensor."""
        if alpha == 2.0:
            return tensor ** 2
        else:
            return torch.abs(tensor) ** alpha
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: Optional closure that reevaluates the model and returns loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_alphas = []
            max_exp_avg_alphas = []
            state_steps = []
            beta1, beta2 = group["betas"]
            alpha = group["alpha"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                
                state = self.state[p]
                exp_avgs.append(state["exp_avg"])
                exp_avg_alphas.append(state["exp_avg_alpha"])
                
                if amsgrad:
                    max_exp_avg_alphas.append(state["max_exp_avg_alpha"])
                
                state["step"] += 1
                state_steps.append(state["step"])
            
            # Update parameters
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_alpha = exp_avg_alphas[i]
                step = state_steps[i]
                
                if weight_decay != 0:
                    grad = grad.add(param, alpha=weight_decay)
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update biased second moment estimate using α-norm
                grad_alpha_norm = self._compute_alpha_norm(grad, alpha)
                exp_avg_alpha.mul_(beta2).add_(grad_alpha_norm, alpha=1 - beta2)
                
                if amsgrad:
                    # Maintain maximum of exp_avg_alpha
                    max_exp_avg_alpha = max_exp_avg_alphas[i]
                    torch.maximum(max_exp_avg_alpha, exp_avg_alpha, out=max_exp_avg_alpha)
                    denom = (max_exp_avg_alpha + eps) ** (1.0 / alpha)
                else:
                    if alpha == 2.0:
                        denom = torch.sqrt(exp_avg_alpha) + eps
                    else:
                        denom = (exp_avg_alpha + eps) ** (1.0 / alpha)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr / bias_correction1
                
                # Update parameters
                param.addcdiv_(exp_avg, denom, value=-step_size * math.sqrt(bias_correction2))
        
        return loss
