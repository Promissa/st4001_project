"""
Core ADOTA-FL training loop.

Round structure (one communication round t):
  1. Server broadcasts w_t to all N clients.
  2. Each client n computes local gradient ∇f_n(w_t) via local SGD.
  3. All clients transmit simultaneously; server receives OTA aggregate
     g_t = (1/N) * [Σ_n h_{n,t} * ∇f_n(w_t)] + ξ_t  via NoisyOracle.
  4. (Optional) MAC clips g_t to remove impulsive outliers.
  5. Server-side optimizer (AdaGrad-OTA / Adam-OTA / FedAvgM / FedAvg)
     updates w_{t+1} using g_t.
"""

import copy
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..channel.ota import NoisyOracle
from ..channel.mac import apply_mac


def _list_norm(tensors: list[torch.Tensor]) -> float:
    total = 0.0
    for tensor in tensors:
        value = float(tensor.detach().float().pow(2).sum().item())
        total += value
    return math.sqrt(total)


def _list_max_abs(tensors: list[torch.Tensor]) -> float:
    max_abs = 0.0
    for tensor in tensors:
        if tensor.numel():
            max_abs = max(max_abs, float(tensor.detach().float().abs().max().item()))
    return max_abs


def _list_finite(tensors: list[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors)


def _optimizer_state_tensors(server_opt) -> list[torch.Tensor]:
    tensors = []
    for name in ("m", "v", "Delta", "buf"):
        value = getattr(server_opt, name, None)
        if isinstance(value, list):
            tensors.extend([tensor for tensor in value if isinstance(tensor, torch.Tensor)])
    return tensors


def local_grad(
    global_model: nn.Module,
    loader: DataLoader,
    local_epochs: int,
    local_lr: float,
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    """
    Client local update: run E epochs of SGD on the local data,
    return the pseudo-gradient (parameter difference).

    The pseudo-gradient Δ_n = w_global - w_local approximates
    +lr * ∇f_n(w_t) accumulated over E steps, and is what each
    client "uploads" in the OTA transmission.

    Args:
        global_model: Current global model w_t (read-only; not modified).
        loader:       Client's local DataLoader.
        local_epochs: Number of local SGD epochs E.
        local_lr:     Local learning rate for client SGD.
        device:       Compute device.

    Returns:
        Tuple of:
          - pseudo-gradient tensors, one per parameter
          - floating-point buffers from the locally trained model
    """
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    opt = torch.optim.SGD(local_model.parameters(), lr=local_lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for x, y in loader:
            # non_blocking=True for async CPU->GPU transfer when pin_memory=True
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            criterion(local_model(x), y).backward()
            opt.step()

    param_deltas = [
        (gp.data - lp.data).detach()
        for lp, gp in zip(local_model.parameters(), global_model.parameters())
    ]
    float_buffers = {
        name: buf.detach().clone()
        for name, buf in local_model.named_buffers()
        if torch.is_floating_point(buf)
    }
    return param_deltas, float_buffers


def run_round(
    global_model: nn.Module,
    client_loaders: list[DataLoader],
    oracle: NoisyOracle,
    server_opt,
    local_epochs: int = 1,
    local_lr: float = 0.01,
    use_mac: bool = False,
    mac_clip: float = 3.0,
    device: torch.device = torch.device("cpu"),
    return_diagnostics: bool = False,
) -> dict | None:
    """
    Execute one FL communication round, updating global_model in-place.

    Args:
        global_model:   Global model w_t; updated in-place to w_{t+1}.
        client_loaders: Per-client DataLoaders.
        oracle:         NoisyOracle implementing OTA aggregation.
        server_opt:     Server-side optimizer (AdaGradOTA/AdamOTA/FedAvgMOTA/FedAvgOTA).
        local_epochs:   Local SGD epochs per client.
        local_lr:       Local SGD learning rate.
        use_mac:        Whether to apply Median Anchored Clipping before optimizer.
        mac_clip:       MAC clipping constant c (τ = c * median deviation).
        device:         Compute device.
        return_diagnostics: Return per-round finite checks and norms.
    """
    N = len(client_loaders)
    if hasattr(oracle, "begin_round"):
        oracle.begin_round()

    # Step 2: client local updates (in practice parallel; simulated sequentially)
    client_updates = [
        local_grad(global_model, loader, local_epochs, local_lr, device)
        for loader in client_loaders
    ]
    all_grads = [grads for grads, _ in client_updates]
    all_buffers = [buffers for _, buffers in client_updates]

    # Step 3: OTA aggregation → g_t (normalized by N)
    agg_grads = []
    for param_grads in zip(*all_grads):
        noisy_sum = oracle.aggregate(list(param_grads))
        agg_grads.append(noisy_sum / N)

    # Step 4: optional MAC pre-processing
    if use_mac:
        agg_grads = apply_mac(agg_grads, clip_factor=mac_clip)

    agg_finite = _list_finite(agg_grads)
    diagnostics = {
        "status": "ok" if agg_finite else "nonfinite_aggregate",
        "agg_finite": agg_finite,
        "agg_norm": _list_norm(agg_grads) if agg_finite else float("inf"),
        "agg_max_abs": _list_max_abs(agg_grads) if agg_finite else float("inf"),
        "use_mac": use_mac,
    }
    if hasattr(oracle, "diagnostics"):
        diagnostics.update(oracle.diagnostics())

    if not agg_finite:
        if return_diagnostics:
            return diagnostics
        raise FloatingPointError("Non-finite OTA aggregate encountered.")

    old_params = None
    if return_diagnostics:
        old_params = [p.detach().clone() for p in global_model.parameters()]

    # Step 5: server-side adaptive update
    server_opt.step(agg_grads)

    model_tensors = [p.detach() for p in global_model.parameters()]
    state_tensors = _optimizer_state_tensors(server_opt)
    model_finite = _list_finite(model_tensors)
    opt_finite = _list_finite(state_tensors) if state_tensors else True
    diagnostics["model_finite"] = model_finite
    diagnostics["optimizer_finite"] = opt_finite
    if not model_finite:
        diagnostics["status"] = "nonfinite_model"
    elif not opt_finite:
        diagnostics["status"] = "nonfinite_optimizer"

    if old_params is not None:
        updates = [
            old - new.detach()
            for old, new in zip(old_params, global_model.parameters())
        ]
        diagnostics["update_norm"] = _list_norm(updates)
        diagnostics["update_to_agg_norm"] = (
            diagnostics["update_norm"] / diagnostics["agg_norm"]
            if diagnostics["agg_norm"] > 0 and math.isfinite(diagnostics["agg_norm"])
            else 0.0
        )
        adaptive_state = getattr(server_opt, "v", None)
        if isinstance(adaptive_state, list) and _list_finite(adaptive_state):
            diagnostics["v_norm"] = _list_norm(adaptive_state)
            diagnostics["v_max_abs"] = _list_max_abs(adaptive_state)

    if not model_finite or not opt_finite:
        if return_diagnostics:
            return diagnostics
        raise FloatingPointError("Non-finite model or optimizer state encountered.")

    # Keep floating buffers (e.g. BatchNorm running stats) in sync with client models.
    # Without this, ResNet evaluation uses stale normalization statistics from round 0.
    if all_buffers:
        with torch.no_grad():
            for name, global_buf in global_model.named_buffers():
                if not torch.is_floating_point(global_buf):
                    continue
                averaged = torch.stack(
                    [client_buffers[name].to(global_buf.device) for client_buffers in all_buffers]
                ).mean(dim=0)
                global_buf.copy_(averaged)

    return diagnostics if return_diagnostics else None


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> tuple[float, float]:
    """
    Evaluate model on a DataLoader.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            total_loss += criterion(out, y).item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total
