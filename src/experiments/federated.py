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
import contextlib
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..channel.ota import NoisyOracle
from ..channel.mac import apply_mac

# Cache local model buffers per global model instance to avoid per-round deepcopy
_local_model_cache: dict[int, nn.Module] = {}


def _get_cached_local_model(global_model: nn.Module, device: torch.device) -> nn.Module:
    model_id = id(global_model)
    if model_id not in _local_model_cache:
        _local_model_cache[model_id] = copy.deepcopy(global_model).to(device)
    local_model = _local_model_cache[model_id]
    with torch.no_grad():
        for p_local, p_global in zip(local_model.parameters(), global_model.parameters()):
            p_local.copy_(p_global)
        for b_local, b_global in zip(local_model.buffers(), global_model.buffers()):
            b_local.copy_(b_global)
    return local_model


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


def _amp_torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def _autocast(device: torch.device, use_amp: bool, amp_dtype: str):
    if use_amp and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=_amp_torch_dtype(amp_dtype))
    return contextlib.nullcontext()


def _make_grad_scaler(device: torch.device, use_amp: bool, amp_dtype: str):
    enabled = bool(use_amp and amp_dtype == "fp16" and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _prepare_loader_batch(
    loader,
    batch,
    device: torch.device,
    channels_last: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(loader, "prepare_batch"):
        return loader.prepare_batch(batch, device=device, channels_last=channels_last)

    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if channels_last and x.ndim == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x, y


def _train_local_model(
    local_model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    local_epochs: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: str,
    channels_last: bool,
    scaler,
) -> None:
    local_model.train()
    for _ in range(local_epochs):
        for batch in loader:
            x, y = _prepare_loader_batch(loader, batch, device, channels_last)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, use_amp, amp_dtype):
                loss = criterion(local_model(x), y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()


def local_grad(
    global_model: nn.Module,
    loader: DataLoader,
    local_epochs: int,
    local_lr: float,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: str = "bf16",
    channels_last: bool = False,
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
    if channels_last:
        local_model = local_model.to(memory_format=torch.channels_last)
    local_model.train()
    opt = torch.optim.SGD(local_model.parameters(), lr=local_lr)
    criterion = nn.CrossEntropyLoss()
    scaler = _make_grad_scaler(device, use_amp, amp_dtype)

    _train_local_model(
        local_model,
        loader,
        opt,
        criterion,
        local_epochs,
        device,
        use_amp,
        amp_dtype,
        channels_last,
        scaler,
    )

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
    use_amp: bool = False,
    amp_dtype: str = "bf16",
    channels_last: bool = False,
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
        oracle.begin_round(collect_diagnostics=return_diagnostics)

    # Step 2: client local updates. Reuse a cached local model buffer to avoid
    # per-round deepcopy overhead on the CPU.
    local_model = _get_cached_local_model(global_model, device)
    if channels_last:
        local_model = local_model.to(memory_format=torch.channels_last)
    local_opt = torch.optim.SGD(local_model.parameters(), lr=local_lr)
    criterion = nn.CrossEntropyLoss()
    scaler = _make_grad_scaler(device, use_amp, amp_dtype)

    delta_sums = [torch.zeros_like(p, device=device) for p in global_model.parameters()]
    buffer_sums = {
        name: torch.zeros_like(buf, device=device)
        for name, buf in global_model.named_buffers()
        if torch.is_floating_point(buf)
    }
    global_params = list(global_model.parameters())
    global_buffers = list(global_model.buffers())

    for loader in client_loaders:
        with torch.no_grad():
            for p_local, p_global in zip(local_model.parameters(), global_params):
                p_local.copy_(p_global)
            for b_local, b_global in zip(local_model.buffers(), global_buffers):
                b_local.copy_(b_global)
        _train_local_model(
            local_model,
            loader,
            local_opt,
            criterion,
            local_epochs,
            device,
            use_amp,
            amp_dtype,
            channels_last,
            scaler,
        )

        with torch.no_grad():
            for delta_sum, global_param, local_param in zip(
                delta_sums,
                global_model.parameters(),
                local_model.parameters(),
            ):
                delta_sum.add_(global_param.detach() - local_param.detach())
            for name, local_buf in local_model.named_buffers():
                if name in buffer_sums:
                    buffer_sums[name].add_(local_buf.detach())

    # Step 3: OTA aggregation → g_t (normalized by N)
    agg_grads = []
    for delta_sum in delta_sums:
        noisy_sum = oracle.aggregate_sum(delta_sum) if hasattr(oracle, "aggregate_sum") else oracle.aggregate([delta_sum])
        agg_grads.append(noisy_sum / N)

    # Step 4: optional MAC pre-processing
    if use_mac:
        agg_grads = apply_mac(agg_grads, clip_factor=mac_clip)

    diagnostics = None
    if return_diagnostics:
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
            return diagnostics

    old_params = None
    if return_diagnostics:
        old_params = [p.detach().clone() for p in global_model.parameters()]

    # Step 5: server-side adaptive update
    server_opt.step(agg_grads)

    if return_diagnostics:
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
            return diagnostics

    # Keep floating buffers (e.g. BatchNorm running stats) in sync with client models.
    # Without this, ResNet evaluation uses stale normalization statistics from round 0.
    if buffer_sums:
        with torch.no_grad():
            for name, global_buf in global_model.named_buffers():
                if name in buffer_sums:
                    global_buf.copy_((buffer_sums[name] / N).to(global_buf.device))

    return diagnostics if return_diagnostics else None


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = False,
    amp_dtype: str = "bf16",
    channels_last: bool = False,
) -> tuple[float, float]:
    """
    Evaluate model on a DataLoader.

    Returns:
        (avg_loss, accuracy)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device)
    total = 0

    with torch.no_grad():
        for batch in loader:
            x, y = _prepare_loader_batch(loader, batch, device, channels_last)
            with _autocast(device, use_amp, amp_dtype):
                out = model(x)
                loss = criterion(out, y)
            total_loss += loss.float() * len(y)
            correct += (out.argmax(dim=1) == y).sum()
            total += len(y)

    return (total_loss / total).item(), (correct / total).item()
