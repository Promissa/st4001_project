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
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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


def resolve_devices(spec: str | None = None) -> list[torch.device]:
    """Resolve a comma-separated CUDA device list against visible GPUs."""
    if not torch.cuda.is_available():
        return [torch.device("cpu")]

    raw_ids = [item.strip() for item in (spec or "0").split(",") if item.strip()]
    ids = []
    for item in raw_ids:
        idx = int(item)
        if idx not in ids:
            ids.append(idx)

    visible = torch.cuda.device_count()
    valid_ids = [idx for idx in ids if 0 <= idx < visible]
    if not valid_ids:
        valid_ids = [0]
    return [torch.device(f"cuda:{idx}") for idx in valid_ids]


def _amp_torch_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    return torch.bfloat16


def _autocast(device: torch.device, use_amp: bool, amp_dtype: str):
    if use_amp and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=_amp_torch_dtype(amp_dtype))
    return nullcontext()


def _prepare_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    channels_last: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if channels_last and x.ndim == 4:
        x = x.contiguous(memory_format=torch.channels_last)
    return x, y


def _prepare_loader_batch(
    loader: DataLoader,
    x: torch.Tensor,
    y: torch.Tensor | None,
    device: torch.device,
    channels_last: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    prepare_batch = getattr(loader, "prepare_batch", None)
    if callable(prepare_batch):
        return prepare_batch(x, y, device, channels_last)
    return _prepare_batch(x, y, device, channels_last)


def _maybe_channels_last(model: nn.Module, enabled: bool) -> nn.Module:
    if enabled:
        model = model.to(memory_format=torch.channels_last)
    return model


def _sum_float_buffers(
    buffer_dicts: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    for buffers in buffer_dicts:
        for name, value in buffers.items():
            if name not in sums:
                sums[name] = torch.zeros_like(value, device=device)
            sums[name].add_(value.to(device))
    return sums


def _loader_num_samples(loader: DataLoader) -> int:
    if hasattr(loader, "num_samples"):
        return int(loader.num_samples)
    targets = getattr(loader, "targets", None)
    if targets is not None:
        return len(targets)
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        return len(dataset)
    try:
        return len(loader)  # type: ignore[arg-type]
    except TypeError:
        return 0


def _balanced_client_shards(
    client_loaders: list[DataLoader],
    num_shards: int,
) -> list[list[DataLoader]]:
    """Greedily balance shards by sample count, not just client count."""
    shards: list[list[DataLoader]] = [[] for _ in range(num_shards)]
    shard_sizes = [0 for _ in range(num_shards)]
    weighted = sorted(
        ((idx, loader, _loader_num_samples(loader)) for idx, loader in enumerate(client_loaders)),
        key=lambda item: (-item[2], item[0]),
    )

    for _, loader, samples in weighted:
        target = min(range(num_shards), key=lambda shard_idx: shard_sizes[shard_idx])
        shards[target].append(loader)
        shard_sizes[target] += samples
    return shards


def _finalize_round(
    global_model: nn.Module,
    agg_grads: list[torch.Tensor],
    oracle: NoisyOracle,
    server_opt,
    use_mac: bool,
    mac_clip: float,
    return_diagnostics: bool,
    buffer_sums: dict[str, torch.Tensor] | None = None,
    buffer_count: int = 0,
) -> dict | None:
    """Apply MAC, server optimizer update, diagnostics, and buffer averaging."""
    if use_mac:
        agg_grads = apply_mac(agg_grads, clip_factor=mac_clip)

    diagnostics = None
    old_params = None
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

        old_params = [p.detach().clone() for p in global_model.parameters()]

    server_opt.step(agg_grads)

    if old_params is not None:
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

    if buffer_sums and buffer_count > 0:
        with torch.no_grad():
            for name, global_buf in global_model.named_buffers():
                if torch.is_floating_point(global_buf) and name in buffer_sums:
                    averaged = buffer_sums[name].to(global_buf.device) / buffer_count
                    global_buf.copy_(averaged)

    return diagnostics


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
    local_model = _maybe_channels_last(local_model, channels_last)
    local_model.train()
    opt = torch.optim.SGD(local_model.parameters(), lr=local_lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for x, y in loader:
            x, y = _prepare_loader_batch(loader, x, y, device, channels_last)
            opt.zero_grad(set_to_none=True)
            with _autocast(device, use_amp, amp_dtype):
                loss = criterion(local_model(x), y)
            loss.backward()
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


def _train_client_shard(
    global_model: nn.Module,
    loaders: list[DataLoader],
    local_epochs: int,
    local_lr: float,
    device: torch.device,
    use_amp: bool,
    amp_dtype: str,
    channels_last: bool,
    collect_stats: bool = True,
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], int, dict]:
    """Train a shard of clients on one GPU, accumulating only summed deltas."""
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if collect_stats:
            torch.cuda.reset_peak_memory_stats(device)

    base_model = copy.deepcopy(global_model).to(device)
    base_model = _maybe_channels_last(base_model, channels_last)
    base_model.eval()

    local_model = copy.deepcopy(base_model).to(device)
    local_model = _maybe_channels_last(local_model, channels_last)
    opt = torch.optim.SGD(local_model.parameters(), lr=local_lr)
    criterion = nn.CrossEntropyLoss()

    delta_sums = [torch.zeros_like(p, device=device) for p in base_model.parameters()]
    buffer_sums = {
        name: torch.zeros_like(buf, device=device)
        for name, buf in base_model.named_buffers()
        if torch.is_floating_point(buf)
    }
    base_state = base_model.state_dict()
    num_batches = 0
    num_samples = 0

    for loader in loaders:
        local_model.load_state_dict(base_state)
        local_model.train()

        for _ in range(local_epochs):
            for x, y in loader:
                num_batches += 1
                num_samples += int(x.numel() if y is None else y.numel())
                x, y = _prepare_loader_batch(loader, x, y, device, channels_last)
                opt.zero_grad(set_to_none=True)
                with _autocast(device, use_amp, amp_dtype):
                    loss = criterion(local_model(x), y)
                loss.backward()
                opt.step()

        with torch.no_grad():
            for acc, bp, lp in zip(delta_sums, base_model.parameters(), local_model.parameters()):
                acc.add_(bp.data - lp.data)
            for name, buf in local_model.named_buffers():
                if torch.is_floating_point(buf) and name in buffer_sums:
                    buffer_sums[name].add_(buf.detach())

    if device.type == "cuda" and collect_stats:
        torch.cuda.synchronize(device)
        memory_allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        memory_allocated_mb = 0.0
        peak_memory_mb = 0.0

    stats = {
        "device": str(device),
        "clients": len(loaders),
        "batches": num_batches,
        "samples": num_samples,
        "local_seconds": time.perf_counter() - started if collect_stats else None,
        "memory_allocated_mb": memory_allocated_mb,
        "peak_memory_mb": peak_memory_mb,
    }
    return delta_sums, buffer_sums, len(loaders), stats


class _ClientShardWorker:
    """Persistent per-GPU worker used by ClientParallelExecutor."""

    def __init__(
        self,
        global_model: nn.Module,
        loaders: list[DataLoader],
        local_lr: float,
        device: torch.device,
        use_amp: bool,
        amp_dtype: str,
        channels_last: bool,
    ):
        self.loaders = loaders
        self.local_lr = local_lr
        self.device = device
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.channels_last = channels_last
        self.samples = sum(_loader_num_samples(loader) for loader in loaders)

        if device.type == "cuda":
            torch.cuda.set_device(device)

        self.base_model = copy.deepcopy(global_model).to(device)
        self.base_model = _maybe_channels_last(self.base_model, channels_last)
        self.base_model.eval()
        self.local_model = copy.deepcopy(self.base_model).to(device)
        self.local_model = _maybe_channels_last(self.local_model, channels_last)
        self.opt = torch.optim.SGD(self.local_model.parameters(), lr=local_lr)
        self.criterion = nn.CrossEntropyLoss()
        self.delta_sums = [torch.zeros_like(p, device=device) for p in self.base_model.parameters()]
        self.buffer_sums = {
            name: torch.zeros_like(buf, device=device)
            for name, buf in self.base_model.named_buffers()
            if torch.is_floating_point(buf)
        }

    def run(
        self,
        global_model: nn.Module,
        local_epochs: int,
        collect_stats: bool,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], int, dict]:
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            if collect_stats:
                torch.cuda.reset_peak_memory_stats(self.device)

        self.base_model.load_state_dict(global_model.state_dict())
        base_state = self.base_model.state_dict()
        for tensor in self.delta_sums:
            tensor.zero_()
        for tensor in self.buffer_sums.values():
            tensor.zero_()

        num_batches = 0
        num_samples = 0

        for loader in self.loaders:
            self.local_model.load_state_dict(base_state)
            self.local_model.train()

            for _ in range(local_epochs):
                for x, y in loader:
                    num_batches += 1
                    if y is None:
                        batch_samples = int(x.numel())
                    else:
                        batch_samples = int(y.numel())
                    num_samples += batch_samples
                    x, y = _prepare_loader_batch(loader, x, y, self.device, self.channels_last)
                    self.opt.zero_grad(set_to_none=True)
                    with _autocast(self.device, self.use_amp, self.amp_dtype):
                        loss = self.criterion(self.local_model(x), y)
                    loss.backward()
                    self.opt.step()

            with torch.no_grad():
                for acc, bp, lp in zip(
                    self.delta_sums,
                    self.base_model.parameters(),
                    self.local_model.parameters(),
                ):
                    acc.add_(bp.data - lp.data)
                for name, buf in self.local_model.named_buffers():
                    if torch.is_floating_point(buf) and name in self.buffer_sums:
                        self.buffer_sums[name].add_(buf.detach())

        if self.device.type == "cuda" and collect_stats:
            torch.cuda.synchronize(self.device)
            memory_allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            memory_allocated_mb = 0.0
            peak_memory_mb = 0.0

        stats = {
            "device": str(self.device),
            "clients": len(self.loaders),
            "samples": num_samples,
            "batches": num_batches,
            "local_seconds": time.perf_counter() - started if collect_stats else None,
            "memory_allocated_mb": memory_allocated_mb,
            "peak_memory_mb": peak_memory_mb,
        }
        return self.delta_sums, self.buffer_sums, len(self.loaders), stats


class ClientParallelExecutor:
    """Persistent multi-GPU client simulator for repeated FL rounds."""

    def __init__(
        self,
        global_model: nn.Module,
        client_loaders: list[DataLoader],
        worker_devices: list[torch.device],
        local_lr: float,
        use_amp: bool,
        amp_dtype: str,
        channels_last: bool,
    ):
        self.worker_devices = worker_devices
        self.shards = _balanced_client_shards(client_loaders, len(worker_devices))
        self.workers = [
            _ClientShardWorker(
                global_model,
                shard,
                local_lr,
                worker_device,
                use_amp,
                amp_dtype,
                channels_last,
            )
            for worker_device, shard in zip(worker_devices, self.shards)
            if shard
        ]
        self.executor = ThreadPoolExecutor(max_workers=len(self.workers))

    def run(
        self,
        global_model: nn.Module,
        local_epochs: int,
        collect_stats: bool,
    ):
        futures = [
            self.executor.submit(worker.run, global_model, local_epochs, collect_stats)
            for worker in self.workers
        ]
        return [future.result() for future in futures]

    def close(self) -> None:
        self.executor.shutdown(wait=True)


def _run_round_parallel(
    global_model: nn.Module,
    client_loaders: list[DataLoader],
    oracle: NoisyOracle,
    server_opt,
    local_epochs: int,
    local_lr: float,
    use_mac: bool,
    mac_clip: float,
    device: torch.device,
    return_diagnostics: bool,
    worker_devices: list[torch.device],
    use_amp: bool,
    amp_dtype: str,
    channels_last: bool,
    parallel_executor: ClientParallelExecutor | None = None,
) -> dict | None:
    N = len(client_loaders)
    collect_worker_stats = return_diagnostics

    if parallel_executor is not None:
        shard_results = parallel_executor.run(
            global_model,
            local_epochs,
            collect_stats=collect_worker_stats,
        )
    else:
        shards = _balanced_client_shards(client_loaders, len(worker_devices))
        with ThreadPoolExecutor(max_workers=len(worker_devices)) as executor:
            futures = [
                executor.submit(
                    _train_client_shard,
                    global_model,
                    shard,
                    local_epochs,
                    local_lr,
                    worker_device,
                    use_amp,
                    amp_dtype,
                    channels_last,
                    collect_worker_stats,
                )
                for worker_device, shard in zip(worker_devices, shards)
                if shard
            ]
            shard_results = [future.result() for future in futures]

    param_sums = [torch.zeros_like(p, device=device) for p in global_model.parameters()]
    buffer_sums: dict[str, torch.Tensor] = {}
    buffer_count = 0
    worker_stats = []

    for delta_sums, shard_buffer_sums, count, stats in shard_results:
        for acc, delta_sum in zip(param_sums, delta_sums):
            acc.add_(delta_sum.to(device, non_blocking=True))
        for name, value in shard_buffer_sums.items():
            if name not in buffer_sums:
                buffer_sums[name] = torch.zeros_like(value, device=device)
            buffer_sums[name].add_(value.to(device, non_blocking=True))
        buffer_count += count
        worker_stats.append(stats)

    if device.type == "cuda":
        # Keep this destination sync so the source worker buffers are not zeroed
        # for the next round before queued cross-GPU reductions have consumed them.
        torch.cuda.synchronize(device)

    agg_grads = []
    for param_sum in param_sums:
        if hasattr(oracle, "aggregate_sum"):
            noisy_sum = oracle.aggregate_sum(param_sum)
        else:
            noisy_sum = oracle.aggregate([param_sum])
        agg_grads.append(noisy_sum / N)

    diagnostics = _finalize_round(
        global_model,
        agg_grads,
        oracle,
        server_opt,
        use_mac,
        mac_clip,
        return_diagnostics,
        buffer_sums=buffer_sums,
        buffer_count=buffer_count,
    )
    if return_diagnostics and diagnostics is not None:
        diagnostics["execution_path"] = "parallel"
        diagnostics["worker_stats"] = worker_stats
    return diagnostics


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
    worker_devices: list[torch.device] | None = None,
    client_parallel: str = "off",
    use_amp: bool = False,
    amp_dtype: str = "bf16",
    channels_last: bool = False,
    parallel_executor: ClientParallelExecutor | None = None,
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
        worker_devices: CUDA devices used for client-local simulation.
        client_parallel: "auto" uses one shard per visible worker GPU; "off" is sequential.
        use_amp:        Use CUDA automatic mixed precision in local/eval passes.
        amp_dtype:      AMP dtype name: "bf16" or "fp16".
        channels_last:  Use NHWC memory format for convolutional models.
    """
    N = len(client_loaders)
    if hasattr(oracle, "begin_round"):
        oracle.begin_round(collect_diagnostics=return_diagnostics)

    worker_devices = worker_devices or [device]
    can_parallel = (
        client_parallel == "auto"
        and len(worker_devices) > 1
        and all(worker_device.type == "cuda" for worker_device in worker_devices)
    )
    if can_parallel:
        return _run_round_parallel(
            global_model,
            client_loaders,
            oracle,
            server_opt,
            local_epochs,
            local_lr,
            use_mac,
            mac_clip,
            device,
            return_diagnostics,
            worker_devices,
            use_amp,
            amp_dtype,
            channels_last,
            parallel_executor,
        )

    # Step 2: client local updates (in practice parallel; simulated sequentially)
    client_updates = [
        local_grad(
            global_model,
            loader,
            local_epochs,
            local_lr,
            device,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            channels_last=channels_last,
        )
        for loader in client_loaders
    ]
    all_grads = [grads for grads, _ in client_updates]
    all_buffers = [buffers for _, buffers in client_updates]
    buffer_sums = _sum_float_buffers(all_buffers, device)

    # Step 3: OTA aggregation → g_t (normalized by N)
    agg_grads = []
    for param_grads in zip(*all_grads):
        noisy_sum = oracle.aggregate(list(param_grads))
        agg_grads.append(noisy_sum / N)

    diagnostics = _finalize_round(
        global_model,
        agg_grads,
        oracle,
        server_opt,
        use_mac,
        mac_clip,
        return_diagnostics,
        buffer_sums=buffer_sums,
        buffer_count=len(all_buffers),
    )
    if return_diagnostics and diagnostics is not None:
        diagnostics["execution_path"] = "sequential"
        diagnostics["worker_stats"] = [{
            "device": str(device),
            "clients": len(client_loaders),
            "samples": sum(_loader_num_samples(loader) for loader in client_loaders),
        }]
    return diagnostics


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
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = _prepare_loader_batch(loader, x, y, device, channels_last)
            with _autocast(device, use_amp, amp_dtype):
                out = model(x)
                loss = criterion(out, y)
            total_loss += loss.item() * len(y)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += len(y)

    return total_loss / total, correct / total
