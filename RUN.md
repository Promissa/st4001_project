# Running ADOTA-FL

Complete command reference for the ADOTA-FL simulation framework.

---

## 1. Environment Setup

The project uses `uv` and Python >= 3.11.

```bash
# Install dependencies (creates .venv automatically)
uv sync
```

All subsequent commands use `uv run ...` so you do not need to activate the virtual environment manually.

---

## 2. Single Training Run (`train.py`)

Use this for quick prototyping or a single-method run with TensorBoard logging.

### Example: CIFAR-10 / ResNet-18 / Adam-OTA / non-IID

```bash
uv run train.py \
  --dataset cifar10 \
  --model resnet18 \
  --optimizer adam_ota \
  --rounds 200 \
  --num_clients 100 \
  --local_epochs 5 \
  --batch_size 128 \
  --server_lr 0.01 \
  --local_lr 0.01 \
  --momentum 0.9 \
  --beta2 0.999 \
  --alpha 2.0 \
  --noise_scale 0.05 \
  --non_iid \
  --dir_conc 0.1 \
  --save_results
```

### Example: MNIST / MLP / AdaGrad-OTA / IID

```bash
uv run train.py \
  --dataset mnist \
  --model mlp \
  --optimizer adagrad_ota \
  --rounds 100 \
  --num_clients 10 \
  --alpha 2.0 \
  --noise_scale 0.05
```

### Key Flags

| Flag | Meaning |
|---|---|
| `--dataset` | `mnist` or `cifar10` |
| `--model` | `mlp`, `convnet`, `resnet18`, `resnet34` |
| `--optimizer` | `fedavg`, `fedavgm`, `adagrad_ota`, `adam_ota` |
| `--rounds` | Number of FL communication rounds |
| `--num_clients` | Number of federated clients `N` |
| `--local_epochs` | Local SGD epochs per client per round |
| `--batch_size` | Local training batch size |
| `--server_lr` | Server-side optimizer learning rate `eta` |
| `--local_lr` | Client local SGD learning rate |
| `--momentum` | Momentum / first-moment decay `beta1` |
| `--beta2` | Adam second-moment decay `beta2` (default: 0.999) |
| `--alpha` | Noise tail index. `2.0` = Gaussian (AWGN). `< 2` = heavy-tailed |
| `--noise_scale` | Noise standard-deviation scale `gamma` |
| `--non_iid` | Enable Dirichlet(0.1) non-IID data partition |
| `--dir_conc` | Dirichlet concentration for non-IID partition |
| `--use_mac` | Enable Median-Anchored Clipping pre-processor |
| `--mac_clip` | MAC clipping threshold (default: 3.0) |
| `--seed` | Random seed (default: 42) |
| `--log_every` | Print interval in rounds (default: 10) |
| `--save_results` | Write JSON to `results/single/` |
| `--use_tensorboard` | Enable TensorBoard logging (default: true) |
| `--log_dir` | TensorBoard log directory (default: `runs`) |

### TensorBoard

```bash
# View live curves
tensorboard --logdir=runs
```

---

## 3. Comparative Analysis (`compare.py`)

Runs all four optimizers under identical seed, data split, and noise realizations.

### Example: CIFAR-10 / ResNet-18 / AWGN

```bash
uv run -m src.experiments.compare \
  --dataset cifar10 \
  --model resnet18 \
  --rounds 200 \
  --num_clients 100 \
  --local_epochs 5 \
  --batch_size 64 \
  --server_lr 0.1 \
  --local_lr 0.01 \
  --momentum 0.9 \
  --beta2 0.999 \
  --alpha 2.0 \
  --noise_scale 0.05 \
  --non_iid \
  --out_dir results/comparison
```

### Example: MNIST / MLP / AWGN

```bash
uv run -m src.experiments.compare \
  --dataset mnist \
  --model mlp \
  --rounds 100 \
  --num_clients 10 \
  --server_lr 0.01 \
  --noise_scale 0.05 \
  --out_dir results/comparison
```

> **Note on `--beta2`:** Pass `--beta2 0.999` explicitly for standard Adam behavior. Some experiment scripts may use a different internal default.

**Output:** `results/comparison/<dataset>_<model>_alpha<alpha>_N<num_clients>.json`

---

## 4. Ablation Studies (`ablation.py`)

### A. Noise-Scale Ablation (`gamma` sweep)

Tests all four optimizers at `gamma in {0.001, 0.005, 0.01, 0.05, 0.1, 0.5}`.

```bash
uv run -m src.experiments.ablation --study noise \
  --dataset cifar10 \
  --model resnet18 \
  --rounds 100 \
  --num_clients 10 \
  --alpha 2.0 \
  --server_lr 1e-4 \
  --local_lr 0.01 \
  --beta2 0.999 \
  --out_dir results/ablation
```

### B. Client-Count Ablation (`N` sweep)

Tests Adam-OTA at `N in {5, 10, 20, 50, 100}`.

```bash
uv run -m src.experiments.ablation --study clients \
  --dataset cifar10 \
  --model resnet18 \
  --rounds 100 \
  --noise_scale 0.05 \
  --alpha 2.0 \
  --server_lr 1e-4 \
  --beta2 0.999 \
  --out_dir results/ablation
```

### C. Run Both in Sequence

```bash
uv run -m src.experiments.ablation --study both \
  --dataset cifar10 \
  --model resnet18 \
  --rounds 100 \
  --alpha 2.0 \
  --server_lr 1e-4 \
  --beta2 0.999 \
  --out_dir results/ablation
```

**Output:**
- `results/ablation/ablation_noise_<dataset>_<model>.json`
- `results/ablation/ablation_clients_<dataset>_<model>.json`

---

## 5. Heavy-Tailed Noise (`alpha < 2`)

To test non-Gaussian channels, reduce the stability index `alpha`.

### Single run with heavy-tailed noise

```bash
uv run train.py \
  --dataset cifar10 \
  --model resnet18 \
  --optimizer adam_ota \
  --alpha 1.5 \
  --noise_scale 0.05 \
  --rounds 200 \
  --num_clients 100 \
  --non_iid \
  --beta2 0.999
```

### Full comparison under heavy-tailed noise

```bash
uv run -m src.experiments.compare \
  --dataset cifar10 \
  --model resnet18 \
  --alpha 1.5 \
  --noise_scale 0.05 \
  --rounds 200 \
  --num_clients 100 \
  --beta2 0.999
```

### Alpha sweep for thesis analysis

Run comparisons at multiple tail indices:

```bash
for alpha in 1.5 1.8 2.0; do
  uv run -m src.experiments.compare \
    --dataset cifar10 \
    --model resnet18 \
    --alpha $alpha \
    --noise_scale 0.05 \
    --rounds 200 \
    --num_clients 100 \
    --beta2 0.999 \
    --out_dir results/comparison
done
```

---

## 6. Plotting (`plot.py`)

Generate publication figures from saved JSONs.

### Comparison Curves (accuracy / loss vs. rounds)

```bash
uv run -m src.experiments.plot \
  --result results/comparison/cifar10_resnet18_alpha2.0_N100.json
```

### Noise Ablation Figure (final accuracy vs. `gamma`)

```bash
uv run -m src.experiments.plot \
  --ablation_noise results/ablation/ablation_noise_cifar10_resnet18.json
```

### Client Ablation Figure (curves colored by `N`)

```bash
uv run -m src.experiments.plot \
  --ablation_clients results/ablation/ablation_clients_cifar10_resnet18.json
```

### Plot All at Once

```bash
uv run -m src.experiments.plot \
  --result results/comparison/cifar10_resnet18_alpha2.0_N100.json \
  --ablation_noise results/ablation/ablation_noise_cifar10_resnet18.json \
  --ablation_clients results/ablation/ablation_clients_cifar10_resnet18.json \
  --out_dir results/figures
```

**Output:** PNGs written to `results/figures/`.

---

## 7. Recommended Full Workflow

Run these in order to produce the complete result set for a thesis:

```bash
# 1. Core comparison under AWGN (alpha = 2.0)
uv run -m src.experiments.compare \
  --dataset cifar10 \
  --model resnet18 \
  --rounds 200 \
  --num_clients 100 \
  --beta2 0.999

# 2. Core comparison under heavy-tailed noise (alpha = 1.5)
uv run -m src.experiments.compare \
  --dataset cifar10 \
  --model resnet18 \
  --alpha 1.5 \
  --rounds 200 \
  --num_clients 100 \
  --beta2 0.999

# 3. Ablation sweeps (noise scale + client count)
uv run -m src.experiments.ablation --study both \
  --dataset cifar10 \
  --model resnet18 \
  --rounds 100 \
  --beta2 0.999

# 4. Generate all figures
uv run -m src.experiments.plot \
  --result results/comparison/cifar10_resnet18_alpha2.0_N100.json \
  --ablation_noise results/ablation/ablation_noise_cifar10_resnet18.json \
  --ablation_clients results/ablation/ablation_clients_cifar10_resnet18.json
```

---

## 8. Quick Reference: Output File Paths

| Script | Output Path |
|---|---|
| `train.py` | `results/single/*.json`, `runs/*/` (TensorBoard) |
| `compare.py` | `results/comparison/<dataset>_<model>_alpha<alpha>_N<N>.json` |
| `ablation.py` | `results/ablation/ablation_noise_*.json`, `results/ablation/ablation_clients_*.json` |
| `plot.py` | `results/figures/*.png` |