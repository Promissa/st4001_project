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

## 4. Legacy AWGN Ablation Studies (`ablation.py`)

These commands are kept for traceability. They are no longer the main thesis
evidence because the revised core experiment is the alpha-stable tail-index
sweep in Section 5.

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

## 5. Core Heavy-Tailed Experiments (`heavy_tail.py`)

The revised thesis core is alpha-stable interference. `alpha=2.0` is AWGN;
smaller values are heavier-tailed.

### A. Fractional-alpha sanity check

```bash
./run_all.sh alpha_sanity
```

This runs a tiny MNIST job at `alpha=1.1` and `alpha=1.3` to check for
NaN/inf before spending compute on CIFAR-10.

### B. CIFAR-10 alpha ablation

```bash
uv run -m src.experiments.heavy_tail --study alpha \
  --dataset cifar10 \
  --model resnet18 \
  --rounds 100 \
  --num_clients 100 \
  --local_epochs 1 \
  --server_lr 0.01 \
  --noise_scale 0.05 \
  --alphas 1.1,1.3,1.5,1.7,1.9,2.0 \
  --seeds 42,43,44 \
  --methods fedavg,fedavgm,adagrad_ota,adam_ota \
  --out_dir results/heavytail
```

Equivalent shortcut:

```bash
./run_all.sh alpha_ablation
```

### C. MAC vs no-MAC

```bash
uv run -m src.experiments.heavy_tail --study mac \
  --dataset cifar10 \
  --model resnet18 \
  --rounds 100 \
  --num_clients 100 \
  --local_epochs 1 \
  --server_lr 0.01 \
  --noise_scale 0.05 \
  --mac_alpha 1.3 \
  --mac_clip 3.0 \
  --seeds 42,43,44 \
  --methods fedavg,fedavgm,adagrad_ota,adam_ota \
  --out_dir results/heavytail
```

Equivalent shortcut:

```bash
./run_all.sh mac_compare
```

**Output:**
- `results/heavytail/alpha_ablation_cifar10_resnet18.json`
- `results/heavytail/mac_compare_cifar10_resnet18_alpha1.3.json`

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

### Alpha Ablation and MAC Figures

```bash
uv run -m src.experiments.plot \
  --alpha_ablation results/heavytail/alpha_ablation_cifar10_resnet18.json \
  --mac_compare results/heavytail/mac_compare_cifar10_resnet18_alpha1.3.json \
  --out_dir results/figures
```

### Generated LaTeX Tables

```bash
uv run -m src.experiments.report \
  --alpha_ablation results/heavytail/alpha_ablation_cifar10_resnet18.json \
  --mac_compare results/heavytail/mac_compare_cifar10_resnet18_alpha1.3.json \
  --out_dir template/data/generated
```

Shortcut:

```bash
./run_all.sh paper_update
```

### Plot All at Once

```bash
uv run -m src.experiments.plot \
  --result results/comparison/cifar10_resnet18_alpha2.0_N100.json \
  --ablation_noise results/ablation/ablation_noise_cifar10_resnet18.json \
  --ablation_clients results/ablation/ablation_clients_cifar10_resnet18.json \
  --alpha_ablation results/heavytail/alpha_ablation_cifar10_resnet18.json \
  --mac_compare results/heavytail/mac_compare_cifar10_resnet18_alpha1.3.json \
  --out_dir results/figures
```

**Output:** PNGs written to `results/figures/`.

---

## 7. Recommended Full Workflow

Run this to produce the revised thesis core result set:

```bash
./run_all.sh core_heavytail
```

For a shorter compute check:

```bash
ABLATION_ROUNDS=2 SEEDS=42 ALPHA_VALUES=1.3,2.0 METHODS=adagrad_ota,adam_ota ./run_all.sh core_heavytail
```

---

## 8. Quick Reference: Output File Paths

| Script | Output Path |
|---|---|
| `train.py` | `results/single/*.json`, `runs/*/` (TensorBoard) |
| `compare.py` | `results/comparison/<dataset>_<model>_alpha<alpha>_N<N>.json` |
| `ablation.py` | `results/ablation/ablation_noise_*.json`, `results/ablation/ablation_clients_*.json` |
| `heavy_tail.py` | `results/heavytail/alpha_ablation_*.json`, `results/heavytail/mac_compare_*.json` |
| `plot.py` | `results/figures/*.png` |
| `report.py` | `template/data/generated/*.tex` |
