# Parameter Settings from `2403.06528v1.pdf`

This document summarizes the parameter settings reported in the original paper,
`documents/2403.06528v1.pdf`, and maps them to the current repo where needed.

## 1. Paper Experiment Matrix

The paper evaluates ADOTA-FL across three learning tasks.

| Task | Dataset | Model | Clients `N` | Partition | Main Methods |
|---|---|---:|---:|---|---|
| Image classification | CIFAR-10 | ResNet-18 | 100 | non-IID Dirichlet, default `Dir = 0.1` | Adam-OTA, AdaGrad-OTA, FedAvgM-OTA |
| Image classification | CIFAR-100 | ResNet-34 | 50 | non-IID Dirichlet, default `Dir = 0.1` | Adam-OTA, AdaGrad-OTA, FedAvgM-OTA |
| Character classification | EMNIST | Logistic regression | 50 | non-IID Dirichlet, default `Dir = 0.1` | Adam-OTA, AdaGrad-OTA, FedAvgM-OTA |

Notes:
- The paper uses FedAvgM-OTA as the main baseline.
- The paper explicitly says FedAvg is not included in its reported main comparison because FedAvgM usually performs better in real-world data settings.
- The current repo supports `mnist` and `cifar10`; it does not yet support `emnist` or `cifar100`.

## 2. Data Partition Settings

The paper uses symmetric Dirichlet partitioning to simulate heterogeneous local data.

| Parameter | Paper Setting | Meaning |
|---|---:|---|
| Partition family | Symmetric Dirichlet | Creates non-IID client datasets |
| Default concentration | `Dir = 0.1` | Highly heterogeneous non-IID split |
| Heterogeneity sweep | `Dir = 0.1, 0.3, 1.0, 3.0, 10.0` | Smaller `Dir` means more non-IID |
| Client-count sweep | `N = 50, 100, 200, 400` | Used for system-scale analysis on CIFAR-10 |
| Client-count sweep partition | `Dir = 0.2` | Used in Fig. 6 |

Repo mapping:
- Use `--non_iid --dir_conc 0.1` for the paper default.
- Use `--num_clients 100` for CIFAR-10.
- Use MNIST only as a lightweight substitute for the paper's EMNIST-style simple task.

## 3. Channel / OTA Settings

The paper models the OTA aggregation channel as:

```text
g_t = (1/N) * sum_n h_{n,t} * grad f_n(w_t) + xi_t
```

| Parameter | Paper Setting | Meaning |
|---|---:|---|
| Channel fading | Rayleigh fading | Client channel gain model |
| Average channel gain | `mu_c = 1` | Mean channel gain |
| Interference distribution | Symmetric alpha-stable | Heavy-tailed electromagnetic interference |
| Default tail index | `alpha = 1.5` | Heavy-tailed noise in main Fig. 2 |
| Default interference scale | `0.1` | Unless otherwise specified |
| Alternative channel setting | `alpha = 1.8`, scale `0.01` | Fig. 3 CIFAR-10 robustness check |

Repo mapping:
- The current `NoisyOracle` supports alpha-stable/AWGN additive noise.
- The current `NoisyOracle` implements Rayleigh fading coefficients `h_{n,t}` with configurable mean `--channel_mean`.
- Use `--alpha 1.5 --noise_scale 0.1` to match the paper's main heavy-tailed noise setting more closely.
- Use `--alpha 2.0` for AWGN-only experiments in this repo.

## 4. Optimizer / Algorithm Parameters

The paper defines two adaptive OTA algorithms.

| Method | First-moment update | Second-moment update | Server update |
|---|---|---|---|
| AdaGrad-OTA | `Delta_t = beta1 * Delta_{t-1} + (1 - beta1) * g_t` | `v_t = v_{t-1} + |Delta_t|^alpha` | `w_{t+1} = w_t - eta * Delta_t / (v_t^(1/alpha) + eps)` |
| Adam-OTA | Same first moment | `v_t = beta2 * v_{t-1} + (1 - beta2) * |Delta_t|^alpha` | Same normalized update |
| FedAvgM-OTA | Momentum baseline | Static learning rate | Baseline comparison |

Paper-stated optimizer details:
- `beta1` controls the amount of historical smoothing in the first moment.
- `beta2` controls the amortization/historical weighting in Adam-OTA.
- Fig. 4 studies Adam-OTA with `beta1 = 0`.
- Fig. 4 sweeps `beta2 = 0.0, 0.3, 0.7, 0.9`.
- The paper observes `beta2 = 0.3` works well in that sweep.

Not specified in the paper:
- Exact server learning rate `eta`
- Exact local learning rate
- Batch size
- Number of local epochs
- Random seed
- `epsilon`

These must be chosen in the repo configuration.

## 5. Communication Rounds from Figures

The paper reports curves over communication rounds. Exact round counts are visible from figure axes rather than a single table.

| Figure / Study | Dataset / Model | Approx. Round Axis |
|---|---|---:|
| Fig. 2 CIFAR-10 | ResNet-18 / CIFAR-10 | 0 to 600 |
| Fig. 2 CIFAR-100 | ResNet-34 / CIFAR-100 | 0 to 200 |
| Fig. 2 EMNIST | Logistic regression / EMNIST | 0 to 600 |
| Fig. 3 CIFAR-10 alternate noise | ResNet-18 / CIFAR-10 | 0 to 600 |
| Fig. 4 beta2 sweep | Adam-OTA / CIFAR-10 | 0 to 600 |
| Fig. 5 alpha sweep | AdaGrad-OTA / CIFAR-10 | 0 to 800 |
| Fig. 6 client-count sweep | AdaGrad-OTA / CIFAR-10 | 0 to 700 |
| Fig. 7 Dir sweep | AdaGrad-OTA / CIFAR-10 | 0 to 600 |

Repo mapping:
- Use `--rounds 600` for paper-style CIFAR-10 curves.
- Use shorter runs such as `--rounds 100` or `--rounds 200` for local debugging.

## 6. Paper-Style Repo Commands

### Main CIFAR-10 Comparison, Paper-Like Heavy-Tailed Setting

```bash
uv run -m src.experiments.compare \
  --dataset cifar10 \
  --model resnet18 \
  --rounds 600 \
  --num_clients 100 \
  --alpha 1.5 \
  --noise_scale 0.1 \
  --non_iid \
  --dir_conc 0.1 \
  --beta2 0.3 \
  --out_dir results/comparison
```

### CIFAR-10 Alternate Noise Setting from Fig. 3

```bash
uv run -m src.experiments.compare \
  --dataset cifar10 \
  --model resnet18 \
  --rounds 600 \
  --num_clients 100 \
  --alpha 1.8 \
  --noise_scale 0.01 \
  --non_iid \
  --dir_conc 0.1 \
  --beta2 0.3 \
  --out_dir results/comparison
```

### Adam-OTA `beta2` Sweep from Fig. 4

The paper studies `beta2 = 0.0, 0.3, 0.7, 0.9` with `beta1 = 0` on CIFAR-10.
The repo does not currently have a dedicated `beta2` sweep script, so run
single jobs or add a sweep module.

```bash
for beta2 in 0.0 0.3 0.7 0.9; do
  uv run train.py \
    --dataset cifar10 \
    --model resnet18 \
    --optimizer adam_ota \
    --rounds 600 \
    --num_clients 100 \
    --momentum 0.0 \
    --beta2 "$beta2" \
    --alpha 1.5 \
    --noise_scale 0.1 \
    --non_iid \
    --dir_conc 0.1 \
    --save_results
done
```

### Alpha Sweep from Fig. 5

The paper sweeps `alpha = 1.1, 1.3, 1.7, 1.9` using AdaGrad-OTA on CIFAR-10.

```bash
for alpha in 1.1 1.3 1.7 1.9; do
  uv run train.py \
    --dataset cifar10 \
    --model resnet18 \
    --optimizer adagrad_ota \
    --rounds 800 \
    --num_clients 100 \
    --alpha "$alpha" \
    --noise_scale 0.1 \
    --non_iid \
    --dir_conc 0.1 \
    --save_results
done
```

### Client-Count Sweep from Fig. 6

The paper sweeps `N = 50, 100, 200, 400` with `Dir = 0.2` using AdaGrad-OTA on CIFAR-10.

```bash
for n in 50 100 200 400; do
  uv run train.py \
    --dataset cifar10 \
    --model resnet18 \
    --optimizer adagrad_ota \
    --rounds 700 \
    --num_clients "$n" \
    --alpha 1.5 \
    --noise_scale 0.1 \
    --non_iid \
    --dir_conc 0.2 \
    --save_results
done
```

### Data-Heterogeneity Sweep from Fig. 7

The paper sweeps `Dir = 0.1, 0.3, 1.0, 3.0, 10.0` using AdaGrad-OTA on CIFAR-10.

```bash
for dir in 0.1 0.3 1.0 3.0 10.0; do
  uv run train.py \
    --dataset cifar10 \
    --model resnet18 \
    --optimizer adagrad_ota \
    --rounds 600 \
    --num_clients 100 \
    --alpha 1.5 \
    --noise_scale 0.1 \
    --non_iid \
    --dir_conc "$dir" \
    --save_results
done
```

## 7. Recommended Settings for This Repo

Because this repo does not implement CIFAR-100 or EMNIST, use the following
settings for thesis experiments unless you add those datasets.

| Purpose | Dataset | Model | Optimizer(s) | Clients | Rounds | Noise | Partition |
|---|---|---|---|---:|---:|---|---|
| Main paper-like run | CIFAR-10 | ResNet-18 | Adam-OTA, AdaGrad-OTA, FedAvgM-OTA | 100 | 600 | `alpha=1.5`, `noise_scale=0.1` | `Dir=0.1` |
| Fast debug | MNIST | MLP | Any | 10 | 5-100 | `alpha=2.0`, `noise_scale=0.05` | IID or `Dir=0.1` |
| AWGN variant | CIFAR-10 | ResNet-18 | Adam-OTA, AdaGrad-OTA, FedAvgM-OTA | 100 | 200-600 | `alpha=2.0`, `noise_scale=0.05` | `Dir=0.1` |
| Tail-index study | CIFAR-10 | ResNet-18 | AdaGrad-OTA | 100 | 800 | `alpha in {1.1,1.3,1.7,1.9}` | `Dir=0.1` |
| Client-scale study | CIFAR-10 | ResNet-18 | AdaGrad-OTA | `50,100,200,400` | 700 | `alpha=1.5`, `noise_scale=0.1` | `Dir=0.2` |
| Heterogeneity study | CIFAR-10 | ResNet-18 | AdaGrad-OTA | 100 | 600 | `alpha=1.5`, `noise_scale=0.1` | `Dir in {0.1,0.3,1.0,3.0,10.0}` |

## 8. Missing Paper Details

The paper does not state every engineering hyperparameter needed to run this repo.
The following should be documented in your experiment logs whenever you run:

- server learning rate
- local learning rate
- local epochs
- batch size
- seed
- optimizer epsilon
- whether BatchNorm buffers are averaged
- whether Rayleigh fading is enabled or disabled with `--no_fading`
- whether CIFAR-100/EMNIST are omitted or replaced
