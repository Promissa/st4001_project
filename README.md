# ADOTA-FL: Adaptive Federated Learning Over the Air

ST 4001 Senior Thesis — Peiran Wei (ZJU-UIUC Institute, Electrical and Computing Engineering, supervisor: Hao Yang).

A PyTorch simulation framework for **Federated Learning with Analog Over-the-Air aggregation under additive white Gaussian noise (AWGN)**. The project tests the hypothesis that *server-side adaptive optimizers* (AdaGrad, Adam) act as an **implicit noise-mitigation filter**: as channel noise grows, the second-moment accumulator inflates and the effective learning rate contracts, stabilising training without any channel-state information.

## Background

Over-the-Air FL (OTA-FL) exploits the superposition property of the wireless multiple-access channel: clients transmit local updates simultaneously and the server reads back the analog sum, achieving bandwidth that does not scale with the number of users. The trade-off is that the aggregated gradient is corrupted by additive channel noise. With plain server-side SGD (FedAvg), increasing noise variance destabilises the trajectory; with momentum (FedAvgM) it improves but degrades quickly. The question this project investigates is whether the second-moment estimate that AdaGrad and Adam use to normalise per-coordinate step sizes — originally devised for sparse / noisy gradients in centralised ML — also doubles as an effective filter for channel-noise-induced gradient variance.

## System Model

Each communication round t:

1. Server broadcasts global model w_t to N clients.
2. Each client n runs E local SGD epochs and forms a pseudo-gradient Δ_n = w_t − w_n^local.
3. The **Noisy Oracle** simulates analog superposition:
   g_t = (1/N) · Σ_n Δ_n + ξ_t,    ξ_t ~ N(0, γ²·I) (AWGN, default α = 2)
4. *(optional)* **MAC** robust pre-processor (off by default).
5. Server-side optimizer updates w_{t+1}.

Server-side optimizers implemented:

| Optimizer | Update rule |
|---|---|
| FedAvg-OTA | w ← w − η g |
| FedAvgM-OTA | buf ← β·buf + g; w ← w − η·buf |
| AdaGrad-OTA | Δ = β₁Δ + (1−β₁)g; v += Δ²; w ← w − η·Δ/(√v + ε) |
| Adam-OTA | Δ = β₁Δ + (1−β₁)g; v = β₂v + (1−β₂)Δ²; w ← w − η·Δ/(√v + ε) |

(With α = 2 the AdaGrad-OTA / Adam-OTA updates reduce to classical AdaGrad / Adam; the optimizers retain a generic α exponent so that non-Gaussian channels can be explored later.)

Noise generation uses the Chambers–Mallows–Stuck stable-variable sampler, which falls back to a Gaussian draw when α = 2.

## Repository Layout

```
src/
  channel/
    noise.py        # Stable-variable sampler (AWGN when α = 2)
    ota.py          # NoisyOracle: analog OTA aggregation with additive noise
    mac.py          # Optional median-anchored clipping pre-processor
  data/
    datasets.py     # MNIST/CIFAR-10 loaders + IID and Dirichlet partitioning
  models/
    nets.py         # MLP, ConvNet, ResNet-18/34 (CIFAR-10 adapted)
  optimizers/
    baselines.py    # FedAvgOTA, FedAvgMOTA
    adagrad_ota.py  # Server-side AdaGrad on the OTA-aggregated gradient
    adam_ota.py     # Server-side Adam on the OTA-aggregated gradient
  experiments/
    federated.py    # run_round / evaluate — core FL training loop
    compare.py      # Comparative analysis across all four methods
    ablation.py     # Ablations over noise scale γ and client count N
    plot.py         # Figures for comparison and ablations
train.py            # Single-run trainer
run.sh              # Example end-to-end CIFAR-10 / ResNet-18 / Adam-OTA / non-IID run
documents/          # Thesis proposal and task plan
runs/               # TensorBoard event files
```

## Setup

The project uses [`uv`](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
uv sync
```

Core dependencies: PyTorch, torchvision, numpy, scipy, matplotlib, tensorboardX, tqdm.

## Usage

**Single training run**:

```bash
uv run train.py --dataset cifar10 --model resnet18 --optimizer adam_ota \
                --noise_scale 0.05 --rounds 200 --non_iid
```

`run.sh` contains a representative CIFAR-10 / ResNet-18 / 100-client / non-IID configuration.

**Comparative analysis** (FedAvg vs FedAvgM vs AdaGrad-OTA vs Adam-OTA):

```bash
uv run -m src.experiments.compare --dataset cifar10 --model resnet18 --rounds 200
```

**Ablation studies**:

```bash
uv run -m src.experiments.ablation --study noise    # γ ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5}
uv run -m src.experiments.ablation --study clients  # N ∈ {5, 10, 20, 50, 100}
```

**Figures**:

```bash
uv run -m src.experiments.plot --result           results/comparison/<file>.json
uv run -m src.experiments.plot --ablation_noise   results/ablation/<file>.json
uv run -m src.experiments.plot --ablation_clients results/ablation/<file>.json
```

**TensorBoard** logging is enabled by default: `tensorboard --logdir=runs`.

## Expected Outcomes

- A working OTA-FL framework that converges under AWGN where plain FedAvg/FedAvgM degrade or diverge as noise grows.
- Empirical evidence that the gap between FedAvg-style and adaptive methods widens monotonically with the noise scale γ — the operational signature of the implicit learning-rate compression provided by the second-moment accumulator.
- Empirical confirmation of the OTA averaging gain: final accuracy improves with N as the per-round noise variance contracts.
- Adam-OTA expected to outperform AdaGrad-OTA on convergence speed thanks to its EMA second-moment design.

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, 2017 — FedAvg.
- Duchi, Hazan, Singer, *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*, JMLR 2011 — AdaGrad.
- Kingma, Ba, *Adam: A Method for Stochastic Optimization*, 2015 — Adam.
- Chambers, Mallows, Stuck, *A Method for Simulating Stable Random Variables*, JASA 1976 — noise sampler.

A fuller bibliography on OTA-FL background is in `documents/ST4001 Project Proposal v1.pdf`.
