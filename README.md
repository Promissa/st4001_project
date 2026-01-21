# Adaptive Federated Learning Over the Air

This project implements and validates an adaptive optimization framework for Over-the-Air Federated Learning (OTA-FL) that functions as an implicit noise-mitigation filter. The project reproduces and extends findings showing that adaptive step-size mechanisms (AdaGrad and Adam variants) can stabilize training trajectories against channel impairments, particularly heavy-tailed interference.

## Project Overview

The project focuses on the intersection of wireless communications (physical layer) and distributed optimization (algorithmic layer) to address reliability challenges in industrial IoT environments where electromagnetic interference is impulsive and heavy-tailed.

## Key Features

- **Noisy Oracle System Model**: Simulates OTA aggregation with α-stable noise injection
- **Adaptive Optimizers**: Modified AdaGrad and Adam using fractional moments (α-norms) for second-moment estimation
- **Heavy-Tailed Noise**: Chambers-Mallows-Stuck method for generating synthetic α-stable noise
- **Robust Statistics**: Median Anchored Clipping (MAC) preprocessing to truncate extreme outliers

## Project Structure

```
st4001/
├── src/
│   ├── ota_fl/
│   │   ├── __init__.py
│   │   ├── noise.py              # α-stable noise generation
│   │   ├── noisy_oracle.py       # OTA aggregation simulation
│   │   ├── optimizers.py         # AdaGrad-OTA, Adam-OTA
│   │   ├── clipping.py           # MAC preprocessing
│   │   └── models.py             # Model architectures
│   └── experiments/
│       ├── __init__.py
│       ├── train.py              # Main training script
│       └── config.py             # Configuration management
├── tests/
├── data/                         # Dataset storage
├── results/                      # Experiment results
└── pyproject.toml
```

## Installation

```bash
# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

## Usage

```bash
# Run training experiments
python -m src.experiments.train --config configs/default.yaml
```

## References

- Wang, C., Chen, Z., Pappas, N., & Yang, H. (2024). Adaptive Federated Learning Over the Air. IEEE Transactions on Signal Processing, 2025.
