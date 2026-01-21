# Quick Start Guide

## Setup

1. **Install dependencies** (if not already done):
   ```bash
   source $HOME/.local/bin/env  # Activate uv
   uv sync
   ```

2. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

3. **Verify the setup**:
   ```bash
   python verify_setup.py
   ```

## Running Experiments

### Basic Training

Run a training experiment with default configuration:
```bash
python -m src.experiments.train --config configs/default.yaml
```

### Custom Configuration

Create your own configuration file or modify `configs/default.yaml`:

```yaml
noise:
  alpha: 1.5  # Tail index (1 < alpha <= 2)
  noise_scale: 0.1
  use_clipping: true

optimizer:
  name: "adamota"  # or "adagradota"
  lr: 0.001
  alpha: 1.5  # Should match noise alpha

data:
  name: "mnist"  # or "cifar10"
  num_clients: 100
  data_split: "iid"  # or "non-iid"

training:
  num_rounds: 100
  local_epochs: 1
```

## Project Structure

- `src/ota_fl/`: Core OTA-FL implementation
  - `noise.py`: α-stable noise generation
  - `noisy_oracle.py`: OTA aggregation simulation
  - `optimizers.py`: AdaGrad-OTA and Adam-OTA
  - `clipping.py`: Median Anchored Clipping
  - `models.py`: Model architectures

- `src/experiments/`: Experiment scripts
  - `train.py`: Main training script
  - `config.py`: Configuration management

- `configs/`: Configuration files
- `data/`: Dataset storage (auto-created)
- `results/`: Experiment results (auto-created)

## Key Features

1. **α-stable Noise**: Chambers-Mallows-Stuck method for generating heavy-tailed interference
2. **Noisy Oracle**: Simulates OTA aggregation with channel fading and interference
3. **Adaptive Optimizers**: Modified AdaGrad and Adam using α-norms for robust training
4. **MAC Preprocessing**: Median Anchored Clipping for outlier removal

## Next Steps

1. Run baseline experiments with different noise parameters (α values)
2. Compare AdaGrad-OTA vs Adam-OTA convergence rates
3. Experiment with different datasets (MNIST, CIFAR-10)
4. Analyze the effect of MAC clipping on robustness
