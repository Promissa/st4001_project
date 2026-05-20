#!/usr/bin/env bash
set -euo pipefail

# Project runner for ADOTA-FL.
#
# This file follows the revised ST4001 project positioning:
#   - core evidence: alpha-stable heavy-tailed OTA-FL on CIFAR-10
#   - AWGN is the alpha=2.0 special case
#   - MAC is a measured robust pre-processing baseline
#
# Usage:
#   ./run_all.sh help
#   ./run_all.sh runall
#   ./run_all.sh workflow
#   ./run_all.sh compare
#   ./run_all.sh ablation

UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}"
export UV_CACHE_DIR

DATASET="${DATASET:-cifar10}"
MODEL="${MODEL:-resnet18}"
NUM_CLIENTS="${NUM_CLIENTS:-100}"
ROUNDS="${ROUNDS:-200}"
ABLATION_ROUNDS="${ABLATION_ROUNDS:-100}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
COMPARE_BATCH_SIZE="${COMPARE_BATCH_SIZE:-64}"
SERVER_LR="${SERVER_LR:-0.01}"
COMPARE_SERVER_LR="${COMPARE_SERVER_LR:-0.1}"
ABLATION_SERVER_LR="${ABLATION_SERVER_LR:-1e-4}"
LOCAL_LR="${LOCAL_LR:-0.01}"
MOMENTUM="${MOMENTUM:-0.9}"
BETA2="${BETA2:-0.999}"
ALPHA="${ALPHA:-2.0}"
ALPHA_VALUES="${ALPHA_VALUES:-1.1,1.3,1.5,1.7,1.9,2.0}"
SEEDS="${SEEDS:-42,43,44}"
MAC_ALPHA="${MAC_ALPHA:-1.3}"
MAC_CLIP="${MAC_CLIP:-3.0}"
NOISE_SCALE="${NOISE_SCALE:-0.05}"
DIR_CONC="${DIR_CONC:-0.1}"
HEAVYTAIL_LOCAL_EPOCHS="${HEAVYTAIL_LOCAL_EPOCHS:-1}"
AMP="${AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
CHANNELS_LAST="${CHANNELS_LAST:-1}"
FAST_DATA="${FAST_DATA:-auto}"
EVAL_EVERY="${EVAL_EVERY:-5}"
COMPARE_EVAL_EVERY="${COMPARE_EVAL_EVERY:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
DIAGNOSTICS_EVERY="${DIAGNOSTICS_EVERY:-0}"
MAX_EVAL_LOSS="${MAX_EVAL_LOSS:-1e6}"

COMPARISON_DIR="${COMPARISON_DIR:-results/comparison}"
ABLATION_DIR="${ABLATION_DIR:-results/ablation}"
HEAVYTAIL_DIR="${HEAVYTAIL_DIR:-results/heavytail1}"
FIGURE_DIR="${FIGURE_DIR:-results/figures}"
SINGLE_DIR="${SINGLE_DIR:-results/single}"
GENERATED_TEX_DIR="${GENERATED_TEX_DIR:-template/data/generated}"

ACCELERATOR_FLAGS=(
  --amp_dtype "$AMP_DTYPE"
  --fast_data "$FAST_DATA"
  --eval_every "$EVAL_EVERY"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --diagnostics_every "$DIAGNOSTICS_EVERY"
)

HEAVYTAIL_GUARD_FLAGS=(
  --max_eval_loss "$MAX_EVAL_LOSS"
)

case "$AMP" in
  1|true|TRUE|yes|YES) ACCELERATOR_FLAGS+=(--amp) ;;
  0|false|FALSE|no|NO) ACCELERATOR_FLAGS+=(--no_amp) ;;
esac

case "$CHANNELS_LAST" in
  1|true|TRUE|yes|YES) ACCELERATOR_FLAGS+=(--channels_last) ;;
  0|false|FALSE|no|NO) ACCELERATOR_FLAGS+=(--no_channels_last) ;;
esac

run_cmd() {
  printf '\n==> %s\n' "$*"
  "$@"
}

setup() {
  run_cmd uv sync
}

data() {
  run_cmd uv run prepare_data.py
  run_cmd uv run prepare_data.py --dataset mnist --partition iid --num_clients 10
  run_cmd uv run prepare_data.py --dataset cifar10 --partition non_iid \
    --num_clients 100 --dir_conc 0.1
}

train_cifar() {
  run_cmd uv run train.py \
    --dataset cifar10 \
    --model resnet18 \
    --optimizer adam_ota \
    --rounds "$ROUNDS" \
    --num_clients "$NUM_CLIENTS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --server_lr "$SERVER_LR" \
    --local_lr "$LOCAL_LR" \
    --momentum "$MOMENTUM" \
    --beta2 "$BETA2" \
    --alpha 2.0 \
    --noise_scale "$NOISE_SCALE" \
    --non_iid \
    --dir_conc "$DIR_CONC" \
    --save_results \
    --out_dir "$SINGLE_DIR" \
    "${ACCELERATOR_FLAGS[@]}"
}

train_mnist() {
  run_cmd uv run train.py \
    --dataset mnist \
    --model mlp \
    --optimizer adagrad_ota \
    --rounds 100 \
    --num_clients 10 \
    --alpha 2.0 \
    --noise_scale "$NOISE_SCALE" \
    --save_results \
    --out_dir "$SINGLE_DIR" \
    "${ACCELERATOR_FLAGS[@]}"
}

train() {
  train_mnist
  train_cifar
}

compare_cifar() {
  run_cmd uv run -m src.experiments.compare \
    --dataset cifar10 \
    --model resnet18 \
    --rounds "$ROUNDS" \
    --num_clients "$NUM_CLIENTS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch_size "$COMPARE_BATCH_SIZE" \
    --server_lr "$COMPARE_SERVER_LR" \
    --local_lr "$LOCAL_LR" \
    --momentum "$MOMENTUM" \
    --beta2 "$BETA2" \
    --alpha 2.0 \
    --noise_scale "$NOISE_SCALE" \
    --non_iid \
    --dir_conc "$DIR_CONC" \
    --out_dir "$COMPARISON_DIR" \
    "${ACCELERATOR_FLAGS[@]}" \
    --eval_every "$COMPARE_EVAL_EVERY"
}

compare_mnist() {
  run_cmd uv run -m src.experiments.compare \
    --dataset mnist \
    --model mlp \
    --rounds 100 \
    --num_clients 10 \
    --server_lr "$SERVER_LR" \
    --noise_scale "$NOISE_SCALE" \
    --beta2 "$BETA2" \
    --out_dir "$COMPARISON_DIR" \
    "${ACCELERATOR_FLAGS[@]}" \
    --eval_every "$COMPARE_EVAL_EVERY"
}

compare() {
  compare_mnist
  compare_cifar
}

ablation_noise() {
  run_cmd uv run -m src.experiments.ablation --study noise \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --rounds "$ABLATION_ROUNDS" \
    --num_clients 10 \
    --alpha 2.0 \
    --server_lr "$ABLATION_SERVER_LR" \
    --local_lr "$LOCAL_LR" \
    --beta2 "$BETA2" \
    --out_dir "$ABLATION_DIR"
}

ablation_clients() {
  run_cmd uv run -m src.experiments.ablation --study clients \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --rounds "$ABLATION_ROUNDS" \
    --noise_scale "$NOISE_SCALE" \
    --alpha 2.0 \
    --server_lr "$ABLATION_SERVER_LR" \
    --beta2 "$BETA2" \
    --out_dir "$ABLATION_DIR"
}

ablation() {
  run_cmd uv run -m src.experiments.ablation --study both \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --rounds "$ABLATION_ROUNDS" \
    --alpha 2.0 \
    --server_lr "$ABLATION_SERVER_LR" \
    --beta2 "$BETA2" \
    --out_dir "$ABLATION_DIR"
}

plot() {
  run_cmd uv run -m src.experiments.plot \
    --result "$COMPARISON_DIR/cifar10_resnet18_alpha2.0_N${NUM_CLIENTS}.json" \
    --ablation_noise "$ABLATION_DIR/ablation_noise_${DATASET}_${MODEL}.json" \
    --ablation_clients "$ABLATION_DIR/ablation_clients_${DATASET}_${MODEL}.json" \
    --out_dir "$FIGURE_DIR"
}

workflow() {
  data
  core_heavytail
}

runall() {
  setup
  data
  train
  compare
  core_heavytail
}

quickcheck() {
  ROUNDS=2 ABLATION_ROUNDS=1 NUM_CLIENTS=5 MODEL=mlp DATASET=mnist runall
}

alpha_sanity() {
  run_cmd uv run -m src.experiments.heavy_tail --study alpha \
    --dataset mnist \
    --model mlp \
    --rounds "${SANITY_ROUNDS:-2}" \
    --num_clients 5 \
    --local_epochs 1 \
    --batch_size 64 \
    --server_lr "$SERVER_LR" \
    --local_lr "$LOCAL_LR" \
    --momentum "$MOMENTUM" \
    --beta2 "$BETA2" \
    --noise_scale "$NOISE_SCALE" \
    --alphas "1.1,1.3" \
    --seeds "42" \
    --methods "adagrad_ota,adam_ota" \
    --log_every 1 \
    --save_diagnostics \
    --out_dir "$HEAVYTAIL_DIR/sanity" \
    "${HEAVYTAIL_GUARD_FLAGS[@]}" \
    "${ACCELERATOR_FLAGS[@]}"
}

alpha_ablation() {
  run_cmd uv run -m src.experiments.heavy_tail --study alpha \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --rounds "$ABLATION_ROUNDS" \
    --num_clients "$NUM_CLIENTS" \
    --local_epochs "${HEAVYTAIL_LOCAL_EPOCHS}" \
    --batch_size "$COMPARE_BATCH_SIZE" \
    --server_lr "0.001" \
    --local_lr "$LOCAL_LR" \
    --momentum "$MOMENTUM" \
    --beta2 "$BETA2" \
    --noise_scale "$NOISE_SCALE" \
    --alphas "$ALPHA_VALUES" \
    --seeds "43,44" \
    --methods "${METHODS:-fedavg,adagrad_ota,adam_ota}" \
    --dir_conc "$DIR_CONC" \
    --out_dir "$HEAVYTAIL_DIR" \
    "${HEAVYTAIL_GUARD_FLAGS[@]}" \
    "${ACCELERATOR_FLAGS[@]}"
}

mac_compare() {
  run_cmd uv run -m src.experiments.heavy_tail --study mac \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --rounds "$ABLATION_ROUNDS" \
    --num_clients "$NUM_CLIENTS" \
    --local_epochs "${HEAVYTAIL_LOCAL_EPOCHS}" \
    --batch_size "$COMPARE_BATCH_SIZE" \
    --server_lr "$SERVER_LR" \
    --local_lr "$LOCAL_LR" \
    --momentum "$MOMENTUM" \
    --beta2 "$BETA2" \
    --noise_scale "$NOISE_SCALE" \
    --mac_alpha "$MAC_ALPHA" \
    --mac_clip "$MAC_CLIP" \
    --seeds "$SEEDS" \
    --methods "${METHODS:-fedavg,fedavgm,adagrad_ota,adam_ota}" \
    --dir_conc "$DIR_CONC" \
    --out_dir "$HEAVYTAIL_DIR" \
    "${HEAVYTAIL_GUARD_FLAGS[@]}" \
    "${ACCELERATOR_FLAGS[@]}"
}

paper_update() {
  run_cmd uv run -m src.experiments.plot \
    --alpha_ablation "$HEAVYTAIL_DIR/alpha_ablation_${DATASET}_${MODEL}.json" \
    --mac_compare "$HEAVYTAIL_DIR/mac_compare_${DATASET}_${MODEL}_alpha${MAC_ALPHA}.json" \
    --out_dir "$FIGURE_DIR"

  run_cmd uv run -m src.experiments.report \
    --alpha_ablation "$HEAVYTAIL_DIR/alpha_ablation_${DATASET}_${MODEL}.json" \
    --mac_compare "$HEAVYTAIL_DIR/mac_compare_${DATASET}_${MODEL}_alpha${MAC_ALPHA}.json" \
    --out_dir "$GENERATED_TEX_DIR"
}

mechanism() {
  # Run a short heavy-tail trace with --save_diagnostics so the mechanism
  # plot can show v_t, ‖ḡ_t‖, and the effective server step contracting on
  # impulsive rounds. Defaults to a single seed and alpha=1.3 to keep this
  # cheap; override MECH_ROUNDS / MECH_SEEDS / MECH_ALPHAS to broaden it.
  local mech_alphas="${MECH_ALPHAS:-1.3}"
  local mech_seeds="${MECH_SEEDS:-42}"
  local mech_rounds="${MECH_ROUNDS:-50}"
  run_cmd uv run -m src.experiments.heavy_tail --study alpha \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --rounds "$mech_rounds" \
    --num_clients "$NUM_CLIENTS" \
    --local_epochs "${HEAVYTAIL_LOCAL_EPOCHS}" \
    --batch_size "$COMPARE_BATCH_SIZE" \
    --server_lr "$SERVER_LR" \
    --local_lr "$LOCAL_LR" \
    --momentum "$MOMENTUM" \
    --beta2 "$BETA2" \
    --noise_scale "$NOISE_SCALE" \
    --alphas "$mech_alphas" \
    --seeds "$mech_seeds" \
    --methods "${METHODS:-fedavg,fedavgm,adagrad_ota,adam_ota}" \
    --dir_conc "$DIR_CONC" \
    --save_diagnostics \
    --out_dir "$HEAVYTAIL_DIR/mechanism" \
    "${HEAVYTAIL_GUARD_FLAGS[@]}" \
    "${ACCELERATOR_FLAGS[@]}"

  # Mechanism plots are one per alpha key inside the JSON.
  local mech_json="$HEAVYTAIL_DIR/mechanism/alpha_ablation_${DATASET}_${MODEL}.json"
  IFS=',' read -r -a mech_alpha_arr <<< "$mech_alphas"
  for alpha_value in "${mech_alpha_arr[@]}"; do
    run_cmd uv run -m src.experiments.plot_mechanism \
      --diagnostics "$mech_json" \
      --alpha "$alpha_value" \
      --out_dir "$FIGURE_DIR"
  done
}

lr_sweep() {
  local lr_alphas="${LR_SWEEP_ALPHA:-1.3}"
  local lr_values="${LR_SWEEP_LRS:-1e-3,3e-3,1e-2,3e-2,1e-1}"
  run_cmd uv run -m src.experiments.lr_sweep \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --rounds "$ABLATION_ROUNDS" \
    --num_clients "$NUM_CLIENTS" \
    --local_epochs "${HEAVYTAIL_LOCAL_EPOCHS}" \
    --batch_size "$COMPARE_BATCH_SIZE" \
    --server_lrs "$lr_values" \
    --local_lr "$LOCAL_LR" \
    --momentum "$MOMENTUM" \
    --beta2 "$BETA2" \
    --noise_scale "$NOISE_SCALE" \
    --mac_alpha "$lr_alphas" \
    --seeds "$SEEDS" \
    --methods "${METHODS:-fedavg,fedavgm,adagrad_ota,adam_ota}" \
    --dir_conc "$DIR_CONC" \
    --out_dir "$HEAVYTAIL_DIR" \
    "${HEAVYTAIL_GUARD_FLAGS[@]}" \
    "${ACCELERATOR_FLAGS[@]}"

  run_cmd uv run -m src.experiments.plot \
    --lr_sweep "$HEAVYTAIL_DIR/lr_sweep_${DATASET}_${MODEL}_alpha${lr_alphas}.json" \
    --out_dir "$FIGURE_DIR"
}

mac_sweep() {
  local mac_ks="${MAC_K_SWEEP:-0,1,3,5}"
  local mac_gammas="${MAC_GAMMA_SWEEP:-0.05,0.1,0.2}"
  run_cmd uv run -m src.experiments.heavy_tail --study mac_sweep \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --rounds "$ABLATION_ROUNDS" \
    --num_clients "$NUM_CLIENTS" \
    --local_epochs "${HEAVYTAIL_LOCAL_EPOCHS}" \
    --batch_size "$COMPARE_BATCH_SIZE" \
    --server_lr "$SERVER_LR" \
    --local_lr "$LOCAL_LR" \
    --momentum "$MOMENTUM" \
    --beta2 "$BETA2" \
    --noise_scale "$NOISE_SCALE" \
    --mac_alpha "$MAC_ALPHA" \
    --mac_clips "$mac_ks" \
    --mac_noise_scales "$mac_gammas" \
    --seeds "$SEEDS" \
    --methods "${METHODS:-fedavg,fedavgm,adagrad_ota,adam_ota}" \
    --dir_conc "$DIR_CONC" \
    --out_dir "$HEAVYTAIL_DIR" \
    "${HEAVYTAIL_GUARD_FLAGS[@]}" \
    "${ACCELERATOR_FLAGS[@]}"

  run_cmd uv run -m src.experiments.plot \
    --mac_sweep "$HEAVYTAIL_DIR/mac_sweep_${DATASET}_${MODEL}_alpha${MAC_ALPHA}.json" \
    --out_dir "$FIGURE_DIR"
}

new_sweep() {
    lr_sweep
    mac_sweep
}

core_heavytail() {
  alpha_sanity
  alpha_ablation
  mac_compare
  paper_update
}

heavy_tail_compare() {
  run_cmd uv run -m src.experiments.compare \
    --dataset cifar10 \
    --model resnet18 \
    --rounds "$ROUNDS" \
    --num_clients "$NUM_CLIENTS" \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch_size "$COMPARE_BATCH_SIZE" \
    --server_lr "$COMPARE_SERVER_LR" \
    --local_lr "$LOCAL_LR" \
    --momentum "$MOMENTUM" \
    --beta2 "$BETA2" \
    --alpha "${HEAVY_ALPHA:-1.5}" \
    --noise_scale "${HEAVY_NOISE_SCALE:-0.05}" \
    --non_iid \
    --dir_conc "$DIR_CONC" \
    --out_dir "$COMPARISON_DIR" \
    "${ACCELERATOR_FLAGS[@]}"
}

paper_like_compare() {
  run_cmd uv run -m src.experiments.compare \
    --dataset cifar10 \
    --model resnet18 \
    --rounds "${PAPER_ROUNDS:-600}" \
    --num_clients 100 \
    --local_epochs "$LOCAL_EPOCHS" \
    --batch_size "$COMPARE_BATCH_SIZE" \
    --server_lr "$COMPARE_SERVER_LR" \
    --local_lr "$LOCAL_LR" \
    --momentum "$MOMENTUM" \
    --beta2 0.3 \
    --alpha 1.5 \
    --noise_scale 0.1 \
    --non_iid \
    --dir_conc 0.1 \
    --out_dir "$COMPARISON_DIR/paper_like" \
    "${ACCELERATOR_FLAGS[@]}"
}

help() {
  cat <<'EOF'
Usage:
  ./run_all.sh <module>

Core project modules:
  runall              Run setup, data, legacy comparisons, heavy-tail core runs, and plots.
  workflow            Run the revised thesis workflow: data -> alpha/MAC heavy-tail runs -> plots.
  quickcheck          Very short smoke run for command validation.
  setup               Install/sync dependencies with uv.
  data                Prepare/verify MNIST and CIFAR-10.
  train               Run MNIST sanity training and CIFAR-10 main single training.
  train_mnist         Run MNIST / MLP sanity training.
  train_cifar         Run CIFAR-10 / ResNet-18 / Adam-OTA main single training.
  compare             Run MNIST and CIFAR-10 four-method comparisons.
  compare_mnist       Run MNIST comparison.
  compare_cifar       Run CIFAR-10 comparison.
  ablation            Run AWGN noise-scale and client-count ablations.
  ablation_noise      Run only AWGN noise-scale ablation.
  ablation_clients    Run only AWGN client-count ablation.
  plot                Generate core thesis figures.
  alpha_sanity        1-2 round fractional-alpha finite-value smoke test.
  alpha_ablation      Run alpha-stable tail-index sweep.
  mac_compare         Run MAC vs no-MAC under alpha-stable interference.
  core_heavytail      Run alpha_sanity, alpha_ablation, mac_compare, paper_update.
  paper_update        Generate alpha/MAC figures and LaTeX tables from JSON.
  mechanism           Run a single-seed heavy-tail trace with diagnostics on
                      and plot v_t, ‖ḡ_t‖, effective step contraction.
  lr_sweep            Sweep server learning rate per method at fixed (α, γ).
  mac_sweep           Sweep MAC clip factor k and noise scale γ at fixed α.

Optional extension modules:
  heavy_tail_compare  Legacy one-off alpha-stable comparison.
  paper_like_compare  Optional partial paper-style CIFAR-10 run.
  help                Show this message.

Common overrides:
  ROUNDS=20 ABLATION_ROUNDS=10 ./run_all.sh workflow
  NUM_CLIENTS=10 MODEL=convnet ROUNDS=20 ./run_all.sh compare_cifar
  ABLATION_ROUNDS=20 SEEDS=42 ALPHA_VALUES=1.3,1.7,2.0 ./run_all.sh core_heavytail
  CUDA_VISIBLE_DEVICES=0 FAST_DATA=on AMP=1 AMP_DTYPE=bf16 CHANNELS_LAST=1 EVAL_EVERY=5 ./run_all.sh alpha_ablation
  COMPARE_BATCH_SIZE=128 HEAVYTAIL_LOCAL_EPOCHS=2 ./run_all.sh alpha_ablation
EOF
}

main() {
  local module="${1:-help}"
  case "$module" in
    runall|workflow|quickcheck|setup|data|train|train_mnist|train_cifar|compare|compare_mnist|compare_cifar|ablation|ablation_noise|ablation_clients|plot|alpha_sanity|alpha_ablation|mac_compare|core_heavytail|paper_update|mechanism|lr_sweep|mac_sweep|heavy_tail_compare|paper_like_compare|help|new_sweep)
      "$module"
      ;;
    *)
      printf 'Unknown module: %s\n\n' "$module" >&2
      help >&2
      exit 2
      ;;
  esac
}

main "$@"
