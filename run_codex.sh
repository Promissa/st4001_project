#!/usr/bin/env bash
set -euo pipefail

# Codex-prioritized experiment runner for the repaired ST4001 code.
#
# Default behavior is help only. Run explicit stages, for example:
#   ./run_codex.sh sanity
#   ./run_codex.sh core
#   ./run_codex.sh beta2
#
# Priority:
#   1. sanity          cheap pipeline check
#   2. alpha           core proposal-aligned heavy-tail evidence
#   3. mac             robust pre-processing comparison
#   4. mechanism       diagnostics for adaptive-step explanation
#   5. beta2           Adam beta2 sensitivity
#   6. mac_sweep       optional MAC k/gamma sensitivity
#   7. legacy          older CIFAR comparison refresh
#   8. build_pdf       rebuild template/main.pdf

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-.cache/matplotlib}"
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR"

# Hardware/runtime defaults for Ryzen 9 9950X + RTX 4090.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export AMP="${AMP:-1}"
export AMP_DTYPE="${AMP_DTYPE:-bf16}"
export CHANNELS_LAST="${CHANNELS_LAST:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"

# Final CIFAR-10 / ResNet-18 heavy-tail setting.
export DATASET="${DATASET:-cifar10}"
export MODEL="${MODEL:-resnet18}"
export NUM_CLIENTS="${NUM_CLIENTS:-100}"
export ABLATION_ROUNDS="${ABLATION_ROUNDS:-100}"
export HEAVYTAIL_LOCAL_EPOCHS="${HEAVYTAIL_LOCAL_EPOCHS:-1}"
export COMPARE_BATCH_SIZE="${COMPARE_BATCH_SIZE:-64}"
export SERVER_LR="${SERVER_LR:-0.01}"
export LOCAL_LR="${LOCAL_LR:-0.01}"
export MOMENTUM="${MOMENTUM:-0.9}"
export BETA2="${BETA2:-0.999}"
export NOISE_SCALE="${NOISE_SCALE:-0.05}"
export DIR_CONC="${DIR_CONC:-0.1}"
export ALPHA_VALUES="${ALPHA_VALUES:-1.1,1.3,1.5,1.7,1.9,2.0}"
export MAC_ALPHA="${MAC_ALPHA:-1.3}"
export MAC_CLIP="${MAC_CLIP:-3.0}"
export SEEDS="${SEEDS:-42,43,44}"
export METHODS="${METHODS:-fedavg,fedavgm,adagrad_ota,adam_ota}"
export HEAVYTAIL_DIR="${HEAVYTAIL_DIR:-results/heavytail}"
export FIGURE_DIR="${FIGURE_DIR:-results/figures}"
export GENERATED_TEX_DIR="${GENERATED_TEX_DIR:-template/data/generated}"
export MAX_EVAL_LOSS="${MAX_EVAL_LOSS:-1e6}"

# Keep paper text aligned with "evaluation after each communication round".
# If this is too slow, set EVAL_EVERY=5 and update the paper wording.
export EVAL_EVERY="${EVAL_EVERY:-1}"

run_logged() {
  local log_file="$1"
  shift
  printf '\n==> %s\n' "$*"
  "$@" 2>&1 | tee "$log_file"
}

show_config() {
  cat <<EOF
Current Codex experiment defaults:
  device: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, AMP=$AMP, AMP_DTYPE=$AMP_DTYPE, CHANNELS_LAST=$CHANNELS_LAST
  core:   DATASET=$DATASET, MODEL=$MODEL, NUM_CLIENTS=$NUM_CLIENTS, ROUNDS=$ABLATION_ROUNDS
  FL:     local_epochs=$HEAVYTAIL_LOCAL_EPOCHS, batch=$COMPARE_BATCH_SIZE, server_lr=$SERVER_LR, local_lr=$LOCAL_LR
  Adam:   beta1/momentum=$MOMENTUM, beta2=$BETA2
  noise:  alpha_values=$ALPHA_VALUES, gamma=$NOISE_SCALE, MAC alpha=$MAC_ALPHA, MAC k=$MAC_CLIP
  data:   Dirichlet concentration=$DIR_CONC, seeds=$SEEDS
  eval:   EVAL_EVERY=$EVAL_EVERY, EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE
  guard:  MAX_EVAL_LOSS=$MAX_EVAL_LOSS
  logs:   $LOG_DIR/
EOF
}

setup() {
  run_logged "$LOG_DIR/00_setup.log" uv sync
}

data() {
  run_logged "$LOG_DIR/00_data.log" ./run_all.sh data
}

sanity() {
  # MNIST sanity must not use the CIFAR-only fast loader.
  run_logged "$LOG_DIR/00_alpha_sanity.log" env \
    FAST_DATA=auto \
    EVAL_EVERY=1 \
    EVAL_BATCH_SIZE=1024 \
    SANITY_ROUNDS="${SANITY_ROUNDS:-1}" \
    ./run_all.sh alpha_sanity
}

alpha() {
  run_logged "$LOG_DIR/01_alpha_ablation.log" env \
    FAST_DATA=on \
    ./run_all.sh alpha_ablation
}

paper_update_alpha() {
  run_logged "$LOG_DIR/02_alpha_plot.log" uv run -m src.experiments.plot \
    --alpha_ablation "$HEAVYTAIL_DIR/alpha_ablation_${DATASET}_${MODEL}.json" \
    --out_dir "$FIGURE_DIR"

  run_logged "$LOG_DIR/02_alpha_report.log" uv run -m src.experiments.report \
    --alpha_ablation "$HEAVYTAIL_DIR/alpha_ablation_${DATASET}_${MODEL}.json" \
    --out_dir "$GENERATED_TEX_DIR"
}

mac() {
  run_logged "$LOG_DIR/03_mac_compare.log" env \
    FAST_DATA=on \
    ./run_all.sh mac_compare
}

paper_update_mac() {
  run_logged "$LOG_DIR/04_mac_plot.log" uv run -m src.experiments.plot \
    --mac_compare "$HEAVYTAIL_DIR/mac_compare_${DATASET}_${MODEL}_alpha${MAC_ALPHA}.json" \
    --out_dir "$FIGURE_DIR"

  run_logged "$LOG_DIR/04_mac_report.log" uv run -m src.experiments.report \
    --mac_compare "$HEAVYTAIL_DIR/mac_compare_${DATASET}_${MODEL}_alpha${MAC_ALPHA}.json" \
    --out_dir "$GENERATED_TEX_DIR"
}

mechanism() {
  run_logged "$LOG_DIR/05_mechanism.log" env \
    FAST_DATA=on \
    MECH_ROUNDS="${MECH_ROUNDS:-50}" \
    MECH_ALPHAS="${MECH_ALPHAS:-1.3}" \
    MECH_SEEDS="${MECH_SEEDS:-42}" \
    ./run_all.sh mechanism
}

beta2() {
  local beta
  for beta in ${BETA2_SWEEP:-0.9 0.99 0.999}; do
    run_logged "$LOG_DIR/06_beta2_${beta}.log" uv run -m src.experiments.heavy_tail --study alpha \
      --dataset cifar10 \
      --model resnet18 \
      --rounds "${BETA2_ROUNDS:-100}" \
      --num_clients 100 \
      --local_epochs 1 \
      --batch_size 64 \
      --server_lr 0.01 \
      --local_lr 0.01 \
      --momentum 0.9 \
      --beta2 "$beta" \
      --noise_scale 0.05 \
      --alphas "${BETA2_ALPHAS:-1.3,2.0}" \
      --seeds "$SEEDS" \
      --methods adam_ota \
      --dir_conc 0.1 \
      --out_dir "results/heavytail/beta2_${beta}" \
      --max_eval_loss "$MAX_EVAL_LOSS" \
      --amp \
      --amp_dtype bf16 \
      --channels_last \
      --fast_data on \
      --eval_every "$EVAL_EVERY" \
      --eval_batch_size "$EVAL_BATCH_SIZE"
  done
}

mac_sweep() {
  run_logged "$LOG_DIR/07_mac_sweep.log" env \
    FAST_DATA=on \
    ABLATION_ROUNDS="${MAC_SWEEP_ROUNDS:-50}" \
    SEEDS="${MAC_SWEEP_SEEDS:-42}" \
    METHODS="${MAC_SWEEP_METHODS:-adagrad_ota,adam_ota}" \
    MAC_K_SWEEP="${MAC_K_SWEEP:-0,1,3,5}" \
    MAC_GAMMA_SWEEP="${MAC_GAMMA_SWEEP:-0.05,0.1,0.2}" \
    ./run_all.sh mac_sweep
}

legacy() {
  run_logged "$LOG_DIR/08_compare_cifar.log" env \
    FAST_DATA=on \
    ROUNDS="${LEGACY_ROUNDS:-200}" \
    LOCAL_EPOCHS="${LEGACY_LOCAL_EPOCHS:-5}" \
    COMPARE_SERVER_LR="${COMPARE_SERVER_LR:-0.1}" \
    COMPARE_BATCH_SIZE="$COMPARE_BATCH_SIZE" \
    ./run_all.sh compare_cifar

  run_logged "$LOG_DIR/09_legacy_plot.log" ./run_all.sh plot
}

build_pdf() {
  run_logged "$LOG_DIR/10_xelatex_1.log" bash -lc "cd template && xelatex -interaction=nonstopmode -halt-on-error main.tex"
  run_logged "$LOG_DIR/10_bibtex.log" bash -lc "cd template && bibtex main"
  run_logged "$LOG_DIR/10_xelatex_2.log" bash -lc "cd template && xelatex -interaction=nonstopmode -halt-on-error main.tex"
  run_logged "$LOG_DIR/10_xelatex_3.log" bash -lc "cd template && xelatex -interaction=nonstopmode -halt-on-error main.tex"
}

core() {
  show_config
  sanity
  alpha
  paper_update_alpha
  mac
  paper_update_mac
  mechanism
}

all() {
  show_config
  setup
  data
  core
  beta2
  mac_sweep
  legacy
  build_pdf
}

help() {
  show_config
  cat <<'EOF'

Usage:
  ./run_codex.sh <stage>

Stages:
  help                Show this message.
  config              Print current defaults.
  setup               uv sync.
  data                Prepare/download datasets.
  sanity              One-round MNIST smoke test. Uses FAST_DATA=auto.
  alpha               Full CIFAR alpha-stable tail-index ablation.
  paper_alpha         Generate tables/figures after alpha.
  mac                 MAC vs no-MAC comparison at alpha=1.3.
  paper_mac           Generate tables/figures after MAC.
  mechanism           Short diagnostic run and mechanism plot.
  beta2               Adam beta2 sensitivity over 0.9, 0.99, 0.999.
  mac_sweep           Optional smaller MAC k/gamma sweep.
  legacy              Optional older CIFAR comparison and legacy plot.
  build_pdf           Rebuild template/main.pdf.
  core                Priority run: sanity -> alpha -> paper -> MAC -> paper -> mechanism.
  all                 setup -> data -> core -> beta2 -> mac_sweep -> legacy -> build_pdf.

Recommended sequence for the 3-day deadline:
  ./run_codex.sh sanity
  ./run_codex.sh alpha
  ./run_codex.sh paper_alpha
  ./run_codex.sh mac
  ./run_codex.sh paper_mac
  ./run_codex.sh mechanism
  ./run_codex.sh beta2

Useful overrides:
  EVAL_EVERY=5 ./run_codex.sh alpha
  SEEDS=42 ALPHA_VALUES=1.3,2.0 ./run_codex.sh alpha
  BETA2_SWEEP="0.99 0.999" ./run_codex.sh beta2
  MAC_SWEEP_ROUNDS=30 ./run_codex.sh mac_sweep
EOF
}

main() {
  local stage="${1:-help}"
  case "$stage" in
    help) help ;;
    config) show_config ;;
    setup) setup ;;
    data) data ;;
    sanity) sanity ;;
    alpha) alpha ;;
    paper_alpha) paper_update_alpha ;;
    mac) mac ;;
    paper_mac) paper_update_mac ;;
    mechanism) mechanism ;;
    beta2) beta2 ;;
    mac_sweep) mac_sweep ;;
    legacy) legacy ;;
    build_pdf) build_pdf ;;
    core) core ;;
    all) all ;;
    *)
      printf 'Unknown stage: %s\n\n' "$stage" >&2
      help >&2
      exit 2
      ;;
  esac
}

main "$@"
