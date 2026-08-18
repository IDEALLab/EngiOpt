#!/bin/bash
# Evaluate LV-MMD metrics across multiple generator models and seeds.
# Run after generator sweeps complete.
#
# Usage: bash run_generator_sweep_eval.sh [problem_id]
# Default: beams2d

PROBLEM_ID="${1:-beams2d}"

# Generator types that have trained models on WandB
GEN_MODELS=("cgan_cnn_2d" "gan_cnn_2d" "diffusion_2d_cond" "vqgan")

# Seeds to evaluate (match sweep config)
SEEDS=(1 2 3 4 5 6 7 8 9 10)

# LVAE seeds to average over
LVAE_SEEDS="1,2,3"

echo "=== LV-MMD Generator Sweep Evaluation ==="
echo "Problem: ${PROBLEM_ID}"
echo "Generators: ${GEN_MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"

for MODEL in "${GEN_MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- ${MODEL} seed=${SEED} ---"
        python engiopt/evaluate_lv_only.py \
            --problem-id "${PROBLEM_ID}" \
            --gen-model "${MODEL}" \
            --gen-seed "${SEED}" \
            --n-samples 50 \
            --lvae-seeds "${LVAE_SEEDS}" \
            --log-to-wandb \
            2>&1 || echo "  SKIPPED (model not found)"
    done
done

echo ""
echo "=== Done ==="
