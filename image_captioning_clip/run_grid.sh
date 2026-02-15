#!/bin/bash
# Grid search: model_size x vocab_size
# Usage: bash run_grid.sh [variant]
#   e.g. bash run_grid.sh snow

set -e

VARIANT="${1:-snow}"
MODEL_SIZES=("base" "small" "tiny" "micro")
VOCAB_SIZES=(8000 4000 2000 1000)

echo "=== Grid Search: ${VARIANT} ==="
echo "  Model sizes: ${MODEL_SIZES[*]}"
echo "  Vocab sizes: ${VOCAB_SIZES[*]}"
echo "  Total runs: $(( ${#MODEL_SIZES[@]} * ${#VOCAB_SIZES[@]} ))"

# Step 0: Cache CLIP features (one-time, shared across all runs)
echo ""
echo "########################################"
echo "  Step 0: Caching CLIP features"
echo "########################################"
python cache_features.py --variant "$VARIANT" --split both

# Step 1: Train all tokenizers first (shared across model sizes)
echo ""
echo "########################################"
echo "  Step 1: Training tokenizers"
echo "########################################"
for VSIZE in "${VOCAB_SIZES[@]}"; do
    echo ""
    echo "--- Tokenizer: vocab_size=${VSIZE} ---"
    python tokenizer/train_tokenizer.py --variant "$VARIANT" --vocab_size "$VSIZE"
done

# Step 2: Train and evaluate all combinations
# Results are automatically appended to outputs/grid_results_{variant}.csv by train.py
echo ""
echo "########################################"
echo "  Step 2: Train & Evaluate grid"
echo "########################################"

for SIZE in "${MODEL_SIZES[@]}"; do
    for VSIZE in "${VOCAB_SIZES[@]}"; do
        echo ""
        echo "======================================"
        echo "  Training: ${VARIANT} / ${SIZE} / v${VSIZE}"
        echo "======================================"
        python train.py --variant "$VARIANT" --model_size "$SIZE" --vocab_size "$VSIZE" --use_cache --auto_batch

        # Determine checkpoint path
        RUN_NAME="run_${VARIANT}_${SIZE}"
        if [ "$VSIZE" -ne 8000 ]; then
            RUN_NAME="${RUN_NAME}_v${VSIZE}"
        fi
        CKPT="$(dirname "$(pwd)")/outputs/${RUN_NAME}/checkpoints/best.pt"

        if [ -f "$CKPT" ]; then
            echo ""
            echo "======================================"
            echo "  Evaluating: ${VARIANT} / ${SIZE} / v${VSIZE}"
            echo "======================================"
            python evaluate.py --variant "$VARIANT" --model_size "$SIZE" --vocab_size "$VSIZE" --checkpoint "$CKPT"
        else
            echo "WARNING: Checkpoint not found at $CKPT, skipping evaluation."
        fi
    done
done

RESULTS_CSV="$(dirname "$(pwd)")/outputs/grid_results_${VARIANT}.csv"
echo ""
echo "======================================"
echo "  Grid Search Complete: ${VARIANT}"
echo "======================================"
if [ -f "$RESULTS_CSV" ]; then
    echo ""
    echo "=== Results Summary ==="
    column -t -s',' "$RESULTS_CSV"
fi
