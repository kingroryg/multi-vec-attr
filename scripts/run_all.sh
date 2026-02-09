#!/bin/bash
# Run all CDW experiments
# ======================

set -e  # Exit on error

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-results}"
NUM_GPUS="${NUM_GPUS:-1}"
SEED="${SEED:-42}"

echo "========================================"
echo "CDW Experiments - Full Run"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "Random seed: $SEED"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Experiment 1: Vendor Scaling
echo ""
echo "[1/8] Running Experiment 1: Vendor Scaling..."
python experiments/exp1_vendor_scaling.py \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --num-vendors 4 8 16 32 64 \
    --num-images 500 \
    --num-runs 3

# Experiment 2: Robustness
echo ""
echo "[2/8] Running Experiment 2: Robustness..."
python experiments/exp2_robustness.py \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --num-vendors 32 \
    --num-images 100

# Experiment 3: Image Quality
echo ""
echo "[3/8] Running Experiment 3: Image Quality..."
python experiments/exp3_image_quality.py \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --num-images 10000

# Experiment 4: Code Length Ablation
echo ""
echo "[4/8] Running Experiment 4: Code Length Ablation..."
python experiments/exp4_code_length.py \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    --code-lengths 16 32 64 128 256

# Experiment 5: Code Family Comparison
echo ""
echo "[5/8] Running Experiment 5: Code Family Comparison..."
python experiments/exp5_code_families.py \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"

# Experiment 6: Theory Validation
echo ""
echo "[6/8] Running Experiment 6: Theory Validation..."
python experiments/exp6_theory_validation.py \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"

# Experiment 7: Security Evaluation
echo ""
echo "[7/8] Running Experiment 7: Security Evaluation..."
python experiments/exp7_security.py \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"

# Experiment 8: Scalability
echo ""
echo "[8/8] Running Experiment 8: Scalability..."
python experiments/exp8_scalability.py \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"

# Generate paper figures
echo ""
echo "Generating paper figures..."
python analysis/paper_figures.py \
    --results-dir "$OUTPUT_DIR" \
    --output-dir "$OUTPUT_DIR/figures"

echo ""
echo "========================================"
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"
