# Code-Division Watermarking (CDW) Experiments

Production-grade experiments for validating Code-Division Watermarking for diffusion models.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Test](#quick-test)
4. [Running All Experiments](#running-all-experiments)
5. [Running Individual Experiments](#running-individual-experiments)
6. [Generating Paper Figures & Tables](#generating-paper-figures--tables)
7. [Expected Results](#expected-results)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

**Hardware Requirements:**
- GPU: NVIDIA GPU with at least 10GB VRAM (RTX 3080+ recommended)
- RAM: 32GB minimum
- Storage: 50GB free space

**Software Requirements:**
- Python 3.9+
- CUDA 11.8+ (for GPU acceleration)
- Git

---

## Installation

Run these commands in order:

```bash
# 1. Navigate to experiment directory
cd /Users/kingroryg/workspace/papers/cdma_exp

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install the cdma_exp package in editable mode
pip install -e .

# 6. Create data directories
mkdir -p data/prompts data/generated results

# 7. Download COCO captions for prompts (choose one method):

# Method A: Using curl
curl -L "https://raw.githubusercontent.com/tylin/coco-caption/master/annotations/captions_train2014.json" | \
  python -c "import json,sys; caps=json.load(sys.stdin); print('\n'.join([a['caption'] for a in caps['annotations'][:50000]]))" \
  > data/prompts/coco_captions.txt

# Method B: Create simple test prompts (for quick testing)
cat > data/prompts/coco_captions.txt << 'EOF'
A photo of a cat sitting on a windowsill
A beautiful sunset over the ocean
A city street at night with neon lights
A forest path in autumn with fallen leaves
A mountain landscape with snow-capped peaks
A still life painting of fruits on a table
A portrait of a person with dramatic lighting
An abstract colorful geometric pattern
A dog running through a grassy field
A cozy living room with a fireplace
EOF

# 8. Verify installation
python -c "from cdw import CDWEmbedder, CDWDetector; print('Installation successful!')"
```

---

## Quick Test

Before running full experiments, verify everything works:

```bash
# Quick test with minimal settings (takes ~5 minutes on GPU)
python experiments/exp1_vendor_scaling.py \
  --num-vendors 4 \
  --num-images 5 \
  --output-dir results/test \
  --seed 42

# Check the results
cat results/test/exp1_vendor_scaling/*/results.json | python -m json.tool
```

---

## Running All Experiments

### Option A: Run Everything at Once

```bash
# Full run - approximately 128 GPU-hours on A100
# Set environment variables as needed
export OUTPUT_DIR="results"
export SEED=42

./scripts/run_all.sh
```

### Option B: Run in Sequence (Recommended)

Run experiments one by one to monitor progress:

```bash
# Set output directory
export OUTPUT_DIR="results"
export SEED=42

# Experiment 1: Vendor Scaling (~40 GPU-hours)
echo "Starting Experiment 1: Vendor Scaling..."
python experiments/exp1_vendor_scaling.py \
  --output-dir $OUTPUT_DIR \
  --seed $SEED \
  --num-vendors 4 8 16 32 64 \
  --num-images 500 \
  --num-runs 3

# Experiment 2: Robustness (~8 GPU-hours)
echo "Starting Experiment 2: Robustness..."
python experiments/exp2_robustness.py \
  --output-dir $OUTPUT_DIR \
  --seed $SEED \
  --num-vendors 32 \
  --num-images 100

# Experiment 3: Image Quality (~25 GPU-hours)
echo "Starting Experiment 3: Image Quality..."
python experiments/exp3_image_quality.py \
  --output-dir $OUTPUT_DIR \
  --seed $SEED \
  --num-images 1000 \
  --num-vendors 32

# Experiment 4: Code Length Ablation (~10 GPU-hours)
echo "Starting Experiment 4: Code Length..."
python experiments/exp4_code_length.py \
  --output-dir $OUTPUT_DIR \
  --seed $SEED \
  --num-vendors 32 \
  --num-images 100

# Experiment 5: Code Family Comparison (~15 GPU-hours)
echo "Starting Experiment 5: Code Families..."
python experiments/exp5_code_families.py \
  --output-dir $OUTPUT_DIR \
  --seed $SEED \
  --num-images 100

# Experiment 6: Theory Validation (~8 GPU-hours)
echo "Starting Experiment 6: Theory Validation..."
python experiments/exp6_theory_validation.py \
  --output-dir $OUTPUT_DIR \
  --seed $SEED \
  --num-images 200

# Experiment 7: Security Evaluation (~20 GPU-hours)
echo "Starting Experiment 7: Security..."
python experiments/exp7_security.py \
  --output-dir $OUTPUT_DIR \
  --seed $SEED \
  --num-vendors 32

# Experiment 8: Scalability Benchmarks (~2 GPU-hours)
echo "Starting Experiment 8: Scalability..."
python experiments/exp8_scalability.py \
  --output-dir $OUTPUT_DIR \
  --seed $SEED \
  --num-images 50
```

### Option C: Reduced Settings for Faster Results

For quicker iteration (results in ~10 GPU-hours total):

```bash
export OUTPUT_DIR="results/fast"
export SEED=42

python experiments/exp1_vendor_scaling.py --output-dir $OUTPUT_DIR --seed $SEED --num-vendors 4 8 16 32 --num-images 50 --num-runs 1
python experiments/exp2_robustness.py --output-dir $OUTPUT_DIR --seed $SEED --num-vendors 16 --num-images 20
python experiments/exp3_image_quality.py --output-dir $OUTPUT_DIR --seed $SEED --num-images 100 --num-vendors 16
python experiments/exp4_code_length.py --output-dir $OUTPUT_DIR --seed $SEED --num-vendors 16 --num-images 50
python experiments/exp5_code_families.py --output-dir $OUTPUT_DIR --seed $SEED --num-images 50
python experiments/exp6_theory_validation.py --output-dir $OUTPUT_DIR --seed $SEED --num-images 100
python experiments/exp7_security.py --output-dir $OUTPUT_DIR --seed $SEED --num-vendors 16
python experiments/exp8_scalability.py --output-dir $OUTPUT_DIR --seed $SEED --num-images 20
```

---

## Running Individual Experiments

### Experiment 1: Vendor Scaling

Tests how identification accuracy changes with number of vendors.

```bash
python experiments/exp1_vendor_scaling.py \
  --num-vendors 4 8 16 32 64 \
  --num-images 500 \
  --num-runs 3 \
  --output-dir results \
  --seed 42
```

**Generates:** `results/exp1_vendor_scaling/<timestamp>/results.json`

### Experiment 2: Robustness

Tests watermark survival under various attacks.

```bash
python experiments/exp2_robustness.py \
  --num-vendors 32 \
  --num-images 100 \
  --output-dir results \
  --seed 42
```

**Generates:** `results/exp2_robustness/<timestamp>/results.json`

### Experiment 3: Image Quality

Measures FID, LPIPS, PSNR, SSIM across watermark strengths.

```bash
python experiments/exp3_image_quality.py \
  --num-images 1000 \
  --num-vendors 32 \
  --output-dir results \
  --seed 42
```

**Generates:** `results/exp3_image_quality/<timestamp>/results.json`

### Experiment 4: Code Length Ablation

Tests accuracy with different code lengths (L).

```bash
python experiments/exp4_code_length.py \
  --num-vendors 32 \
  --num-images 100 \
  --output-dir results \
  --seed 42
```

**Generates:** `results/exp4_code_length/<timestamp>/results.json`

### Experiment 5: Code Family Comparison

Compares Walsh-Hadamard vs Gold vs Random codes.

```bash
python experiments/exp5_code_families.py \
  --num-images 100 \
  --output-dir results \
  --seed 42
```

**Generates:** `results/exp5_code_families/<timestamp>/results.json`

### Experiment 6: Theory Validation

Validates Theorem 1 predictions match empirical results.

```bash
python experiments/exp6_theory_validation.py \
  --num-images 200 \
  --output-dir results \
  --seed 42
```

**Generates:** `results/exp6_theory_validation/<timestamp>/results.json`

### Experiment 7: Security Evaluation

Tests forgery, code recovery, and removal attacks.

```bash
python experiments/exp7_security.py \
  --num-vendors 32 \
  --output-dir results \
  --seed 42
```

**Generates:** `results/exp7_security/<timestamp>/results.json`

### Experiment 8: Scalability

Benchmarks timing and memory usage.

```bash
python experiments/exp8_scalability.py \
  --num-images 50 \
  --output-dir results \
  --seed 42
```

**Generates:** `results/exp8_scalability/<timestamp>/results.json`

---

## Generating Paper Figures & Tables

After running experiments, generate all figures and tables:

```bash
# Generate all figures and LaTeX tables
python -m analysis.paper_figures \
  --results-dir results \
  --output-dir results/figures

# Or use the module directly
python analysis/paper_figures.py \
  --results-dir results \
  --output-dir results/figures
```

**Output files:**

| File | Description | Paper Reference |
|------|-------------|-----------------|
| `fig1_vendor_scaling.pdf` | Accuracy vs N vendors | Figure 1 |
| `fig2_robustness_heatmap.pdf` | Attack robustness heatmap | Figure 2 |
| `fig3_quality_tradeoff.pdf` | Accuracy-quality Pareto | Figure 3 |
| `fig4_theory_validation.pdf` | Theory vs empirical scatter | Figure 4 |
| `fig5_code_length.pdf` | Accuracy vs code length | Figure 5 |
| `fig6_code_families.pdf` | Code family comparison | Figure 6 |
| `tables/table1_main_results.tex` | Main results table | Table 1 |
| `tables/table2_robustness.tex` | Robustness table | Table 2 |

### Generate Individual Figures

```bash
# Start Python
python

# Generate specific figures
from analysis.plots import *
import json

# Load results
with open("results/exp1_vendor_scaling/<timestamp>/results.json") as f:
    exp1 = json.load(f)

# Generate Figure 1
plot_accuracy_vs_vendors(exp1, output_path="fig1.pdf", show=False)
```

---

## Complete Workflow Summary

```bash
# === STEP 1: SETUP ===
cd /Users/kingroryg/workspace/papers/cdma_exp
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt && pip install -e .
mkdir -p data/prompts results

# === STEP 2: PREPARE DATA ===
# Create test prompts (or download COCO captions)
echo -e "A photo of a cat\nA sunset over ocean\nA city at night" > data/prompts/coco_captions.txt

# === STEP 3: QUICK TEST ===
python experiments/exp1_vendor_scaling.py --num-vendors 4 --num-images 5 --output-dir results/test

# === STEP 4: RUN ALL EXPERIMENTS ===
./scripts/run_all.sh

# === STEP 5: GENERATE FIGURES ===
python analysis/paper_figures.py --results-dir results --output-dir results/figures

# === STEP 6: CHECK OUTPUTS ===
ls -la results/figures/
```

---

## Expected Results

| Metric | Expected Value | Condition |
|--------|----------------|-----------|
| Identification Accuracy | >95% | N=32, L=64, clean |
| Identification Accuracy | >90% | N=64, L=64, clean |
| Accuracy under JPEG-50 | >80% | N=32, L=64 |
| Accuracy under Noise σ=0.05 | >85% | N=32, L=64 |
| FID Increase | <2 | α=0.1 |
| LPIPS | <0.02 | α=0.1 |
| Theory-Empirical R² | >0.95 | All conditions |
| Forgery FPR | <1% | Random codes |

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size or image count:
```bash
python experiments/exp1_vendor_scaling.py --num-images 50 --batch-size 1
```

### Model Download Issues

Pre-download the Stable Diffusion model:
```bash
python -c "from diffusers import StableDiffusionPipeline; StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')"
```

### Missing Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate
pip install lpips torchmetrics scikit-image
```

### Check GPU Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## Project Structure

```
cdma_exp/
├── cdw/                    # Core CDW implementation
│   ├── codes.py            # Walsh-Hadamard, Gold, Random codes
│   ├── embedding.py        # Fourier-domain watermark embedding
│   ├── detection.py        # DDIM inversion + correlation detection
│   ├── ring_mask.py        # Annular Fourier regions
│   └── utils.py            # Utilities
├── attacks/                # Image perturbation suite
│   ├── compression.py      # JPEG, WebP
│   ├── noise.py            # Gaussian, salt-pepper, speckle
│   ├── geometric.py        # Resize, crop, rotate
│   ├── color.py            # Brightness, contrast, saturation
│   ├── filtering.py        # Blur, sharpen, median
│   └── registry.py         # Unified attack interface
├── metrics/                # Evaluation metrics
│   ├── identification.py   # Accuracy, confusion matrix, ROC
│   ├── quality.py          # FID, LPIPS, PSNR, SSIM
│   ├── theoretical.py      # Theory vs empirical comparison
│   └── statistical.py      # Confidence intervals, t-tests
├── experiments/            # Experiment scripts (exp1-exp8)
├── analysis/               # Plotting and table generation
│   ├── plots.py            # Publication-quality figures
│   ├── tables.py           # LaTeX table generation
│   └── paper_figures.py    # Generate all paper figures
├── baselines/              # Baseline implementations
├── configs/                # YAML configurations
├── scripts/                # Shell scripts
├── data/                   # Data directory
├── results/                # Output directory
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
├── README.md               # This file
└── CLAUDE.md               # Detailed experiment documentation
```

---

## Citation

```bibtex
@inproceedings{munshi2025cdw,
  title={Code-Division Watermarking: Scalable Multi-Vendor Attribution for Diffusion Models},
  author={Munshi, Sarthak},
  booktitle={...},
  year={2025}
}
```

---

## Documentation

For comprehensive experiment documentation including theoretical background, expected failure modes, and detailed hyperparameter explanations, see [CLAUDE.md](CLAUDE.md).
