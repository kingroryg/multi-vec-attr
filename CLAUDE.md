# Code-Division Watermarking (CDW) Experiments

## Overview

This repository contains production-grade experiments for validating **Code-Division Watermarking (CDW)**, a CDMA-based multi-vendor watermarking framework for diffusion models.

**Paper Title:** Code-Division Watermarking: Scalable Multi-Vendor Attribution for Diffusion Models

## Core Hypothesis

Multi-vendor watermarking can be formulated as a Code-Division Multiple Access (CDMA) problem. By assigning each vendor an orthogonal spreading code and embedding in the Fourier domain of the initial latent, we can:
1. Support N vendors with code length L (N ≤ L for Walsh-Hadamard)
2. Achieve identification accuracy predicted by Theorem 1
3. Inherit Tree-Ring's robustness properties
4. Provide formal guarantees on interference and capacity

## Directory Structure

```
cdma_exp/
├── CLAUDE.md                 # This file - experiment documentation
├── README.md                 # Quick start guide
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
│
├── cdw/                      # Core CDW implementation
│   ├── __init__.py
│   ├── codes.py              # Walsh-Hadamard, Gold code generation
│   ├── embedding.py          # Fourier-domain watermark embedding
│   ├── detection.py          # DDIM inversion + correlation detection
│   ├── ring_mask.py          # Ring geometry utilities
│   └── utils.py              # FFT, conjugate symmetry helpers
│
├── baselines/                # Baseline implementations
│   ├── __init__.py
│   ├── tree_ring.py          # Original Tree-Ring (single-key)
│   ├── ring_id.py            # RingID multi-key extension
│   └── random_codes.py       # Random (non-orthogonal) codes
│
├── attacks/                  # Image perturbation suite
│   ├── __init__.py
│   ├── compression.py        # JPEG compression at various qualities
│   ├── noise.py              # Gaussian, salt-pepper, speckle noise
│   ├── geometric.py          # Resize, crop, rotation
│   ├── color.py              # Brightness, contrast, saturation
│   ├── filtering.py          # Blur, sharpen, median filter
│   └── adversarial.py        # Targeted watermark removal attempts
│
├── metrics/                  # Evaluation metrics
│   ├── __init__.py
│   ├── identification.py     # Accuracy, precision, recall, confusion matrix
│   ├── quality.py            # FID, LPIPS, PSNR, SSIM
│   ├── theoretical.py        # Compare empirical vs theoretical predictions
│   └── statistical.py        # Confidence intervals, significance tests
│
├── experiments/              # Main experiment scripts
│   ├── __init__.py
│   ├── exp1_vendor_scaling.py      # Accuracy vs number of vendors
│   ├── exp2_robustness.py          # Robustness under attacks
│   ├── exp3_image_quality.py       # Quality metrics (FID, LPIPS)
│   ├── exp4_code_length.py         # Ablation on code length L
│   ├── exp5_code_families.py       # Walsh-Hadamard vs Gold vs Random
│   ├── exp6_theory_validation.py   # Empirical vs theoretical accuracy
│   ├── exp7_security.py            # Forgery, recovery, removal attacks
│   └── exp8_scalability.py         # Timing and memory benchmarks
│
├── analysis/                 # Result analysis and plotting
│   ├── __init__.py
│   ├── plots.py              # Publication-quality figures
│   ├── tables.py             # LaTeX table generation
│   ├── statistics.py         # Statistical analysis
│   └── paper_figures.py      # Generate all figures for paper
│
├── configs/                  # Experiment configurations
│   ├── default.yaml          # Default hyperparameters
│   ├── exp1_config.yaml      # Vendor scaling experiment
│   ├── exp2_config.yaml      # Robustness experiment
│   ├── exp3_config.yaml      # Quality experiment
│   ├── exp4_config.yaml      # Code length ablation
│   ├── exp5_config.yaml      # Code family comparison
│   ├── exp6_config.yaml      # Theory validation
│   ├── exp7_config.yaml      # Security evaluation
│   └── exp8_config.yaml      # Scalability benchmarks
│
├── scripts/                  # Shell scripts for running experiments
│   ├── setup_env.sh          # Environment setup
│   ├── download_data.sh      # Download COCO captions, models
│   ├── run_all.sh            # Run all experiments
│   ├── run_exp1.sh           # Run individual experiments
│   ├── run_exp2.sh
│   ├── ...
│   └── generate_paper.sh     # Generate all paper figures/tables
│
├── data/                     # Data directory (downloaded)
│   ├── prompts/              # Text prompts for generation
│   └── generated/            # Generated images (gitignored)
│
└── results/                  # Experiment results (gitignored)
    ├── exp1/
    ├── exp2/
    └── ...
```

## Experiments

### Experiment 1: Vendor Scaling (exp1_vendor_scaling.py)

**Goal:** Validate that CDW scales to many vendors with graceful accuracy degradation.

**Setup:**
- Model: Stable Diffusion v1.5
- Code length: L = 64 (Walsh-Hadamard)
- Number of vendors: N ∈ {4, 8, 16, 32, 64}
- Images per vendor: 500 (total 2000-32000 images)
- Prompts: Random sample from COCO captions

**Metrics:**
- Top-1 identification accuracy
- Top-3 identification accuracy
- Per-vendor precision/recall
- Confusion matrix

**Baselines:**
- Random codes (same L, no orthogonality)
- RingID (state-of-the-art multi-key)

**Expected Outcome:**
- CDW: >95% accuracy for N≤32, >90% for N=64
- Random codes: Degrades significantly as N increases
- RingID: Comparable to CDW but theoretically less principled

**Statistical Rigor:**
- 3 independent runs with different random seeds
- Report mean ± std
- Paired t-test for significance vs baselines

---

### Experiment 2: Robustness (exp2_robustness.py)

**Goal:** Measure watermark survival under realistic image perturbations.

**Setup:**
- Fixed: N = 32 vendors, L = 64, α = 0.1
- Images: 100 per vendor (3200 total)
- Apply each attack independently

**Attacks:**
| Attack | Parameters |
|--------|------------|
| JPEG | Q ∈ {90, 75, 50, 25} |
| Gaussian Noise | σ ∈ {0.01, 0.03, 0.05, 0.10} |
| Resize | scale ∈ {0.75, 0.50, 0.25} (then back to original) |
| Crop | keep ∈ {0.90, 0.75, 0.50} (center crop, then resize) |
| Rotation | θ ∈ {5°, 15°, 30°} (with padding) |
| Brightness | factor ∈ {0.8, 1.2} |
| Contrast | factor ∈ {0.8, 1.2} |
| Gaussian Blur | kernel ∈ {3, 5, 7} |
| Combined | JPEG-75 + Noise-0.03 + Resize-0.75 |

**Metrics:**
- Identification accuracy per attack
- Bit error rate (if applicable)
- Correlation strength distribution

**Baselines:**
- Tree-Ring (single-key, upper bound)
- RingID

**Expected Outcome:**
- JPEG-75: >90% accuracy
- JPEG-50: >80% accuracy
- Gaussian noise σ=0.05: >85% accuracy
- Resize 50%: >80% accuracy
- Severe attacks (JPEG-25, Resize-25%): graceful degradation

---

### Experiment 3: Image Quality (exp3_image_quality.py)

**Goal:** Verify watermarking doesn't degrade perceptual quality.

**Setup:**
- Generate 10,000 images with and without watermark (same seeds)
- Vary watermark strength: α ∈ {0.05, 0.10, 0.15, 0.20, 0.30}
- N = 32 vendors

**Metrics:**
| Metric | Description | Target |
|--------|-------------|--------|
| FID | Fréchet Inception Distance vs unwatermarked | <5 increase |
| LPIPS | Perceptual similarity to unwatermarked | <0.05 |
| PSNR | Peak signal-to-noise ratio | >35 dB |
| SSIM | Structural similarity | >0.95 |

**Analysis:**
- Quality vs accuracy tradeoff curve (Pareto frontier)
- Per-strength metrics table
- Visual examples at each strength level

**Expected Outcome:**
- α = 0.10: FID increase <2, LPIPS <0.02, PSNR >40dB
- Quality degrades gracefully with α
- Visually indistinguishable at recommended α = 0.10

---

### Experiment 4: Code Length Ablation (exp4_code_length.py)

**Goal:** Validate theoretical scaling with code length.

**Setup:**
- Fixed: N = 32 vendors
- Code lengths: L ∈ {16, 32, 64, 128, 256}
- For L < N, use subset of vendors
- Images: 200 per vendor

**Analysis:**
- Plot accuracy vs L
- Overlay theoretical prediction from Theorem 1
- Compute empirical noise parameter σ̂

**Expected Outcome:**
- Accuracy increases with L
- Empirical matches theoretical prediction
- Diminishing returns beyond L = 128

---

### Experiment 5: Code Family Comparison (exp5_code_families.py)

**Goal:** Compare Walsh-Hadamard, Gold, and random codes.

**Setup:**
- Walsh-Hadamard: L = 64 (supports 64 vendors)
- Gold: L = 63 (supports 65 vendors)
- Random: L = 64 (supports unlimited, but interference)
- Test N ∈ {16, 32, 64}

**Metrics:**
- Identification accuracy
- Cross-correlation distribution (empirical)
- Interference analysis

**Expected Outcome:**
- Walsh-Hadamard: Best for N ≤ L (perfect orthogonality)
- Gold: Slightly worse, but more vendors possible
- Random: Degrades significantly at high N

---

### Experiment 6: Theory Validation (exp6_theory_validation.py)

**Goal:** Verify Theorem 1 prediction matches empirical accuracy.

**Setup:**
- Estimate noise σ from clean extraction errors
- Compute theoretical P(correct) from Eq. (10)
- Compare with observed accuracy across conditions

**Analysis:**
- Scatter plot: predicted vs observed accuracy
- Regression line (should be y = x)
- Residual analysis

**Expected Outcome:**
- Strong correlation (R² > 0.95)
- Systematic bias, if any, indicates model misspecification
- Theory provides useful upper bound

---

### Experiment 7: Security Evaluation (exp7_security.py)

**Goal:** Validate security claims against realistic attacks.

**7a. Forgery Attack:**
- Adversary generates images with random codes
- Measure false positive rate (incorrectly attributed to real vendor)
- Expected: FPR < 0.01 for threshold τ chosen at 1% FPR on clean images

**7b. Code Recovery Attack:**
- Give adversary M ∈ {10, 100, 1000, 10000} watermarked images
- Adversary estimates code via averaging
- Measure: (1) code estimation error, (2) forgery success with estimated code
- Expected: Recovery requires M > 10,000 for reasonable success

**7c. Removal Attack:**
- Apply aggressive processing: heavy blur, JPEG-10, noise σ=0.2
- Measure: watermark detection rate, image quality after attack
- Expected: Removal destroys image quality (PSNR < 20dB)

---

### Experiment 8: Scalability (exp8_scalability.py)

**Goal:** Benchmark computational costs.

**Metrics:**
| Operation | Measurement |
|-----------|-------------|
| Code generation | Time for N=1000 Walsh-Hadamard codes |
| Embedding | Time per image (FFT + modulation + IFFT) |
| Detection | Time per image (DDIM inversion + N correlations) |
| Memory | Peak GPU memory during detection |

**Expected Outcome:**
- Embedding: <10ms overhead (negligible vs diffusion sampling)
- Detection: ~30s per image (dominated by DDIM inversion)
- Memory: <8GB (fits on consumer GPU)

---

## Hardware Requirements

**Minimum (for development):**
- GPU: NVIDIA RTX 3080 (10GB VRAM)
- RAM: 32GB
- Storage: 100GB SSD

**Recommended (for full experiments):**
- GPU: NVIDIA A100 (40GB VRAM) or 4x RTX 4090
- RAM: 128GB
- Storage: 500GB SSD

**Cloud Instances:**
- AWS: p4d.24xlarge (8x A100) or g5.4xlarge (1x A10G)
- GCP: a2-highgpu-1g (1x A100)
- Lambda Labs: 1x A100 instance

**Estimated Compute:**
| Experiment | Images | GPU Hours (A100) |
|------------|--------|------------------|
| Exp 1 | 32,000 | 40 |
| Exp 2 | 3,200 | 8 |
| Exp 3 | 20,000 | 25 |
| Exp 4 | 6,400 | 10 |
| Exp 5 | 9,600 | 15 |
| Exp 6 | 5,000 | 8 |
| Exp 7 | 15,000 | 20 |
| Exp 8 | 1,000 | 2 |
| **Total** | ~92,000 | **~128** |

With 4x A100, full experiments complete in ~32 hours.

---

## Key Hyperparameters

```yaml
# Model
model_id: "runwayml/stable-diffusion-v1-5"
scheduler: "ddim"
num_inference_steps: 50
guidance_scale: 7.5

# Watermark
code_length: 64                    # L
code_family: "walsh_hadamard"      # or "gold", "random"
watermark_strength: 0.1            # α
watermark_channel: 3               # Which latent channel to embed

# Ring geometry (in 64x64 latent space)
ring_r_min: 4                      # Inner radius (pixels)
ring_r_max: 20                     # Outer radius (pixels)
ring_num_rings: 8                  # Number of concentric rings

# Detection
detection_threshold: 0.3           # τ (calibrated on validation set)
ddim_inversion_steps: 50           # Same as generation

# Experiment
num_images_per_vendor: 500         # Statistical power
num_runs: 3                        # For confidence intervals
random_seed: 42                    # Reproducibility
```

---

## Evaluation Protocol

### Statistical Rigor

1. **Multiple Runs:** All experiments run 3x with different seeds
2. **Confidence Intervals:** Report mean ± 1.96*std (95% CI)
3. **Significance Tests:** Paired t-test for baseline comparisons, p < 0.05
4. **Effect Sizes:** Cohen's d for practical significance

### Reproducibility

1. **Fixed Seeds:** Document all random seeds
2. **Version Pinning:** Exact package versions in requirements.txt
3. **Checksums:** SHA256 of model weights and datasets
4. **Logging:** Full experiment logs saved to results/

### Honest Reporting

1. **Negative Results:** Report failures (e.g., attacks that break watermark)
2. **Limitations:** Document conditions where CDW underperforms
3. **Cherry-Picking Prevention:** Report all metrics, not just favorable ones
4. **Baseline Fairness:** Use official implementations, tune hyperparameters fairly

---

## Expected Results Summary

Based on theoretical analysis, we expect:

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
| Code Recovery | Fail | M<10,000 images |

---

## Failure Modes to Watch

1. **DDIM Inversion Failure:** Some images may not invert cleanly
   - Mitigation: Report inversion quality metrics, exclude severe failures

2. **Prompt Dependence:** Accuracy may vary by prompt type
   - Mitigation: Use diverse prompt set, report per-category results

3. **Model Dependence:** Results may not transfer to SDXL or other models
   - Mitigation: Acknowledge limitation, test on SD 2.1 if time permits

4. **Threshold Sensitivity:** Detection threshold τ affects TPR/FPR tradeoff
   - Mitigation: Report ROC curve, pick τ at fixed FPR

---

## Running the Experiments

### Quick Start

```bash
# 1. Setup environment
cd cdma_exp
./scripts/setup_env.sh

# 2. Download data and models
./scripts/download_data.sh

# 3. Run a single experiment (e.g., vendor scaling)
python experiments/exp1_vendor_scaling.py --config configs/exp1_config.yaml

# 4. Run all experiments
./scripts/run_all.sh

# 5. Generate paper figures
python analysis/paper_figures.py
```

### Cloud Deployment

```bash
# On cloud instance with GPU
git clone <repo>
cd cdma_exp
pip install -e .
./scripts/run_all.sh --gpus 4 --output-dir /data/results
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{munshi2025cdw,
  title={Code-Division Watermarking: Scalable Multi-Vendor Attribution for Diffusion Models},
  author={Munshi, Sarthak and ...},
  booktitle={...},
  year={2025}
}
```

---

## Contact

For questions about the experiments, please open an issue or contact [email].
