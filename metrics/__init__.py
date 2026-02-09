"""
Evaluation Metrics for CDW Experiments

Comprehensive metrics for watermark detection accuracy, image quality,
and theoretical validation.
"""

from .identification import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_per_vendor_metrics,
    compute_roc_curve,
    compute_auc,
)

from .quality import (
    compute_fid,
    compute_lpips,
    compute_psnr,
    compute_ssim,
    compute_all_quality_metrics,
)

from .theoretical import (
    theoretical_accuracy,
    estimate_noise_sigma,
    compare_theory_vs_empirical,
    compute_correlation_statistics,
)

from .statistical import (
    compute_confidence_interval,
    paired_t_test,
    compute_effect_size,
    bootstrap_confidence_interval,
)

__all__ = [
    # Identification
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_per_vendor_metrics",
    "compute_roc_curve",
    "compute_auc",
    # Quality
    "compute_fid",
    "compute_lpips",
    "compute_psnr",
    "compute_ssim",
    "compute_all_quality_metrics",
    # Theoretical
    "theoretical_accuracy",
    "estimate_noise_sigma",
    "compare_theory_vs_empirical",
    "compute_correlation_statistics",
    # Statistical
    "compute_confidence_interval",
    "paired_t_test",
    "compute_effect_size",
    "bootstrap_confidence_interval",
]
