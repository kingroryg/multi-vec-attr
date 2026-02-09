"""
Theoretical Validation Metrics

Compare empirical results with theoretical predictions from the paper.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple


def theoretical_accuracy(
    alpha: float,
    sigma: float,
    code_length: int,
    num_vendors: int,
) -> float:
    """
    Compute theoretical identification accuracy from Theorem 1.

    P(correct) = [Phi(alpha * sqrt(L) / sigma)]^(N-1)

    Args:
        alpha: Watermark strength
        sigma: Noise standard deviation
        code_length: L (length of spreading code)
        num_vendors: N (number of vendors)

    Returns:
        Predicted accuracy
    """
    if sigma <= 0:
        return 1.0 if num_vendors <= 1 else 0.0

    snr = alpha * np.sqrt(code_length) / sigma
    prob_pairwise = norm.cdf(snr)
    accuracy = prob_pairwise ** max(num_vendors - 1, 0)

    return accuracy


def theoretical_accuracy_with_interference(
    alpha: float,
    sigma: float,
    code_length: int,
    num_vendors: int,
    cross_correlation: float = 0.0,
) -> float:
    """
    Theoretical accuracy accounting for code cross-correlation.

    For non-orthogonal codes (e.g., Gold codes), cross-correlation
    introduces additional interference.

    Args:
        alpha: Watermark strength
        sigma: Noise standard deviation
        code_length: L
        num_vendors: N
        cross_correlation: Maximum |c_i^T c_j| / L for i != j

    Returns:
        Predicted accuracy
    """
    if sigma <= 0:
        return 1.0 if num_vendors <= 1 else 0.0

    # Effective signal is reduced by cross-correlation interference
    effective_alpha = alpha * (1 - cross_correlation)

    # Noise is increased by interference from other vendors
    interference_variance = (num_vendors - 1) * (alpha * cross_correlation) ** 2
    effective_sigma = np.sqrt(sigma ** 2 + interference_variance)

    return theoretical_accuracy(effective_alpha, effective_sigma, code_length, num_vendors)


def estimate_noise_sigma(
    correlations_or_signals: np.ndarray,
    codes: np.ndarray,
    alpha: float = None,
    vendor_ids: List[int] = None,
) -> float:
    """
    Estimate noise parameter sigma from extraction errors or correlations.

    Can be called in two ways:
    1. estimate_noise_sigma(signals, codes, alpha, vendor_ids) - full version
    2. estimate_noise_sigma(correlations, codes) - simplified from correlations

    Args:
        correlations_or_signals: Either:
            - Array of shape (num_samples, code_length) for extracted signals
            - Array of shape (num_samples, num_vendors) for correlations
        codes: Array of shape (num_vendors, code_length)
        alpha: Watermark strength used for embedding (optional)
        vendor_ids: True vendor ID for each sample (optional)

    Returns:
        Estimated sigma
    """
    num_samples = correlations_or_signals.shape[0]
    num_features = correlations_or_signals.shape[1]
    num_vendors, code_length = codes.shape

    # Determine which mode we're in based on array shape
    if num_features == num_vendors and alpha is None:
        # Simplified mode: estimate from correlation variance
        # For correct detection, correlation with true vendor should be high
        # The noise in correlations reflects sigma
        correlations = correlations_or_signals

        # Get off-diagonal correlations (incorrect vendors)
        # These should be zero-mean with variance proportional to sigma^2/L
        all_incorrect = []
        for i in range(num_samples):
            max_idx = np.argmax(correlations[i])
            for j in range(num_vendors):
                if j != max_idx:
                    all_incorrect.append(correlations[i, j])

        if all_incorrect:
            # Variance of incorrect correlations ≈ sigma^2 / L
            sigma_sq_over_L = np.var(all_incorrect)
            sigma = np.sqrt(sigma_sq_over_L * code_length)
        else:
            sigma = 0.1  # Default fallback

        return float(sigma)

    else:
        # Full mode: compute from extraction errors
        if alpha is None or vendor_ids is None:
            raise ValueError("alpha and vendor_ids required for signal-based estimation")

        errors = []
        for signal, vid in zip(correlations_or_signals, vendor_ids):
            expected = alpha * codes[vid]
            error = signal - expected
            errors.append(error)

        errors = np.array(errors)
        sigma = np.std(errors)

        return float(sigma)


def compare_theory_vs_empirical(
    empirical_accuracy: float,
    alpha: float,
    estimated_sigma: float,
    code_length: int,
    num_vendors: int,
) -> Dict[str, float]:
    """
    Compare empirical accuracy with theoretical prediction.

    Args:
        empirical_accuracy: Observed accuracy
        alpha: Watermark strength
        estimated_sigma: Estimated noise sigma
        code_length: L
        num_vendors: N

    Returns:
        Dict with theoretical prediction, empirical result, and difference
    """
    predicted = theoretical_accuracy(alpha, estimated_sigma, code_length, num_vendors)

    return {
        "theoretical": predicted,
        "empirical": empirical_accuracy,
        "absolute_error": abs(predicted - empirical_accuracy),
        "relative_error": abs(predicted - empirical_accuracy) / max(empirical_accuracy, 1e-8),
        "alpha": alpha,
        "sigma": estimated_sigma,
        "snr": alpha * np.sqrt(code_length) / estimated_sigma if estimated_sigma > 0 else float('inf'),
        "code_length": code_length,
        "num_vendors": num_vendors,
    }


def compute_correlation_statistics(
    correlations: np.ndarray,
    vendor_ids: List[int],
) -> Dict[str, float]:
    """
    Compute statistics of correlation values for analysis.

    Args:
        correlations: Array of shape (num_samples, num_vendors)
        vendor_ids: True vendor ID for each sample

    Returns:
        Dict with correlation statistics
    """
    num_samples, num_vendors = correlations.shape

    # Correlations with correct vendor
    correct_corrs = [correlations[i, v] for i, v in enumerate(vendor_ids)]

    # Correlations with incorrect vendors
    incorrect_corrs = []
    for i, v in enumerate(vendor_ids):
        for j in range(num_vendors):
            if j != v:
                incorrect_corrs.append(correlations[i, j])

    return {
        "correct_mean": float(np.mean(correct_corrs)),
        "correct_std": float(np.std(correct_corrs)),
        "correct_min": float(np.min(correct_corrs)),
        "correct_max": float(np.max(correct_corrs)),
        "incorrect_mean": float(np.mean(incorrect_corrs)),
        "incorrect_std": float(np.std(incorrect_corrs)),
        "incorrect_min": float(np.min(incorrect_corrs)),
        "incorrect_max": float(np.max(incorrect_corrs)),
        "separation": float(np.mean(correct_corrs) - np.mean(incorrect_corrs)),
        "d_prime": float(
            (np.mean(correct_corrs) - np.mean(incorrect_corrs)) /
            np.sqrt(0.5 * (np.var(correct_corrs) + np.var(incorrect_corrs)))
        ),
    }


def theoretical_capacity(
    code_family: str,
    code_length: int,
) -> int:
    """
    Compute theoretical capacity (max vendors) for a code family.

    Args:
        code_family: "walsh_hadamard", "gold", or "random"
        code_length: L

    Returns:
        Maximum number of distinguishable vendors
    """
    if code_family == "walsh_hadamard":
        # Exactly L orthogonal codes
        return code_length
    elif code_family == "gold":
        # L + 2 codes for L = 2^n - 1
        return code_length + 2
    else:
        # Random codes: theoretically unlimited but interference grows
        # Practical limit depends on acceptable accuracy
        return code_length * 2  # Rough heuristic


def compute_required_code_length(
    num_vendors: int,
    target_accuracy: float,
    alpha: float,
    sigma: float,
) -> int:
    """
    Compute required code length to achieve target accuracy.

    From Corollary 1: L = O((sigma^2 / alpha^2) * log^2(N))

    Args:
        num_vendors: N
        target_accuracy: Target identification accuracy
        alpha: Watermark strength
        sigma: Expected noise level

    Returns:
        Required code length (rounded up to power of 2)
    """
    # From Theorem 1: accuracy = Phi(alpha * sqrt(L) / sigma)^(N-1)
    # Invert to find required SNR for target accuracy

    if target_accuracy >= 1.0:
        target_accuracy = 0.999  # Numerical stability

    # Required pairwise probability
    prob_pairwise = target_accuracy ** (1 / (num_vendors - 1)) if num_vendors > 1 else 1.0

    # Required SNR
    required_snr = norm.ppf(prob_pairwise)

    # Required L
    required_L = (required_snr * sigma / alpha) ** 2

    # Round up to nearest power of 2
    L = 1
    while L < required_L:
        L *= 2

    return L
