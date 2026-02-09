"""
Statistical Analysis Utilities

Confidence intervals, significance tests, and effect sizes.
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for mean.

    Args:
        data: List of observations
        confidence: Confidence level (default 95%)

    Returns:
        (mean, lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)

    # t-distribution critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_crit * std_err

    return float(mean), float(mean - margin), float(mean + margin)


def bootstrap_confidence_interval(
    data: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean",
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Args:
        data: List of observations
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        statistic: "mean" or "median"
        seed: Random seed

    Returns:
        (statistic_value, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    data = np.array(data)
    n = len(data)

    stat_fn = np.mean if statistic == "mean" else np.median
    observed = stat_fn(data)

    # Bootstrap
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_fn(sample))

    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return float(observed), float(lower), float(upper)


def paired_t_test(
    data1: List[float],
    data2: List[float],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Perform paired t-test.

    Args:
        data1: First set of observations
        data2: Second set of observations (paired with data1)
        alternative: "two-sided", "greater", or "less"

    Returns:
        Dict with t-statistic, p-value, and effect size
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    t_stat, p_value = stats.ttest_rel(data1, data2, alternative=alternative)

    # Effect size (Cohen's d for paired samples)
    diff = data1 - data2
    d = np.mean(diff) / np.std(diff, ddof=1)

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(d),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
    }


def independent_t_test(
    data1: List[float],
    data2: List[float],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Perform independent samples t-test.

    Args:
        data1: First set of observations
        data2: Second set of observations
        alternative: "two-sided", "greater", or "less"

    Returns:
        Dict with t-statistic, p-value, and effect size
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    t_stat, p_value = stats.ttest_ind(data1, data2, alternative=alternative)

    # Effect size (Cohen's d)
    n1, n2 = len(data1), len(data2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1)) /
        (n1 + n2 - 2)
    )
    d = (np.mean(data1) - np.mean(data2)) / pooled_std

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(d),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
    }


def compute_effect_size(
    data1: List[float],
    data2: List[float],
    paired: bool = False,
) -> Dict[str, float]:
    """
    Compute multiple effect size measures.

    Args:
        data1: First set of observations
        data2: Second set of observations
        paired: Whether data is paired

    Returns:
        Dict with various effect size measures
    """
    data1 = np.array(data1)
    data2 = np.array(data2)

    mean_diff = np.mean(data1) - np.mean(data2)

    if paired:
        diff = data1 - data2
        cohens_d = mean_diff / np.std(diff, ddof=1)
    else:
        n1, n2 = len(data1), len(data2)
        pooled_std = np.sqrt(
            ((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1)) /
            (n1 + n2 - 2)
        )
        cohens_d = mean_diff / pooled_std

    # Hedges' g (bias-corrected Cohen's d)
    n = len(data1) + len(data2)
    correction = 1 - 3 / (4 * n - 9)
    hedges_g = cohens_d * correction

    # Glass's delta (using control group std)
    glass_delta = mean_diff / np.std(data2, ddof=1)

    return {
        "mean_difference": float(mean_diff),
        "cohens_d": float(cohens_d),
        "hedges_g": float(hedges_g),
        "glass_delta": float(glass_delta),
        "interpretation": interpret_cohens_d(cohens_d),
    }


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compute_correlation(
    x: List[float],
    y: List[float],
) -> Dict[str, float]:
    """
    Compute correlation coefficient with significance test.

    Args:
        x, y: Data to correlate

    Returns:
        Dict with correlation coefficient and p-value
    """
    x = np.array(x)
    y = np.array(y)

    # Pearson correlation
    r, p_pearson = stats.pearsonr(x, y)

    # Spearman correlation (rank-based, more robust)
    rho, p_spearman = stats.spearmanr(x, y)

    return {
        "pearson_r": float(r),
        "pearson_p": float(p_pearson),
        "spearman_rho": float(rho),
        "spearman_p": float(p_spearman),
        "r_squared": float(r ** 2),
    }


def multiple_comparison_correction(
    p_values: List[float],
    method: str = "bonferroni",
) -> List[float]:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: List of p-values
        method: "bonferroni", "holm", or "fdr_bh"

    Returns:
        Corrected p-values
    """
    from scipy.stats import false_discovery_control

    p_values = np.array(p_values)
    n = len(p_values)

    if method == "bonferroni":
        return list(np.minimum(p_values * n, 1.0))
    elif method == "holm":
        # Holm-Bonferroni step-down
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        corrected = np.zeros(n)
        for i, p in enumerate(sorted_p):
            corrected[sorted_idx[i]] = min(p * (n - i), 1.0)
        return list(corrected)
    elif method == "fdr_bh":
        # Benjamini-Hochberg FDR
        try:
            return list(false_discovery_control(p_values, method='bh'))
        except:
            # Fallback to Bonferroni if scipy version doesn't support FDR
            return list(np.minimum(p_values * n, 1.0))
    else:
        raise ValueError(f"Unknown method: {method}")
