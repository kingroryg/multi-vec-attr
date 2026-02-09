"""
Random Codes Baseline

Non-orthogonal random codes for comparison with CDW's orthogonal codes.
This ablation demonstrates the importance of orthogonality for multi-vendor
watermarking.
"""

import numpy as np
from typing import Optional


class RandomCodeWatermark:
    """
    Random (non-orthogonal) spreading codes baseline.

    These codes have expected cross-correlation of zero but high variance,
    leading to interference at high vendor counts.
    """

    def __init__(
        self,
        num_vendors: int,
        code_length: int,
        seed: Optional[int] = None
    ):
        """
        Initialize random codes.

        Args:
            num_vendors: Number of vendors
            code_length: Length of spreading codes
            seed: Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)
        self.codes = rng.choice([-1, 1], size=(num_vendors, code_length)).astype(np.float32)
        self.num_vendors = num_vendors
        self.code_length = code_length

    def get_codes(self) -> np.ndarray:
        """Return the code matrix."""
        return self.codes

    def get_code(self, vendor_id: int) -> np.ndarray:
        """Return a specific vendor's code."""
        return self.codes[vendor_id]

    def compute_cross_correlations(self) -> dict:
        """
        Compute cross-correlation statistics.

        Returns:
            Dict with correlation statistics
        """
        gram = self.codes @ self.codes.T

        # Off-diagonal elements
        mask = ~np.eye(self.num_vendors, dtype=bool)
        cross_corr = gram[mask] / self.code_length

        return {
            "max_cross_correlation": float(np.max(np.abs(cross_corr))),
            "mean_cross_correlation": float(np.mean(np.abs(cross_corr))),
            "std_cross_correlation": float(np.std(cross_corr)),
        }
