"""
Baseline Watermarking Methods

Implementations of baseline methods for comparison:
- Tree-Ring: Original single-key Fourier watermark
- RingID: Multi-key extension with discretization
- Random: Non-orthogonal random codes (ablation)
"""

from .tree_ring import TreeRingWatermark
from .random_codes import RandomCodeWatermark

__all__ = [
    "TreeRingWatermark",
    "RandomCodeWatermark",
]
