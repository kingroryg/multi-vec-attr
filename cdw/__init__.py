"""
Code-Division Watermarking (CDW) for Diffusion Models

A CDMA-based multi-vendor watermarking framework that embeds orthogonal
spreading codes in the Fourier domain of diffusion model latents.
"""

from .codes import (
    generate_walsh_hadamard_codes,
    generate_gold_codes,
    generate_random_codes,
    generate_codes,
    compute_code_statistics,
    get_code_properties,
    CodeFamily,
)
from .embedding import CDWEmbedder
from .detection import CDWDetector
from .ring_mask import RingMask

__version__ = "0.1.0"
__all__ = [
    "generate_walsh_hadamard_codes",
    "generate_gold_codes",
    "generate_random_codes",
    "generate_codes",
    "compute_code_statistics",
    "get_code_properties",
    "CodeFamily",
    "CDWEmbedder",
    "CDWDetector",
    "RingMask",
]
