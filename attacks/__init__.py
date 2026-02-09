"""
Image Attack Suite for Watermark Robustness Testing

Comprehensive set of image perturbations to evaluate watermark survival
under realistic transformations.
"""

from .compression import jpeg_compress, webp_compress
from .noise import gaussian_noise, salt_pepper_noise, speckle_noise
from .geometric import resize_attack, crop_attack, rotate_attack
from .color import brightness_attack, contrast_attack, saturation_attack
from .filtering import gaussian_blur, median_filter, sharpen
from .combined import combined_attack, attack_pipeline

from .registry import (
    ATTACK_REGISTRY,
    get_attack,
    list_attacks,
    apply_attack,
    AttackConfig,
)

__all__ = [
    # Individual attacks
    "jpeg_compress",
    "webp_compress",
    "gaussian_noise",
    "salt_pepper_noise",
    "speckle_noise",
    "resize_attack",
    "crop_attack",
    "rotate_attack",
    "brightness_attack",
    "contrast_attack",
    "saturation_attack",
    "gaussian_blur",
    "median_filter",
    "sharpen",
    "combined_attack",
    "attack_pipeline",
    # Registry
    "ATTACK_REGISTRY",
    "get_attack",
    "list_attacks",
    "apply_attack",
    "AttackConfig",
]
