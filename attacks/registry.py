"""
Attack Registry

Centralized registry for all attacks with unified interface.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Callable, Union, Optional
import torch
import numpy as np
from PIL import Image

from .compression import jpeg_compress, webp_compress
from .noise import gaussian_noise, salt_pepper_noise, speckle_noise, uniform_noise
from .geometric import resize_attack, crop_attack, rotate_attack, flip_attack
from .color import brightness_attack, contrast_attack, saturation_attack
from .filtering import gaussian_blur, median_filter, sharpen, edge_enhance
from .combined import combined_attack, social_media_attack, attack_pipeline


@dataclass
class AttackConfig:
    """Configuration for a single attack."""
    name: str
    function: Callable
    default_params: Dict[str, Any] = field(default_factory=dict)
    param_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    description: str = ""
    category: str = "other"


# Global attack registry
ATTACK_REGISTRY: Dict[str, AttackConfig] = {}


def register_attack(
    name: str,
    function: Callable,
    default_params: Dict[str, Any] = None,
    param_ranges: Dict[str, List[Any]] = None,
    description: str = "",
    category: str = "other",
):
    """Register an attack in the global registry."""
    ATTACK_REGISTRY[name] = AttackConfig(
        name=name,
        function=function,
        default_params=default_params or {},
        param_ranges=param_ranges or {},
        description=description,
        category=category,
    )


# Register all attacks
register_attack(
    "jpeg",
    jpeg_compress,
    default_params={"quality": 75},
    param_ranges={"quality": [90, 75, 50, 25]},
    description="JPEG compression",
    category="compression",
)

register_attack(
    "webp",
    webp_compress,
    default_params={"quality": 75},
    param_ranges={"quality": [90, 75, 50, 25]},
    description="WebP compression",
    category="compression",
)

register_attack(
    "gaussian_noise",
    gaussian_noise,
    default_params={"sigma": 0.05},
    param_ranges={"sigma": [0.01, 0.03, 0.05, 0.10]},
    description="Additive Gaussian noise",
    category="noise",
)

register_attack(
    "salt_pepper",
    salt_pepper_noise,
    default_params={"prob": 0.05},
    param_ranges={"prob": [0.01, 0.03, 0.05, 0.10]},
    description="Salt and pepper noise",
    category="noise",
)

register_attack(
    "speckle",
    speckle_noise,
    default_params={"sigma": 0.1},
    param_ranges={"sigma": [0.05, 0.1, 0.2]},
    description="Multiplicative speckle noise",
    category="noise",
)

register_attack(
    "resize",
    resize_attack,
    default_params={"scale": 0.5, "restore_size": True},
    param_ranges={"scale": [0.75, 0.50, 0.25]},
    description="Downscale then upscale",
    category="geometric",
)

register_attack(
    "crop",
    crop_attack,
    default_params={"keep_ratio": 0.75, "restore_size": True},
    param_ranges={"keep_ratio": [0.90, 0.75, 0.50]},
    description="Center crop then resize",
    category="geometric",
)

register_attack(
    "rotate",
    rotate_attack,
    default_params={"angle": 15},
    param_ranges={"angle": [5, 15, 30, 45]},
    description="Rotation with padding",
    category="geometric",
)

register_attack(
    "flip_h",
    flip_attack,
    default_params={"horizontal": True},
    description="Horizontal flip",
    category="geometric",
)

register_attack(
    "flip_v",
    flip_attack,
    default_params={"horizontal": False},
    description="Vertical flip",
    category="geometric",
)

register_attack(
    "brightness",
    brightness_attack,
    default_params={"factor": 1.2},
    param_ranges={"factor": [0.7, 0.8, 1.2, 1.3]},
    description="Brightness adjustment",
    category="color",
)

register_attack(
    "contrast",
    contrast_attack,
    default_params={"factor": 1.2},
    param_ranges={"factor": [0.7, 0.8, 1.2, 1.3]},
    description="Contrast adjustment",
    category="color",
)

register_attack(
    "saturation",
    saturation_attack,
    default_params={"factor": 1.5},
    param_ranges={"factor": [0.5, 0.8, 1.2, 1.5]},
    description="Saturation adjustment",
    category="color",
)

register_attack(
    "blur",
    gaussian_blur,
    default_params={"kernel_size": 5, "sigma": 1.0},
    param_ranges={"sigma": [0.5, 1.0, 2.0, 3.0]},
    description="Gaussian blur",
    category="filtering",
)

register_attack(
    "median",
    median_filter,
    default_params={"kernel_size": 3},
    param_ranges={"kernel_size": [3, 5, 7]},
    description="Median filter",
    category="filtering",
)

register_attack(
    "sharpen",
    sharpen,
    default_params={"factor": 2.0},
    param_ranges={"factor": [1.5, 2.0, 3.0]},
    description="Sharpening",
    category="filtering",
)

register_attack(
    "combined",
    combined_attack,
    default_params={"jpeg_quality": 75, "noise_sigma": 0.03, "resize_scale": 0.75},
    description="Combined JPEG + noise + resize",
    category="combined",
)

register_attack(
    "social_media",
    social_media_attack,
    default_params={"platform": "generic"},
    param_ranges={"platform": ["twitter", "instagram", "facebook", "generic"]},
    description="Social media simulation",
    category="combined",
)


def get_attack(name: str) -> AttackConfig:
    """Get attack configuration by name."""
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {name}. Available: {list(ATTACK_REGISTRY.keys())}")
    return ATTACK_REGISTRY[name]


def list_attacks(category: Optional[str] = None) -> List[str]:
    """List all registered attacks, optionally filtered by category."""
    if category is None:
        return list(ATTACK_REGISTRY.keys())
    return [name for name, config in ATTACK_REGISTRY.items() if config.category == category]


def apply_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    attack_name: str,
    params: Optional[Dict[str, Any]] = None,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Apply an attack by name with optional parameter override.

    Args:
        image: Input image
        attack_name: Name of attack in registry
        params: Optional parameter override (merges with defaults)

    Returns:
        Attacked image
    """
    config = get_attack(attack_name)

    # Merge default params with provided params
    final_params = {**config.default_params}
    if params:
        final_params.update(params)

    return config.function(image, **final_params)


def get_attack_suite(
    categories: Optional[List[str]] = None,
    severity: str = "moderate",
) -> List[tuple]:
    """
    Get a suite of attacks for comprehensive testing.

    Args:
        categories: List of categories to include (None = all)
        severity: "mild", "moderate", or "severe" determines param selection

    Returns:
        List of (attack_name, params) tuples
    """
    severity_index = {"mild": 0, "moderate": 1, "severe": 2}
    idx = severity_index.get(severity, 1)

    suite = []
    for name, config in ATTACK_REGISTRY.items():
        if categories is not None and config.category not in categories:
            continue

        # Select parameters based on severity
        params = {}
        for param_name, param_range in config.param_ranges.items():
            # Pick parameter at appropriate severity level
            param_idx = min(idx, len(param_range) - 1)
            params[param_name] = param_range[param_idx]

        if not params:
            params = config.default_params

        suite.append((name, params))

    return suite


if __name__ == "__main__":
    # Test registry
    print("Registered attacks:")
    for category in ["compression", "noise", "geometric", "color", "filtering", "combined"]:
        attacks = list_attacks(category)
        print(f"  {category}: {attacks}")

    print("\nAttack suite (moderate severity):")
    suite = get_attack_suite(severity="moderate")
    for name, params in suite:
        print(f"  {name}: {params}")
