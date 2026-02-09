"""
Combined Attacks

Pipelines of multiple attacks applied sequentially.
"""

from typing import Union, List, Callable, Any
import torch
import numpy as np
from PIL import Image

from .compression import jpeg_compress
from .noise import gaussian_noise
from .geometric import resize_attack, crop_attack, rotate_attack
from .color import brightness_attack, contrast_attack
from .filtering import gaussian_blur


def combined_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    jpeg_quality: int = 75,
    noise_sigma: float = 0.03,
    resize_scale: float = 0.75,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Apply a combination of common attacks.

    This simulates a realistic scenario where an image might be
    shared on social media (compressed, resized, etc.)

    Args:
        image: Input image
        jpeg_quality: JPEG compression quality
        noise_sigma: Gaussian noise level
        resize_scale: Resize scale factor

    Returns:
        Attacked image
    """
    # Apply attacks in sequence
    result = image

    # Resize (with restoration)
    result = resize_attack(result, scale=resize_scale, restore_size=True)

    # Add noise
    result = gaussian_noise(result, sigma=noise_sigma)

    # JPEG compression
    result = jpeg_compress(result, quality=jpeg_quality)

    return result


def social_media_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    platform: str = "generic",
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Simulate social media processing for specific platforms.

    Args:
        image: Input image
        platform: "twitter", "instagram", "facebook", "generic"

    Returns:
        Processed image
    """
    if platform == "twitter":
        # Twitter: JPEG 85, max 4096px
        result = jpeg_compress(image, quality=85)
    elif platform == "instagram":
        # Instagram: JPEG 70, aggressive compression
        result = jpeg_compress(image, quality=70)
        result = resize_attack(result, scale=0.9, restore_size=True)
    elif platform == "facebook":
        # Facebook: JPEG 75, slight sharpening
        result = jpeg_compress(image, quality=75)
    else:
        # Generic: moderate JPEG
        result = jpeg_compress(image, quality=80)

    return result


def attack_pipeline(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    attacks: List[tuple],
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Apply a sequence of attacks with specified parameters.

    Args:
        image: Input image
        attacks: List of (attack_function, kwargs_dict) tuples

    Returns:
        Attacked image

    Example:
        attacks = [
            (jpeg_compress, {"quality": 75}),
            (gaussian_noise, {"sigma": 0.05}),
            (rotate_attack, {"angle": 5}),
        ]
        result = attack_pipeline(image, attacks)
    """
    result = image
    for attack_fn, kwargs in attacks:
        result = attack_fn(result, **kwargs)
    return result


# Predefined attack sequences for benchmarking
ATTACK_SEQUENCES = {
    "mild": [
        ("jpeg_compress", {"quality": 90}),
    ],
    "moderate": [
        ("jpeg_compress", {"quality": 75}),
        ("gaussian_noise", {"sigma": 0.02}),
    ],
    "severe": [
        ("jpeg_compress", {"quality": 50}),
        ("gaussian_noise", {"sigma": 0.05}),
        ("resize_attack", {"scale": 0.5}),
    ],
    "geometric": [
        ("rotate_attack", {"angle": 10}),
        ("crop_attack", {"keep_ratio": 0.8}),
    ],
    "color": [
        ("brightness_attack", {"factor": 1.2}),
        ("contrast_attack", {"factor": 0.8}),
    ],
    "social_media": [
        ("resize_attack", {"scale": 0.8}),
        ("jpeg_compress", {"quality": 70}),
        ("gaussian_noise", {"sigma": 0.01}),
    ],
}


def get_attack_sequence(name: str) -> List[tuple]:
    """Get a predefined attack sequence by name."""
    if name not in ATTACK_SEQUENCES:
        raise ValueError(f"Unknown attack sequence: {name}. Available: {list(ATTACK_SEQUENCES.keys())}")
    return ATTACK_SEQUENCES[name]
