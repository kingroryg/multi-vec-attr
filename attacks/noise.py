"""
Noise Attacks

Various noise injection methods.
"""

import torch
import numpy as np
from typing import Union, Optional
from PIL import Image


def gaussian_noise(
    image: Union[torch.Tensor, np.ndarray],
    sigma: float = 0.05,
    seed: Optional[int] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Add Gaussian noise to image.

    Args:
        image: Input image (tensor or numpy array)
        sigma: Standard deviation of noise (relative to [0, 1] range)
        seed: Random seed for reproducibility

    Returns:
        Noisy image in same format
    """
    if seed is not None:
        if isinstance(image, torch.Tensor):
            torch.manual_seed(seed)
        else:
            np.random.seed(seed)

    if isinstance(image, torch.Tensor):
        noise = torch.randn_like(image) * sigma
        # Assume values in [-1, 1], noise scaled accordingly
        noisy = image + noise * 2  # Scale to [-1, 1] range
        return noisy.clamp(-1, 1)
    else:
        noise = np.random.randn(*image.shape).astype(np.float32) * sigma
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            noisy = np.clip(image + noise, 0, 1)
            return (noisy * 255).astype(np.uint8)
        else:
            return np.clip(image + noise, 0, 1)


def salt_pepper_noise(
    image: Union[torch.Tensor, np.ndarray],
    prob: float = 0.05,
    seed: Optional[int] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Add salt and pepper noise.

    Args:
        image: Input image
        prob: Probability of each pixel being affected
        seed: Random seed

    Returns:
        Noisy image
    """
    if seed is not None:
        if isinstance(image, torch.Tensor):
            torch.manual_seed(seed)
        else:
            np.random.seed(seed)

    if isinstance(image, torch.Tensor):
        mask = torch.rand_like(image[:, :1, :, :]).expand_as(image)
        noisy = image.clone()
        noisy[mask < prob / 2] = -1  # Salt (black in [-1,1])
        noisy[mask > 1 - prob / 2] = 1  # Pepper (white in [-1,1])
        return noisy
    else:
        mask = np.random.rand(*image.shape[:2])
        noisy = image.copy()
        if image.dtype == np.uint8:
            noisy[mask < prob / 2] = 0
            noisy[mask > 1 - prob / 2] = 255
        else:
            noisy[mask < prob / 2] = 0
            noisy[mask > 1 - prob / 2] = 1
        return noisy


def speckle_noise(
    image: Union[torch.Tensor, np.ndarray],
    sigma: float = 0.1,
    seed: Optional[int] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Add multiplicative speckle noise.

    Args:
        image: Input image
        sigma: Standard deviation of multiplicative noise
        seed: Random seed

    Returns:
        Noisy image
    """
    if seed is not None:
        if isinstance(image, torch.Tensor):
            torch.manual_seed(seed)
        else:
            np.random.seed(seed)

    if isinstance(image, torch.Tensor):
        noise = 1 + torch.randn_like(image) * sigma
        noisy = image * noise
        return noisy.clamp(-1, 1)
    else:
        noise = 1 + np.random.randn(*image.shape).astype(np.float32) * sigma
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            noisy = np.clip(image * noise, 0, 1)
            return (noisy * 255).astype(np.uint8)
        else:
            return np.clip(image * noise, 0, 1)


def uniform_noise(
    image: Union[torch.Tensor, np.ndarray],
    magnitude: float = 0.1,
    seed: Optional[int] = None,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Add uniform noise.

    Args:
        image: Input image
        magnitude: Half-width of uniform distribution
        seed: Random seed

    Returns:
        Noisy image
    """
    if seed is not None:
        if isinstance(image, torch.Tensor):
            torch.manual_seed(seed)
        else:
            np.random.seed(seed)

    if isinstance(image, torch.Tensor):
        noise = (torch.rand_like(image) - 0.5) * 2 * magnitude
        noisy = image + noise * 2  # Scale for [-1, 1]
        return noisy.clamp(-1, 1)
    else:
        noise = (np.random.rand(*image.shape).astype(np.float32) - 0.5) * 2 * magnitude
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            noisy = np.clip(image + noise, 0, 1)
            return (noisy * 255).astype(np.uint8)
        else:
            return np.clip(image + noise, 0, 1)
