"""
Filtering Attacks

Blur, sharpen, median filter, and other convolution-based attacks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from typing import Union
from scipy import ndimage


def gaussian_blur(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Apply Gaussian blur.

    Args:
        image: Input image
        kernel_size: Size of Gaussian kernel (must be odd)
        sigma: Standard deviation of Gaussian

    Returns:
        Blurred image
    """
    input_type = type(image)
    device = None

    if isinstance(image, torch.Tensor):
        device = image.device
        if image.dim() == 4:
            image = image[0]
        image = ((image.cpu().float() + 1) / 2 * 255).clamp(0, 255).byte()
        image = Image.fromarray(image.permute(1, 2, 0).numpy())
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    blurred = image.filter(ImageFilter.GaussianBlur(radius=sigma))

    if input_type == torch.Tensor:
        blurred = np.array(blurred).astype(np.float32) / 255.0
        blurred = torch.from_numpy(blurred).permute(2, 0, 1)
        blurred = blurred * 2 - 1
        blurred = blurred.unsqueeze(0)
        if device is not None:
            blurred = blurred.to(device)
        return blurred
    elif input_type == np.ndarray:
        return np.array(blurred)
    else:
        return blurred


def median_filter(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    kernel_size: int = 3,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Apply median filter.

    Args:
        image: Input image
        kernel_size: Size of median filter kernel

    Returns:
        Filtered image
    """
    input_type = type(image)
    device = None

    if isinstance(image, torch.Tensor):
        device = image.device
        if image.dim() == 4:
            image = image[0]
        image = ((image.cpu().float() + 1) / 2 * 255).clamp(0, 255).byte()
        image = Image.fromarray(image.permute(1, 2, 0).numpy())
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    filtered = image.filter(ImageFilter.MedianFilter(size=kernel_size))

    if input_type == torch.Tensor:
        filtered = np.array(filtered).astype(np.float32) / 255.0
        filtered = torch.from_numpy(filtered).permute(2, 0, 1)
        filtered = filtered * 2 - 1
        filtered = filtered.unsqueeze(0)
        if device is not None:
            filtered = filtered.to(device)
        return filtered
    elif input_type == np.ndarray:
        return np.array(filtered)
    else:
        return filtered


def sharpen(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    factor: float = 2.0,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Apply sharpening filter.

    Args:
        image: Input image
        factor: Sharpening strength (1.0 = no change, >1 = sharper)

    Returns:
        Sharpened image
    """
    input_type = type(image)
    device = None

    if isinstance(image, torch.Tensor):
        device = image.device
        if image.dim() == 4:
            image = image[0]
        image = ((image.cpu().float() + 1) / 2 * 255).clamp(0, 255).byte()
        image = Image.fromarray(image.permute(1, 2, 0).numpy())
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    from PIL import ImageEnhance
    enhancer = ImageEnhance.Sharpness(image)
    sharpened = enhancer.enhance(factor)

    if input_type == torch.Tensor:
        sharpened = np.array(sharpened).astype(np.float32) / 255.0
        sharpened = torch.from_numpy(sharpened).permute(2, 0, 1)
        sharpened = sharpened * 2 - 1
        sharpened = sharpened.unsqueeze(0)
        if device is not None:
            sharpened = sharpened.to(device)
        return sharpened
    elif input_type == np.ndarray:
        return np.array(sharpened)
    else:
        return sharpened


def edge_enhance(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Apply edge enhancement filter.

    Args:
        image: Input image

    Returns:
        Edge-enhanced image
    """
    input_type = type(image)
    device = None

    if isinstance(image, torch.Tensor):
        device = image.device
        if image.dim() == 4:
            image = image[0]
        image = ((image.cpu().float() + 1) / 2 * 255).clamp(0, 255).byte()
        image = Image.fromarray(image.permute(1, 2, 0).numpy())
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    enhanced = image.filter(ImageFilter.EDGE_ENHANCE)

    if input_type == torch.Tensor:
        enhanced = np.array(enhanced).astype(np.float32) / 255.0
        enhanced = torch.from_numpy(enhanced).permute(2, 0, 1)
        enhanced = enhanced * 2 - 1
        enhanced = enhanced.unsqueeze(0)
        if device is not None:
            enhanced = enhanced.to(device)
        return enhanced
    elif input_type == np.ndarray:
        return np.array(enhanced)
    else:
        return enhanced
