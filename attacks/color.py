"""
Color Attacks

Brightness, contrast, saturation adjustments.
"""

import torch
import numpy as np
from PIL import Image, ImageEnhance
from typing import Union


def brightness_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    factor: float = 1.2,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Adjust image brightness.

    Args:
        image: Input image
        factor: Brightness factor (1.0 = no change, >1 = brighter)

    Returns:
        Adjusted image
    """
    input_type = type(image)
    device = None

    # Convert to PIL
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

    # Apply brightness
    enhancer = ImageEnhance.Brightness(image)
    adjusted = enhancer.enhance(factor)

    # Convert back
    if input_type == torch.Tensor:
        adjusted = np.array(adjusted).astype(np.float32) / 255.0
        adjusted = torch.from_numpy(adjusted).permute(2, 0, 1)
        adjusted = adjusted * 2 - 1
        adjusted = adjusted.unsqueeze(0)
        if device is not None:
            adjusted = adjusted.to(device)
        return adjusted
    elif input_type == np.ndarray:
        return np.array(adjusted)
    else:
        return adjusted


def contrast_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    factor: float = 1.2,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Adjust image contrast.

    Args:
        image: Input image
        factor: Contrast factor (1.0 = no change, >1 = more contrast)

    Returns:
        Adjusted image
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

    enhancer = ImageEnhance.Contrast(image)
    adjusted = enhancer.enhance(factor)

    if input_type == torch.Tensor:
        adjusted = np.array(adjusted).astype(np.float32) / 255.0
        adjusted = torch.from_numpy(adjusted).permute(2, 0, 1)
        adjusted = adjusted * 2 - 1
        adjusted = adjusted.unsqueeze(0)
        if device is not None:
            adjusted = adjusted.to(device)
        return adjusted
    elif input_type == np.ndarray:
        return np.array(adjusted)
    else:
        return adjusted


def saturation_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    factor: float = 1.5,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Adjust image saturation.

    Args:
        image: Input image
        factor: Saturation factor (1.0 = no change, 0 = grayscale, >1 = more saturated)

    Returns:
        Adjusted image
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

    enhancer = ImageEnhance.Color(image)
    adjusted = enhancer.enhance(factor)

    if input_type == torch.Tensor:
        adjusted = np.array(adjusted).astype(np.float32) / 255.0
        adjusted = torch.from_numpy(adjusted).permute(2, 0, 1)
        adjusted = adjusted * 2 - 1
        adjusted = adjusted.unsqueeze(0)
        if device is not None:
            adjusted = adjusted.to(device)
        return adjusted
    elif input_type == np.ndarray:
        return np.array(adjusted)
    else:
        return adjusted


def hue_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    shift: float = 0.1,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Shift image hue.

    Args:
        image: Input image
        shift: Hue shift in [0, 1] (0.5 = 180 degree shift)

    Returns:
        Adjusted image
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

    # Convert to HSV, shift hue, convert back
    import colorsys
    pixels = np.array(image).astype(np.float32) / 255.0
    hsv = np.zeros_like(pixels)

    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            r, g, b = pixels[i, j]
            h, s, v = colorsys.rgb_to_hsv(r, g, b)
            h = (h + shift) % 1.0
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            pixels[i, j] = [r, g, b]

    adjusted = (pixels * 255).clip(0, 255).astype(np.uint8)
    adjusted = Image.fromarray(adjusted)

    if input_type == torch.Tensor:
        adjusted = np.array(adjusted).astype(np.float32) / 255.0
        adjusted = torch.from_numpy(adjusted).permute(2, 0, 1)
        adjusted = adjusted * 2 - 1
        adjusted = adjusted.unsqueeze(0)
        if device is not None:
            adjusted = adjusted.to(device)
        return adjusted
    elif input_type == np.ndarray:
        return np.array(adjusted)
    else:
        return adjusted
