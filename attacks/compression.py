"""
Compression Attacks

JPEG and WebP compression at various quality levels.
"""

import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Union


def jpeg_compress(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    quality: int = 75,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Apply JPEG compression to image.

    Args:
        image: Input image (tensor, PIL Image, or numpy array)
        quality: JPEG quality (1-100, lower = more compression)

    Returns:
        Compressed image in same format as input
    """
    input_type = type(image)
    input_tensor_device = None

    # Convert to PIL
    if isinstance(image, torch.Tensor):
        input_tensor_device = image.device
        # Assume (B, C, H, W) or (C, H, W), values in [-1, 1]
        if image.dim() == 4:
            image = image[0]
        image = ((image.cpu().float() + 1) / 2 * 255).clamp(0, 255).byte()
        image = image.permute(1, 2, 0).numpy()
        image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    # Apply JPEG compression
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    compressed.load()  # Force load before buffer closes

    # Convert back to input type
    if input_type == torch.Tensor:
        compressed = np.array(compressed).astype(np.float32) / 255.0
        compressed = torch.from_numpy(compressed).permute(2, 0, 1)
        compressed = compressed * 2 - 1  # Back to [-1, 1]
        compressed = compressed.unsqueeze(0)
        if input_tensor_device is not None:
            compressed = compressed.to(input_tensor_device)
        return compressed
    elif input_type == np.ndarray:
        return np.array(compressed)
    else:
        return compressed


def webp_compress(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    quality: int = 75,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Apply WebP compression to image.

    Args:
        image: Input image
        quality: WebP quality (1-100)

    Returns:
        Compressed image in same format as input
    """
    input_type = type(image)
    input_tensor_device = None

    # Convert to PIL
    if isinstance(image, torch.Tensor):
        input_tensor_device = image.device
        if image.dim() == 4:
            image = image[0]
        image = ((image.cpu().float() + 1) / 2 * 255).clamp(0, 255).byte()
        image = image.permute(1, 2, 0).numpy()
        image = Image.fromarray(image)
    elif isinstance(image, np.ndarray):
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    # Apply WebP compression
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer)
    compressed.load()

    # Convert back
    if input_type == torch.Tensor:
        compressed = np.array(compressed).astype(np.float32) / 255.0
        compressed = torch.from_numpy(compressed).permute(2, 0, 1)
        compressed = compressed * 2 - 1
        compressed = compressed.unsqueeze(0)
        if input_tensor_device is not None:
            compressed = compressed.to(input_tensor_device)
        return compressed
    elif input_type == np.ndarray:
        return np.array(compressed)
    else:
        return compressed
