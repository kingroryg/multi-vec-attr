"""
Geometric Attacks

Resize, crop, rotation, and other geometric transformations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional


def resize_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    scale: float = 0.5,
    restore_size: bool = True,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Resize image (downscale then optionally upscale back).

    Args:
        image: Input image
        scale: Scale factor (< 1 for downscaling)
        restore_size: If True, resize back to original dimensions

    Returns:
        Processed image
    """
    input_type = type(image)
    original_size = None
    device = None

    # Get original size and convert to tensor
    if isinstance(image, torch.Tensor):
        device = image.device
        if image.dim() == 3:
            image = image.unsqueeze(0)
        original_size = (image.shape[2], image.shape[3])  # H, W
    elif isinstance(image, Image.Image):
        original_size = image.size[::-1]  # PIL uses W, H
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    elif isinstance(image, np.ndarray):
        original_size = image.shape[:2]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    # Compute new size
    new_h = int(original_size[0] * scale)
    new_w = int(original_size[1] * scale)

    # Downscale
    resized = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # Restore size if requested
    if restore_size:
        resized = F.interpolate(resized, size=original_size, mode='bilinear', align_corners=False)

    # Convert back to original format
    if input_type == torch.Tensor:
        if device is not None:
            resized = resized.to(device)
        return resized
    elif input_type == Image.Image:
        resized = (resized[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(resized)
    else:
        return resized[0].permute(1, 2, 0).numpy()


def crop_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    keep_ratio: float = 0.75,
    position: str = "center",
    restore_size: bool = True,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Crop image and optionally resize back.

    Args:
        image: Input image
        keep_ratio: Fraction of image to keep (0-1)
        position: Crop position ("center", "random", or (top, left) tuple)
        restore_size: If True, resize back to original dimensions

    Returns:
        Processed image
    """
    input_type = type(image)
    device = None

    # Convert to tensor
    if isinstance(image, torch.Tensor):
        device = image.device
        if image.dim() == 3:
            image = image.unsqueeze(0)
        original_size = (image.shape[2], image.shape[3])
    elif isinstance(image, Image.Image):
        original_size = image.size[::-1]
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    else:
        original_size = image.shape[:2]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    h, w = original_size
    new_h = int(h * np.sqrt(keep_ratio))
    new_w = int(w * np.sqrt(keep_ratio))

    # Determine crop position
    if position == "center":
        top = (h - new_h) // 2
        left = (w - new_w) // 2
    elif position == "random":
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
    else:
        top, left = position

    # Crop
    cropped = image[:, :, top:top+new_h, left:left+new_w]

    # Restore size
    if restore_size:
        cropped = F.interpolate(cropped, size=original_size, mode='bilinear', align_corners=False)

    # Convert back
    if input_type == torch.Tensor:
        if device is not None:
            cropped = cropped.to(device)
        return cropped
    elif input_type == Image.Image:
        cropped = (cropped[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(cropped)
    else:
        return cropped[0].permute(1, 2, 0).numpy()


def rotate_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    angle: float = 15.0,
    fill_mode: str = "reflect",
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Rotate image by given angle.

    Args:
        image: Input image
        angle: Rotation angle in degrees (positive = counterclockwise)
        fill_mode: How to fill empty regions ("reflect", "constant", "replicate")

    Returns:
        Rotated image
    """
    input_type = type(image)
    device = None

    # Convert to PIL for rotation (handles padding well)
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

    # Rotate
    if fill_mode == "reflect":
        # PIL doesn't support reflect, use expand and crop
        rotated = image.rotate(angle, resample=Image.BILINEAR, expand=True)
        # Center crop back to original size
        orig_w, orig_h = image.size
        rot_w, rot_h = rotated.size
        left = (rot_w - orig_w) // 2
        top = (rot_h - orig_h) // 2
        rotated = rotated.crop((left, top, left + orig_w, top + orig_h))
    else:
        fill_color = (128, 128, 128) if fill_mode == "constant" else None
        rotated = image.rotate(angle, resample=Image.BILINEAR, fillcolor=fill_color)

    # Convert back
    if input_type == torch.Tensor:
        rotated = np.array(rotated).astype(np.float32) / 255.0
        rotated = torch.from_numpy(rotated).permute(2, 0, 1)
        rotated = rotated * 2 - 1
        rotated = rotated.unsqueeze(0)
        if device is not None:
            rotated = rotated.to(device)
        return rotated
    elif input_type == np.ndarray:
        return np.array(rotated)
    else:
        return rotated


def flip_attack(
    image: Union[torch.Tensor, Image.Image, np.ndarray],
    horizontal: bool = True,
) -> Union[torch.Tensor, Image.Image, np.ndarray]:
    """
    Flip image horizontally or vertically.

    Args:
        image: Input image
        horizontal: If True, flip horizontally; else vertically

    Returns:
        Flipped image
    """
    if isinstance(image, torch.Tensor):
        dim = 3 if horizontal else 2
        if image.dim() == 3:
            dim -= 1
        return torch.flip(image, [dim])
    elif isinstance(image, Image.Image):
        if horizontal:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return image.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        if horizontal:
            return np.fliplr(image)
        else:
            return np.flipud(image)
