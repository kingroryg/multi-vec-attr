"""
Utility Functions for CDW

Helper functions for FFT operations, image processing, and reproducibility.
"""

import torch
import numpy as np
import random
import os
from typing import Optional, Tuple, Union
from PIL import Image


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic operations (may slow down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def image_to_tensor(
    image: Union[Image.Image, np.ndarray],
    normalize: bool = True,
) -> torch.Tensor:
    """
    Convert PIL Image or numpy array to tensor.

    Args:
        image: PIL Image or numpy array (H, W, 3) in [0, 255]
        normalize: If True, normalize to [-1, 1], else [0, 1]

    Returns:
        Tensor of shape (1, 3, H, W)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensure float and correct range
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    # HWC to CHW
    if image.ndim == 3 and image.shape[-1] == 3:
        image = np.transpose(image, (2, 0, 1))

    tensor = torch.from_numpy(image).float()

    # Add batch dimension
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    # Normalize to [-1, 1] if requested
    if normalize:
        tensor = tensor * 2 - 1

    return tensor


def tensor_to_image(
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> Image.Image:
    """
    Convert tensor to PIL Image.

    Args:
        tensor: Tensor of shape (B, 3, H, W) or (3, H, W), values in [-1, 1] or [0, 1]
        denormalize: If True, assume input is in [-1, 1]

    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.cpu().float()

    if denormalize:
        tensor = (tensor + 1) / 2

    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).byte()

    # CHW to HWC
    image_np = tensor.permute(1, 2, 0).numpy()

    return Image.fromarray(image_np)


def compute_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        img1, img2: Tensors of same shape, values in [0, max_val]
        max_val: Maximum pixel value

    Returns:
        PSNR in dB
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
) -> float:
    """
    Compute Structural Similarity Index.

    Simplified implementation - for production use torchmetrics.

    Args:
        img1, img2: Tensors of shape (B, C, H, W)
        window_size: Size of Gaussian window

    Returns:
        Mean SSIM
    """
    try:
        from torchmetrics.functional import structural_similarity_index_measure
        return structural_similarity_index_measure(img1, img2).item()
    except ImportError:
        # Fallback to simple correlation-based similarity
        img1_flat = img1.flatten()
        img2_flat = img2.flatten()
        correlation = torch.corrcoef(torch.stack([img1_flat, img2_flat]))[0, 1]
        return correlation.item()


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_tensor(tensor: torch.Tensor, path: str):
    """Save tensor to file."""
    ensure_dir(os.path.dirname(path))
    torch.save(tensor, path)


def load_tensor(path: str) -> torch.Tensor:
    """Load tensor from file."""
    return torch.load(path)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Timer:
    """Simple timer for profiling."""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0

    def start(self):
        import time
        self.start_time = time.perf_counter()

    def stop(self):
        import time
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.start_time = None
        return self.elapsed

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def theoretical_accuracy(
    alpha: float,
    sigma: float,
    code_length: int,
    num_vendors: int,
) -> float:
    """
    Compute theoretical identification accuracy from Theorem 1.

    P(correct) = [Phi(alpha * sqrt(L) / sigma)]^(N-1)

    Args:
        alpha: Watermark strength
        sigma: Noise standard deviation
        code_length: L
        num_vendors: N

    Returns:
        Predicted accuracy
    """
    from scipy.stats import norm

    snr = alpha * np.sqrt(code_length) / sigma
    prob_pairwise = norm.cdf(snr)
    accuracy = prob_pairwise ** (num_vendors - 1)

    return accuracy


def estimate_noise_sigma(
    clean_signals: np.ndarray,
    watermarked_signals: np.ndarray,
    codes: np.ndarray,
    alpha: float,
) -> float:
    """
    Estimate noise parameter sigma from extraction errors.

    Args:
        clean_signals: Extracted signals from unwatermarked images
        watermarked_signals: Extracted signals from watermarked images
        codes: Vendor codes used for watermarking
        alpha: Watermark strength

    Returns:
        Estimated sigma
    """
    # The extraction error is the difference between extracted and expected
    # For watermarked: expected = alpha * code
    # Error = extracted - expected

    errors = []
    for i, (signal, code) in enumerate(zip(watermarked_signals, codes)):
        expected = alpha * code
        error = signal - expected
        errors.append(error)

    errors = np.array(errors)
    sigma = np.std(errors)

    return sigma
