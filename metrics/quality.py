"""
Image Quality Metrics

FID, LPIPS, PSNR, SSIM for evaluating watermark imperceptibility.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Union
from PIL import Image


def _to_numpy(img: Union[torch.Tensor, np.ndarray, Image.Image]) -> np.ndarray:
    """Convert image to numpy array."""
    if isinstance(img, Image.Image):
        return np.array(img).astype(np.float32) / 255.0
    if isinstance(img, torch.Tensor):
        return img.cpu().numpy()
    return img


def _to_tensor(img: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
    """Convert image to torch tensor."""
    if isinstance(img, Image.Image):
        arr = np.array(img).astype(np.float32) / 255.0
        # (H, W, C) -> (C, H, W)
        if arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).unsqueeze(0)
    if isinstance(img, np.ndarray):
        if img.ndim == 3 and img.shape[2] in [1, 3, 4]:
            img = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(img).unsqueeze(0) if img.ndim == 3 else torch.from_numpy(img)
    return img


def compute_psnr(
    img1: Union[torch.Tensor, np.ndarray, Image.Image],
    img2: Union[torch.Tensor, np.ndarray, Image.Image],
    max_val: float = None,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.

    Args:
        img1, img2: Images to compare (same shape)
        max_val: Maximum pixel value. Auto-detected if None.

    Returns:
        PSNR in dB
    """
    img1 = _to_numpy(img1)
    img2 = _to_numpy(img2)

    # Auto-detect max_val
    if max_val is None:
        max_val = 1.0 if img1.max() <= 1.0 else 255.0

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ssim(
    img1: Union[torch.Tensor, np.ndarray, Image.Image],
    img2: Union[torch.Tensor, np.ndarray, Image.Image],
    data_range: float = None,
) -> float:
    """
    Compute Structural Similarity Index.

    Args:
        img1, img2: Images to compare
        data_range: Range of pixel values

    Returns:
        SSIM score (higher is better, max 1.0)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
        img1 = _to_numpy(img1)
        img2 = _to_numpy(img2)

        # Handle batch dimension
        if img1.ndim == 4:
            img1 = img1[0]
        if img2.ndim == 4:
            img2 = img2[0]

        # Handle channel dimension (C, H, W) -> (H, W, C)
        if img1.ndim == 3 and img1.shape[0] in [1, 3, 4]:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))

        # Auto-detect data_range
        if data_range is None:
            data_range = 1.0 if img1.max() <= 1.0 else 255.0

        return ssim(img1, img2, data_range=data_range, channel_axis=-1)
    except ImportError:
        # Fallback: simple correlation-based similarity
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()
        corr = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
        return float(corr)


def compute_lpips(
    img1: Union[torch.Tensor, np.ndarray, Image.Image],
    img2: Union[torch.Tensor, np.ndarray, Image.Image],
    net: str = "alex",
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity.

    Args:
        img1, img2: Image tensors of shape (B, C, H, W), values in [-1, 1]
        net: Network to use ("alex", "vgg", "squeeze")
        device: Torch device

    Returns:
        LPIPS distance (lower is more similar)
    """
    try:
        import lpips

        # Convert to tensor if needed
        img1 = _to_tensor(img1)
        img2 = _to_tensor(img2)

        if device is None:
            device = img1.device if isinstance(img1, torch.Tensor) else torch.device("cpu")

        # Ensure 4D tensor (B, C, H, W)
        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)

        # LPIPS expects [-1, 1] range
        if img1.max() <= 1.0 and img1.min() >= 0:
            img1 = img1 * 2 - 1
            img2 = img2 * 2 - 1

        # Initialize LPIPS model (cached)
        if not hasattr(compute_lpips, "_model") or compute_lpips._net != net:
            compute_lpips._model = lpips.LPIPS(net=net).to(device)
            compute_lpips._net = net

        model = compute_lpips._model.to(device)

        with torch.no_grad():
            distance = model(img1.to(device), img2.to(device))

        return float(distance.mean().item())
    except ImportError:
        # Fallback: use simple MSE
        img1 = _to_tensor(img1)
        img2 = _to_tensor(img2)
        mse = torch.mean((img1 - img2) ** 2).item()
        return mse


def compute_fid(
    real_images: Union[List[torch.Tensor], List[Image.Image], torch.Tensor],
    generated_images: Union[List[torch.Tensor], List[Image.Image], torch.Tensor],
    device: Optional[torch.device] = None,
    batch_size: int = 32,
) -> float:
    """
    Compute Fréchet Inception Distance.

    Args:
        real_images: Real/reference images
        generated_images: Generated/watermarked images
        device: Torch device
        batch_size: Batch size for feature extraction

    Returns:
        FID score (lower is better)
    """
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        fid = FrechetInceptionDistance(normalize=True).to(device)

        # Handle list of images (including PIL Images)
        if isinstance(real_images, list):
            real_images = torch.stack([_to_tensor(img).squeeze(0) for img in real_images])
        if isinstance(generated_images, list):
            generated_images = torch.stack([_to_tensor(img).squeeze(0) for img in generated_images])

        # Ensure correct format: (B, C, H, W), values in [0, 1]
        if real_images.min() < 0:
            real_images = (real_images + 1) / 2
        if generated_images.min() < 0:
            generated_images = (generated_images + 1) / 2

        # Convert to uint8 for FID computation
        real_uint8 = (real_images * 255).byte()
        gen_uint8 = (generated_images * 255).byte()

        # Update in batches
        for i in range(0, len(real_uint8), batch_size):
            batch_real = real_uint8[i:i+batch_size].to(device)
            batch_gen = gen_uint8[i:i+batch_size].to(device)
            fid.update(batch_real, real=True)
            fid.update(batch_gen, real=False)

        return float(fid.compute().item())

    except ImportError:
        # Fallback: compute mean feature distance (rough approximation)
        if isinstance(real_images, list):
            real_images = torch.stack(real_images)
        if isinstance(generated_images, list):
            generated_images = torch.stack(generated_images)

        real_mean = real_images.mean(dim=0)
        gen_mean = generated_images.mean(dim=0)
        return float(torch.mean((real_mean - gen_mean) ** 2).item() * 1000)


def compute_all_quality_metrics(
    original: torch.Tensor,
    watermarked: torch.Tensor,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Compute all image quality metrics.

    Args:
        original: Original unwatermarked images
        watermarked: Watermarked images
        device: Torch device

    Returns:
        Dict with all metrics
    """
    # Ensure same device
    if device is None:
        device = original.device

    original = original.to(device)
    watermarked = watermarked.to(device)

    metrics = {}

    # PSNR (convert to [0, 1] range)
    orig_01 = (original + 1) / 2
    wm_01 = (watermarked + 1) / 2
    metrics["psnr"] = compute_psnr(orig_01, wm_01, max_val=1.0)

    # SSIM
    metrics["ssim"] = compute_ssim(orig_01, wm_01, data_range=1.0)

    # LPIPS
    metrics["lpips"] = compute_lpips(original, watermarked, device=device)

    return metrics


def compute_quality_over_batch(
    originals: List[torch.Tensor],
    watermarkeds: List[torch.Tensor],
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute quality metrics over a batch, returning mean and std.

    Args:
        originals: List of original images
        watermarkeds: List of watermarked images
        device: Torch device

    Returns:
        Dict with mean and std for each metric
    """
    all_metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
    }

    for orig, wm in zip(originals, watermarkeds):
        m = compute_all_quality_metrics(orig, wm, device)
        for k, v in m.items():
            all_metrics[k].append(v)

    results = {}
    for k, values in all_metrics.items():
        results[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    return results
