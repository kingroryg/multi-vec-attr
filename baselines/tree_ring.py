"""
Tree-Ring Watermark Baseline

Implementation of Tree-Ring watermarking (Wen et al., NeurIPS 2023)
for comparison with CDW.

Tree-Ring embeds concentric ring patterns in the Fourier domain of
the initial noise, using DDIM inversion for detection.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TreeRingConfig:
    """Configuration for Tree-Ring watermark."""
    latent_size: int = 64
    num_rings: int = 8
    r_min: float = 4.0
    r_max: float = 20.0
    watermark_channel: int = 3


class TreeRingWatermark:
    """
    Tree-Ring watermark for single-key verification.

    This baseline implements the original Tree-Ring method, which
    embeds a fixed ring pattern and detects by measuring pattern
    correlation after DDIM inversion.
    """

    def __init__(self, config: TreeRingConfig, seed: int = 42):
        self.config = config
        self.seed = seed

        # Generate ring mask
        self._create_ring_mask()

        # Generate watermark pattern (fixed for all images)
        self._generate_pattern()

    def _create_ring_mask(self):
        """Create the ring mask in Fourier domain."""
        h = w = self.config.latent_size
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)

        self.ring_mask = (dist >= self.config.r_min) & (dist <= self.config.r_max)
        self.ring_positions = list(zip(*np.where(self.ring_mask)))

    def _generate_pattern(self):
        """Generate the watermark pattern."""
        np.random.seed(self.seed)

        h = w = self.config.latent_size
        self.pattern = np.zeros((h, w), dtype=np.float32)

        # Set ring values to fixed pattern (zeros, as in original Tree-Ring)
        for y, x in self.ring_positions:
            self.pattern[y, x] = 0.0  # Tree-Ring sets rings to zero

        self.pattern_tensor = torch.from_numpy(self.pattern)

    def embed(
        self,
        latent: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Embed Tree-Ring watermark into latent.

        Args:
            latent: Initial latent of shape (B, C, H, W)
            strength: Watermark strength (not used for Tree-Ring)

        Returns:
            Watermarked latent
        """
        watermarked = latent.clone()
        channel = self.config.watermark_channel

        for b in range(latent.shape[0]):
            channel_data = watermarked[b, channel]

            # FFT
            fft_data = torch.fft.fftshift(torch.fft.fft2(channel_data))

            # Set ring values to zero (Tree-Ring approach)
            for y, x in self.ring_positions:
                fft_data[y, x] = 0.0

            # IFFT
            watermarked[b, channel] = torch.fft.ifft2(
                torch.fft.ifftshift(fft_data)
            ).real

        return watermarked

    def detect(
        self,
        latent: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[bool, float]:
        """
        Detect Tree-Ring watermark.

        Args:
            latent: Inverted latent of shape (B, C, H, W)
            threshold: Detection threshold

        Returns:
            (is_watermarked, score)
        """
        channel = self.config.watermark_channel
        channel_data = latent[0, channel]

        # FFT
        fft_data = torch.fft.fftshift(torch.fft.fft2(channel_data))

        # Compute L1 distance in ring region
        ring_values = []
        for y, x in self.ring_positions:
            ring_values.append(abs(fft_data[y, x].item()))

        # Tree-Ring score: lower is more likely watermarked
        # (rings should be near zero)
        score = np.mean(ring_values)

        # Invert so higher = more likely watermarked (for consistency with CDW)
        normalized_score = 1.0 / (1.0 + score)

        return normalized_score > threshold, normalized_score


class RandomCodeWatermark:
    """
    Random code watermark baseline.

    Uses random (non-orthogonal) spreading codes for comparison
    with orthogonal CDW codes.
    """

    def __init__(self, num_vendors: int, code_length: int, seed: int = 42):
        np.random.seed(seed)
        self.codes = np.random.choice(
            [-1, 1],
            size=(num_vendors, code_length)
        ).astype(np.float32)

    def get_codes(self) -> np.ndarray:
        return self.codes
