"""
CDW Watermark Embedding

Embeds spreading codes into the Fourier domain of diffusion model latents.
The embedding modulates ring regions with the vendor's code, creating a
spread-spectrum watermark that is robust to image transformations.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from .ring_mask import RingMask, create_default_mask
from .codes import generate_codes, CodeFamily


@dataclass
class EmbeddingConfig:
    """Configuration for CDW embedding."""
    code_length: int = 64
    watermark_strength: float = 0.1  # alpha parameter
    watermark_channel: int = 3  # Which latent channel to embed (0-3 for SD)
    r_min: float = 4.0
    r_max: float = 20.0
    latent_size: int = 64


class CDWEmbedder:
    """
    Embeds spreading codes into diffusion model latents.

    The embedder modifies the initial latent z_T in the Fourier domain,
    adding a spread-spectrum signal that encodes the vendor's identity.

    Usage:
        embedder = CDWEmbedder(config)
        embedder.set_vendor_code(vendor_id, code)
        watermarked_latent = embedder.embed(latent, vendor_id)
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedder.

        Args:
            config: Embedding configuration
        """
        self.config = config

        # Create ring mask
        self.mask = create_default_mask(
            latent_size=config.latent_size,
            r_min=config.r_min,
            r_max=config.r_max,
            code_length=config.code_length,
        )

        # Precompute sign pattern for conjugate symmetry
        self.sign_pattern = self.mask.get_sign_pattern()

        # Vendor codes storage
        self.vendor_codes = {}

        # Device management
        self.device = None

    def set_vendor_code(self, vendor_id: int, code: np.ndarray):
        """
        Register a vendor's spreading code.

        Args:
            vendor_id: Unique identifier for the vendor
            code: np.ndarray of shape (code_length,) with values in {-1, +1}
        """
        if len(code) != self.config.code_length:
            raise ValueError(
                f"Code length {len(code)} doesn't match config {self.config.code_length}"
            )
        self.vendor_codes[vendor_id] = code.astype(np.float32)

    def set_all_codes(self, codes: np.ndarray):
        """
        Register codes for vendors 0, 1, ..., N-1.

        Args:
            codes: np.ndarray of shape (num_vendors, code_length)
        """
        for vendor_id, code in enumerate(codes):
            self.set_vendor_code(vendor_id, code)

    def to(self, device: torch.device) -> "CDWEmbedder":
        """Move embedder tensors to device."""
        self.device = device
        return self

    def _get_code_tensor(self, vendor_id: int) -> torch.Tensor:
        """Get vendor's code as a torch tensor on the correct device."""
        code = self.vendor_codes[vendor_id]
        tensor = torch.from_numpy(code)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor

    def embed(
        self,
        latent: torch.Tensor,
        vendor_id: int,
        strength: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Embed watermark into latent tensor.

        Args:
            latent: Initial latent z_T of shape (B, C, H, W)
            vendor_id: Which vendor's code to embed
            strength: Override watermark strength (default: config value)

        Returns:
            Watermarked latent of same shape
        """
        if vendor_id not in self.vendor_codes:
            raise ValueError(f"Vendor {vendor_id} not registered")

        alpha = strength if strength is not None else self.config.watermark_strength
        code = self._get_code_tensor(vendor_id)
        channel = self.config.watermark_channel

        # Clone to avoid modifying input
        watermarked = latent.clone()

        # Process each batch element
        for b in range(latent.shape[0]):
            # Extract the channel to watermark
            channel_data = watermarked[b, channel]  # Shape: (H, W)

            # FFT (shift so DC is at center)
            fft_data = torch.fft.fftshift(torch.fft.fft2(channel_data))

            # Embed code in each ring
            for ring_idx in range(self.config.code_length):
                positions = self.mask.get_ring_positions(ring_idx)
                code_value = code[ring_idx] * alpha

                for y, x in positions:
                    sign = self.sign_pattern[y, x]
                    fft_data[y, x] = fft_data[y, x] + code_value * sign

            # Inverse FFT
            watermarked[b, channel] = torch.fft.ifft2(
                torch.fft.ifftshift(fft_data)
            ).real

        return watermarked

    def embed_batch(
        self,
        latents: torch.Tensor,
        vendor_ids: Union[int, list],
        strength: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Embed watermarks into a batch of latents.

        Args:
            latents: Batch of latents, shape (B, C, H, W)
            vendor_ids: Single vendor ID (applied to all) or list of B vendor IDs
            strength: Override watermark strength

        Returns:
            Watermarked latents of same shape
        """
        batch_size = latents.shape[0]

        if isinstance(vendor_ids, int):
            vendor_ids = [vendor_ids] * batch_size

        if len(vendor_ids) != batch_size:
            raise ValueError(
                f"vendor_ids length {len(vendor_ids)} doesn't match batch size {batch_size}"
            )

        # Process each latent with its vendor code
        watermarked = latents.clone()
        for b, vendor_id in enumerate(vendor_ids):
            watermarked[b:b+1] = self.embed(
                latents[b:b+1], vendor_id, strength
            )

        return watermarked

    def get_embedding_info(self) -> dict:
        """Return information about the embedding configuration."""
        return {
            "config": {
                "code_length": self.config.code_length,
                "watermark_strength": self.config.watermark_strength,
                "watermark_channel": self.config.watermark_channel,
                "latent_size": self.config.latent_size,
            },
            "mask": self.mask.to_dict(),
            "num_vendors": len(self.vendor_codes),
            "vendor_ids": list(self.vendor_codes.keys()),
        }


def create_embedder_with_codes(
    num_vendors: int,
    code_family: CodeFamily = CodeFamily.WALSH_HADAMARD,
    config: Optional[EmbeddingConfig] = None,
    seed: Optional[int] = None,
) -> Tuple[CDWEmbedder, np.ndarray, dict]:
    """
    Convenience function to create embedder with generated codes.

    Args:
        num_vendors: Number of vendors to support
        code_family: Which code family to use
        config: Embedding configuration (default if None)
        seed: Random seed for reproducibility

    Returns:
        embedder: Configured CDWEmbedder
        codes: np.ndarray of shape (num_vendors, code_length)
        code_properties: dict of code family statistics
    """
    if config is None:
        config = EmbeddingConfig()

    # Generate codes
    codes, properties = generate_codes(
        code_family=code_family,
        num_codes=num_vendors,
        code_length=config.code_length,
        seed=seed,
    )

    # Ensure code length matches
    actual_length = codes.shape[1]
    if actual_length != config.code_length:
        # Adjust config to match generated codes
        config.code_length = actual_length

    # Create embedder and register codes
    embedder = CDWEmbedder(config)
    embedder.set_all_codes(codes)

    return embedder, codes, properties


if __name__ == "__main__":
    # Test embedding
    print("Testing CDW Embedding...")

    # Create embedder with 8 vendors
    embedder, codes, props = create_embedder_with_codes(
        num_vendors=8,
        code_family=CodeFamily.WALSH_HADAMARD,
    )
    print(f"Code properties: {props}")

    # Create dummy latent
    latent = torch.randn(1, 4, 64, 64)
    print(f"Original latent stats: mean={latent.mean():.4f}, std={latent.std():.4f}")

    # Embed watermark
    watermarked = embedder.embed(latent, vendor_id=0)
    print(f"Watermarked latent stats: mean={watermarked.mean():.4f}, std={watermarked.std():.4f}")

    # Check perturbation magnitude
    diff = watermarked - latent
    print(f"Perturbation: mean={diff.mean():.6f}, std={diff.std():.6f}, max={diff.abs().max():.6f}")

    print("\nEmbedding info:")
    print(embedder.get_embedding_info())
