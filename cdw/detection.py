"""
CDW Watermark Detection

Detects watermarks via DDIM inversion followed by correlation with vendor codes.
The detector inverts the diffusion process to recover the initial latent, then
correlates the Fourier coefficients with each vendor's spreading code.
"""

import torch
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

from .ring_mask import RingMask, create_default_mask
from .embedding import EmbeddingConfig


@dataclass
class DetectionResult:
    """Result of watermark detection for a single image."""
    detected_vendor: Optional[int]  # None if no watermark detected
    correlations: np.ndarray  # Correlation with each vendor's code
    max_correlation: float  # Highest correlation value
    confidence: float  # Detection confidence (max_corr - threshold)
    extracted_signal: np.ndarray  # Raw extracted signal from Fourier domain


class CDWDetector:
    """
    Detects CDW watermarks via DDIM inversion and correlation.

    The detector:
    1. Inverts the image to recover the initial latent z_T
    2. Computes FFT of the watermarked channel
    3. Extracts signal from ring positions
    4. Correlates with each vendor's code
    5. Returns vendor with highest correlation (if above threshold)

    Usage:
        detector = CDWDetector(config, pipe)
        detector.set_all_codes(codes)
        result = detector.detect(image)
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        pipe: Any,  # StableDiffusionPipeline
        detection_threshold: float = 0.3,
        num_inversion_steps: int = 50,
    ):
        """
        Initialize the detector.

        Args:
            config: Embedding configuration (must match embedder)
            pipe: Diffusion pipeline for DDIM inversion
            detection_threshold: Minimum correlation for positive detection
            num_inversion_steps: Number of DDIM inversion steps
        """
        self.config = config
        self.pipe = pipe
        self.threshold = detection_threshold
        self.num_inversion_steps = num_inversion_steps

        # Create ring mask (must match embedder)
        self.mask = create_default_mask(
            latent_size=config.latent_size,
            r_min=config.r_min,
            r_max=config.r_max,
            code_length=config.code_length,
        )

        # Sign pattern for extraction
        self.sign_pattern = self.mask.get_sign_pattern()

        # Vendor codes
        self.vendor_codes = {}
        self.codes_matrix = None  # Shape: (num_vendors, code_length)

        # Device
        self.device = None

    def set_vendor_code(self, vendor_id: int, code: np.ndarray):
        """Register a vendor's spreading code."""
        self.vendor_codes[vendor_id] = code.astype(np.float32)
        self._update_codes_matrix()

    def set_all_codes(self, codes: np.ndarray):
        """Register codes for vendors 0, 1, ..., N-1."""
        for vendor_id, code in enumerate(codes):
            self.vendor_codes[vendor_id] = code.astype(np.float32)
        self._update_codes_matrix()

    def _update_codes_matrix(self):
        """Update the codes matrix for vectorized correlation."""
        if len(self.vendor_codes) > 0:
            max_id = max(self.vendor_codes.keys())
            self.codes_matrix = np.zeros(
                (max_id + 1, self.config.code_length),
                dtype=np.float32
            )
            for vid, code in self.vendor_codes.items():
                self.codes_matrix[vid] = code

    def to(self, device: torch.device) -> "CDWDetector":
        """Move detector to device."""
        self.device = device
        return self

    @torch.no_grad()
    def ddim_inversion(
        self,
        image: torch.Tensor,
        prompt: str = "",
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Invert image to initial latent via DDIM inversion.

        Args:
            image: Image tensor of shape (B, 3, H, W), values in [-1, 1]
            prompt: Text prompt (use empty for unconditional)
            guidance_scale: CFG scale (1.0 for deterministic inversion)

        Returns:
            Estimated initial latent z_T of shape (B, 4, h, w)
        """
        # Ensure correct device
        if self.device is not None:
            image = image.to(self.device)

        # Encode image to latent
        latent = self.pipe.vae.encode(image).latent_dist.mean
        latent = latent * self.pipe.vae.config.scaling_factor

        # Get text embeddings
        text_input = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]

        # Unconditional embeddings for CFG
        uncond_input = self.pipe.tokenizer(
            "",
            padding="max_length",
            max_length=text_input.input_ids.shape[-1],
            return_tensors="pt",
        )
        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Expand for batch
        batch_size = image.shape[0]
        text_embeddings = text_embeddings.expand(batch_size, -1, -1)
        uncond_embeddings = uncond_embeddings.expand(batch_size, -1, -1)

        # DDIM inversion (reverse the denoising)
        self.pipe.scheduler.set_timesteps(self.num_inversion_steps)
        timesteps = reversed(self.pipe.scheduler.timesteps)

        for t in timesteps:
            # Predict noise
            if guidance_scale > 1.0:
                latent_input = torch.cat([latent, latent])
                text_input = torch.cat([uncond_embeddings, text_embeddings])
                noise_pred = self.pipe.unet(latent_input, t, encoder_hidden_states=text_input).sample
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = self.pipe.unet(latent, t, encoder_hidden_states=text_embeddings).sample

            # Inverse DDIM step
            alpha_prod_t = self.pipe.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.pipe.scheduler.alphas_cumprod[t - 1]
                if t > 0
                else self.pipe.scheduler.final_alpha_cumprod
            )

            # Compute x_t from x_{t-1} (inverse of denoising)
            # x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * eps
            # We have x_{t-1} and want x_t
            pred_x0 = (latent - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
            latent = torch.sqrt(alpha_prod_t_prev) * pred_x0 + torch.sqrt(1 - alpha_prod_t_prev) * noise_pred

        return latent

    def extract_signal(self, latent: torch.Tensor) -> np.ndarray:
        """
        Extract watermark signal from latent's Fourier domain.

        Args:
            latent: Latent tensor of shape (B, C, H, W)

        Returns:
            signal: np.ndarray of shape (B, code_length)
        """
        batch_size = latent.shape[0]
        channel = self.config.watermark_channel
        code_length = self.config.code_length

        signals = np.zeros((batch_size, code_length), dtype=np.float32)

        for b in range(batch_size):
            channel_data = latent[b, channel].cpu().numpy()

            # FFT with shift
            fft_data = np.fft.fftshift(np.fft.fft2(channel_data))

            # Extract signal from each ring
            for ring_idx in range(code_length):
                positions = self.mask.get_ring_positions(ring_idx)
                if len(positions) == 0:
                    continue

                ring_sum = 0.0
                for y, x in positions:
                    sign = self.sign_pattern[y, x]
                    ring_sum += np.real(fft_data[y, x]) * sign

                signals[b, ring_idx] = ring_sum / len(positions)

        return signals

    def correlate(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute correlation between extracted signal and all vendor codes.

        Args:
            signal: np.ndarray of shape (B, code_length)

        Returns:
            correlations: np.ndarray of shape (B, num_vendors)
        """
        # Normalize signal
        signal_norm = signal / (np.linalg.norm(signal, axis=1, keepdims=True) + 1e-8)

        # Normalize codes
        codes_norm = self.codes_matrix / (
            np.linalg.norm(self.codes_matrix, axis=1, keepdims=True) + 1e-8
        )

        # Compute correlations (dot product of normalized vectors)
        correlations = signal_norm @ codes_norm.T

        return correlations

    def detect(
        self,
        image: torch.Tensor,
        prompt: str = "",
        return_details: bool = False,
    ) -> DetectionResult:
        """
        Detect watermark in a single image.

        Args:
            image: Image tensor of shape (1, 3, H, W) or (3, H, W), values in [-1, 1]
            prompt: Text prompt used for generation (helps inversion accuracy)
            return_details: If True, include extracted signal in result

        Returns:
            DetectionResult with vendor attribution
        """
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # DDIM inversion
        latent = self.ddim_inversion(image, prompt)

        # Extract signal
        signal = self.extract_signal(latent)

        # Correlate with codes
        correlations = self.correlate(signal)[0]  # Remove batch dim

        # Find best match
        max_corr = float(np.max(correlations))
        detected_vendor = int(np.argmax(correlations)) if max_corr > self.threshold else None

        return DetectionResult(
            detected_vendor=detected_vendor,
            correlations=correlations,
            max_correlation=max_corr,
            confidence=max_corr - self.threshold,
            extracted_signal=signal[0] if return_details else None,
        )

    def detect_batch(
        self,
        images: torch.Tensor,
        prompts: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> List[DetectionResult]:
        """
        Detect watermarks in a batch of images.

        Args:
            images: Image tensor of shape (B, 3, H, W), values in [-1, 1]
            prompts: List of prompts (one per image), or None for empty prompts
            show_progress: Show progress bar

        Returns:
            List of DetectionResult, one per image
        """
        batch_size = images.shape[0]

        if prompts is None:
            prompts = [""] * batch_size

        results = []
        iterator = range(batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Detecting watermarks")

        for i in iterator:
            result = self.detect(images[i:i+1], prompts[i])
            results.append(result)

        return results

    def compute_metrics(
        self,
        results: List[DetectionResult],
        true_vendors: List[int],
    ) -> Dict[str, float]:
        """
        Compute detection metrics.

        Args:
            results: List of DetectionResult from detect_batch
            true_vendors: Ground truth vendor IDs

        Returns:
            Dict with accuracy, per-vendor precision/recall, etc.
        """
        n = len(results)
        correct = sum(
            1 for r, t in zip(results, true_vendors)
            if r.detected_vendor == t
        )

        # Per-vendor metrics
        vendor_ids = sorted(set(true_vendors))
        per_vendor = {}
        for vid in vendor_ids:
            true_positives = sum(
                1 for r, t in zip(results, true_vendors)
                if t == vid and r.detected_vendor == vid
            )
            false_positives = sum(
                1 for r, t in zip(results, true_vendors)
                if t != vid and r.detected_vendor == vid
            )
            false_negatives = sum(
                1 for r, t in zip(results, true_vendors)
                if t == vid and r.detected_vendor != vid
            )

            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)

            per_vendor[vid] = {"precision": precision, "recall": recall}

        return {
            "accuracy": correct / n,
            "num_samples": n,
            "num_correct": correct,
            "per_vendor": per_vendor,
            "mean_max_correlation": float(np.mean([r.max_correlation for r in results])),
        }


def create_detector(
    config: EmbeddingConfig,
    pipe: Any,
    codes: np.ndarray,
    detection_threshold: float = 0.3,
) -> CDWDetector:
    """
    Convenience function to create a configured detector.

    Args:
        config: Embedding configuration
        pipe: Diffusion pipeline
        codes: Vendor codes array of shape (num_vendors, code_length)
        detection_threshold: Minimum correlation for positive detection

    Returns:
        Configured CDWDetector
    """
    detector = CDWDetector(
        config=config,
        pipe=pipe,
        detection_threshold=detection_threshold,
    )
    detector.set_all_codes(codes)
    return detector
