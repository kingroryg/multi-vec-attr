"""
Experiment 3: Image Quality Assessment

Measures the impact of watermarking on perceptual image quality.
Tests FID, LPIPS, PSNR, SSIM across watermark strengths.
"""

import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.base import ExperimentBase, ExperimentConfig
from cdw.codes import CodeFamily
from cdw.embedding import EmbeddingConfig, create_embedder_with_codes
from cdw.detection import CDWDetector
from cdw.utils import set_seed, tensor_to_image
from metrics.quality import compute_fid, compute_lpips, compute_psnr, compute_ssim
from metrics.identification import compute_accuracy


@dataclass
class Exp3Config(ExperimentConfig):
    """Configuration for Experiment 3."""
    experiment_name: str = "exp3_image_quality"
    num_vendors: int = 32
    num_images: int = 1000  # Total images (not per-vendor)

    # Watermark strengths to test
    watermark_strengths: List[float] = field(default_factory=lambda: [
        0.05, 0.10, 0.15, 0.20, 0.30
    ])


class ImageQualityExperiment(ExperimentBase):
    """Experiment 3: Image quality assessment."""

    def __init__(self, config: Exp3Config):
        super().__init__(config)
        self.config: Exp3Config = config

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting Experiment 3: Image Quality")

        pipe = self._load_pipeline()
        set_seed(self.config.random_seed)

        # Load prompts
        prompts = self._load_prompts(self.config.num_images)

        # Generate random latents once (shared across strengths for fair comparison)
        self.logger.info("Pre-generating latent codes...")
        latents = [
            torch.randn(1, 4, 64, 64, device=self.device)
            for _ in range(self.config.num_images)
        ]
        vendor_ids = [i % self.config.num_vendors for i in range(self.config.num_images)]

        results = {"strengths": {}, "summary": {}}

        # Generate unwatermarked baseline images
        self.logger.info("Generating baseline (unwatermarked) images...")
        baseline_images = self._generate_images_from_latents(
            pipe, latents, prompts, embedder=None, vendor_ids=None
        )

        # Test each watermark strength
        for strength in self.config.watermark_strengths:
            self.logger.info(f"Testing strength = {strength}...")

            # Create embedder with this strength
            embedder, codes, _ = create_embedder_with_codes(
                num_vendors=self.config.num_vendors,
                code_family=CodeFamily.WALSH_HADAMARD,
                config=EmbeddingConfig(
                    code_length=self.config.code_length,
                    watermark_strength=strength,
                    watermark_channel=self.config.watermark_channel,
                ),
                seed=self.config.random_seed,
            )
            embedder.to(self.device)

            # Create detector
            detector = CDWDetector(
                config=EmbeddingConfig(
                    code_length=self.config.code_length,
                    watermark_strength=strength,
                    watermark_channel=self.config.watermark_channel,
                ),
                pipe=pipe,
                detection_threshold=self.config.detection_threshold,
            )
            detector.set_all_codes(codes)
            detector.to(self.device)

            # Generate watermarked images with same latents
            watermarked_images = self._generate_images_from_latents(
                pipe, latents, prompts, embedder, vendor_ids
            )

            # Compute quality metrics
            self.logger.info("  Computing quality metrics...")
            fid = compute_fid(baseline_images, watermarked_images)
            lpips_scores = []
            psnr_scores = []
            ssim_scores = []

            for base_img, wm_img in zip(baseline_images, watermarked_images):
                lpips_scores.append(compute_lpips(base_img, wm_img))
                psnr_scores.append(compute_psnr(base_img, wm_img))
                ssim_scores.append(compute_ssim(base_img, wm_img))

            # Compute detection accuracy
            self.logger.info("  Computing detection accuracy...")
            predictions = []
            for img, vid, prompt in zip(watermarked_images, vendor_ids, prompts):
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)
                result = detector.detect(img_tensor, prompt)
                predictions.append(result.detected_vendor)

            accuracy = compute_accuracy(predictions, vendor_ids)["accuracy"]

            # Store results
            results["strengths"][str(strength)] = {
                "strength": strength,
                "fid": float(fid),
                "lpips_mean": float(np.mean(lpips_scores)),
                "lpips_std": float(np.std(lpips_scores)),
                "psnr_mean": float(np.mean(psnr_scores)),
                "psnr_std": float(np.std(psnr_scores)),
                "ssim_mean": float(np.mean(ssim_scores)),
                "ssim_std": float(np.std(ssim_scores)),
                "accuracy": float(accuracy),
            }

            self.logger.info(f"  FID: {fid:.2f}")
            self.logger.info(f"  LPIPS: {np.mean(lpips_scores):.4f}")
            self.logger.info(f"  PSNR: {np.mean(psnr_scores):.2f} dB")
            self.logger.info(f"  SSIM: {np.mean(ssim_scores):.4f}")
            self.logger.info(f"  Accuracy: {accuracy:.4f}")

        # Find Pareto-optimal points (best accuracy-quality tradeoff)
        strengths_data = list(results["strengths"].values())
        results["summary"] = {
            "recommended_strength": self._find_optimal_strength(strengths_data),
            "num_images": self.config.num_images,
            "num_vendors": self.config.num_vendors,
        }

        self.save_results(results)
        self.log_summary(results["summary"])
        return results

    def _generate_images_from_latents(
        self,
        pipe,
        latents: List[torch.Tensor],
        prompts: List[str],
        embedder,
        vendor_ids: List[int],
    ):
        """Generate images from pre-specified latents."""
        images = []
        pbar = tqdm(total=len(latents), desc="Generating")

        for i, (latent, prompt) in enumerate(zip(latents, prompts)):
            with torch.no_grad():
                if embedder is not None:
                    vid = vendor_ids[i]
                    wm_latent = embedder.embed(latent, vid)
                else:
                    wm_latent = latent

                image = pipe(
                    prompt=prompt,
                    latents=wm_latent.half() if self.device.type == "cuda" else wm_latent,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                ).images[0]

            images.append(image)
            pbar.update(1)

        pbar.close()
        return images

    def _find_optimal_strength(self, strengths_data: List[Dict]) -> float:
        """Find optimal strength balancing accuracy and quality."""
        # Simple heuristic: highest accuracy with acceptable quality
        # (FID increase < 5, LPIPS < 0.05)
        acceptable = [
            s for s in strengths_data
            if s["fid"] < 5.0 and s["lpips_mean"] < 0.05
        ]

        if acceptable:
            # Return highest accuracy among acceptable
            return max(acceptable, key=lambda x: x["accuracy"])["strength"]
        else:
            # Fallback: lowest strength with >90% accuracy
            high_acc = [s for s in strengths_data if s["accuracy"] > 0.9]
            if high_acc:
                return min(high_acc, key=lambda x: x["strength"])["strength"]
            return 0.10  # Default


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: Image Quality")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--num-vendors", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Exp3Config(
        num_images=args.num_images,
        num_vendors=args.num_vendors,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    experiment = ImageQualityExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
