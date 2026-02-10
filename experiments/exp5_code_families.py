"""
Experiment 5: Code Family Comparison

Compares Walsh-Hadamard, Gold, and Random codes.
Tests orthogonality benefits for multi-vendor watermarking.
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
from cdw.codes import CodeFamily, generate_codes, compute_code_statistics
from cdw.embedding import EmbeddingConfig, create_embedder_with_codes
from cdw.detection import CDWDetector
from cdw.utils import set_seed, image_to_tensor
from metrics.identification import compute_accuracy, compute_confusion_matrix


@dataclass
class Exp5Config(ExperimentConfig):
    """Configuration for Experiment 5."""
    experiment_name: str = "exp5_code_families"
    num_images_per_vendor: int = 100

    # Code families to test
    code_families: List[str] = field(default_factory=lambda: [
        "walsh_hadamard", "gold", "random"
    ])

    # Vendor counts to test
    vendor_counts: List[int] = field(default_factory=lambda: [
        16, 32, 64
    ])


class CodeFamilyExperiment(ExperimentBase):
    """Experiment 5: Code family comparison."""

    def __init__(self, config: Exp5Config):
        super().__init__(config)
        self.config: Exp5Config = config

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting Experiment 5: Code Family Comparison")

        pipe = self._load_pipeline()
        set_seed(self.config.random_seed)

        results = {"families": {}, "summary": {}}

        for family_name in self.config.code_families:
            results["families"][family_name] = {}

            for num_vendors in self.config.vendor_counts:
                self.logger.info(f"Testing {family_name} with N = {num_vendors}...")

                # Get code family enum
                if family_name == "walsh_hadamard":
                    code_family = CodeFamily.WALSH_HADAMARD
                elif family_name == "gold":
                    code_family = CodeFamily.GOLD
                else:
                    code_family = CodeFamily.RANDOM

                # Check if this configuration is valid
                # Walsh-Hadamard requires code_length >= num_vendors
                code_length = self.config.code_length
                if code_family == CodeFamily.WALSH_HADAMARD and num_vendors > code_length:
                    self.logger.warning(
                        f"  Skipping: Walsh-Hadamard requires L >= N "
                        f"(L={code_length}, N={num_vendors})"
                    )
                    continue

                # Create embedder
                try:
                    embedder, codes, _ = create_embedder_with_codes(
                        num_vendors=num_vendors,
                        code_family=code_family,
                        config=EmbeddingConfig(
                            code_length=code_length,
                            watermark_strength=self.config.watermark_strength,
                            watermark_channel=self.config.watermark_channel,
                        ),
                        seed=self.config.random_seed,
                    )
                    embedder.to(self.device)
                except Exception as e:
                    self.logger.error(f"  Failed to create embedder: {e}")
                    continue

                # Compute code statistics
                code_stats = compute_code_statistics(codes)

                # Create detector
                detector = CDWDetector(
                    config=EmbeddingConfig(
                        code_length=code_length,
                        watermark_strength=self.config.watermark_strength,
                        watermark_channel=self.config.watermark_channel,
                    ),
                    pipe=pipe,
                    detection_threshold=self.config.detection_threshold,
                )
                detector.set_all_codes(codes)
                detector.to(self.device)

                # Generate and test images
                images, vendors, prompts = self._generate_images(
                    pipe, embedder, num_vendors
                )

                # Detect
                predictions = self._detect_all(images, prompts, detector)
                accuracy_result = compute_accuracy(predictions, vendors)
                confusion = compute_confusion_matrix(predictions, vendors, num_vendors)

                results["families"][family_name][str(num_vendors)] = {
                    "num_vendors": num_vendors,
                    "accuracy": float(accuracy_result["accuracy"]),
                    "max_cross_correlation": float(code_stats["max_cross_correlation"]),
                    "mean_cross_correlation": float(code_stats["mean_cross_correlation"]),
                    "confusion_matrix": confusion.tolist(),
                }

                self.logger.info(f"  Accuracy: {accuracy_result['accuracy']:.4f}")
                self.logger.info(f"  Max cross-correlation: {code_stats['max_cross_correlation']:.4f}")

        # Summarize best family per vendor count
        summary = {}
        for num_vendors in self.config.vendor_counts:
            best_family = None
            best_accuracy = -1

            for family_name in self.config.code_families:
                if str(num_vendors) in results["families"].get(family_name, {}):
                    acc = results["families"][family_name][str(num_vendors)]["accuracy"]
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_family = family_name

            summary[f"best_at_N{num_vendors}"] = {
                "family": best_family,
                "accuracy": best_accuracy,
            }

        results["summary"] = summary

        self.save_results(results)
        self.log_summary(summary)
        return results

    def _generate_images(self, pipe, embedder, num_vendors: int):
        """Generate watermarked images for testing."""
        total = num_vendors * self.config.num_images_per_vendor
        prompts = self._load_prompts(total)
        images = []
        vendors = []

        pbar = tqdm(total=total, desc="Generating")
        for vid in range(num_vendors):
            for i in range(self.config.num_images_per_vendor):
                idx = vid * self.config.num_images_per_vendor + i
                prompt = prompts[idx]

                with torch.no_grad():
                    latent = torch.randn(1, 4, 64, 64, device=self.device)
                    wm_latent = embedder.embed(latent, vid)

                    image = pipe(
                        prompt=prompt,
                        latents=wm_latent.half() if self.device.type == "cuda" else wm_latent,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                    ).images[0]

                images.append(image)
                vendors.append(vid)
                pbar.update(1)

        pbar.close()
        return images, vendors, prompts

    def _detect_all(self, images, prompts, detector):
        """Detect vendor for all images."""
        predictions = []

        for img, prompt in tqdm(zip(images, prompts), total=len(images), desc="Detecting"):
            img_tensor = image_to_tensor(img).to(self.device)
            result = detector.detect(img_tensor, prompt)
            predictions.append(result.detected_vendor)

        return predictions


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Code Families")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Exp5Config(
        num_images_per_vendor=args.num_images,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    experiment = CodeFamilyExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
