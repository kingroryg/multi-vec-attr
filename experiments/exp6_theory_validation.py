"""
Experiment 6: Theory Validation

Validates that Theorem 1's accuracy predictions match empirical results.
Tests across multiple conditions to verify the theoretical framework.
"""

import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.base import ExperimentBase, ExperimentConfig
from cdw.codes import CodeFamily
from cdw.embedding import EmbeddingConfig, create_embedder_with_codes
from cdw.detection import CDWDetector
from cdw.utils import set_seed, image_to_tensor
from metrics.identification import compute_accuracy
from metrics.theoretical import theoretical_accuracy, estimate_noise_sigma


@dataclass
class Exp6Config(ExperimentConfig):
    """Configuration for Experiment 6."""
    experiment_name: str = "exp6_theory_validation"
    num_images_per_condition: int = 200

    # Conditions to test (varied combinations)
    vendor_counts: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    watermark_strengths: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20])
    code_lengths: List[int] = field(default_factory=lambda: [32, 64, 128])


class TheoryValidationExperiment(ExperimentBase):
    """Experiment 6: Theory validation."""

    def __init__(self, config: Exp6Config):
        super().__init__(config)
        self.config: Exp6Config = config

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting Experiment 6: Theory Validation")

        pipe = self._load_pipeline()
        set_seed(self.config.random_seed)

        results = {
            "conditions": [],
            "scatter_data": {"empirical": [], "theoretical": []},
            "summary": {},
        }

        # Generate test conditions
        conditions = self._generate_conditions()
        self.logger.info(f"Testing {len(conditions)} conditions...")

        for i, (num_vendors, strength, code_length) in enumerate(conditions):
            self.logger.info(
                f"[{i+1}/{len(conditions)}] "
                f"N={num_vendors}, alpha={strength}, L={code_length}"
            )

            # Skip invalid combinations
            if num_vendors > code_length:
                self.logger.info("  Skipping: N > L for Walsh-Hadamard")
                continue

            # Create embedder
            embedder, codes, _ = create_embedder_with_codes(
                num_vendors=num_vendors,
                code_family=CodeFamily.WALSH_HADAMARD,
                config=EmbeddingConfig(
                    code_length=code_length,
                    watermark_strength=strength,
                    watermark_channel=self.config.watermark_channel,
                ),
                seed=self.config.random_seed,
            )
            embedder.to(self.device)

            # Create detector
            detector = CDWDetector(
                config=EmbeddingConfig(
                    code_length=code_length,
                    watermark_strength=strength,
                    watermark_channel=self.config.watermark_channel,
                ),
                pipe=pipe,
                detection_threshold=self.config.detection_threshold,
            )
            detector.set_all_codes(codes)
            detector.to(self.device)

            # Generate images
            images, vendors, prompts, correlations = self._generate_and_detect(
                pipe, embedder, detector, num_vendors
            )

            # Compute empirical accuracy
            predictions = [
                int(np.argmax(corr)) if np.max(corr) > self.config.detection_threshold else None
                for corr in correlations
            ]
            empirical_acc = compute_accuracy(predictions, vendors)["accuracy"]

            # Estimate noise and compute theoretical prediction
            sigma_hat = estimate_noise_sigma(np.array(correlations), codes)
            theoretical_acc = theoretical_accuracy(
                alpha=strength,
                sigma=sigma_hat,
                code_length=code_length,
                num_vendors=num_vendors,
            )

            # Store results
            condition_result = {
                "num_vendors": num_vendors,
                "watermark_strength": strength,
                "code_length": code_length,
                "empirical_accuracy": float(empirical_acc),
                "theoretical_accuracy": float(theoretical_acc),
                "sigma_hat": float(sigma_hat),
                "absolute_error": abs(empirical_acc - theoretical_acc),
            }
            results["conditions"].append(condition_result)
            results["scatter_data"]["empirical"].append(empirical_acc)
            results["scatter_data"]["theoretical"].append(theoretical_acc)

            self.logger.info(f"  Empirical: {empirical_acc:.4f}")
            self.logger.info(f"  Theoretical: {theoretical_acc:.4f}")
            self.logger.info(f"  Error: {condition_result['absolute_error']:.4f}")

        # Compute overall statistics
        empirical = np.array(results["scatter_data"]["empirical"])
        theoretical = np.array(results["scatter_data"]["theoretical"])

        if len(empirical) > 1:
            correlation = np.corrcoef(empirical, theoretical)[0, 1]
            r_squared = correlation ** 2
            mae = np.mean(np.abs(empirical - theoretical))
            rmse = np.sqrt(np.mean((empirical - theoretical) ** 2))

            # Linear regression
            slope, intercept = np.polyfit(theoretical, empirical, 1)
        else:
            r_squared = mae = rmse = 0.0
            slope, intercept = 1.0, 0.0

        results["summary"] = {
            "r_squared": float(r_squared),
            "mean_absolute_error": float(mae),
            "rmse": float(rmse),
            "regression_slope": float(slope),
            "regression_intercept": float(intercept),
            "num_conditions": len(results["conditions"]),
            "theory_valid": r_squared > 0.9 and abs(slope - 1.0) < 0.1,
        }

        self.save_results(results)
        self.log_summary(results["summary"])
        return results

    def _generate_conditions(self) -> List[Tuple[int, float, int]]:
        """Generate all test conditions."""
        conditions = []

        for num_vendors in self.config.vendor_counts:
            for strength in self.config.watermark_strengths:
                for code_length in self.config.code_lengths:
                    # Only valid if num_vendors <= code_length
                    if num_vendors <= code_length:
                        conditions.append((num_vendors, strength, code_length))

        return conditions

    def _generate_and_detect(
        self,
        pipe,
        embedder,
        detector,
        num_vendors: int,
    ) -> Tuple[List, List, List, List]:
        """Generate images and get detection correlations."""
        total = self.config.num_images_per_condition
        prompts = self._load_prompts(total)
        images = []
        vendors = []
        correlations = []

        pbar = tqdm(total=total, desc="Generate+Detect")

        for i in range(total):
            vid = i % num_vendors
            prompt = prompts[i]

            with torch.no_grad():
                latent = torch.randn(1, 4, 64, 64, device=self.device)
                wm_latent = embedder.embed(latent, vid)

                image = pipe(
                    prompt=prompt,
                    latents=wm_latent.half() if self.device.type == "cuda" else wm_latent,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                ).images[0]

                # Immediate detection
                img_tensor = image_to_tensor(image).to(self.device)
                result = detector.detect(img_tensor, prompt)

            images.append(image)
            vendors.append(vid)
            correlations.append(result.correlations)
            pbar.update(1)

        pbar.close()
        return images, vendors, prompts, correlations


def main():
    parser = argparse.ArgumentParser(description="Experiment 6: Theory Validation")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Exp6Config(
        num_images_per_condition=args.num_images,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    experiment = TheoryValidationExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
