"""
Experiment 4: Code Length Ablation

Studies how code length L affects identification accuracy.
Validates theoretical prediction from Theorem 1.
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
from cdw.utils import set_seed, image_to_tensor
from metrics.identification import compute_accuracy
from metrics.theoretical import theoretical_accuracy, estimate_noise_sigma


@dataclass
class Exp4Config(ExperimentConfig):
    """Configuration for Experiment 4."""
    experiment_name: str = "exp4_code_length"
    num_vendors: int = 32
    num_images_per_vendor: int = 100

    # Code lengths to test
    code_lengths: List[int] = field(default_factory=lambda: [
        16, 32, 64, 128, 256
    ])


class CodeLengthExperiment(ExperimentBase):
    """Experiment 4: Code length ablation."""

    def __init__(self, config: Exp4Config):
        super().__init__(config)
        self.config: Exp4Config = config

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting Experiment 4: Code Length Ablation")

        pipe = self._load_pipeline()
        set_seed(self.config.random_seed)

        results = {"code_lengths": {}, "theory_comparison": []}

        for code_length in self.config.code_lengths:
            self.logger.info(f"Testing code length L = {code_length}...")

            # For Walsh-Hadamard, num_vendors must be <= code_length
            # If code_length < num_vendors, use subset of vendors
            effective_vendors = min(self.config.num_vendors, code_length)

            self.logger.info(f"  Using {effective_vendors} vendors")

            # Create embedder
            embedder, codes, _ = create_embedder_with_codes(
                num_vendors=effective_vendors,
                code_family=CodeFamily.WALSH_HADAMARD,
                config=EmbeddingConfig(
                    code_length=code_length,
                    watermark_strength=self.config.watermark_strength,
                    watermark_channel=self.config.watermark_channel,
                ),
                seed=self.config.random_seed,
            )
            embedder.to(self.device)

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
                pipe, embedder, effective_vendors
            )

            # Detect and compute accuracy
            predictions, correlations = self._detect_all(images, prompts, detector)
            accuracy_result = compute_accuracy(predictions, vendors)

            # Estimate noise parameter
            sigma_hat = estimate_noise_sigma(correlations, codes)

            # Theoretical prediction
            theory_acc = theoretical_accuracy(
                alpha=self.config.watermark_strength,
                sigma=sigma_hat,
                code_length=code_length,
                num_vendors=effective_vendors,
            )

            results["code_lengths"][str(code_length)] = {
                "code_length": code_length,
                "num_vendors": effective_vendors,
                "empirical_accuracy": float(accuracy_result["accuracy"]),
                "theoretical_accuracy": float(theory_acc),
                "sigma_hat": float(sigma_hat),
            }

            results["theory_comparison"].append({
                "code_length": code_length,
                "empirical": float(accuracy_result["accuracy"]),
                "theoretical": float(theory_acc),
            })

            self.logger.info(f"  Empirical accuracy: {accuracy_result['accuracy']:.4f}")
            self.logger.info(f"  Theoretical accuracy: {theory_acc:.4f}")
            self.logger.info(f"  Estimated sigma: {sigma_hat:.4f}")

        # Compute correlation between theory and empirical
        empirical = [r["empirical"] for r in results["theory_comparison"]]
        theoretical = [r["theoretical"] for r in results["theory_comparison"]]

        if len(empirical) > 1:
            correlation = np.corrcoef(empirical, theoretical)[0, 1]
            r_squared = correlation ** 2
        else:
            r_squared = 1.0

        results["summary"] = {
            "theory_empirical_r_squared": float(r_squared),
            "best_code_length": max(
                results["code_lengths"].values(),
                key=lambda x: x["empirical_accuracy"]
            )["code_length"],
        }

        self.save_results(results)
        self.log_summary(results["summary"])
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
        """Detect vendor for all images and collect correlations."""
        predictions = []
        correlations = []

        for img, prompt in tqdm(zip(images, prompts), total=len(images), desc="Detecting"):
            img_tensor = image_to_tensor(img).to(self.device)
            result = detector.detect(img_tensor, prompt)
            predictions.append(result.detected_vendor)
            correlations.append(result.correlations)

        return predictions, np.array(correlations)


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Code Length")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--num-vendors", type=int, default=32)
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Exp4Config(
        num_vendors=args.num_vendors,
        num_images_per_vendor=args.num_images,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    experiment = CodeLengthExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
