"""
Experiment 1: Vendor Scaling

Validates that CDW scales to many vendors with graceful accuracy degradation.

Key questions:
1. How does identification accuracy change as N increases?
2. How does CDW compare to random codes and RingID?
3. Does empirical accuracy match theoretical predictions?
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.base import ExperimentBase, ExperimentConfig
from cdw import CDWEmbedder, CDWDetector, RingMask
from cdw.codes import generate_codes, CodeFamily, get_code_properties
from cdw.embedding import EmbeddingConfig, create_embedder_with_codes
from cdw.detection import create_detector
from cdw.utils import set_seed, image_to_tensor, tensor_to_image
from metrics.identification import compute_accuracy, compute_confusion_matrix, compute_topk_accuracy
from metrics.theoretical import theoretical_accuracy, estimate_noise_sigma, compare_theory_vs_empirical
from metrics.statistical import compute_confidence_interval, paired_t_test


@dataclass
class Exp1Config(ExperimentConfig):
    """Configuration for Experiment 1."""
    experiment_name: str = "exp1_vendor_scaling"

    # Vendor counts to test
    vendor_counts: List[int] = None

    def __post_init__(self):
        if self.vendor_counts is None:
            self.vendor_counts = [4, 8, 16, 32, 64]


class VendorScalingExperiment(ExperimentBase):
    """
    Experiment 1: Vendor Scaling

    Tests how identification accuracy scales with number of vendors.
    """

    def __init__(self, config: Exp1Config):
        super().__init__(config)
        self.config: Exp1Config = config

    def run(self) -> Dict[str, Any]:
        """Run the vendor scaling experiment."""
        self.logger.info("Starting Experiment 1: Vendor Scaling")

        # Load pipeline
        pipe = self._load_pipeline()

        # Results storage
        all_results = {
            "config": self.config.to_dict(),
            "runs": [],
            "summary": {},
        }

        # Run for each vendor count
        for num_vendors in self.config.vendor_counts:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing with N = {num_vendors} vendors")
            self.logger.info(f"{'='*60}")

            vendor_results = self._run_for_vendor_count(pipe, num_vendors)
            all_results["runs"].append({
                "num_vendors": num_vendors,
                **vendor_results
            })

        # Compute summary statistics
        all_results["summary"] = self._compute_summary(all_results["runs"])

        # Save and return
        self.save_results(all_results)
        self.log_summary(all_results["summary"])

        return all_results

    def _run_for_vendor_count(
        self,
        pipe,
        num_vendors: int,
    ) -> Dict[str, Any]:
        """Run experiment for a specific number of vendors."""
        results = {
            "cdw": {"accuracies": [], "correlations": []},
            "random": {"accuracies": [], "correlations": []},
        }

        # Run multiple times for statistical significance
        for run_idx in range(self.config.num_runs):
            seed = self.config.random_seed + run_idx
            set_seed(seed)
            self.logger.info(f"Run {run_idx + 1}/{self.config.num_runs} (seed={seed})")

            # Create embedders
            cdw_embedder, cdw_codes, cdw_props = self._create_embedder(
                num_vendors, CodeFamily.WALSH_HADAMARD, seed
            )
            random_embedder, random_codes, random_props = self._create_embedder(
                num_vendors, CodeFamily.RANDOM, seed
            )

            # Generate and detect
            cdw_result = self._generate_and_detect(
                pipe, cdw_embedder, cdw_codes, num_vendors, seed
            )
            random_result = self._generate_and_detect(
                pipe, random_embedder, random_codes, num_vendors, seed + 1000
            )

            results["cdw"]["accuracies"].append(cdw_result["accuracy"])
            results["cdw"]["correlations"].extend(cdw_result["correlations"])
            results["random"]["accuracies"].append(random_result["accuracy"])
            results["random"]["correlations"].extend(random_result["correlations"])

        # Compute statistics
        for method in ["cdw", "random"]:
            accs = results[method]["accuracies"]
            mean, lower, upper = compute_confidence_interval(accs)
            results[method]["accuracy_mean"] = mean
            results[method]["accuracy_std"] = np.std(accs)
            results[method]["accuracy_ci_lower"] = lower
            results[method]["accuracy_ci_upper"] = upper

        # Statistical comparison
        if len(results["cdw"]["accuracies"]) > 1:
            ttest = paired_t_test(
                results["cdw"]["accuracies"],
                results["random"]["accuracies"],
                alternative="greater"
            )
            results["cdw_vs_random_ttest"] = ttest

        # Theoretical comparison
        results["theoretical_accuracy"] = theoretical_accuracy(
            alpha=self.config.watermark_strength,
            sigma=0.1,  # Will be estimated from data
            code_length=self.config.code_length,
            num_vendors=num_vendors,
        )

        return results

    def _create_embedder(
        self,
        num_vendors: int,
        code_family: CodeFamily,
        seed: int,
    ):
        """Create CDW embedder with specified code family."""
        config = EmbeddingConfig(
            code_length=self.config.code_length,
            watermark_strength=self.config.watermark_strength,
            watermark_channel=self.config.watermark_channel,
            r_min=self.config.ring_r_min,
            r_max=self.config.ring_r_max,
            latent_size=self.config.latent_size,
        )

        embedder, codes, props = create_embedder_with_codes(
            num_vendors=num_vendors,
            code_family=code_family,
            config=config,
            seed=seed,
        )

        embedder.to(self.device)
        return embedder, codes, props

    def _generate_and_detect(
        self,
        pipe,
        embedder: CDWEmbedder,
        codes: np.ndarray,
        num_vendors: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Generate watermarked images and detect watermarks."""
        set_seed(seed)

        # Load prompts
        total_images = num_vendors * self.config.num_images_per_vendor
        prompts = self._load_prompts(total_images)

        # Create detector
        detector = CDWDetector(
            config=EmbeddingConfig(
                code_length=self.config.code_length,
                watermark_strength=self.config.watermark_strength,
                watermark_channel=self.config.watermark_channel,
                r_min=self.config.ring_r_min,
                r_max=self.config.ring_r_max,
                latent_size=self.config.latent_size,
            ),
            pipe=pipe,
            detection_threshold=self.config.detection_threshold,
            num_inversion_steps=self.config.ddim_inversion_steps,
        )
        detector.set_all_codes(codes)
        detector.to(self.device)

        # Generate and detect
        predictions = []
        ground_truth = []
        all_correlations = []

        pbar = tqdm(
            total=total_images,
            desc=f"Generating (N={num_vendors})",
            leave=False
        )

        for vendor_id in range(num_vendors):
            for img_idx in range(self.config.num_images_per_vendor):
                prompt_idx = vendor_id * self.config.num_images_per_vendor + img_idx
                prompt = prompts[prompt_idx]

                # Generate with watermark
                with torch.no_grad():
                    # Get initial latent
                    latent = torch.randn(
                        1, 4, self.config.latent_size, self.config.latent_size,
                        device=self.device,
                        dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    )

                    # Embed watermark
                    watermarked_latent = embedder.embed(latent.float(), vendor_id)
                    watermarked_latent = watermarked_latent.half() if self.device.type == "cuda" else watermarked_latent

                    # Generate image
                    image = pipe(
                        prompt=prompt,
                        latents=watermarked_latent,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                    ).images[0]

                    # Convert to tensor
                    image_tensor = image_to_tensor(image).to(self.device)

                    # Detect
                    result = detector.detect(image_tensor, prompt)

                    predictions.append(result.detected_vendor)
                    ground_truth.append(vendor_id)
                    all_correlations.append(result.correlations.tolist())

                pbar.update(1)

        pbar.close()

        # Compute accuracy
        accuracy_metrics = compute_accuracy(predictions, ground_truth)

        return {
            "accuracy": accuracy_metrics["accuracy"],
            "predictions": predictions,
            "ground_truth": ground_truth,
            "correlations": all_correlations,
            **accuracy_metrics,
        }

    def _compute_summary(self, runs: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics across all runs."""
        summary = {
            "vendor_counts": [],
            "cdw_accuracies": [],
            "random_accuracies": [],
            "improvement": [],
        }

        for run in runs:
            summary["vendor_counts"].append(run["num_vendors"])
            summary["cdw_accuracies"].append(run["cdw"]["accuracy_mean"])
            summary["random_accuracies"].append(run["random"]["accuracy_mean"])
            summary["improvement"].append(
                run["cdw"]["accuracy_mean"] - run["random"]["accuracy_mean"]
            )

        return summary


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Vendor Scaling")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--num-vendors", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Create config
    if args.config:
        config = Exp1Config.from_yaml(args.config)
    else:
        config = Exp1Config(
            vendor_counts=args.num_vendors,
            num_images_per_vendor=args.num_images,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            random_seed=args.seed,
        )

    # Run experiment
    experiment = VendorScalingExperiment(config)
    results = experiment.run()

    print("\nExperiment completed successfully!")
    print(f"Results saved to: {experiment.output_path}")


if __name__ == "__main__":
    main()
