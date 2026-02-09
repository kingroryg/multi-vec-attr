"""
Experiment 2: Robustness Under Attacks

Measures watermark survival under realistic image perturbations.
Tests JPEG, noise, resize, crop, rotation, color changes, blur.
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
from attacks import apply_attack, list_attacks, ATTACK_REGISTRY
from metrics.identification import compute_accuracy


@dataclass
class Exp2Config(ExperimentConfig):
    """Configuration for Experiment 2."""
    experiment_name: str = "exp2_robustness"
    num_vendors: int = 32
    num_images_per_vendor: int = 100

    # Attacks to test
    attacks: Dict[str, List[Any]] = field(default_factory=lambda: {
        "jpeg": [90, 75, 50, 25],
        "gaussian_noise": [0.01, 0.03, 0.05, 0.10],
        "resize": [0.75, 0.50, 0.25],
        "crop": [0.90, 0.75, 0.50],
        "rotate": [5, 15, 30],
        "brightness": [0.8, 1.2],
        "contrast": [0.8, 1.2],
        "blur": [1.0, 2.0, 3.0],
    })


class RobustnessExperiment(ExperimentBase):
    """Experiment 2: Robustness under attacks."""

    def __init__(self, config: Exp2Config):
        super().__init__(config)
        self.config: Exp2Config = config

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting Experiment 2: Robustness")

        pipe = self._load_pipeline()
        set_seed(self.config.random_seed)

        # Create embedder and detector
        embedder, codes, _ = create_embedder_with_codes(
            num_vendors=self.config.num_vendors,
            code_family=CodeFamily.WALSH_HADAMARD,
            config=EmbeddingConfig(
                code_length=self.config.code_length,
                watermark_strength=self.config.watermark_strength,
                watermark_channel=self.config.watermark_channel,
            ),
            seed=self.config.random_seed,
        )
        embedder.to(self.device)

        detector = CDWDetector(
            config=EmbeddingConfig(
                code_length=self.config.code_length,
                watermark_strength=self.config.watermark_strength,
                watermark_channel=self.config.watermark_channel,
            ),
            pipe=pipe,
            detection_threshold=self.config.detection_threshold,
        )
        detector.set_all_codes(codes)
        detector.to(self.device)

        # Generate watermarked images
        self.logger.info("Generating watermarked images...")
        images, vendors, prompts = self._generate_images(pipe, embedder)

        # Test each attack
        results = {"attacks": {}, "baseline_accuracy": None}

        # Baseline (no attack)
        self.logger.info("Testing baseline (no attack)...")
        baseline_acc = self._test_attack(images, vendors, prompts, detector, None, None)
        results["baseline_accuracy"] = baseline_acc
        self.logger.info(f"Baseline accuracy: {baseline_acc:.4f}")

        # Each attack
        for attack_name, param_values in self.config.attacks.items():
            results["attacks"][attack_name] = {}

            for param_val in param_values:
                self.logger.info(f"Testing {attack_name} = {param_val}...")

                # Determine param name from registry
                attack_config = ATTACK_REGISTRY.get(attack_name)
                if attack_config:
                    param_name = list(attack_config.param_ranges.keys())[0] if attack_config.param_ranges else list(attack_config.default_params.keys())[0]
                    params = {param_name: param_val}
                else:
                    params = {"quality": param_val} if attack_name == "jpeg" else {"sigma": param_val}

                acc = self._test_attack(images, vendors, prompts, detector, attack_name, params)
                results["attacks"][attack_name][str(param_val)] = {
                    "accuracy": acc,
                    "param": param_val,
                }
                self.logger.info(f"  Accuracy: {acc:.4f}")

        self.save_results(results)
        self.log_summary({"baseline": results["baseline_accuracy"], "attacks": len(results["attacks"])})
        return results

    def _generate_images(self, pipe, embedder):
        """Generate watermarked images for testing."""
        total = self.config.num_vendors * self.config.num_images_per_vendor
        prompts = self._load_prompts(total)
        images = []
        vendors = []

        pbar = tqdm(total=total, desc="Generating")
        for vid in range(self.config.num_vendors):
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

    def _test_attack(self, images, vendors, prompts, detector, attack_name, params):
        """Test detection accuracy under a specific attack."""
        predictions = []

        for img, vid, prompt in zip(images, vendors, prompts):
            # Apply attack
            if attack_name:
                img_tensor = image_to_tensor(img)
                attacked = apply_attack(img_tensor, attack_name, params)
            else:
                attacked = image_to_tensor(img)

            # Detect
            attacked = attacked.to(self.device)
            result = detector.detect(attacked, prompt)
            predictions.append(result.detected_vendor)

        acc = compute_accuracy(predictions, vendors)["accuracy"]
        return acc


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Robustness")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--num-vendors", type=int, default=32)
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Exp2Config(
        num_vendors=args.num_vendors,
        num_images_per_vendor=args.num_images,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    experiment = RobustnessExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
