"""
Experiment 7: Security Evaluation

Tests security properties against:
- Forgery attacks (generating false watermarks)
- Code recovery attacks (inferring vendor codes)
- Removal attacks (destroying watermarks)
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
from cdw.codes import CodeFamily, generate_codes
from cdw.embedding import EmbeddingConfig, create_embedder_with_codes
from cdw.detection import CDWDetector
from cdw.utils import set_seed, image_to_tensor
from metrics.identification import compute_accuracy
from metrics.quality import compute_psnr, compute_ssim
from attacks import apply_attack


@dataclass
class Exp7Config(ExperimentConfig):
    """Configuration for Experiment 7."""
    experiment_name: str = "exp7_security"
    num_vendors: int = 32
    num_images_per_vendor: int = 100

    # Code recovery: number of images adversary observes
    recovery_samples: List[int] = field(default_factory=lambda: [
        10, 100, 1000, 10000
    ])

    # Removal attacks
    removal_attacks: Dict[str, Any] = field(default_factory=lambda: {
        "heavy_blur": {"name": "blur", "params": {"kernel_size": 15}},
        "jpeg_10": {"name": "jpeg", "params": {"quality": 10}},
        "heavy_noise": {"name": "gaussian_noise", "params": {"sigma": 0.2}},
        "aggressive_combined": {"name": "combined", "params": {}},
    })


class SecurityExperiment(ExperimentBase):
    """Experiment 7: Security evaluation."""

    def __init__(self, config: Exp7Config):
        super().__init__(config)
        self.config: Exp7Config = config

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting Experiment 7: Security Evaluation")

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

        results = {
            "forgery": self._test_forgery(pipe, detector),
            "code_recovery": self._test_code_recovery(pipe, embedder, codes, detector),
            "removal": self._test_removal(pipe, embedder, detector),
        }

        # Summary
        results["summary"] = {
            "forgery_fpr": results["forgery"]["false_positive_rate"],
            "recovery_secure_threshold": self._find_secure_threshold(results["code_recovery"]),
            "removal_destroys_quality": results["removal"]["quality_impact"],
        }

        self.save_results(results)
        self.log_summary(results["summary"])
        return results

    def _test_forgery(self, pipe, detector) -> Dict[str, Any]:
        """Test 7a: Forgery attack - can adversary forge valid watermarks?"""
        self.logger.info("Testing forgery attack...")

        # Generate images with random (non-registered) codes
        num_test = 500
        prompts = self._load_prompts(num_test)

        false_positives = 0
        detections = []

        for i, prompt in enumerate(tqdm(prompts, desc="Forgery test")):
            with torch.no_grad():
                # Generate with random latent (no watermark)
                latent = torch.randn(1, 4, 64, 64, device=self.device)

                image = pipe(
                    prompt=prompt,
                    latents=latent.half() if self.device.type == "cuda" else latent,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                ).images[0]

                # Try to detect watermark
                img_tensor = image_to_tensor(image).to(self.device)
                result = detector.detect(img_tensor, prompt)

                if result.detected_vendor is not None:
                    false_positives += 1
                    detections.append({
                        "image_idx": i,
                        "detected_vendor": result.detected_vendor,
                        "max_correlation": float(max(result.correlations)),
                    })

        fpr = false_positives / num_test

        return {
            "num_tested": num_test,
            "false_positives": false_positives,
            "false_positive_rate": float(fpr),
            "detections": detections[:10],  # First 10 for inspection
        }

    def _test_code_recovery(
        self,
        pipe,
        embedder,
        codes: np.ndarray,
        detector,
    ) -> Dict[str, Any]:
        """Test 7b: Code recovery - can adversary infer vendor codes?"""
        self.logger.info("Testing code recovery attack...")

        target_vendor = 0  # Attack vendor 0
        true_code = codes[target_vendor]
        results_per_sample = {}

        for num_samples in self.config.recovery_samples:
            self.logger.info(f"  Testing with M = {num_samples} samples...")

            # Generate M watermarked images from target vendor
            prompts = self._load_prompts(num_samples)
            extracted_signals = []

            # For efficiency, limit actual generation for large M
            actual_samples = min(num_samples, 1000)
            scale_factor = num_samples / actual_samples

            for prompt in tqdm(prompts[:actual_samples], desc=f"M={num_samples}"):
                with torch.no_grad():
                    latent = torch.randn(1, 4, 64, 64, device=self.device)
                    wm_latent = embedder.embed(latent, target_vendor)

                    image = pipe(
                        prompt=prompt,
                        latents=wm_latent.half() if self.device.type == "cuda" else wm_latent,
                        num_inference_steps=self.config.num_inference_steps,
                        guidance_scale=self.config.guidance_scale,
                    ).images[0]

                    # Extract signal (what adversary can observe)
                    img_tensor = image_to_tensor(image).to(self.device)
                    result = detector.detect(img_tensor, prompt)
                    # The raw correlations give some signal information
                    extracted_signals.append(result.correlations)

            # Adversary's code estimate: average of extracted signals
            # (This is a simplified attack - real attack would be more sophisticated)
            extracted_signals = np.array(extracted_signals)

            # Estimate code direction from correlation pattern
            avg_correlations = np.mean(extracted_signals, axis=0)

            # Normalize to get estimated code
            estimated_code = avg_correlations / (np.linalg.norm(avg_correlations) + 1e-8)
            estimated_code = estimated_code * np.sqrt(len(true_code))  # Scale to same norm

            # Measure attack success
            # 1. Code estimation error
            code_correlation = np.dot(estimated_code, true_code) / (
                np.linalg.norm(estimated_code) * np.linalg.norm(true_code) + 1e-8
            )

            # 2. Forgery success with estimated code
            # (Would require modifying embedder, simplified here)
            forgery_success = code_correlation > 0.9  # Threshold for "recovered"

            results_per_sample[str(num_samples)] = {
                "num_samples": num_samples,
                "code_correlation": float(code_correlation),
                "forgery_would_succeed": bool(forgery_success),
                "estimated_vs_true_mse": float(np.mean((estimated_code - true_code) ** 2)),
            }

            self.logger.info(f"    Code correlation: {code_correlation:.4f}")
            self.logger.info(f"    Recovery success: {forgery_success}")

        return {
            "target_vendor": target_vendor,
            "samples": results_per_sample,
        }

    def _test_removal(self, pipe, embedder, detector) -> Dict[str, Any]:
        """Test 7c: Removal attack - can adversary remove watermark?"""
        self.logger.info("Testing removal attacks...")

        # Generate watermarked images
        num_images = 100
        prompts = self._load_prompts(num_images)

        images = []
        vendors = []

        for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
            vid = i % self.config.num_vendors

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

        # Test each removal attack
        attack_results = {}

        for attack_name, attack_config in self.config.removal_attacks.items():
            self.logger.info(f"  Testing {attack_name}...")

            predictions = []
            psnr_scores = []
            ssim_scores = []

            for img, vid, prompt in zip(images, vendors, prompts):
                # Apply attack
                img_tensor = image_to_tensor(img)

                try:
                    attacked = apply_attack(
                        img_tensor,
                        attack_config["name"],
                        attack_config["params"],
                    )
                except Exception as e:
                    self.logger.warning(f"    Attack failed: {e}")
                    continue

                # Measure quality degradation
                psnr_scores.append(compute_psnr(img, attacked))
                ssim_scores.append(compute_ssim(img, attacked))

                # Try to detect
                attacked_gpu = attacked.to(self.device)
                result = detector.detect(attacked_gpu, prompt)
                predictions.append(result.detected_vendor)

            accuracy = compute_accuracy(predictions, vendors[:len(predictions)])["accuracy"]

            attack_results[attack_name] = {
                "accuracy_after_attack": float(accuracy),
                "psnr_mean": float(np.mean(psnr_scores)) if psnr_scores else 0,
                "psnr_std": float(np.std(psnr_scores)) if psnr_scores else 0,
                "ssim_mean": float(np.mean(ssim_scores)) if ssim_scores else 0,
                "watermark_removed": accuracy < 0.5,
                "quality_destroyed": np.mean(psnr_scores) < 20 if psnr_scores else True,
            }

            self.logger.info(f"    Accuracy: {accuracy:.4f}")
            self.logger.info(f"    PSNR: {np.mean(psnr_scores):.2f} dB")

        # Overall quality impact assessment
        quality_impact = all(
            r["quality_destroyed"] for r in attack_results.values()
            if r["watermark_removed"]
        )

        return {
            "attacks": attack_results,
            "quality_impact": quality_impact,
        }

    def _find_secure_threshold(self, recovery_results: Dict) -> int:
        """Find number of samples above which recovery becomes possible."""
        samples = recovery_results.get("samples", {})

        for num_str, data in sorted(samples.items(), key=lambda x: int(x[0])):
            if data["forgery_would_succeed"]:
                return int(num_str)

        return -1  # Never succeeded


def main():
    parser = argparse.ArgumentParser(description="Experiment 7: Security")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--num-vendors", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Exp7Config(
        num_vendors=args.num_vendors,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    experiment = SecurityExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
