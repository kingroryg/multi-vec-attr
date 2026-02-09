"""
Experiment 8: Scalability Benchmarks

Measures computational costs:
- Code generation time
- Embedding overhead
- Detection time
- Memory usage
"""

import sys
import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import gc

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.base import ExperimentBase, ExperimentConfig
from cdw.codes import CodeFamily, generate_codes
from cdw.embedding import EmbeddingConfig, create_embedder_with_codes
from cdw.detection import CDWDetector
from cdw.utils import set_seed, image_to_tensor


@dataclass
class Exp8Config(ExperimentConfig):
    """Configuration for Experiment 8."""
    experiment_name: str = "exp8_scalability"

    # Benchmark configurations
    vendor_counts_for_timing: List[int] = field(default_factory=lambda: [
        8, 16, 32, 64, 128, 256, 512, 1000
    ])

    num_images_for_benchmark: int = 50
    num_warmup: int = 5
    num_timing_runs: int = 10


class ScalabilityExperiment(ExperimentBase):
    """Experiment 8: Scalability benchmarks."""

    def __init__(self, config: Exp8Config):
        super().__init__(config)
        self.config: Exp8Config = config

    def run(self) -> Dict[str, Any]:
        self.logger.info("Starting Experiment 8: Scalability Benchmarks")

        pipe = self._load_pipeline()
        set_seed(self.config.random_seed)

        results = {
            "code_generation": self._benchmark_code_generation(),
            "embedding": self._benchmark_embedding(pipe),
            "detection": self._benchmark_detection(pipe),
            "memory": self._benchmark_memory(pipe),
            "end_to_end": self._benchmark_end_to_end(pipe),
        }

        # Summary
        results["summary"] = {
            "code_gen_1000_vendors_ms": results["code_generation"].get("1000", {}).get("time_ms", 0),
            "embedding_overhead_ms": results["embedding"]["overhead_ms"],
            "detection_per_image_ms": results["detection"]["per_image_ms"],
            "peak_memory_gb": results["memory"]["peak_memory_gb"],
        }

        self.save_results(results)
        self.log_summary(results["summary"])
        return results

    def _benchmark_code_generation(self) -> Dict[str, Any]:
        """Benchmark code generation time."""
        self.logger.info("Benchmarking code generation...")

        results = {}

        for num_vendors in self.config.vendor_counts_for_timing:
            # Find valid code length (power of 2 >= num_vendors)
            code_length = 1
            while code_length < num_vendors:
                code_length *= 2

            times = []

            for _ in range(self.config.num_timing_runs):
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

                start = time.perf_counter()
                codes = generate_codes(
                    num_codes=num_vendors,
                    code_length=code_length,
                    family=CodeFamily.WALSH_HADAMARD,
                )
                end = time.perf_counter()

                times.append((end - start) * 1000)  # ms

            results[str(num_vendors)] = {
                "num_vendors": num_vendors,
                "code_length": code_length,
                "time_ms": float(np.mean(times)),
                "time_std_ms": float(np.std(times)),
            }

            self.logger.info(f"  N={num_vendors}: {np.mean(times):.2f} ms")

        return results

    def _benchmark_embedding(self, pipe) -> Dict[str, Any]:
        """Benchmark watermark embedding overhead."""
        self.logger.info("Benchmarking embedding overhead...")

        # Create embedder
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

        # Generate test latents
        latents = [
            torch.randn(1, 4, 64, 64, device=self.device)
            for _ in range(self.config.num_images_for_benchmark)
        ]

        # Warmup
        for i in range(self.config.num_warmup):
            with torch.no_grad():
                _ = embedder.embed(latents[i % len(latents)], 0)

        # Time embedding only
        embedding_times = []

        for latent in tqdm(latents, desc="Embedding"):
            vid = np.random.randint(0, self.config.num_vendors)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            with torch.no_grad():
                _ = embedder.embed(latent, vid)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()

            embedding_times.append((end - start) * 1000)

        return {
            "overhead_ms": float(np.mean(embedding_times)),
            "overhead_std_ms": float(np.std(embedding_times)),
            "num_samples": len(latents),
        }

    def _benchmark_detection(self, pipe) -> Dict[str, Any]:
        """Benchmark detection time."""
        self.logger.info("Benchmarking detection...")

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

        # Generate test images
        prompts = self._load_prompts(self.config.num_images_for_benchmark)
        images = []

        self.logger.info("  Generating test images...")
        for i, prompt in enumerate(tqdm(prompts[:10], desc="Gen images")):
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

            images.append((image, prompt))

        # Warmup detection
        for i in range(min(3, len(images))):
            img, prompt = images[i]
            img_tensor = image_to_tensor(img).to(self.device)
            _ = detector.detect(img_tensor, prompt)

        # Time detection
        detection_times = []
        inversion_times = []
        correlation_times = []

        for img, prompt in tqdm(images, desc="Detection"):
            img_tensor = image_to_tensor(img).to(self.device)

            torch.cuda.synchronize() if torch.cuda.is_available() else None

            # Time full detection
            start = time.perf_counter()
            result = detector.detect(img_tensor, prompt)
            end = time.perf_counter()

            detection_times.append((end - start) * 1000)

            # Try to get component times if available
            if hasattr(result, 'inversion_time'):
                inversion_times.append(result.inversion_time * 1000)
            if hasattr(result, 'correlation_time'):
                correlation_times.append(result.correlation_time * 1000)

        return {
            "per_image_ms": float(np.mean(detection_times)),
            "per_image_std_ms": float(np.std(detection_times)),
            "min_ms": float(np.min(detection_times)),
            "max_ms": float(np.max(detection_times)),
            "inversion_ms": float(np.mean(inversion_times)) if inversion_times else None,
            "correlation_ms": float(np.mean(correlation_times)) if correlation_times else None,
            "num_samples": len(images),
        }

    def _benchmark_memory(self, pipe) -> Dict[str, Any]:
        """Benchmark memory usage."""
        self.logger.info("Benchmarking memory...")

        results = {}

        if torch.cuda.is_available():
            # Reset memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

            baseline_memory = torch.cuda.memory_allocated() / 1e9  # GB

            # Create embedder
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

            after_embedder = torch.cuda.memory_allocated() / 1e9

            # Create detector
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

            after_detector = torch.cuda.memory_allocated() / 1e9

            # Generate one image
            with torch.no_grad():
                latent = torch.randn(1, 4, 64, 64, device=self.device)
                wm_latent = embedder.embed(latent, 0)
                image = pipe(
                    prompt="test",
                    latents=wm_latent.half(),
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                ).images[0]

            after_generation = torch.cuda.memory_allocated() / 1e9
            peak_memory = torch.cuda.max_memory_allocated() / 1e9

            # Detect
            img_tensor = image_to_tensor(image).to(self.device)
            _ = detector.detect(img_tensor, "test")

            after_detection = torch.cuda.memory_allocated() / 1e9
            peak_after_detection = torch.cuda.max_memory_allocated() / 1e9

            results = {
                "baseline_memory_gb": float(baseline_memory),
                "after_embedder_gb": float(after_embedder),
                "after_detector_gb": float(after_detector),
                "after_generation_gb": float(after_generation),
                "after_detection_gb": float(after_detection),
                "peak_memory_gb": float(peak_after_detection),
                "embedder_overhead_gb": float(after_embedder - baseline_memory),
                "detector_overhead_gb": float(after_detector - after_embedder),
            }
        else:
            # CPU-only fallback
            import resource
            peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # MB to GB (approx)

            results = {
                "peak_memory_gb": float(peak_memory),
                "note": "CPU-only benchmark, limited accuracy",
            }

        self.logger.info(f"  Peak memory: {results.get('peak_memory_gb', 0):.2f} GB")
        return results

    def _benchmark_end_to_end(self, pipe) -> Dict[str, Any]:
        """Benchmark complete end-to-end throughput."""
        self.logger.info("Benchmarking end-to-end throughput...")

        # Create embedder
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

        prompts = self._load_prompts(self.config.num_images_for_benchmark)

        # Time generation with watermarking
        generation_times = []

        for prompt in tqdm(prompts[:20], desc="E2E"):
            vid = np.random.randint(0, self.config.num_vendors)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            with torch.no_grad():
                latent = torch.randn(1, 4, 64, 64, device=self.device)
                wm_latent = embedder.embed(latent, vid)

                _ = pipe(
                    prompt=prompt,
                    latents=wm_latent.half() if self.device.type == "cuda" else wm_latent,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                ).images[0]

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()

            generation_times.append(end - start)

        # Compare to generation without watermarking
        baseline_times = []

        for prompt in tqdm(prompts[:20], desc="Baseline"):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.perf_counter()

            with torch.no_grad():
                latent = torch.randn(1, 4, 64, 64, device=self.device)

                _ = pipe(
                    prompt=prompt,
                    latents=latent.half() if self.device.type == "cuda" else latent,
                    num_inference_steps=self.config.num_inference_steps,
                    guidance_scale=self.config.guidance_scale,
                ).images[0]

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.perf_counter()

            baseline_times.append(end - start)

        return {
            "generation_with_wm_sec": float(np.mean(generation_times)),
            "generation_baseline_sec": float(np.mean(baseline_times)),
            "overhead_percent": float(
                (np.mean(generation_times) - np.mean(baseline_times))
                / np.mean(baseline_times) * 100
            ),
            "throughput_images_per_minute": float(60.0 / np.mean(generation_times)),
        }


def main():
    parser = argparse.ArgumentParser(description="Experiment 8: Scalability")
    parser.add_argument("--config", type=str, help="Config YAML path")
    parser.add_argument("--num-images", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Exp8Config(
        num_images_for_benchmark=args.num_images,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    experiment = ScalabilityExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
