"""
Base Experiment Class

Common infrastructure for all experiments.
"""

import os
import json
import yaml
import torch
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from cdw.utils import set_seed, get_device, ensure_dir


@dataclass
class ExperimentConfig:
    """Base configuration for experiments."""
    # Experiment identification
    experiment_name: str = "experiment"
    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Model settings
    model_id: str = "runwayml/stable-diffusion-v1-5"
    scheduler: str = "ddim"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    # Watermark settings
    code_length: int = 64
    code_family: str = "walsh_hadamard"
    watermark_strength: float = 0.1
    watermark_channel: int = 3

    # Ring geometry
    ring_r_min: float = 4.0
    ring_r_max: float = 20.0
    latent_size: int = 64

    # Detection
    detection_threshold: float = 0.3
    ddim_inversion_steps: int = 50

    # Experiment settings
    num_vendors: int = 32
    num_images_per_vendor: int = 100
    num_runs: int = 3
    random_seed: int = 42

    # Paths
    output_dir: str = "results"
    data_dir: str = "data"
    prompts_file: str = "data/prompts/coco_captions.txt"

    # Hardware
    device: str = "auto"
    batch_size: int = 4
    num_workers: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def save(self, path: str):
        ensure_dir(os.path.dirname(path))
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


class ExperimentBase(ABC):
    """Base class for all experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.logger = self._setup_logging()
        self.device = self._setup_device()

        # Create output directory
        self.output_path = Path(config.output_dir) / config.experiment_name / config.run_id
        ensure_dir(str(self.output_path))

        # Save config
        self.config.save(str(self.output_path / "config.yaml"))

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the experiment."""
        logger = logging.getLogger(self.config.experiment_name)
        logger.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        return logger

    def _setup_device(self) -> torch.device:
        """Set up compute device."""
        if self.config.device == "auto":
            return get_device()
        return torch.device(self.config.device)

    def _load_pipeline(self):
        """Load the diffusion pipeline."""
        from diffusers import StableDiffusionPipeline, DDIMScheduler

        self.logger.info(f"Loading model: {self.config.model_id}")

        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            safety_checker=None,
        )

        # Set scheduler
        if self.config.scheduler == "ddim":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)

        return pipe

    def _load_prompts(self, num_prompts: int) -> List[str]:
        """Load text prompts for generation."""
        prompts_path = self.config.prompts_file

        if os.path.exists(prompts_path):
            with open(prompts_path, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Default prompts if file doesn't exist
            self.logger.warning(f"Prompts file not found: {prompts_path}. Using default prompts.")
            prompts = [
                "A photo of a cat sitting on a windowsill",
                "A beautiful sunset over the ocean",
                "A city street at night with neon lights",
                "A forest path in autumn with fallen leaves",
                "A mountain landscape with snow-capped peaks",
                "A still life painting of fruits on a table",
                "A portrait of a person with dramatic lighting",
                "An abstract colorful geometric pattern",
            ]

        # Repeat prompts if needed
        while len(prompts) < num_prompts:
            prompts = prompts + prompts

        return prompts[:num_prompts]

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Run the experiment. Must be implemented by subclasses."""
        pass

    def save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        self.results = results

        # Save as JSON
        results_path = self.output_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to: {results_path}")

    def log_summary(self, results: Dict[str, Any]):
        """Log a summary of results."""
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 60)

        for key, value in results.items():
            if isinstance(value, dict):
                self.logger.info(f"{key}:")
                for k, v in value.items():
                    if isinstance(v, float):
                        self.logger.info(f"  {k}: {v:.4f}")
                    else:
                        self.logger.info(f"  {k}: {v}")
            elif isinstance(value, float):
                self.logger.info(f"{key}: {value:.4f}")
            else:
                self.logger.info(f"{key}: {value}")

        self.logger.info("=" * 60)
