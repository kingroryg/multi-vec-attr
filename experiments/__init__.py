"""
CDW Experiment Scripts

Each experiment validates a specific claim from the paper.
"""

from .base import ExperimentBase, ExperimentConfig

from .exp1_vendor_scaling import VendorScalingExperiment, Exp1Config
from .exp2_robustness import RobustnessExperiment, Exp2Config
from .exp3_image_quality import ImageQualityExperiment, Exp3Config
from .exp4_code_length import CodeLengthExperiment, Exp4Config
from .exp5_code_families import CodeFamilyExperiment, Exp5Config
from .exp6_theory_validation import TheoryValidationExperiment, Exp6Config
from .exp7_security import SecurityExperiment, Exp7Config
from .exp8_scalability import ScalabilityExperiment, Exp8Config

__all__ = [
    "ExperimentBase",
    "ExperimentConfig",
    "VendorScalingExperiment",
    "Exp1Config",
    "RobustnessExperiment",
    "Exp2Config",
    "ImageQualityExperiment",
    "Exp3Config",
    "CodeLengthExperiment",
    "Exp4Config",
    "CodeFamilyExperiment",
    "Exp5Config",
    "TheoryValidationExperiment",
    "Exp6Config",
    "SecurityExperiment",
    "Exp7Config",
    "ScalabilityExperiment",
    "Exp8Config",
]
