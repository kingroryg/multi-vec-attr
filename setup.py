"""
Setup script for CDW experiments package.
"""

from setuptools import setup, find_packages

setup(
    name="cdw-experiments",
    version="0.1.0",
    description="Code-Division Watermarking experiments for diffusion models",
    author="Sarthak Munshi",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pillow>=9.5.0",
        "diffusers>=0.21.0",
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "quality": [
            "lpips>=0.1.4",
            "torchmetrics>=1.0.0",
            "scikit-image>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cdw-exp1=experiments.exp1_vendor_scaling:main",
        ],
    },
)
