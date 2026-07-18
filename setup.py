#!/usr/bin/env python3
"""
Setup script for OBC CubeSat Predictive Maintenance
Install dependencies with: pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="obc-cubesat-soh-prediction",
    version="1.0.0",
    author="Hnainia Mohamed Amine",
    description="Battery State of Health (SOH) prediction for CubeSats using LSTM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HnainiaMohamedAmine29/OBC_Cubesat_Predictive_Maintenance",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=0.24.0",
        "joblib>=1.0.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "ml": [
            "torch>=1.9.0",  # or tensorflow>=2.6.0
            "tensorflow>=2.6.0",
        ],
        "dev": [
            "jupyter>=1.0.0",
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            # Optional CLI commands
        ],
    },
)
