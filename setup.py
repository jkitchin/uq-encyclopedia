"""Setup script for UQ Encyclopedia."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.org"
if readme_file.exists():
    with open(readme_file, 'r') as f:
        long_description = f.read()
else:
    long_description = "UQ Encyclopedia - Comprehensive uncertainty quantification benchmark"

setup(
    name="uq-encyclopedia",
    version="0.1.0",
    description="Comprehensive benchmark framework for uncertainty quantification methods",
    long_description=long_description,
    long_description_content_type="text/plain",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/uq-encyclopedia",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "mapie>=0.8.0",
        "h5py>=3.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "notebooks": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "ipywidgets>=8.0.0",
        ],
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
