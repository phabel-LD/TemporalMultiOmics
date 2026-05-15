"""Setup script for TMO library and tmopy high‑level API."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tmo",                          # core package name; tmopy is also installed
    version="1.0.0",                     # update to final version
    author="Phabel A. Lopez-Delgado",
    author_email="phabel@lcg.unam.mx",
    description="Temporal Multi-Omics (TMO) foundation model for single-cell ATAC+RNA data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phabel-LD/tmo",
    packages=find_packages(include=['tmo', 'tmo.*', 'tmopy', 'tmopy.*']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "scanpy>=1.9.0",
        "anndata>=0.9.0",
        "scikit-learn>=1.2.0",
        "scipy>=1.10.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.12.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "tqdm>=4.65.0",
        "muon>=0.1.0",          # for reading 10x Multiome data (used by tmopy)
        "gseapy>=1.0.0",        # for GO enrichment (used by tmopy)
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "wandb>=0.15.0",
        ],
        "plotting": [
            "seaborn>=0.12.0",
        ],
    },
)