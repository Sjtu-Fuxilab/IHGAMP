"""Setup script for IHGAMP."""

from setuptools import setup, find_packages

setup(
    name="ihgamp",
    version="0.1.0",
    author="Sanwal Ahmad Zafar",
    author_email="sanwal@sjtu.edu.cn",
    description="Integrative Histopathology-Genomic Analysis for Molecular Phenotyping",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Sjtu-Fuxilab/IHGAMP",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "torch>=2.0.0",
        "openslide-python>=1.2.0",
    ],
)
