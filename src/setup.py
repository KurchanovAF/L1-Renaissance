"""
Setup script for L1-Renaissance package.
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="l1renaissance",
    version="0.1.0",
    author="Alexey Kurchanov",
    author_email="",  # Optional
    description="Modern revival of 2006 L1-adaptive filtering algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KurchanovAF/L1-Renaissance",
    project_urls={
        "Bug Tracker": "https://github.com/KurchanovAF/L1-Renaissance/issues",
        "Documentation": "https://github.com/KurchanovAF/L1-Renaissance#readme",
        "Source Code": "https://github.com/KurchanovAF/L1-Renaissance",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "examples": [
            "matplotlib>=3.5.0",
            "scipy>=1.9.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "l1-demo=l1renaissance.examples.basic_demo:main",
        ],
    },
)
