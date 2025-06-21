from setuptools import setup, find_packages

setup(
    name="universal-lora",
    version="0.1.0",
    description="Parameter-efficient LoRA wrapper for PyTorch models",
    author="Foivos Ntelemis",
    author_email="f.ntelemis@outlook.com",
    packages=find_packages(include=["universal_lora*"]),
    install_requires=[
        "torch>=2.0",
        "timm>=0.9.12"
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "ruff", "black"]
    },
)