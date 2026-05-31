"""
setup.py  —  MedQuAD-Classify
Allows  pip install -e .  so  'from src.predict import predict'  works
from any directory in the project, not just the root.
"""

from setuptools import setup, find_packages

setup(
    name="medquad_classify",
    version="1.0.0",
    description="Medical question type & department classifier (MedQuAD)",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn>=1.3",
        "pandas>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
        "joblib>=1.3",
        "matplotlib>=3.7",
        "seaborn>=0.12",
    ],
)