from setuptools import setup, find_packages

setup(
    author="Samuel JS Wright",
    description="Functions for generating summary plots and statistics from sklearn models",
    name="modelingutils",
    version="0.1.0",
    packages=find_packages(include=["modelingutils", "modelingutils.*"]),
    install_requires=[
        "scikit-learn >= 1.2.0",
        "matplotlib >= 3.7.1",
        "pandas >= 1.5.2",
    ],
    python_requires=">=3.9",
)
