from setuptools import setup, find_packages

setup(
    name="protconv",
    version="0.1.9",
    description="Protein fragment geometry and deep learning utilities",
    author="Patrik Florek",
    packages=find_packages(),
    python_requires=">=3.8",
    include_package_data=True,
)
