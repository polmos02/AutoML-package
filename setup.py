from setuptools import setup, find_packages

setup(
    name="AutoMushroom",
    version="0.0.0",
    packages=find_packages(),
    description="A Python package for classification of mushrooms",
    url="https://github.com/polmos02/AutoML-package/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
)