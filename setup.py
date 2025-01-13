from setuptools import setup, find_packages

setup(
    name="AutoMushroom",
    version="0.0.0",
    description="A Python package for classification of mushrooms",
    url="https://github.com/polmos02/AutoML-package/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    packages=find_packages(),
    install_requires=[
        "pandas>=2.2.2",
        "scikit-learn>=1.5.1",
        "seaborn>=0.13.2",
        "matplotlib>=3.9.2",
    ]
)