from setuptools import setup, find_packages

setup(
    name="deepmistake",              # Replace with your package name
    version="0.1.0",                       # Initial version
    description="DeepMistake Model",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/nvanva/deepmistake",  # URL to your repo
    packages=find_packages(),              # Automatically finds packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',               # Minimum Python version requirement
    install_requires=[                     # List your package dependencies
        "fire==0.6.0",
        "numpy==1.23.3",
        "pandas==1.5.0",
        "scikit_learn==1.1.2",
        "scipy==1.9.1",
        "torch==1.12.1+cu116",
        "tqdm==4.64.1",
        "transformers==4.22.2",
    ],
)
