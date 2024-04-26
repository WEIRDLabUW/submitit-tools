from setuptools import setup, find_packages

requirements = [
    "submitit",
    "tqdm",
    "wandb",
    "torch",
    "fsspec",
]

setup(
    name='submitit_tools',
    packages=find_packages(),
    version='0.1.0',
    description='A package to manage submitit jobs',
    author='Weird Lab',
    requires=requirements
)
