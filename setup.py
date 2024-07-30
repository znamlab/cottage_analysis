from setuptools import setup, find_packages

setup(
    name="cottage_analysis",
    version="v2.0",
    packages=find_packages(),
    url="https://github.com/znamlab/cottage_analysis",
    license="MIT",
    author="Antonin Blot, Yiran He, Petr Znamenskyi",
    author_email="antonin.blot@crick.ac.uk",
    description="Common functions for analysis",
    install_requires=[
        "numpy",
        "pandas",
        "pathlib",
        "flexiznam @ git+ssh://git@github.com/znamlab/flexiznam.git",
        "znamutils @ git+ssh://git@github.com/znamlab/znamutils.git",
        "matplotlib",
        "scipy",
        "tables",
        "scikit-learn",
        "tqdm",
        "numba",
        "numba_progress",
        "scikit-image",
        "defopt",
    ],
)
