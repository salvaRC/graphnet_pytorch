from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='graphnet',
    version='0.1.0',
    packages=find_packages(),
    url='https://github.com/salvaRC7/graphnet_pytorch',
    license='CC BY 4.0',
    author='Salva Rühling Cachay',
    author_email='',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
