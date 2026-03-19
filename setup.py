from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core requirements (excludes camb which needs Fortran compiler)
install_requires = [
    "healpy>=1.16.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",
    "matplotlib>=3.7.0",
    "astropy>=5.3.0",
    "requests>=2.28.0",
    "tqdm>=4.65.0",
]

setup(
    name="cmb-planck",
    version="1.0.0",
    author="Om Arora",
    description="CMB power spectrum analysis with Planck 2018 public data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Vector-Pi/cmb-planck",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "theory": ["camb>=1.5.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
