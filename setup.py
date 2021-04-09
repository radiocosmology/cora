import os

from setuptools import setup, Extension, find_packages
from distutils import sysconfig

from Cython.Build import cythonize

import re

import numpy as np

import versioneer


# Decide whether to use OpenMP or not
if ("CORA_NO_OPENMP" in os.environ) or (
    re.search("gcc", sysconfig.get_config_var("CC")) is None
):
    print("Not using OpenMP")
    args = []
else:
    args = ["-fopenmp"]


# Cubic spline extension
cs_ext = Extension(
    "cora.util.cubicspline",
    ["cora/util/cubicspline.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=args,
    extra_link_args=args,
)

# Bi-linear map extension
bm_ext = Extension(
    "cora.util.bilinearmap",
    ["cora/util/bilinearmap.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=args,
    extra_link_args=args,
)

# coord extension
cr_ext = Extension(
    "cora.util.coord",
    ["cora/util/coord.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=args,
    extra_link_args=args,
)

# Load the requirements list
with open("requirements.txt", "r") as fh:
    requires = fh.read().split()

# Load the description
with open("README.rst", "r") as fh:
    long_description = fh.read()


setup(
    name="cora",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    ext_modules=cythonize([cs_ext, bm_ext, cr_ext]),
    python_requires=">=3.7",
    install_requires=requires,
    package_data={
        "cora.signal": ["data/ps_z1.5.dat", "data/corr_z1.5.dat"],
        "cora.foreground": ["data/skydata.npz", "data/combinedps.dat"],
    },
    entry_points="""
        [console_scripts]
        cora-makesky=cora.scripts.makesky:cli
    """,
    # metadata for upload to PyPI
    author="J. Richard Shaw",
    author_email="richard@phas.ubc.ca",
    description="Simulation and modelling of low frequency radio skies.",
    long_description=long_description,
    license="MIT",
    url="http://github.com/radiocosmology/cora",
)
