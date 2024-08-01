"""Build cython extensions.

The full project config can be found in `pyproject.toml`. `setup.py` is still
required to build cython extensions.
"""

import os
import re
import sysconfig

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# Decide whether to use OpenMP or not
if ("CORA_NO_OPENMP" in os.environ) or (
    re.search("gcc", sysconfig.get_config_var("CC")) is None
):
    print("Not using OpenMP")
    omp_args = ["-std=c99"]
else:
    omp_args = ["-std=c99", "-fopenmp"]

extensions = [
    # Cubic spline extension
    Extension(
        "cora.util.cubicspline",
        ["cora/util/cubicspline.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=omp_args,
        extra_link_args=omp_args,
    ),
    # Bi-linear map extension
    Extension(
        "cora.util.bilinearmap",
        ["cora/util/bilinearmap.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=omp_args,
        extra_link_args=omp_args,
    ),
    # coord extension
    Extension(
        "cora.util.coord",
        ["cora/util/coord.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=omp_args,
        extra_link_args=omp_args,
    ),
    # particle-mesh gridding extension
    Extension(
        "cora.util.pmesh",
        ["cora/util/pmesh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=omp_args,
        extra_link_args=omp_args,
    ),
]

setup(
    name="cora",  # required
    ext_modules=cythonize(extensions),
)
