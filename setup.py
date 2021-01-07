
import os
import warnings

from setuptools import setup, Extension, find_packages
from distutils import sysconfig

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


# Cython extensions
try:
    from Cython.Distutils import build_ext

    HAVE_CYTHON = True
except ImportError as e:
    warnings.warn("Cython not installed.")
    from distutils.command import build_ext

    HAVE_CYTHON = False


def cython_file(filename):
    return filename + (".pyx" if HAVE_CYTHON else ".c")


# Cubic spline extension
# args = to_native(args)  # TODO: Python 3
cs_ext = Extension(
    "cora.util.cubicspline",
    [cython_file("cora/util/cubicspline")],
    include_dirs=[np.get_include()],
    extra_compile_args=args,
    extra_link_args=args,
)

# Bi-linear map extension
bm_ext = Extension(
    "cora.util.bilinearmap",
    [cython_file("cora/util/bilinearmap")],
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
    cmdclass=versioneer.get_cmdclass({"build_ext": build_ext}),
    packages=find_packages(),
    ext_modules=[cs_ext, bm_ext],
    python_requires=">=3.6",
    install_requires=requires,
    extras_require={"sphfunc": ["pygsl"]},
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
