# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import os
import warnings

from setuptools import setup, Extension, find_packages
from distutils import sysconfig

import re

import numpy as np

import versioneer


# TODO: Python 3 - can remove when moved entirely
def to_native(s):
    """Recursively transform strings in structure to native type.
    """
    from future.utils import text_type, PY2

    # Transform compound types
    if isinstance(s, dict):
        return {to_native(k): to_native(v) for k, v in s.items()}
    if isinstance(s, list):
        return [to_native(v) for v in s]
    if isinstance(s, tuple):
        return tuple(to_native(v) for v in s)

    # Transform strings
    try:
        if PY2 and isinstance(s, text_type):  # PY2
            return s.encode("ascii")
    except NameError:
        pass

    return s


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
    filename = filename + (".pyx" if HAVE_CYTHON else ".c")
    return to_native(filename)


# Cubic spline extension
# args = to_native(args)  # TODO: Python 3
cs_ext = Extension(
    to_native("cora.util.cubicspline"),
    [cython_file("cora/util/cubicspline")],
    include_dirs=[np.get_include()],
    extra_compile_args=args,
    extra_link_args=args,
)

# Bi-linear map extension
bm_ext = Extension(
    to_native("cora.util.bilinearmap"),
    [cython_file("cora/util/bilinearmap")],
    include_dirs=[np.get_include()],
    extra_compile_args=args,
    extra_link_args=args,
)

# Particle mesh extension
pm_ext = Extension(
    to_native("cora.util.pmesh"),
    [cython_file("cora/util/pmesh")],
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
    ext_modules=[cs_ext, bm_ext, pm_ext],
    install_requires=requires,
    extras_require={"sphfunc": ["pygsl"]},
    package_data=to_native(
        {
            "cora.signal": ["data/ps_z1.5.dat", "data/corr_z1.5.dat"],
            "cora.foreground": ["data/skydata.npz", "data/combinedps.dat"],
        }
    ),
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
