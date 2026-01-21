"""Build cython extensions.

The full project config can be found in `pyproject.toml`. `setup.py` is still
required to build cython extensions.
"""

import os
import tempfile
import sysconfig
import subprocess

import numpy
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

# Subset of `-ffast-math` compiler flags which should
# preserve IEEE compliance
FAST_MATH_ARGS = ["-O3", "-fno-math-errno", "-fno-trapping-math", "-march=native"]


def _compiler_supports_openmp():
    cc = os.environ.get("CC", sysconfig.get_config_var("CC"))
    if cc is None:
        return False

    test_code = r"""
    #include <omp.h>
    int main(void) {
        return omp_get_max_threads();
    }
    """

    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "test.c")
        exe = os.path.join(d, "test")
        with open(src, "w") as f:
            f.write(test_code)

        cmd = [cc, src, "-fopenmp", "-o", exe]
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except Exception:  # noqa: BLE001
            return False

    return True


OMP_ARGS = ["-std=c99"]

if not os.environ.get("CORA_NO_OPENMP"):
    if _compiler_supports_openmp():
        OMP_ARGS.append("-fopenmp")
    else:
        cc = os.environ.get("CC", sysconfig.get_config_var("CC"))
        print(
            f"Compiler `{cc}` does not support OpenMP. "
            "If an OpenMP-supporting compiler is available, "
            "add it to your PATH or use `CC=<compiler> pip install ...` "
            "Alternatively, to suppress this warning, set the environment "
            "variable CORA_NO_OPENMP to any truth-like value."
        )

extensions = [
    # Cubic spline extension
    Extension(
        "cora.util.cubicspline",
        ["cora/util/cubicspline.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[*FAST_MATH_ARGS, *OMP_ARGS],
        extra_link_args=[*FAST_MATH_ARGS, *OMP_ARGS],
    ),
    # Bi-linear map extension
    Extension(
        "cora.util.bilinearmap",
        ["cora/util/bilinearmap.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[*FAST_MATH_ARGS, *OMP_ARGS],
        extra_link_args=[*FAST_MATH_ARGS, *OMP_ARGS],
    ),
    # particle-mesh gridding extension
    Extension(
        "cora.util.pmesh",
        ["cora/util/pmesh.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[*FAST_MATH_ARGS, *OMP_ARGS],
        extra_link_args=[*FAST_MATH_ARGS, *OMP_ARGS],
    ),
]

setup(
    name="cora",  # required
    ext_modules=cythonize(extensions),
)
