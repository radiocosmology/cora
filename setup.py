import warnings

from setuptools import setup, Extension, find_packages
from distutils import sysconfig

import re

import numpy as np

import cora


if re.search('gcc', sysconfig.get_config_var('CC')) is None:
    args = []
else:
    args = ['-fopenmp']


try:
    from Cython.Distutils import build_ext

    HAVE_CYTHON = True

except ImportError as e:
    warnings.warn("Cython not installed.")
    from distutils.command import build_ext

    HAVE_CYTHON = False


def cython_file(filename):
    filename = filename + ('.pyx' if HAVE_CYTHON else '.c')
    return filename


# Cubic spline extension
cs_ext = Extension('cora.util.cubicspline', [ cython_file('cora/util/cubicspline') ],
                   include_dirs=[ np.get_include() ],
                   extra_compile_args=args,
                   extra_link_args=args)

# Tri-linear map extension
tm_ext = Extension('cora.util.trilinearmap', [cython_file('cora/util/trilinearmap')],
                   include_dirs=[np.get_include()])


setup(
    name='cora',
    version=cora.__version__,

    packages=find_packages(),
    ext_modules=[cs_ext, tm_ext],
    install_requires=['numpy>=1.7', 'scipy>=0.10', 'healpy>=1.8', 'h5py', 'click'],
    extras_require={
        'sphfunc': ["pygsl"]
    },
    package_data={
        'cora.signal': ['data/ps_z1.5.dat', 'data/corr_z1.5.dat'],
        'cora.foreground': ['data/skydata.npz', 'data/combinedps.dat']
    },
    entry_points="""
        [console_scripts]
        cora-makesky=cora.scripts.makesky:cli
    """,

    # metadata for upload to PyPI
    author="J. Richard Shaw",
    author_email="richard@phas.ubc.ca",
    description="Simulation and modelling of low frequency radio skies.",
    license="MIT",
    url="http://github.com/jrs65/cora",

    cmdclass={'build_ext': build_ext}
)
