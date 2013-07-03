import warnings

from setuptools import setup, Extension, find_packages
from distutils import sysconfig

import re

import numpy as np

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

# Spherical bessel extension (not built by default)
sb_ext = Extension('cora.util._sphbessel_c', [cython_file('cora/util/_sphbessel_c')],
                   include_dirs=[np.get_include()],
                   libraries=['gsl', 'gslcblas'])

# Tri-linear map extension
tm_ext = Extension('cora.util.trilinearmap', [cython_file('cora/util/trilinearmap')],
                   include_dirs=[np.get_include()])


setup(
    name = 'cora',
    version = 0.1,

    packages = find_packages(),
    ext_modules = [ cs_ext, tm_ext],
    requires = ['numpy', 'scipy', 'healpy', 'h5py'],
    package_data = {'cora.signal' : ['data/ps_z1.5.dat'], 'cora.foreground' : ['data/skydata.npz']},
    scripts = ['scripts/cora-makesky'],

    # metadata for upload to PyPI
    author = "J. Richard Shaw",
    author_email = "jrs65@cita.utoronto.ca",
    description = "Simulation and modelling of low frequency radio skies.",
    license = "GPL v3.0",
    url = "http://github.com/jrs65/cora",

    cmdclass = {'build_ext': build_ext}
)
