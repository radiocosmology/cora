from distutils.core import setup  
from distutils.extension import Extension
from distutils import sysconfig
from Cython.Distutils import build_ext  
import re

import numpy as np

if re.search('gcc', sysconfig.get_config_var('CC')) is None:
    args = []
else:
    args = ['-fopenmp']


setup(  
   name = 'CubicSpline',  
   ext_modules=[ Extension('cubicspline', ['cubicspline.pyx'],
                           include_dirs=[np.get_include()],
                           extra_compile_args=args,
                           extra_link_args=args,
                           ),
                 Extension('_sphbessel_c', ['_sphbessel_c.pyx'],
                           include_dirs=[np.get_include()],
                           libraries=['gsl', 'gslcblas']),
                 Extension('trilinearmap', ['trilinearmap.pyx'],
                           include_dirs=[np.get_include()])
                 ],  
   cmdclass = {'build_ext': build_ext}  
)
