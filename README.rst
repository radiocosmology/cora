=========
Cosmo21cm
=========

A package for simulating skies for 21cm Intensity Mapping, with a lot of bonus
utility code for dealing with Healpix maps, spherical co-ordinates etc.

This package has many dependencies. The following are essential:

* `Python <http://www.python.org/>`_ 2.7 or later
* `Numpy <http://scipy.org/>`_ (probably version 1.5 or later)
* `Scipy <http://scipy.org/>`_
* `Cython <http://cython.org/>`_

It is **strongly** recommended that you use optimized versions of `Numpy` and
`Scipy` compiled against a multithreaded `BLAS` library such as the `Intel MKL
<http://www.intel.com/software/products/mkl/>`_ `ACML
<http://developer.amd.com/libraries/acml>`_.

For full functionality there are many other packages which are required:

* `GSL <http://www.gnu.org/software/gsl/>`_, the Gnu Scientific Library (GSL),
  for compiling spherical bessel routines. Ensure   it's in your `LIBRARY_PATH`,
  `LD_LIBRARY_PATH` and `CPATH`.
* `Healpy <https://github.com/healpy/healpy>`_

There are several Cython modules which need to be built, to do that just::

    > cd cosmoutils
    > python setup.py build_ext --inplace



