=========
Cosmo21cm
=========

A package for simulating skies for 21cm Intensity Mapping, with a lot of bonus
utility code for dealing with Healpix maps, spherical co-ordinates etc.

This package has many dependencies. The following are essential:

* Python 2.7 or later
* Numpy (probably version 1.5 or later)
* Scipy
* Cython

For full functionality there are many other packages which are required:

* Gnu Scientific Library (GSL), for compiling spherical bessel routines. Ensure
  it's in your `LIBRARY_PATH`, `LD_LIBRARY_PATH` and `CPATH`.
* Healpy

There are several Cython modules which need to be built, to do that just::

    > cd cosmoutils
    > python setup.py build_ext --inplace

