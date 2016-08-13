==================================
cora - Cosmology in the Radio Band
==================================

A package for simulating skies for 21cm Intensity Mapping, with a lot of bonus
utility code for dealing with Healpix maps, spherical co-ordinates etc.

This package has many dependencies. The following are essential:

* `Python <http://www.python.org/>`_ 2.7 or later
* `Numpy <http://scipy.org/>`_ (version 1.7 or later)
* `Scipy <http://scipy.org/>`_
* `Cython <http://cython.org/>`_

It is **strongly** recommended that you use optimized versions of `Numpy` and
`Scipy` compiled against a multithreaded `BLAS` library such as the `Intel MKL
<http://www.intel.com/software/products/mkl/>`_ or for AMD chips `ACML
<http://developer.amd.com/libraries/acml>`_.

For full functionality there are other packages which are required:

* `Healpy <https://github.com/healpy/healpy>`_ (version 1.8 or later)
* `h5py <http://www.h5py.org/>`_

As there are several Cython modules which need to be built the `setup.py` must
be run in some form. Either the standard ::

    $ python setup.py install [--user]

or to develop the package ::

    $ python setup.py develop [--user]
