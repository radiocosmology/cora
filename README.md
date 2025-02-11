# cora - Cosmology in the Radio Band

A package for simulating skies for 21cm Intensity Mapping, with a lot of bonus
utility code for dealing with Healpix maps, spherical co-ordinates etc.

This package has many dependencies. The following are essential:

* `Python <http://www.python.org/>` 3.9 or later
* `Numpy <http://scipy.org/>` (version 2.0 or later)
* `Scipy <http://scipy.org/>`
* `Cython <http://cython.org/>`

It is **strongly** recommended that you use optimized versions of `Numpy` and
`Scipy` compiled against a multithreaded `BLAS` library such as the `Intel MKL
<http://www.intel.com/software/products/mkl/>` or for AMD chips `ACML
<http://developer.amd.com/libraries/acml>`.

For full functionality there are other packages which are required:

* `Healpy <https://github.com/healpy/healpy>` (version 1.15 or later)
* `h5py <http://www.h5py.org/>`

As there are several Cython modules which need to be built the `setup.py` must
be run in some form. Either the standard ::

    $ python setup.py install [--user]

or to develop the package ::

    $ python setup.py develop [--user]

## Referencing

If you use this package in published scientific work, please cite the corresponding Zenodo listing (https://zenodo.org/records/13181020), along with the two papers that describe various aspects of the simulation implementation: https://arxiv.org/abs/1302.0327 and https://arxiv.org/abs/1401.2095.