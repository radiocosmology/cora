# cora - Cosmology in the Radio Band

A package for simulating skies for 21cm Intensity Mapping, with a lot of bonus
utility code for dealing with Healpix maps, spherical co-ordinates etc.

## Installation

This package has many dependencies. The following are essential:

* `Python <http://www.python.org/>` 3.10 or later
* `Numpy <http://numpy.org/>` (version 1.24 or later)
* `Scipy <http://scipy.org/>`
* `Cython <http://cython.org/>`

It is **strongly** recommended that you use optimized versions of `Numpy` and
`Scipy` compiled against a multithreaded `BLAS` library such as the `Intel MKL
<http://www.intel.com/software/products/mkl/>` or for AMD chips `ACML
<http://developer.amd.com/libraries/acml>`.

For full functionality there are other packages which are required:

* `Healpy <https://github.com/healpy/healpy>` (version 1.17 or later)
* `h5py <http://www.h5py.org/>`

If you are building `healpy` yourself, you will also the following libraries:
* `pkg-config`
* `HEALPix`
* `cfitsio`

See <https://healpy.readthedocs.io/en/latest/install.html#building-against-external-healpix-and-cfitsio>
for more information about building `healpy`.

As there are several Cython modules which need to be built the `setup.py` must
be run in some form. The recommended way to do this is ::

    $ pip install [-e] .

If you just need to recompile any Cython modules ::

    $ python setup.py build_ext --inplace

## Referencing

If you use this package in published scientific work, please cite the corresponding Zenodo listing (https://zenodo.org/records/13181020), along with the two papers that describe various aspects of the simulation implementation: https://arxiv.org/abs/1302.0327 and https://arxiv.org/abs/1401.2095.