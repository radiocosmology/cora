=========
Richard's code
=========

Here's an initial dump of all my code, to be organised into the main python_km tree at a later date.

Currently there is code to generate all sorts of foreground maps:

- Seperable angular/frequency foregrounds: ``C_l(\nu,\nu') = A_l B(\nu, \nu')``. Specific models implemented from Santos, Cooray and Knox. See ``foregroundsck.py``
- LOFAR style synchrotron (Velic et al 2008). See ``lofar.py``.
- Point sources (using Di Matteo et al 2002 double power law model). *Note* that currently point source maps are returned in Janskys (to be fixed). See ``pointsource.py``

There are also routines to calculate correlation functions for both signal and foregrounds:

- A general class for calculating redshift distorted correlation functions is in ``corr.py``.
- Code specifically for 21cm signal is in ``corr21cm.py``.
- Foreground correlation functions (for SCK type foregrounds only) are in ``foregroundsck.py``

Notes
-----
**Important**: remember to compile the cubicspline routines. You'll need to have cython installed. Just run ``python setup.py build_ext --inplace``.

The correlation function routines can be slow as they require lots of integrals for explicit calculations. Caching the results is recommended, and there are methods for generating and loading the cache files. One is included by default ``data/corr0.dat`` which is generated from the matter power spectrum at redshift z=0.

Todo
----
Things still to do are:

- Regularise small scales in the correlation functions (currently they don't quite diverge)
- Make point source maps return temperatures.




