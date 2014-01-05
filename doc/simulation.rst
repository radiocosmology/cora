========================
Simulating the Radio Sky
========================

One of the main purposes of ``cora`` is for generating simulations of the
radio sky with a particular focus on cosmological 21cm observations
(frequencies in the range 100 MHz up to 1.4 GHz). It focuses on three
different sources of radio emission:

- 21cm radiation from neutral hydrogen at cosmological distances. Currently
  the support for reionisation is poor.
- Synchrotron from our galaxy, including the effect of emission from a broad
  range of Faraday depths within our galaxy.
- Radio emission from point sources. This incluces multiple components:
	* Real point sources derived from NVSS and VLSS.
	* A synthetic population of dimmer point sources.
	* A gaussian random field to model an unresolved background of sources.

These models are described in more detail in `arXiv:1302.3267`_ and `arXiv:1401.XXXX`_

.. _`arXiv:1302.3267`: http://arxiv.org/abs/1302.3267
.. _`arXiv:1401.XXXX`: http://arxiv.org/abs/1401.XXXX

Command Line Tools
------------------

The primary way of generating simulations is to use the command line tool
``cora-makesky``, which is able to generate maps of each of these components.
If you installed ``cora`` with ``easy_install``, ``pip`` or used the
``setup.py`` this command should be in your path.

A typical call takes the form of::

	$ cora-makesky COMPONENT NSIDE FREQ_LOWER FREQ_UPPER NFREQ outputmap.hdf5

where the required options are:

- ``COMPONENT`` is the type of sky to simulate. Must be one of:
	* ``galaxy`` - synchrotron from our galaxy.
	* ``pointsource`` - emission from radio point sources.
	* ``foreground`` - both of the above.
	* ``21cm`` - cosmological 21cm emission.
- ``NSIDE`` is the Healpix resolution. Must be a power of two. 256 is a good value to start with.
- ``FREQ_LOWER`` the lowest frequency in MHz.
- ``FREQ_UPPER`` the highest frequency in MHz.
- ``NFREQ`` gives the number of evenly spaced frequency channels with in the band.
- ``outmap.hdf5`` replace with your desired output filename.

Map Format
----------

Maps are stored as ``hdf5`` files which can be easily read in Python using
``healpy``. Currently there is minimal structure to the written files. There
is a single dataset called ``map`` which contains ``Healpix`` maps of the
whole sky for each frequency and polarisation. Within this the files are
packed by frequency, polarisation and Healpix pixel.



Selected API
------------

.. automodule:: cora.foreground.galaxy



