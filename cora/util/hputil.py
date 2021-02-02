"""
============================================
Healpix Utilities (:mod:`~cora.util.hputil`)
============================================

Convenience functions for dealing with Healpix maps.

Makes heavy use of the `healpy`_ module.

.. _`healpy`: http://github.com/healpy

Forward Transform Routines
==========================

.. autosummary::
    :toctree: generated/

    sphtrans_real
    sphtrans_complex
    sphtrans_real_pol
    sphtrans_complex_pol

Backward Transforms
===================

.. autosummary::
    :toctree: generated/

    sphtrans_inv_real
    sphtrans_inv_complex
    sphtrans_inv_real_pol

Sky Transforms
==============

Transform sets of polarised maps at multiple frequencies.

.. autosummary::
    :toctree: generated/

    sphtrans_sky
    sphtrans_inv_sky

Miscellaneous
=============

.. autosummary::
    :toctree: generated/

    ang_positions
    nside_for_lmax
    pack_alm
    unpack_alm

Co-ordinate Transform
=====================
.. autosummary::
    :toctree: generated/

    coord_x2y
    coord_g2c
    coord_c2g

"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import healpy
import numpy as np

_weight = True
_iter = 2

### Healpy SH routines seem to crash occasionally if using OMP, do disable.
# os.environ['OMP_NUM_THREADS'] = '1'


def ang_positions(nside):
    """Fetch the angular position of each pixel in a map.

    Parameters
    ----------
    nside : scalar
        The size of the map (nside is a definition specific to Healpix).

    Returns
    -------
    angpos : np.ndarray
        The angular position (in spherical polars), of each pixel in a
        Healpix map. Packed at `[ [theta1, phi1], [theta2, phi2], ...]`
    """
    npix = healpy.nside2npix(int(nside))

    angpos = np.empty([npix, 2], dtype=np.float64)

    angpos[:, 0], angpos[:, 1] = healpy.pix2ang(nside, np.arange(npix))

    return angpos


def nside_for_lmax(lmax, accuracy_boost=1):
    """Return an nside appropriate for a spherical harmonic decomposition.

    Parameters
    ----------
    lmax : integer
        Maximum multipole in decomposition.

    Returns
    -------
    nside : integer
        Appropriate nside for decomposition. Is power of two.
    """
    nside = int(2 ** (accuracy_boost + np.ceil(np.log((lmax + 1) / 3.0) / np.log(2.0))))
    return nside


def unpack_alm(alm, lmax, fullm=False):
    """Unpack :math:`a_{lm}` from Healpix format into 2D [l,m] array.

    This only unpacks into the

    Parameters
    ----------
    alm : np.ndarray
        a_lm packed in Healpix format.
    lmax : integer
        The maximum multipole in the a_lm's.
    fullm : boolean, optional
        Write out the negative m values.

    Returns
    -------
    almarray : np.ndarray
        a_lm in a 2D array packed as [l,m]. If `fullm` write out the negative
        m's, packed into the second half of the array (they can be indexed as
        [l,-m]).
    """
    almarray = np.zeros((lmax + 1, lmax + 1), dtype=alm.dtype)

    (almarray.T)[np.triu_indices(lmax + 1)] = alm

    if fullm:
        almarray = _make_full_alm(almarray)

    return almarray


def pack_alm(almarray, lmax=None):
    """Pack :math:`a_{lm}` into Healpix format from 2D [l,m] array.

    This only unpacks into the

    Parameters
    ----------
    almarray : np.ndarray
        a_lm packed in a 2D array as [l,m].
    lmax : integer, optional
        The maximum multipole in the a_lm's. If `None` (default) work it out
        from the first index.

    Returns
    -------
    alm : np.ndarray
        a_lm in Healpix packing. If `fullm` write out the negative
        m's, packed into the second half of the array (they can be indexed as
        [l,-m]).
    """
    if (2 * almarray.shape[1] - 1) == almarray.shape[0]:
        almarray = _make_half_alm(almarray)

    if not lmax:
        lmax = almarray.shape[0] - 1

    alm = (almarray.T)[np.triu_indices(lmax + 1)]

    return alm


def _make_full_alm(alm_half, centered=False):
    ## Construct an array of a_{lm} including both positive and
    ## negative m, from one including only positive m.
    lmax, mmax = alm_half.shape[-2:]

    alm = np.zeros(alm_half.shape[:-2] + (lmax, 2 * mmax - 1), dtype=alm_half.dtype)

    alm_neg = alm_half[..., :, :0:-1].conj()
    mfactor = (-1) ** np.arange(mmax)[:0:-1]
    alm_neg = mfactor * alm_neg

    if not centered:
        alm[..., :lmax, :mmax] = alm_half
        alm[..., :lmax, mmax:] = alm_neg
    else:
        alm[..., :lmax, (mmax - 1) :] = alm_half
        alm[..., :lmax, : (mmax - 1)] = alm_neg

    return alm


def _make_half_alm(alm_full):
    ## Construct an array of a_{lm} including only positive m, from one both
    ## positive and negative m.
    lside, mside = alm_full.shape[-2:]

    alm = np.zeros(alm_full.shape[:-2] + (lside, lside), dtype=alm_full.dtype)

    # Copy zero frequency modes
    alm[..., 0] = alm_full[..., :, 0]

    # Project such that only alms corresponding to a real field are included.
    for mi in range(1, lside):
        alm[..., mi] = 0.5 * (
            alm_full[..., mi] + (-1) ** mi * alm_full[..., -mi].conj()
        )

    return alm


def sphtrans_real(hpmap, lmax=None, lside=None):
    """Spherical Harmonic transform of a real map.

    Parameters
    ----------
    hpmap : np.ndarray[npix]
        A Healpix map.
    lmax : scalar, optional
        The maximum l to calculate. If `None` (default), calculate up
        to ``3*NSIDE - 1``.
    lmax : scalar, optional
        Size of array to return.

    Returns
    -------
    alm : np.ndarray[lside+1, lside+1]
        A 2d array of alms, packed as ``alm[l,m]``.

    Notes
    -----
    This only includes m > 0. As this is the transform of a real field:

    .. math:: a_{l -m} = (-1)^m a_{lm}^*
    """
    if lmax is None:
        lmax = 3 * healpy.npix2nside(hpmap.size) - 1

    if lside is None or lside < lmax:
        lside = lmax

    alm = np.zeros([lside + 1, lside + 1], dtype=np.complex128)

    tlm = healpy.map2alm(
        np.ascontiguousarray(hpmap), lmax=lmax, use_weights=_weight, iter=_iter
    )

    alm[np.triu_indices(lmax + 1)] = tlm

    return alm.T


def sphtrans_complex(hpmap, lmax=None, centered=False, lside=None):
    """Spherical harmonic transform of a complex function.

    Parameters
    ----------
    hpmap : np.ndarray
        A complex Healpix map.
    lmax : scalar, optional
        The maximum l to calculate. If `None` (default), calculate up to 3*nside
        - 1.
    centered : boolean, optional
        If False (default) similar to an FFT, alm[l,:lmax+1] contains m >= 0,
        and the latter half alm[l,lmax+1:], contains m < 0. If True the first
        half of alm[l,:] contains m < 0, and the second half m > 0. m = 0 is
        the central column.

    Returns
    -------
    alm : np.ndarray
        A 2d array of alms, packed as alm[l,m].
    """
    if lmax is None:
        lmax = 3 * healpy.npix2nside(hpmap.size) - 1

    rlm = _make_full_alm(
        sphtrans_real(hpmap.real, lmax=lmax, lside=lside), centered=centered
    )
    ilm = _make_full_alm(
        sphtrans_real(hpmap.imag, lmax=lmax, lside=lside), centered=centered
    )

    alm = rlm + 1.0j * ilm

    return alm


def sphtrans_real_pol(hpmaps, lmax=None, lside=None):
    """Spherical Harmonic transform of polarisation functions on the sky.

    Accepts real T, Q, U and optionally V like maps, and returns :math:`a^T_{lm}`,
    :math:`a^E_{lm}`, :math:`a^B_{lm}` and :math:`a^V_{lm}`.

    Parameters
    ----------
    hpmaps : np.ndarray[npol, npix]
        An array of Healpix maps, assumed to be T, Q, U and potentially V.
    lmax : scalar, optional
        The maximum l to calculate. If `None` (default), calculate up to 3*nside
        - 1.

    Returns
    -------
    alms[npol, lmax+1, lmax+1] : np.ndarray
        A 3d array of alms, packed as alm[pol, l, m].

    Notes
    -----
    This only includes m >= 0. As these are the transforms of a real field:

    .. math:: a_{l -m} = (-1)^m a_{lm}^*
    """
    if lmax is None:
        lmax = 3 * healpy.npix2nside(hpmaps[0].size) - 1

    if lside is None or lside < lmax:
        lside = lmax

    npol = len(hpmaps)

    alms = np.zeros([npol, lside + 1, lside + 1], dtype=np.complex128)
    hpmaps = np.ascontiguousarray(hpmaps)

    tlms = healpy.map2alm(
        [hpmap for hpmap in hpmaps[:3]], lmax=lmax, use_weights=_weight, iter=_iter
    )

    for i in range(3):
        alms[i][np.triu_indices(lmax + 1)] = tlms[i]

    # Transform Stokes-V
    if npol == 4:
        alms[3][np.triu_indices(lmax + 1)] = healpy.map2alm(
            np.ascontiguousarray(hpmaps[3]), lmax=lmax, use_weights=_weight, iter=_iter
        )

    return alms.transpose((0, 2, 1))


def sphtrans_complex_pol(hpmaps, lmax=None, centered=False, lside=None):
    """Spherical harmonic transform of the polarisation on the sky (can be
    complex).

    Accepts complex T, Q, U and optionally V like maps, and returns :math:`a^T_{lm}`,
    :math:`a^E_{lm}`, :math:`a^B_{lm}` and :math:`a^V_{lm}`.

    Parameters
    ----------
    hpmaps : np.ndarray
         A list of complex Healpix maps, assumed to be T, Q, U (and optionally V).
    lmax : scalar, optional
        The maximum l to calculate. If `None` (default), calculate up to 3*nside
        - 1.
    centered : boolean, optional
        If False (default) similar to an FFT, alm[l,:lmax+1] contains m >= 0,
        and the latter half alm[l,lmax+1:], contains m < 0. If True the first
        half opf alm[l,:] contains m < 0, and the second half m > 0. m = 0 is
        the central column.

    Returns
    -------
    alm_T, alm_E, alm_B : np.ndarray
        A 2d array of alms, packed as alm[l,m].
    """
    if lmax is None:
        lmax = 3 * healpy.npix2nside(hpmaps[0].size) - 1

    rlms = _make_full_alm(
        sphtrans_real_pol([hpmap.real for hpmap in hpmaps], lmax=lmax, lside=lside),
        centered=centered,
    )
    ilms = _make_full_alm(
        sphtrans_real_pol([hpmap.imag for hpmap in hpmaps], lmax=lmax, lside=lside),
        centered=centered,
    )

    alms = [rlm + 1.0j * ilm for rlm, ilm in zip(rlms, ilms)]

    return alms


def sphtrans_inv_real(alm, nside):
    """Inverse spherical harmonic transform onto a real field.

    Parameters
    ----------
    alm : np.ndarray
        The array of alms. Only expects the real half of the array.
    nside : integer
        The Healpix resolution of the final map.

    Returns
    -------
    hpmaps : np.ndarray
    """
    if alm.shape[1] != alm.shape[0]:
        raise Exception("a_lm array wrong shape.")

    almp = pack_alm(alm)

    return healpy.alm2map(almp, nside, verbose=False)


def sphtrans_inv_real_pol(alm, nside):
    """Inverse spherical harmonic transform onto a real polarised field.

    Parameters
    ----------
    alm : np.ndarray[npol, lmax+1, lmax+1]
        The array of alms. Only expects the real half of the array. The first
        index is over polarisation T, Q, U (and optionally V).
    nside : integer
        The Healpix resolution of the final map.

    Returns
    -------
    hpmaps : np.ndarray[npol, 12*nside**2]
        The T, Q, U and optionally V maps of the sky.
    """
    npol = alm.shape[0]

    if alm.shape[1] != alm.shape[2] or not (npol == 3 or npol == 4):
        raise Exception("a_lm array wrong shape.")

    almp = [pack_alm(alm[0]), pack_alm(alm[1]), pack_alm(alm[2])]

    maps = np.zeros((npol, 12 * nside ** 2), dtype=np.float64)

    maps[:3] = np.array(healpy.alm2map(almp, nside, verbose=False))

    if npol == 4:
        maps[3] = healpy.alm2map(pack_alm(alm[3]), nside, verbose=False)

    return np.array(maps)


def sphtrans_inv_complex(alm, nside):
    """Inverse spherical harmonic transform onto a complex field.

    Parameters
    ----------
    alm : np.ndarray
        The array of alms. Expects both positive and negative alm.
    nside : integer
        The Healpix resolution of the final map.

    Returns
    -------
    hpmaps : np.ndarray
        Complex Healpix maps.
    """
    if alm.shape[1] != (2 * alm.shape[0] - 1):
        raise Exception("a_lm array wrong shape: " + repr(alm.shape))

    almr = _make_half_alm(alm)

    almi = 1.0j * (alm[:, : almr.shape[1]] - almr)

    return sphtrans_inv_real(almr, nside) + 1.0j * sphtrans_inv_real(almi, nside)


def sphtrans_sky(skymap, lmax=None):
    """Transform a 3d skymap, slice by slice.

    This will perform the polarised transformation if there are three
    dimensions, and one is the appropriate length (3 or 4).

    Parameters
    ----------
    skymap : np.ndarray[freq, healpix_index] or np.ndarray[freq, pol, healpix_index]

    lmax : integer, optional
        Maximum l to transform.

    Returns
    -------
    alms : np.ndarray[frequencies, pol, l, m]
    """
    nfreq = skymap.shape[0]
    pol = (len(skymap.shape) == 3) and (skymap.shape[1] >= 3)  # Pol if 3 or 4
    npol = skymap.shape[1]

    if lmax is None:
        lmax = 3 * healpy.npix2nside(skymap.shape[-1]) - 1

    if pol:
        alm_freq = np.empty((nfreq, npol, lmax + 1, lmax + 1), dtype=np.complex128)
    else:
        alm_freq = np.empty((nfreq, lmax + 1, lmax + 1), dtype=np.complex128)

    for i in range(nfreq):
        if pol:
            alm_freq[i] = np.array(
                sphtrans_real_pol(skymap[i].astype(np.float64), lmax)
            )
        else:
            alm_freq[i] = sphtrans_real(skymap[i].astype(np.float64), lmax)

    return alm_freq


def sphtrans_inv_sky(alm, nside):
    """Invert a set of alms back into a 3d sky.

    If the polarisation dimension is length is 3 or 4, this will automatically
    do the polarised transform.

    Parameters
    ----------
    alm : np.ndarray[freq, pol, l, m]
        Expects the full range (both positive and negative m).
    nside : integer
        Resolution of final Healpix maps.

    Returns
    -------
    skymaps : np.ndarray[freq, npol, healpix_index]
    """
    nfreq = alm.shape[0]
    npol = alm.shape[1]
    pol = npol >= 3

    sky_freq = np.empty(
        (nfreq, alm.shape[1], healpy.nside2npix(nside)), dtype=np.float64
    )

    for i in range(nfreq):
        if pol:
            sky_freq[i] = sphtrans_inv_real_pol(alm[i], nside)
        else:
            sky_freq[i, 0] = sphtrans_inv_real(alm[i, 0], nside)

    return sky_freq


def sphtrans_inv_sky_der1(alm, nside):
    """Invert a set of alms back into a 3d sky and angular derivatives.

    No support for polarization.

    Parameters
    ----------
    alm : np.ndarray[freq, 1, l, m]
        Expects the full range (both positive and negative m).
    nside : integer
        Resolution of final Healpix maps.

    Returns
    -------
    skymaps : np.ndarray[freq, 1, healpix_index]
    d_theta :  np.ndarray[freq, 1, healpix_index]
    d_phi   :  np.ndarray[freq, 1, healpix_index]

    """
    nfreq = alm.shape[0]
    #    npol = alm.shape[1]
    #    pol = (npol >= 3)

    sky_freq = np.empty(
        (nfreq, alm.shape[1], healpy.nside2npix(nside)), dtype=np.float64
    )
    d_theta = np.empty(
        (nfreq, alm.shape[1], healpy.nside2npix(nside)), dtype=np.float64
    )
    d_phi = np.empty((nfreq, alm.shape[1], healpy.nside2npix(nside)), dtype=np.float64)

    for i in range(nfreq):
        almp = pack_alm(alm[i, 0])
        sky_freq[i, 0], d_theta[i, 0], d_phi[i, 0] = healpy.alm2map_der1(almp, nside)

    return sky_freq, d_theta, d_phi


def coord_x2y(map, x, y):
    """Rotate a map from galactic co-ordinates into celestial co-ordinates.

    This will operate on a series of maps (provided the Healpix index is last).

    Parameters
    ----------
    map : np.ndarray[..., npix]
        Healpix map.
    x, y : {'C', 'G', 'E'}
        Coordinate system to transform to/from. Celestial, 'C'; Galactic 'G'; or
        Ecliptic 'E'.

    Returns
    -------
    rotmap : np.ndarray
        The rotated map.
    """

    if x not in ["C", "G", "E"] or y not in ["C", "G", "E"]:
        raise Exception("Co-ordinate system invalid.")

    npix = map.shape[-1]

    angpos = ang_positions(healpy.npix2nside(npix))
    theta, phi = healpy.Rotator(coord=[y, x])(angpos[:, 0], angpos[:, 1])

    map_flat = map.reshape((-1, npix))

    for i in range(map_flat.shape[0]):
        map_flat[i] = healpy.get_interp_val(map_flat[i], theta, phi)

    return map_flat.reshape(map.shape)


def coord_g2c(map_):
    """Rotate a map from galactic co-ordinates into celestial co-ordinates.

    This will operate on a series of maps (provided the Healpix index is last).

    Parameters
    ----------
    map : np.ndarray[..., npix]
        Healpix map.

    Returns
    -------
    rotmap : np.ndarray
        The rotated map.
    """

    return coord_x2y(map_, "G", "C")


def coord_c2g(map_):
    """Rotate a map from celestial co-ordinates into galactic co-ordinates.

    This will operate on a series of maps (provided the Healpix index is last).

    Parameters
    ----------
    map : np.ndarray[..., npix]
        Healpix map.

    Returns
    -------
    rotmap : np.ndarray
        The rotated map.
    """

    return coord_x2y(map_, "C", "G")


def sph_ps(map1, map2=None, lmax=None):

    lmax = lmax if lmax is not None else (3 * healpy.get_nside(map1) - 1)

    alm1 = sphtrans_real(map1, lmax)
    alm2 = sphtrans_real(map2, lmax) if map is not None else alm1

    prod = alm1 * alm2.conj()

    s = prod[:, 0] + 2 * prod[:, 1:].sum(axis=1).real

    cl = s / (2.0 * np.arange(lmax + 1) + 1.0)

    return cl
