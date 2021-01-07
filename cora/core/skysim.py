
import numpy as np
import scipy.linalg as la
import scipy.integrate as si
import healpy

from cora.util import hputil, nputil


def clarray(aps, lmax, zarray, zromb=3, zwidth=None):
    """Calculate an array of C_l(z, z').

    Parameters
    ----------
    aps : function
        The angular power spectrum to calculate.
    lmax : integer
        Maximum l to calculate up to.
    zarray : array_like
        Array of z's to calculate at.
    zromb : integer
        The Romberg order for integrating over frequency samples.
    zwidth : scalar, optional
        Width of frequency channel to integrate over. If None (default),
        calculate from the separation of the first two bins.

    Returns
    -------
    aps : np.ndarray[lmax+1, len(zarray), len(zarray)]
        Array of the C_l(z,z') values.
    """

    if zromb == 0:
        return aps(
            np.arange(lmax + 1)[:, np.newaxis, np.newaxis],
            zarray[np.newaxis, :, np.newaxis],
            zarray[np.newaxis, np.newaxis, :],
        )

    else:
        zsort = np.sort(zarray)
        zhalf = np.abs(zsort[1] - zsort[0]) / 2.0 if zwidth is None else zwidth / 2.0
        zlen = zarray.size
        zint = 2 ** zromb + 1
        zspace = 2.0 * zhalf / 2 ** zromb

        za = (
            zarray[:, np.newaxis] + np.linspace(-zhalf, zhalf, zint)[np.newaxis, :]
        ).flatten()

        lsections = np.array_split(np.arange(lmax + 1), lmax // 5)

        cla = np.zeros((lmax + 1, zlen, zlen), dtype=np.float64)

        for lsec in lsections:
            clt = aps(
                lsec[:, np.newaxis, np.newaxis],
                za[np.newaxis, :, np.newaxis],
                za[np.newaxis, np.newaxis, :],
            )

            clt = clt.reshape(-1, zlen, zint, zlen, zint)

            clt = si.romb(clt, dx=zspace, axis=4)
            clt = si.romb(clt, dx=zspace, axis=2)

            cla[lsec] = clt / (2 * zhalf) ** 2  # Normalise

        return cla


def mkfullsky(corr, nside, alms=False):
    """Construct a set of correlated Healpix maps.

    Make a set of full sky gaussian random fields, given the correlation
    structure. Useful for constructing a set of different redshift slices.

    Parameters
    ----------
    corr : np.ndarray (lmax+1, numz, numz)
        The correlation matrix :math:`C_l(z, z')`.
    nside : integer
        The resolution of the Healpix maps.
    alms : boolean, optional
        If True return the alms instead of the sky maps.

    Returns
    -------
    hpmaps : np.ndarray (numz, npix)
        The Healpix maps. hpmaps[i] is the i'th map.
    """

    numz = corr.shape[1]
    maxl = corr.shape[0] - 1

    if corr.shape[2] != numz:
        raise Exception("Correlation matrix is incorrect shape.")

    alm_array = np.zeros((numz, 1, maxl + 1, maxl + 1), dtype=np.complex128)

    # Generate gaussian deviates and transform to have correct correlation
    # structure
    for l in range(maxl + 1):
        # Add in a small diagonal to try and ensure positive definiteness
        cmax = corr[l].diagonal().max() * 1e-14
        corrm = corr[l] + np.identity(numz) * cmax

        trans = nputil.matrix_root_manynull(corrm, truncate=False)
        gaussvars = nputil.complex_std_normal((numz, l + 1))
        alm_array[:, 0, l, : (l + 1)] = np.dot(trans, gaussvars)

    if alms:
        return alm_array

    # Perform the spherical harmonic transform for each z
    sky = hputil.sphtrans_inv_sky(alm_array, nside)
    sky = sky[:, 0]

    return sky


def mkconstrained(corr, constraints, nside):
    """Construct a set of Healpix maps, satisfying given constraints
    on specified frequency slices, by using the lowest eigenmodes.

    Parameters
    ----------
    corr : np.ndarray (lmax+1, numz, numz)
        The correlation matrix :math:`C_l(z, z')`.
    constrains : list
        A list of constraints packed as [[frequency_index, healpix map], ...]
    nside : integer
        The resolution of the Healpix maps.

    Returns
    -------
    hpmaps : np.ndarray (numz, npix)
        The Healpix maps. hpmaps[i] is the i'th map.
    """

    numz = corr.shape[1]
    maxl = corr.shape[0] - 1
    larr, marr = healpy.Alm.getlm(maxl)
    matshape = larr.shape + (numz,)

    # The number of constraints
    nmodes = len(constraints)

    # The frequency slices that are constrained.
    f_ind = [c[0] for c in constraints]

    if corr.shape[2] != numz:
        raise Exception("Correlation matrix is incorrect shape.")

    trans = np.zeros((corr.shape[0], nmodes, corr.shape[2]))
    tmat = np.zeros((corr.shape[0], nmodes, nmodes))
    cmap = np.zeros(larr.shape + (nmodes,), dtype=np.complex128)
    cv = np.zeros((numz,) + larr.shape, dtype=np.complex128)

    # Find eigenmodes, extract the largest nmodes (enough to satisfy
    # constraints), and then pull out coefficients for each constrained
    # frequency.
    for i in range(maxl + 1):
        trans[i] = la.eigh(corr[i])[1][:, -nmodes:].T
        tmat[i] = trans[i][:, f_ind]

    # Switch constraint maps into harmonic space
    for i, cons in enumerate(constraints):
        cmap[:, i] = healpy.map2alm(cons[1], lmax=maxl)

    # Solve for the eigenmode amplitudes to satisfy constraints, and project
    # each mode across the whole frequency range.
    for i, l in enumerate(larr):
        if l == 0:
            cv[:, i] = 0.0
        else:
            cv[:, i] = np.dot(trans[l].T, la.solve(tmat[l].T, cmap[i]))

    hpmaps = np.empty((numz, healpy.nside2npix(nside)))

    for i in range(numz):
        hpmaps[i] = healpy.alm2map(cv[i], nside, verbose=False)

    return hpmaps
