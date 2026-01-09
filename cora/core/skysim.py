import numpy as np
import scipy.linalg as la
import scipy.integrate as si
import scipy.special as ss
import healpy

from caput import mpiarray
from cora.util import hputil, nputil
from cora.signal.lssutil import diff2


def clarray(
    aps,
    lmax,
    zarray,
    gauss_legendre=False,
    zromb=3,
    zwidth=None,
    chunksize=5,
    second_chi_deriv=False,
    chi_func=None,
    channel_profile=None,
):
    """Calculate an array of C_l(z, z').

    The zromb parameter controls whether to integrate the angular
    power spectrum over each frequency channel. Two schemes are
    available: legacy Romberg integration that assumes each channel
    has the same redshift width, or Gauss-Legendre quadrature that
    can either assume a constant width or use the correct per-chennel
    width. The Gauss-Legendre scheme is almost identical to the one
    in cora.signal.corrfunc.corr_to_clarray, except that here the
    integration is in redshift instead of comoving distance.

    Parameters
    ----------
    aps : function
        The angular power spectrum to calculate, with arguments
        (ell, z, z').
    lmax : integer
        Maximum l to calculate up to.
    zarray : array_like, optional
        Array of redshifts to calculate at.
    gauss_legendre : bool, optional
        Use Gauss-Legendre quadrature for channel integration (as in
        cora.signal.corrfunc.corr_to_clarray). If False, use older
        Romberg scheme. Default: False.
    zromb : integer, optional
        The Romberg or Gauss-Legendre order for integrating over
        each channel. If 0, this integration is turned off. Default: 3.
    zwidth : scalar, optional
        Constant redshift width of frequency channel to integrate over.
        If None, calculated from the separation of the first two bins
        for the Romberg scheme, or computed separately for each channel
        in the Gauss-Legendre scheme. Default: None.
    chunksize : int, optional
        Chunk size for performing channel integral in batches of ell values.
        Default: 5.
    second_chi_deriv : bool, optional
        Whether to compute second derivative of computed C_l(z,z') in each
        of z and z', as derivatives in comoving distance. This can be used
        for testing computations involving the Kaiser redshift-space
        distorition term. Default: False.
    chi_func : function, optional
        If `second_chi_deriv` is True, this is the function that converts
        redshift to comoving distance that is used in the finite-difference
        computation. Default: None.
    channel_profile : function, optional
        The frequency channel profile to use in the channel integration.
        The profile function must be defined on [-0.5, 0.5] and integrate
        to unity over this range. Default: None.

    Returns
    -------
    aps : np.ndarray[lmax+1, len(zarray), len(zarray)]
        Array of the C_l(z,z') values.
    """

    if second_chi_deriv and chi_func is None:
        raise RuntimeError("Must specify chi_func if using second_chi_deriv")

    if zromb == 0:
        return aps(
            np.arange(lmax + 1)[:, np.newaxis, np.newaxis],
            zarray[np.newaxis, :, np.newaxis],
            zarray[np.newaxis, np.newaxis, :],
        )

    else:
        # Gauss-Legendre scheme
        if gauss_legendre:

            # Get number of redshifts
            zlen = zarray.size

            # Calculate the half-bin width
            if zwidth is None:
                zhalf = np.ndarray(shape=zarray.shape)
                # Compute width for each channel, using same width for first
                # and second channel
                zhalf[0] = np.abs(zarray[1] - zarray[0]) / 2.0
                zhalf[1:] = np.abs(zarray[1:] - zarray[:-1]) / 2.0
            else:
                # Use constant channel width
                zhalf = np.ones(shape=zarray.shape) * zwidth / 2.0

            # Set number of quadrature points, and get points and weights
            zint = 2**zromb + 1
            z_r, z_w, z_wsum = ss.roots_legendre(zint, mu=True)
            z_w /= z_wsum

            # Calculate the extended xarray with the extra intervals to integrate over
            za = (zarray[:, np.newaxis] + zhalf[:, np.newaxis] * z_r).flatten()

            # If using a channel profile, evaluate it at the integrand sampling points
            if channel_profile is not None:
                za_profile = (
                    np.zeros_like(zarray)[:, np.newaxis]
                    + 0.5 * np.ones_like(zhalf)[:, np.newaxis] * z_r
                ).flatten()
                profile = channel_profile(za_profile)

            # Make array to store final results
            cla = np.zeros((lmax + 1, zlen, zlen), dtype=np.float64)

            # Determine ell chunks
            lsections = np.array_split(np.arange(lmax + 1), lmax // chunksize)

            for lsec in lsections:
                # Get C_ell(z, z') values for this chunk
                clt = aps(
                    lsec[:, np.newaxis, np.newaxis],
                    za[np.newaxis, :, np.newaxis],
                    za[np.newaxis, np.newaxis, :],
                )

                if second_chi_deriv:
                    xarray = chi_func(za)
                    clt = diff2(clt, xarray, axis=1)
                    clt = diff2(clt, xarray, axis=2)

                if channel_profile is not None:
                    clt *= profile[np.newaxis, :, np.newaxis]
                    clt *= profile[np.newaxis, np.newaxis, :]

                # Perform Gauss-Legendre quandrature via matrix multiplications
                clt = clt.reshape(-1, zint)
                clt = np.matmul(clt, z_w).reshape(-1, zlen, zint, zlen)
                clt = np.matmul(clt.transpose(0, 1, 3, 2), z_w)

                cla[lsec] = clt

        # Romberg scheme
        else:
            zsort = np.sort(zarray)
            zhalf = (
                np.abs(zsort[1] - zsort[0]) / 2.0 if zwidth is None else zwidth / 2.0
            )
            zlen = zarray.size
            zint = 2**zromb + 1
            zspace = 2.0 * zhalf / 2**zromb

            za = (
                zarray[:, np.newaxis] + np.linspace(-zhalf, zhalf, zint)[np.newaxis, :]
            ).flatten()

            # If using a channel profile, evaluate it at the integrand sampling points
            if channel_profile is not None:
                za_profile = (
                    np.zeros_like(zarray)[:, np.newaxis]
                    + np.linspace(-0.5, 0.5, zint)[np.newaxis, :]
                ).flatten()
                profile = channel_profile(za_profile)

            lsections = np.array_split(np.arange(lmax + 1), lmax // chunksize)

            cla = np.zeros((lmax + 1, zlen, zlen), dtype=np.float64)

            for lsec in lsections:
                clt = aps(
                    lsec[:, np.newaxis, np.newaxis],
                    za[np.newaxis, :, np.newaxis],
                    za[np.newaxis, np.newaxis, :],
                )

                if channel_profile is not None:
                    clt *= profile[np.newaxis, :, np.newaxis]
                    clt *= profile[np.newaxis, np.newaxis, :]

                clt = clt.reshape(-1, zlen, zint, zlen, zint)

                clt = si.romb(clt, dx=zspace, axis=4)
                clt = si.romb(clt, dx=zspace, axis=2)

                cla[lsec] = clt / (2 * zhalf) ** 2  # Normalise

        return cla


def mkfullsky(corr, nside, alms=False, rng=None):
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
    rng : numpy RNG, optional
        Seeded random number generator to use. Default: None.

    Returns
    -------
    hpmaps : np.ndarray (numz, npix)
        The Healpix maps. hpmaps[i] is the i'th map.
    """

    numz = corr.shape[1]

    isdistributed = isinstance(corr, mpiarray.MPIArray)

    if isdistributed:
        maxl = corr.global_shape[0] - 1
        corr = corr.local_array
    else:
        maxl = corr.shape[0] - 1

    if corr.shape[2] != numz:
        raise Exception("Correlation matrix is incorrect shape.")

    alm_array = mpiarray.zeros(
        (numz, 1, maxl + 1, maxl + 1), dtype=np.complex128, axis=2
    )

    # Generate gaussian deviates and transform to have correct correlation
    # structure
    for lloc, lglob in alm_array.enumerate(axis=2):
        # Add in a small diagonal to try and ensure positive definiteness
        cmax = corr[lloc].diagonal().max() * 1e-14
        corrm = corr[lloc] + np.identity(numz) * cmax

        trans = nputil.matrix_root_manynull(corrm, truncate=False)
        gaussvars = nputil.complex_std_normal((numz, lglob + 1), rng=rng)
        alm_array.local_array[:, 0, lloc, : (lglob + 1)] = np.dot(trans, gaussvars)

    if alms:
        # Return the entire alm array on each rank
        return alm_array.allgather()

    # Perform the spherical harmonic transform for each z
    alm_array = alm_array.redistribute(axis=0)

    sky = hputil.sphtrans_inv_sky(alm_array.local_array, nside)[:, 0]

    if isdistributed:
        # Re-wrap the final array
        sky = mpiarray.MPIArray.wrap(sky, axis=0)

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
    larr, _ = healpy.Alm.getlm(maxl)

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
    for i, ell in enumerate(larr):
        if ell == 0:
            cv[:, i] = 0.0
        else:
            cv[:, i] = np.dot(trans[ell].T, la.solve(tmat[ell].T, cmap[i]))

    hpmaps = np.empty((numz, healpy.nside2npix(nside)))

    for i in range(numz):
        hpmaps[i] = healpy.alm2map(cv[i], nside)

    return hpmaps
