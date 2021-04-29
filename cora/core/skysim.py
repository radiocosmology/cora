# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import os
import numpy as np
import scipy.linalg as la
import scipy.integrate as si
import healpy

from cora.util import hputil, nputil, units as u

from caput import mpiutil, pipeline, config, mpiarray
from draco.core import containers, task, io


def clarray(aps, lmax, zarray, zromb=3, zwidth=None, b_second_func=None):
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
    b_second_func : function, optional
        Linear bias of second tracer as a function of z or frequency, depending
        on format of aps. Default: None.

    Returns
    -------
    aps : np.ndarray[lmax+1, len(zarray), len(zarray)]
        Array of the C_l(z,z') values.
    """

    if zromb == 0:
        if b_second_func is not None:
            return aps(
                np.arange(lmax + 1)[:, np.newaxis, np.newaxis],
                zarray[np.newaxis, :, np.newaxis],
                zarray[np.newaxis, np.newaxis, :],
                b_nu2=b_second_func(zarray)[np.newaxis, np.newaxis, :]
            )
        else:
            return aps(
                np.arange(lmax + 1)[:, np.newaxis, np.newaxis],
                zarray[np.newaxis, :, np.newaxis],
                zarray[np.newaxis, np.newaxis, :]
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
            if b_second_func is not None:
                clt = aps(
                    lsec[:, np.newaxis, np.newaxis],
                    za[np.newaxis, :, np.newaxis],
                    za[np.newaxis, np.newaxis, :],
                    b_nu2=b_second_func(za)[np.newaxis, np.newaxis, :]
                )
            else:
                clt = aps(
                    lsec[:, np.newaxis, np.newaxis],
                    za[np.newaxis, :, np.newaxis],
                    za[np.newaxis, np.newaxis, :]
                )

            clt = clt.reshape(-1, zlen, zint, zlen, zint)

            clt = si.romb(clt, dx=zspace, axis=4)
            clt = si.romb(clt, dx=zspace, axis=2)

            cla[lsec] = clt / (2 * zhalf) ** 2  # Normalise

        return cla


def mkfullsky(corr, nside, alms=False, gaussvars_list=None):
    """Construct a set of correlated Healpix maps.

    Make a set of full sky gaussian random fields, given the correlation
    structure. Useful for constructing a set of different redshift slices.
    Now also accepts a list of correlation matrices to generate maps from
    with a singe realization.

    Parameters
    ----------
    corr : np.ndarray (lmax+1, numz, numz) or list
        The correlation matrix :math:`C_l(z, z')`, or a list
        of correlation matrices
    nside : integer
        The resolution of the Healpix maps.
    alms : boolean, optional
        If True return the alms instead of the sky maps.
    gaussvars_list : list or array-like
        The complex standard normal variables to be used to generate the
        realization. Default is None which generates them on the fly.
        This is usefull when one needs different maps to be drawn from
        the same realization.

    Returns
    -------
    hpmaps : np.ndarray (numz, npix)
        The Healpix maps. hpmaps[i] is the i'th map.
    """
    if not isinstance(corr, list):
        corr = [corr]
    ncorr = len(corr)

    numz = corr[0].shape[1]
    maxl = corr[0].shape[0] - 1

    if corr[0].shape[2] != numz:
        raise Exception("Correlation matrix is incorrect shape.")

    alm_list = [
        np.zeros((numz, 1, maxl + 1, maxl + 1), dtype=np.complex128)
        for ii in range(ncorr)
    ]

    # Generate gaussian deviates and transform to have correct correlation
    # structure
    for l in range(maxl + 1):
        if gaussvars_list is None:
            gaussvars = nputil.complex_std_normal((numz, l + 1))
        else:
            gaussvars = gaussvars_list[l]
        for ii in range(ncorr):
            # Add in a small diagonal to try and ensure positive definiteness
            cmax = corr[ii][l].diagonal().max() * 1e-14
            corrm = corr[ii][l] + np.identity(numz) * cmax

            trans = nputil.matrix_root_manynull(
                corrm, truncate=False, fixed_ev_sign_convention=True
            )
            alm_list[ii][:, 0, l, : (l + 1)] = np.dot(trans, gaussvars)

    if alms:
        if ncorr == 1:
            return alm_list[0]
        else:
            return alm_list

    sky = []
    for ii in range(ncorr):
        # Perform the spherical harmonic transform for each z
        sky.append(hputil.sphtrans_inv_sky(alm_list[ii], nside))
        sky[ii] = sky[ii][:, 0]

    if ncorr == 1:
        return sky[0]
    else:
        return sky


def mkfullsky_der1(corr, nside, comovd, alms=False, gaussvars_list=None):
    """Construct a set of correlated Healpix maps and their 3D
    spacial derivatives.

    Make a set of full sky gaussian random fields, given the correlation
    structure. Useful for constructing a set of different redshift slices.
    Also accepts a list of correlation matrices to generate maps from
    with a singe realization.

    Parameters
    ----------
    corr : np.ndarray (lmax+1, numz, numz) or list
        The correlation matrix :math:`C_l(z, z')`, or a list
        of correlation matrices
    nside : integer
        The resolution of the Healpix maps.
    alms : boolean, optional
        If True return the alms instead of the sky maps.
    gaussvars_list : list or array-like
        The complex standard normal variables to be used to generate the
        realization. Default is None which generates them on the fly.
        This is usefull when one needs different maps to be drawn from
        the same realization.

    Returns
    -------
    resmaps : list [ncorr]
        The Healpix maps and derivatives.
    """
    if not isinstance(corr, list):
        corr = [corr]
    ncorr = len(corr)

    numz = corr[0].shape[1]
    maxl = corr[0].shape[0] - 1

    if corr[0].shape[2] != numz:
        raise Exception("Correlation matrix is incorrect shape.")

    alm_list = [
        np.zeros((numz, 1, maxl + 1, maxl + 1), dtype=np.complex128)
        for ii in range(ncorr)
    ]

    # Generate gaussian deviates and transform to have correct correlation
    # structure
    for l in range(maxl + 1):
        if gaussvars_list is None:
            gaussvars = nputil.complex_std_normal((numz, l + 1))
        else:
            gaussvars = gaussvars_list[l]
        for ii in range(ncorr):
            # Add in a small diagonal to try and ensure positive definiteness
            cmax = corr[ii][l].diagonal().max() * 1e-14
            corrm = corr[ii][l] + np.identity(numz) * cmax

            trans = nputil.matrix_root_manynull(corrm, truncate=False)
            alm_list[ii][:, 0, l, : (l + 1)] = np.dot(trans, gaussvars)

    if alms:
        if ncorr == 1:
            return alm_list[0]
        else:
            return alm_list

    resmaps = []
    for ii in range(ncorr):
        # Perform the spherical harmonic transform for each z
        sky, d_theta, d_phi = hputil.sphtrans_inv_sky_der1(alm_list[ii], nside)
        # TODO: careful with np.gradient! meaning of second argument (varargs) changed
        # from numpy 1.12 to 1.13. Now it accepts an array of 'x values' along the axis.
        # Before it wanted a scalar representing the 'x-variation' between points.
        # The problem is that if you give it an array in old versions it doesn't crash!
        # I guess it silently takes the first value as 'x-variation'...?
        # spacing = np.gradient(comovd)
        # d_x = np.gradient(sky[:,0],spacing,axis=0)
        d_x = np.gradient(sky[:, 0], comovd, axis=0)
        resmaps.append(
            np.array(
                [
                    sky[:, 0],
                    d_theta[:, 0] / comovd[:, None],
                    d_phi[:, 0] / comovd[:, None],
                    d_x,
                ]
            )
        )

    if ncorr == 1:
        return resmaps[0]
    else:
        return resmaps


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




class FLASKMapsToMapContainer(task.SingleTask):
    """Convert a set of single-frequency FLASK maps into a Map container.

    Attributes
    ----------
    input_map_prefix : str
        Prefix of filenames for input FLASK maps, including final "-".
    nside : int
        Nside of input maps.
    flask_field_num : int, optional
        FLASK field number for input maps (default: 1).
    use_mean_21cmT : bool, optional
        Multiply map by mean 21cm temperature as a function of z (default: False).
    map_prefactor : float, optional
        Constant prefactor to multiply maps by (default: 1).
    """

    input_map_prefix = config.Property(proptype=str)
    nside = config.Property(proptype=int)

    flask_field_num = config.Property(proptype=int, default=1)
    use_mean_21cmT = config.Property(proptype=int, default=False)
    map_prefactor = config.Property(proptype=float, default=1.)

    done = False

    def setup(self, bt):
        """Setup the simulation.

        Parameters
        ----------
        bt : ProductManager or BeamTransfer
            Beam Transfer maanger.
        """
        self.beamtransfer = io.get_beamtransfer(bt)
        self.telescope = io.get_telescope(bt)

    def process(self) -> containers.Map:
        """Read FLASK maps and assemble into Map container.

        Returns
        -------
        out_map : Map
            Output Map container.
        """

        # Form frequency map in appropriate format to feed into Map container
        freq = self.telescope.frequencies
        n_freq = len(freq)
        freqmap = np.zeros(n_freq, dtype=[("centre", np.float64), ("width", np.float64)])
        freqmap["centre"][:] = freq
        freqmap["width"][:] = np.abs(np.diff(freq)[0])

        # Make new map container and copy delta into Stokes I component
        m = containers.Map(
            freq=freqmap,
            polarisation=True,
            distributed=True,
            pixel=np.arange(healpy.nside2npix(self.nside))
        )

        # Get indexing info for this rank
        loff = m.map.local_offset[0]
        lshape = m.map.local_shape[0]
        gshape = m.map.global_shape[0]

        # For each frequency on this rank, read FLASK map and insert into
        # Map container
        for fi_local in range(lshape):
            # print("Rank %d: fi_local, fi_global = %d, %d" % (mpiutil.rank, fi_local, loff+fi_local))

            in_file = (
                self.input_map_prefix
                + "f%dz%d.fits" % (self.flask_field_num, loff + fi_local + 1)
            )
            if not os.path.isfile(in_file):
                raise RuntimeError("File %s not found!" % in_file)

            m.map[fi_local, 0, :] = healpy.read_map(in_file)
            self.log.debug(f"Rank %d: Read file %s" % (mpiutil.rank, in_file))

        # Multiply map by specific prefactor, if desired
        if self.map_prefactor != 1:
            self.log.info("Multiplying map by %g" % self.map_prefactor)
            m.map[:] *= self.map_prefactor

        # If desired, multiply by Tb(z)
        if self.use_mean_21cmT:
            from cora.signal.corr21cm import Corr21cm
            cr = Corr21cm()

            z_local = u.nu21 / freq[loff : loff+lshape] - 1
            m.map[:, 0] *= cr.T_b(z_local)[:, np.newaxis]

        self.done = True

        return m

