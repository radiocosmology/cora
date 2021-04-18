# cython: profile=True

""" Module for deriving a density filed from a distribution of particles. """

cimport cython
from libc.math cimport exp

import numpy as np
from scipy.interpolate import interp1d
import healpy as hp

# TODO: Apparently unused variable nslices. Delete?
def za_density(psi,rho0,nside,comovd,nside_factor,ndiv_radial,nslices=2):
    """
    psi : array-like [3,nz,npix]
        Coordinate displacements
    rho0 : array-like [nz,npix]
        The underlying density field to be displaced. Usually the Gaussian
        matter density field linearly evolved to the correct redshifts and
        biased with a Lagrangian bias (with mean 1, not 0).
    """

    def interp_ugd(f,ndiv):
        n = len(f)
        # TODO: my version of scipy doesn't have extrapolate (starts in 0.17.0).
        # I implement it myself.
        fi = interp1d(np.arange(n), f, kind='cubic')#,fill_value='extrapolate')
        sep = 1./float(ndiv)
        x_new = np.arange(-0.5+sep*0.5,n-0.5,sep)
        # Remove out of range points because I don't have extrapolation:
        nextrap = int(ndiv/2)
        f_new = fi(x_new[nextrap:-nextrap])
        # Add back extrapolation points:
        if ndiv%2 == 0:
            sttslc = np.s_[0:nextrap]
            stpslc = np.s_[-nextrap:]
        else:
            sttslc = np.s_[1:1+nextrap]
            stpslc = np.s_[-nextrap-1:-1]
        extrap_stt = (2.*f[0]-f_new[sttslc])[::-1]
        extrap_stp = (2.*f[-1]-f_new[stpslc])[::-1]
        f_new = np.concatenate((extrap_stt,f_new,extrap_stp))
        return f_new

    # Comoving distance bins for digitizing:
    # Does not include edges due to way np.digitize works.
    # This actually adds all displacements past the last bins to
    # the last bins. It doesn't matter with redshift padding,
    # because the last bins get discarded.
    comov_bins = (comovd[1:]+comovd[:-1])*0.5

    npix = hp.pixelfunc.nside2npix(nside) # In original size maps
    nz = len(comovd)
    #mapdtype = psi.dtype
    mapdtype = rho0.dtype
    # Number of pixels in upgraded resolution per pixel of original resolution:
    ndiv_ang = nside_factor**2
    # Number of voxels in upgraded resolution per voxel of original resolution:
    n_subvoxels = ndiv_ang*ndiv_radial
    # Fraction of an original voxel occupied by a voxel of the upd resolution:
    fracfactor = 1./float(n_subvoxels)

    # Initial radial positions in higher resolution grid:
    comov0 = interp_ugd(comovd,ndiv_radial)
    # Index corresponding to comov0 values in lower resolution grid:
    comov0_idx = np.arange(len(comov0),dtype=int)
    comov0_idx = comov0_idx//ndiv_radial
    # Initial angular positions in higher resolution grid:
    ang0 = np.array(hp.pixelfunc.pix2ang(nside*nside_factor,
                    np.arange(hp.pixelfunc.nside2npix(nside*nside_factor))))
    # Index corresponding to ang0 values in lower resolution grid:
    ang0_idx = hp.pixelfunc.ang2pix(nside, *ang0)

    # Array to hold the density obtained from the ZA approximation
    delta_za = np.zeros((nz,npix),dtype=mapdtype)

    for ii in range(len(comov0)):
        # Initial density
        rho0_slc = rho0[comov0_idx[ii],ang0_idx]
        # Coordinate displacements
        psi_slc = psi[:,comov0_idx[ii],ang0_idx]
        # Final angles:
        ang1 = ang0[:,:]+psi_slc[1:]
        # Wrap theta around pi:
        wrap_idxs = np.where(np.logical_or(ang1[0]>np.pi,ang1[0]<0))
        ang1[0][wrap_idxs] = np.pi -  ang1[0][wrap_idxs]%np.pi
        ang1[1][wrap_idxs] = (ang1[1][wrap_idxs] + np.pi)%(2.*np.pi)
        # Wrap phi around 2pi
        wrap_idxs = np.where(np.logical_or(ang1[1]>2.*np.pi,ang1[1]<0))
        ang1[1][wrap_idxs] = ang1[1][wrap_idxs]%(2.*np.pi)
        # Final radial positions:
        comov1 = comov0[ii,np.newaxis] + psi_slc[0]
        # Indices of final positions in lower resolution grid:
        ang_idx = hp.pixelfunc.ang2pix(nside, *ang1)
        radial_idx = np.digitize(comov1,comov_bins)

        fill_delta_za(delta_za, radial_idx.flatten(), ang_idx.flatten(),
                      rho0_slc.flatten())

    return delta_za*fracfactor - 1.


# I tried to use 'nogil' in this function call but apparently it
# degraded the performance slightly.
cpdef void fill_delta_za(double[:,:] delta_za, long[:] rad_idx,
                         long[:] ang_idx, double[:] rho0):
# 'delta_za' is defined in the argument as a cython memoryview.
# It can take a numpy array as parameter and modifying one changes the other.
    cdef long ii, jj, kk
    for kk in range(len(rad_idx)):
        ii = rad_idx[kk]
        jj = ang_idx[kk]
        #delta_za[ii,jj] += 1. # TODO: delete
        delta_za[ii,jj] += rho0[kk]


def calculate_positions(angpos, displacement):
    """Apply an angular displacement, accounting for wrapping.

    Parameters
    ----------
    angpos : np.ndarray[2, npix]
        The original angular positions ordered as theta, phi
    displacement : np.ndarray[2, npix]
        The shift to apply to each coordinate.

    Returns
    -------
    new_angpos
    """

    new_angpos = angpos + displacement

    # Wrap out of bounds values in theta
    wrap_ind = np.where(np.logical_or(new_angpos[0] > np.pi, new_angpos[0] < 0))
    new_angpos[0][wrap_ind] = np.pi - new_angpos[0][wrap_ind] % np.pi
    new_angpos[1][wrap_ind] += np.pi

    # Wrap out of bounds in phi
    new_angpos[1] = new_angpos[1] % (2 * np.pi)

    return new_angpos


cpdef _same_shape(arr1, arr2):
    ndim = arr1.ndim
    return arr2.ndim == ndim and tuple(arr1.shape[:ndim]) == tuple(arr2.shape[:ndim])


cpdef _check_shape(arr, shape):
    return arr.ndim == len(shape) and tuple(arr.shape[:len(shape)]) == shape


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _pixel_weights(
    long[::1] new_ang_ind,
    double[:, ::1] new_ang_vec,
    double[::1] scaling,
    double sigma,
    long[:, ::1] nn_ind,
    double[:, :, ::1] nn_vec,
    # Outputs
    int[:, ::1] pixel_ind,
    double[:, ::1] pixel_weight,
):
    """Calculate the pixel weights for the SPH scheme.

    The applies a Gaussian mass distribution amongst the nearest neighbour pixels.

    Parameters
    ----------
    new_ang_ind
        The index of pixel containing the new particle angular position.
    new_ang_vec
        The actual new position of the particle as a 3D vector. Array is packed as
        `[pixel, coordinate]`.
    scaling
        The scaling of the particle.
    sigma
        The nominal angular size of the particle, i.e. when scaling = 1.
    nn_ind
        The indices of the nearest neighbours of each pixel. The first entry is the
        actual pixel itself, and the subsequent 8 are the nearest neighbours. Array
        is packed as `[pixel, index of neighbour]`. If there is no nearest neighbour
        in a position, it should be given an index < 0 (default behaviour for
        `healpy.get_all_neighbours`).
    nn_vec
        The position of each entry in the `nn_ind` array as a 3D vector. Packed as
        `[pixel, index of neighbour, coordinate]`
    pixel_ind
        An output array giving the indices of all the pixels the particle mass should
        be distributed into. Packed as `[particle, output pixel index]`.
    pixel_weight
        Output array for the weight to be given to the distribution of mass amongst
        pixels. Packed the same as `pixel_ind`.
    """

    cdef int ii, jj, kk
    cdef int npix, npart, nn

    cdef double dist2
    cdef double inv_sigma2
    cdef double wsum
    cdef int ind

    npart = new_ang_ind.shape[0]
    npix = nn_ind.shape[0]
    nn = nn_ind.shape[1]

    if not _check_shape(new_ang_vec, (npart, 3)):
        raise ValueError("new_ang_vec has wrong shape")

    if not _check_shape(nn_ind, (npix, nn)):
        raise ValueError("nn_ind has wrong shape")

    if not _check_shape(nn_vec, (npix, nn, 3)):
        raise ValueError("nn_vec has wrong shape")

    if not _check_shape(scaling, (npart,)):
        raise ValueError("sigma_arr has wrong shape")

    if not _check_shape(pixel_ind, (npart, nn)):
        raise ValueError("pixel_ind has wrong shape")

    if not _check_shape(pixel_weight, (npart, nn)):
        raise ValueError("pixel_weight has wrong shape")


    for ii in range(npart):

        ind = new_ang_ind[ii]

        # Regularise ind so we can't have any out of bound accesses, this shouldn't
        # happen if all the other stages were correct, so the branch predictor should
        # make this test have minimal effect
        if ind < 0 or ind >= npix:
            ind = 0

        # Calculate the inverse squared size of the particle (use the inverse as it
        # will be faster in subsequent calculations)
        inv_sigma2 = (scaling[ii] * sigma)**-2

        for jj in range(nn):

            dist2 = 0.0

            # Calculate the dot product and transform this into the squared distance
            # (actually sin^2 theta)
            for kk in range(3):
                dist2 += nn_vec[ind, jj, kk] * new_ang_vec[ii, kk]
            dist2 = 1.0 - dist2 * dist2

            # Set the output pixel value
            pixel_ind[ii, jj] = <int>nn_ind[ind, jj]

            # On a Healpix grid sometimes there is no valid nearest neighbour and so
            # the index given is less than zero. In this case we need to ensure that
            # the weight is set to zero, and the index is made valid.
            if pixel_ind[ii, jj] >= 0:
                pixel_weight[ii, jj] = exp(-0.5 * dist2 * inv_sigma2)
            else:
                pixel_weight[ii, jj] = 0.0
                pixel_ind[ii, jj] = 0

        # Normalise by the sum of weights to ensure mass is conserved
        wsum = 0.0
        for jj in range(nn):
            wsum += pixel_weight[ii, jj]

        for jj in range(nn):
            pixel_weight[ii, jj] /= wsum


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _radial_weights(
    long[::1] new_chi_ind,
    double[::1] new_chi,
    double[::1] scaling,
    double sigma,
    int nnh,
    double[::1] chi,
    # Outputs
    int[:, ::1] radial_ind,
    double[:, ::1] radial_weight,
):
    """Calculate the radial weights for the SPH scheme.

    The applies a Gaussian mass distribution amongst the nearest `nnh` radial bins.
    The full two sided window will not extend beyond the upper or lower radial edges,
    which means that particles close to (or beyond) the edges will have their mass
    distributed onto the interior side.

    Parameters
    ----------
    new_chi_ind
        The index of the radial bin containing the new particle.
    new_chi
        The actual new radial position of the particle.
    scaling
        The scaling of the particle.
    sigma
        The nominal angular size of the particle, i.e. when scaling = 1.
    nnh
        The number of radial bins each side to distribute across.
    chi
        The radial coordinates of all bins.
    radial_ind
        An output array giving the indices of all the radial bins the particle mass
        should be distributed into. Packed as `[particle, output radial index]`.
    radial_weight
        Output array for the weight to be given to the distribution of mass amongst
        radial bins. Packed the same as `radial_ind`.
    """

    cdef int ii, jj, kk
    cdef int npart, nchi
    cdef int nn

    cdef double dchi
    cdef double inv_sigma2
    cdef double wsum
    cdef int ind
    cdef int low

    npart = new_chi_ind.shape[0]
    nchi = chi.shape[0]
    nn = 2 * nnh + 1

    if not _check_shape(new_chi, (npart,)):
        raise ValueError("new_chi has incorrect shape.")

    if not _check_shape(scaling, (npart,)):
        raise ValueError("scaling has incorrect shape.")

    if not _check_shape(radial_ind, (npart, nn)):
        raise ValueError("radial_ind has incorrect shape.")

    if not _check_shape(radial_weight, (npart, nn)):
        raise ValueError("radial_weight has incorrect shape.")

    for ii in range(npart):

        ind = new_chi_ind[ii]

        # The lowest radial index of the window needs to be clipped to ensure the whole
        # window remains within the range of radial bins
        low = min(max(0, ind - nnh), nchi - nn)

        inv_sigma2 = (scaling[ii] * sigma)**-2

        for jj in range(nn):
            radial_ind[ii, jj] = low + jj

            dchi = chi[low + jj] - new_chi[ii]

            radial_weight[ii, jj] = exp(-0.5 * dchi**2 * inv_sigma2)

        wsum = 0.0
        for jj in range(nn):
            wsum += radial_weight[ii, jj]

        for jj in range(nn):
            radial_weight[ii, jj] /= wsum


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void _bin_delta(
    double[::1] rho,
    int[:, ::1] pixel_ind, double[:, ::1] pixel_weight,
    int[:, ::1] radial_ind, double[:, ::1] radial_weight,
    double[:, ::1] out
):
    """Take the particle masses and bin them each voxel.

    Parameters
    ----------
    rho
        Mass of each particle.
    pixel_ind
        The index of each pixel to distribute into. Array is packed as
        `[particle, output pixel]`.
    pixel_weight
        The weight with which to distribute the mass between pixels. Array is packed
        the same as `pixel_ind`.
    radial_ind
        The index of each radial bin to distribute into. Array is packed as
        `[particle, output bin]`.
    radial_weight
        The weight with which to distribute the mass between radial bins. Array is
        packed the same as `radial_ind`.
    out
        The 3D output array.
    """

    cdef int ipart, ipix, irad
    cdef int npart, nchi
    cdef int len_pixel
    cdef int len_radial
    cdef int pi, ri
    cdef double pw, rw
    cdef double v

    len_pixel = pixel_ind.shape[1]
    len_radial = radial_ind.shape[1]
    nchi = out.shape[0]
    npart = rho.shape[0]

    if not _same_shape(pixel_ind, pixel_weight):
        raise ValueError("Pixel arrays are the wrong shape.")

    if not _same_shape(radial_ind, radial_weight):
        raise ValueError("Radial arrays are the wrong shape.")

    if radial_ind.shape[0] != pixel_ind.shape[0]:
        raise ValueError(
            "Pixel and radial arrays don't have the same number of source pixels."
        )

    if pixel_ind.shape[0] != npart:
        raise ValueError("Weight arrays do not have the same length as the `rho` array")

    for ipart in range(npart):

        v = rho[ipart]

        for ipix in range(len_pixel):
            pi = pixel_ind[ipart, ipix]
            pw = pixel_weight[ipart, ipix]

            # if pi < 0 or pi >= npix:
            #     raise RuntimeError("out of bounds index %i" % pi)

            for irad in range(len_radial):
                ri = radial_ind[ipart, irad]
                rw = radial_weight[ipart, irad]

                # if ri < 0 or ri >= nchi:
                #     raise RuntimeError("out of bounds index chi %i" % ri)

                if rw < 0:
                    continue

                out[ri, pi] += v * pw * rw