# cython: profile=True

""" Module for deriving a density field from a distribution of particles. """

cimport cython
from cython.parallel import prange

from libc.math cimport exp

import numpy as np
from scipy.interpolate import interp1d
import healpy as hp


cdef extern from "pmesh_util.c":
    void _bin_delta_c(
            double* rho,
            int* pixel_ind,
            double* pixel_weight,
            int* radial_ind,
            double* radial_weight,
            double* out,
            int npart,
            int npix,
            int nrad
    )


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


    for ii in prange(npart, nogil=True):

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

    for ii in prange(npart, nogil=True):

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

    _bin_delta_c(
            &rho[0],
        &pixel_ind[0, 0],
        &pixel_weight[0, 0],
        &radial_ind[0, 0],
        &radial_weight[0, 0],
        &out[0, 0],
        npart,
        len_pixel,
        len_radial
    )

