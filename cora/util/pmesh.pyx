# cython: profile=True

""" Module for deriving a density filed from a distribution of particles. """

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
        ang1 = ang0[:,:]+psi_slc[:2]
        # Wrap theta around pi:
        wrap_idxs = np.where(np.logical_or(ang1[0]>np.pi,ang1[0]<0))
        ang1[0][wrap_idxs] = np.pi -  ang1[0][wrap_idxs]%np.pi
        ang1[1][wrap_idxs] = (ang1[1][wrap_idxs] + np.pi)%(2.*np.pi)
        # Wrap phi around 2pi
        wrap_idxs = np.where(np.logical_or(ang1[1]>2.*np.pi,ang1[1]<0))
        ang1[1][wrap_idxs] = ang1[1][wrap_idxs]%(2.*np.pi)
        # Final radial positions:
        comov1 = comov0[ii,np.newaxis] + psi_slc[2]
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
