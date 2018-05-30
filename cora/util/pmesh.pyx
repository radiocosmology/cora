""" Module for deriving a density filed from a distribution of particles. """

import numpy as np

# Cython import
#cimport numpy as np
#cimport cython

def za_density(psi,nside,comovd,nside_factor,ndiv_radial,nslices=2):
    """
    """
        
    from collections import Counter
    from scipy.interpolate import interp1d
    import healpy as hp

    def interp_subdivide(f,ndiv):
        from scipy.interpolate import interp1d
        n = len(f)
        # TODO: my version of scipy doesn't have extrapolate (starts in 0.17.0).
        # Could implement myself...
        fi = interp1d(np.arange(n), f, kind='cubic')#,fill_value='extrapolate')
        sep = 1./float(ndiv)
        x_new = np.arange(-0.5+sep*0.5,n-0.5,sep)
        # Remove out of range points because I don't have extrapolation yet
        nexclude = int(ndiv/2)
        x_new = x_new[nexclude:-nexclude]
        return fi(x_new)#, x_new

    def interp_binlimits(f):
        from scipy.interpolate import interp1d
        n = len(f)
        # TODO: my version of scipy doesn't have extrapolate (starts in 0.17.0).
        # I implement it myself...
        fi = interp1d(np.arange(n), f, kind='linear')#,fill_value='extrapolate')
        # Remove out of range points because I don't have extrapolation yet
        x_new = np.arange(n-1)+0.5
        f_new = fi(x_new)
        # Do not include edges due to way np.digitize works:
        # # Insert zeroeth extrapolation:
        # f_new = np.insert(f_new,0,2*f[0]-f_new[0])
        # # Insert last extrapolation:
        # f_new = np.append(f_new,2*f[-1]-f_new[-1])
        return f_new#, x_new

    comov_bins = interp_binlimits(comovd) # comoving distance bins for digitizing
    npix = hp.pixelfunc.nside2npix(nside) # In original size maps
    nz = len(comovd)
    mapdtype = psi.dtype
    # Number of pixels in upgraded resolution per pixel of original resolution:
    ndiv_ang = nside_factor**2
    # Number of voxels in upgraded resolution per voxel of original resolution:
    n_subvoxels = ndiv_ang*ndiv_radial

    # Initial positions in higher resolution grid:
    comov0 = interp_subdivide(comovd,ndiv_radial)
    ang0 = np.array(hp.pixelfunc.pix2ang(nside*nside_factor, 
                    np.arange(hp.pixelfunc.nside2npix(nside*nside_factor))))

    # TODO: This might be slow too. Might include in the cython loop below?
    # (Actually only contains the small for loops, so it should be fine)
    # Interpolate angular displacements to higher resolution grid:
    psi = np.array([[ hp.pixelfunc.get_interp_val(psi[ii,jj],*ang0)
                    for jj in range(psi.shape[1]) ]
                    for ii in range(psi.shape[0]) ])

    # TODO: I am renaming psi 2 times in the function! One above and once
    # below, in the loop. This is confusing, and the second re-naming 
    # might even be creating problems.

    # Output of psi_int has shape: 
    # (ndim=3,length of input argument, npix*nside_factor**2)
    psi_int = interp1d(comovd, psi, axis=1, kind='linear')#,fill_value='extrapolate')
    delta_za = np.zeros((nz,npix),dtype=mapdtype)
    npasses = len(comov0)//nslices
    if len(comov0)%nslices > 0:
        npasses += 1
    for passi in range(npasses):
        slci = np.s_[passi*nslices:(passi+1)*nslices]
        psi_slc = psi_int(comov0[slci])
        # Final angles:
        ang1 = ang0[:,None,:]+psi_slc[:2]
        # Wrap theta around pi:
        wrap_idxs = np.where(np.logical_or(ang1[0]>np.pi,ang1[0]<0))
        ang1[0][wrap_idxs] = np.pi -  ang1[0][wrap_idxs]%np.pi
        ang1[1][wrap_idxs] = (ang1[1][wrap_idxs] + np.pi)%(2.*np.pi)
        # Wrap phi around 2pi
        wrap_idxs = np.where(np.logical_or(ang1[1]>2.*np.pi,ang1[1]<0))
        ang1[1][wrap_idxs] = ang1[1][wrap_idxs]%(2.*np.pi)
        # Final radial positions:
        comov1 = comov0[slci,None] + psi_slc[2]

        ang_idx = hp.pixelfunc.ang2pix(nside, *ang1)
        radial_idx = np.digitize(comov1,comov_bins)

        # TODO: I am not passing 'maps' as an argument yet
        idxs = zip(radial_idx.flatten(),ang_idx.flatten())
        counts = Counter(idxs)

        assign_counts(delta_za,counts,comovd,npix)

#        for ii in range(len(comovd)):
#            for jj in range(npix):
#                # TODO: Check if this is still necessary!!
##                 if (ii==0) or (ii==len(comovd)-1):
##                     # TODO: The 4 should be a variable?
##                     delta_za[ii,jj] = (counts[(ii,jj)]+4.)/float(n_subvoxels) - 1.
##                 else:
#                
#                delta_za[ii,jj] += counts[(ii,jj)]
##                 if ii==0 and jj==0:
##                     print counts[(ii,jj)]
##                     print delta_za[ii,jj]
##                     print delta_za[ii,jj]/float(n_subvoxels) - 1.

    return delta_za/float(n_subvoxels) - 1.

def assign_counts(delta_za,counts,comovd,npix):
    for ii in range(len(comovd)):
        for jj in range(npix):
            delta_za[ii,jj] += counts[(ii,jj)]


