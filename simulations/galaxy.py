

from os.path import join, dirname

import numpy as np
import h5py
import healpy

from cosmoutils import hputil, nputil
import foregroundsck
import maps
import skysim



_datadir = join(dirname(__file__), "data")



class ConstrainedGalaxy(maps.Sky3d):


    def getsky(self, debug=False, celestial=True):

        # Read in data files.
        haslam = healpy.ud_grade(healpy.read_map(join(_datadir, "haslam.fits")), self.nside) #hputil.coord_g2c()
        f = h5py.File(join(_datadir, 'skydata.hdf5'), 'r')
        s400 = healpy.ud_grade(f['/sky_400MHz'][:], self.nside)
        s800 = healpy.ud_grade(f['/sky_800MHz'][:], self.nside)
        f.close()

        beam = 1.0
        syn = foregroundsck.Synchrotron()

        lmax = 3*self.nside - 1

        efreq = np.concatenate((np.array([400.0, 800.0]), self.nu_pixels))

        cla = skysim.clarray(syn.angular_powerspectrum, lmax, efreq)

        fg = skysim.mkfullsky(cla, self.nside)

        sub4 = healpy.smoothing(fg[0], sigma=beam, degree=True)
        sub8 = healpy.smoothing(fg[1], sigma=beam, degree=True)
    
        fgs = skysim.mkconstrained(cla, [(0, sub4), (1, sub8)], self.nside)

        fgt = fg - fgs

        sc = np.log(s800 / s400) / np.log(2.0)

        fg2 = (haslam[np.newaxis, :] * (((efreq / 400.0)[:, np.newaxis]**sc) + (0.25 * fgt / fgs[0].std())))[2:]

        if celestial:
            for i in range(fg2.shape[0]):
                fg2[i] = hputil.coord_g2c(fg2[i])
    
        if debug:
            return fg2, fgt, fg, fgs, sc, sub4, sub8, s400, s800
        else:
            return fg2
