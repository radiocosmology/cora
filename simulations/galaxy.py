

from os.path import join, dirname, exists

import numpy as np
import h5py
import healpy

from cosmoutils import hputil, nputil
import foregroundsck
import maps
import skysim


_datadir = join(dirname(__file__), "data")


class FullSkySynchrotron(foregroundsck.Synchrotron):
    """Match up Synchrotron amplitudes to thise found in La Porta et al. 2008,
    for galactic latitudes abs(b) > 5 degrees"""
    A = 6.6e-3
    beta = 2.8
    nu_0 = 408.0
    l_0 = 100.0



def map_variance(input_map, nside):

    inp_nside = healpy.get_nside(input_map)

    # Convert to NESTED and then take advantage of this to group into lower
    # resolution pixels
    map_nest = healpy.reorder(input_map, r2n=True)
    map_nest = map_nest.reshape(-1, (inp_nside / nside)**2)

    # Calculate the variance in each low resolution pixel
    var_map = map_nest.var(axis=1)

    # Convert back to RING and return
    return healpy.reorder(var_map, n2r=True)





class ConstrainedGalaxy(maps.Sky3d):


    def __init__(self):

        self._load_data()

        vm = map_variance(healpy.smoothing(self._haslam, sigma=0.5, degree=True), 16)
        self._amp_map = healpy.smoothing(healpy.ud_grade(vm**0.5, 512), sigma=2.0, degree=True)


    def _load_data(self):
        print "Loading data for galaxy simulation."

        _haslam_file = join(_datadir, "haslam.fits")
        _sp_ind_file = join(_datadir, "spectral.hdf5")
        
        if not exists(_haslam_file):
            print "Needed data files missing! Try and fetch the Haslam map from LAMBDA [y/n]?"
            
            choice = raw_input().lower()
            if choice == 'y':
                import urllib
                _haslam_url = 'http://lambda.gsfc.nasa.gov/data/foregrounds/haslam/lambda_haslam408_dsds.fits'
                
                print "Downloading %s ...." % _haslam_url
                urllib.urlretrieve(_haslam_url, _haslam_file)
                print "Done."

            elif choice == 'n':
                raise Exception("No Haslam map found. Can not continue.")
            else:
                print "Please respond with 'y' or 'n'."
        
        self._haslam = healpy.read_map(join(_datadir, "haslam.fits"))
        self._sp_ind = h5py.File(_sp_ind_file)['spectral_index'][:]

        # Upgrade the map resolution to the same as the Healpix map (nside=512).
        self._sp_ind = healpy.smoothing(healpy.ud_grade(self._sp_ind, 512),  degree=True, sigma=1.0)


    def getsky(self, debug=False, celestial=True):

        # Read in data files.
        haslam = healpy.smoothing(healpy.ud_grade(self._haslam, self.nside), degree=True, fwhm=3.0) #hputil.coord_g2c()
        
        beam = 1.0
        syn = FullSkySynchrotron()

        lmax = 3*self.nside - 1

        efreq = np.concatenate((np.array([408.0, 1420.0]), self.nu_pixels))

        cla = skysim.clarray(syn.angular_powerspectrum, lmax, efreq)

        fg = skysim.mkfullsky(cla, self.nside)

        sub408 = healpy.smoothing(fg[0], fwhm=3.0, degree=True)
        sub1420 = healpy.smoothing(fg[1], fwhm=5.8, degree=True)
    
        fgs = skysim.mkconstrained(cla, [(0, sub408), (1, sub1420)], self.nside)

        sc = healpy.ud_grade(self._sp_ind, self.nside)
        am = healpy.ud_grade(self._amp_map, self.nside)
        mv = healpy.smoothing(map_variance(healpy.smoothing(fg[0], sigma=0.5, degree=True), 16)**0.5, degree=True, sigma=1.0).mean()

        fgt = (am / mv) * (fg - fgs)

        fg2 = (haslam[np.newaxis, :] * ((efreq / 408.0)[:, np.newaxis]**sc) + fgt)[2:]

        if celestial:
            for i in range(fg2.shape[0]):
                fg2[i] = hputil.coord_g2c(fg2[i])
    
        if debug:
            return fg2, fg, fgs, fgt

        return fg2
