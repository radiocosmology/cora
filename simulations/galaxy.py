"""Generate semi-realistic simulations of the full sky synchrotron emission
from the Milky Way."""

from os.path import join, dirname

import numpy as np
import h5py
import healpy

from cosmoutils import hputil, nputil
import foregroundsck
import maps
import skysim


_datadir = join(dirname(__file__), "data")


class FullSkySynchrotron(foregroundsck.Synchrotron):
    """Match up Synchrotron amplitudes to those found in La Porta et al. 2008,
    for galactic latitudes abs(b) > 5 degrees"""
    A = 6.6e-3
    beta = 2.8
    nu_0 = 408.0
    l_0 = 100.0


class FullSkyPolarisedSynchrotron(foregroundsck.Synchrotron):
    """Polarised Synchrotron amplitudes. Use same spectral shape,
    but with polarisation fraction=0.5, and a reduced correlation
    length (zeta=0.64), estimated from Faraday rotation=2pi, with RM=16.7 from
    http://adsabs.harvard.edu/abs/2009ApJ...702.1230T"""

    A = 1.65e-3
    beta = 2.8
    nu_0 = 408.0
    l_0 = 100.0
    zeta = 0.64


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
    """Create realistic simulations of the polarised sky.
    """

    def __init__(self):

        self._load_data()

        vm = map_variance(healpy.smoothing(self._haslam, sigma=np.radians(0.5)), 16)
        self._amp_map = healpy.smoothing(healpy.ud_grade(vm**0.5, 512), sigma=np.radians(2.0))


    def _load_data(self):
        print "Loading data for galaxy simulation."

        _data_file = join(_datadir, "skydata.npz")
        
        with np.load(_data_file) as f:
            self._haslam = f['haslam']
            self._sp_ind = f['spectral']
            self._faraday = f['faraday']

        # Upgrade the map resolution to the same as the Healpix map (nside=512).
        #self._sp_ind = healpy.smoothing(healpy.ud_grade(self._sp_ind, 512), sigma=np.radians(1.0))


    def getsky(self, debug=False, celestial=True):
        """Create a realisation of the unpolarised sky.

        Parameters
        ----------
        debug : boolean, optional
            Return intermediate products for debugging. Default is False.
        celestial : boolean, optional
            Maps are returned in celestial co-ordinates.

        Returns
        -------
        skymap : np.ndarray[freq, pixel]
        """
        # Read in data files.
        haslam = healpy.ud_grade(self._haslam, self.nside)

        syn = FullSkySynchrotron()

        lmax = 3*self.nside - 1

        efreq = np.concatenate((np.array([408.0, 1420.0]), self.nu_pixels))

        cla = skysim.clarray(syn.angular_powerspectrum, lmax, efreq, zwidth=(self.nu_pixels[1] - self.nu_pixels[0]))

        fg = skysim.mkfullsky(cla, self.nside)

        sub408 = healpy.smoothing(fg[0], fwhm=np.radians(1.0))
        sub1420 = healpy.smoothing(fg[1], fwhm=np.radians(5.8))
    
        fgs = skysim.mkconstrained(cla, [(0, sub408), (1, sub1420)], self.nside)

        sc = healpy.ud_grade(self._sp_ind, self.nside)
        am = healpy.ud_grade(self._amp_map, self.nside)

        mv = healpy.smoothing(map_variance(healpy.smoothing(fg[0], sigma=np.radians(0.5)), 16)**0.5, sigma=np.radians(2.0)).mean()

        fgt = (am / mv) * (fg - fgs)

        fgsmooth = haslam[np.newaxis, :] * ((efreq / 408.0)[:, np.newaxis]**sc)

        fg2 = (fgsmooth + fgt)[2:]

        if celestial:
            fg2 = hputil.coord_g2c(fg2)
    
        if debug:
            return fg2, fg, fgs, fgt, fgsmooth

        return fg2



    def getpolsky(self, debug=False, celestial=True):
        """Create a realisation of the unpolarised sky.

        Parameters
        ----------
        debug : boolean, optional
            Return intermediate products for debugging. Default is False.
        celestial : boolean, optional
            Maps are returned in celestial co-ordinates.

        Returns
        -------
        skymap : np.ndarray[freq, pixel]
        """

        # Load and smooth the Faraday emission
        sigma_phi = healpy.ud_grade(healpy.smoothing(np.abs(self._faraday), fwhm=np.radians(10.0)), self.nside)

        # Get the Haslam map as a base for the unpolarised emission
        haslam = healpy.smoothing(healpy.ud_grade(self._haslam, self.nside), fwhm=np.radians(3.0))

        # Create a map of the correlation length in phi
        xiphi = 3.0
        xiphimap = np.minimum(sigma_phi / 20.0, xiphi)

        lmax = 3*self.nside - 1
        la = np.arange(lmax+1)

        # The angular powerspectrum of polarisation fluctuations
        def angular(l):
            l[np.where(l == 0)] = 1.0e16
            return (l / 100.0)**-2.8

        c6 = 3e2 # Speed of light in million m/s
        xf = 1.5 # Factor to exand the region in lambda^2 by.

        ## Calculate the range in lambda^2 to generate
        l2l = (c6 / self.nu_upper)**2
        l2h = (c6 / self.nu_lower)**2

        l2a = 0.5 * (l2h + l2l)
        l2d = 0.5 * (l2h - l2l)

        # Bounds and number of points in lambda^2
        l2l = l2a - xf * l2d
        l2h = l2a + xf * l2d
        nl2 = int(xf * self.nu_num)

        l2 = np.linspace(l2l, l2h, nl2) # Grid in lambda^2 to use

        ## Make a realisation of c(n, l2)

        # Generate random numbers and weight by powerspectrum
        w = (np.random.standard_normal((nl2, lmax+1, 2*lmax+1)) + 1.0J * np.random.standard_normal((nl2, lmax+1, 2*lmax+1))) / 2**0.5
        w *= angular(la[np.newaxis, :, np.newaxis])**0.5

        # Transform from spherical harmonics to maps
        map1 = np.zeros((nl2, 12*self.nside**2), dtype=np.complex128)
        print "Making maps"
        for i in range(nl2):
            map1[i] = hputil.sphtrans_inv_complex(w[i], self.nside)

        # Weight frequencies for the internal Faraday depolarisation
        map1 *= np.exp(-2 * (xiphimap[np.newaxis, :] * l2[:, np.newaxis])**2) / 100.0


        if not debug:
            del w
 
        # Transform into phispace
        map2 = np.fft.fft(map1, axis=0)


        # Calculate the phi samples
        phifreq = np.pi * np.fft.fftfreq(nl2, d=(l2[1] - l2[0]))

        if not debug:
            del map1

        # Create weights for smoothing the emission (corresponding to the
        # Gaussian region of emission).
        w = np.exp(-0.25 * (phifreq[:, np.newaxis] / sigma_phi[np.newaxis, :])**2)

        # Calculate the normalisation explicitly (required when Faraday depth
        # is small, as grid is too large).
        w /= w.sum(axis=0)[np.newaxis, :]

        # When the spacing between phi samples is too large we don't get the
        # decorrelation from the independent regions correct.
        xiphimap = np.maximum(xiphimap, np.pi / (l2h - l2l))
        xiphimap = np.minimum(xiphimap, sigma_phi)
        w *= 0.2 * (sigma_phi / xiphimap)**0.5 * (10.0 / haslam)**0.5

        # Additional weighting to account for finite frequency bin width
        # (assume channels are gaussian). Approximate by performing in
        # lambda^2 not nu.
        dnu_nu = np.diff(self.nu_pixels).mean() / self.nu_pixels.mean()

        # Calculate the spacing lambda^2
        dl2 = 2*l2.mean() * dnu_nu / (8*np.log(2.0))**0.5
        w *= np.exp(-2.0 * dl2**2 * phifreq**2)[:, np.newaxis]

        # Weight map, and transform back into lambda^2
        map3 = np.fft.ifft(map2 * w, axis=0)

        if not debug:
            del map2, w

        # Array to hold frequency maps in.
        map4 = np.zeros((self.nu_num, 4, 12*self.nside**2), dtype=np.float64)

        ## Interpolate lambda^2 sampling into regular frequency grid.
        ## Use a piecewise linear method
        for i in range(self.nu_num):

            # Find the lambda^2 for the current frequency
            l2i = (c6 / self.nu_pixels[i])**2

            # Find the bounding array indices in the lambda^2 array
            ih = np.searchsorted(l2, l2i)
            il = ih - 1

            # Calculate the interpolating coefficient
            alpha = (l2i - l2[il]) / (l2[ih] - l2[il])

            # Interpolate each map at the same time.
            mapint = map3[ih] * alpha + (1.0 - alpha) * map3[il]

            # Separate real and imaginary parts into polarised matrix
            map4[i, 1] = mapint.real
            map4[i, 2] = mapint.imag

        if not debug:
            del map3

        # Unflatten the intensity by multiplying by the unpolarised realisation
        map4[:, 0] = self.getsky(celestial=False)
        map4[:, 1:3] *= map4[:, 0, np.newaxis, :]

        # Rotate to celestial co-ordinates if required.
        if celestial:
            map4 = hputil.coord_g2c(map4)

        if debug:
            return map4, map1, map2, map3, w, sigma_phi
        else:
            return map4
