"""
=====================================================
Galactic Synchrotron (:mod:`~cora.foreground.galaxy`)
=====================================================

.. currentmodule:: cora.foreground.galaxy

Generate semi-realistic simulations of the full sky synchrotron emission
from the Milky Way.

Classes
=======

.. autosummary::
    :toctree: generated/
   
    ConstrainedGalaxy
"""

from os.path import join, dirname

import numpy as np
import h5py
import healpy

from cora.core import maps, skysim
from cora.util import hputil, nputil
from cora.foreground import gaussianfg

_datadir = join(dirname(__file__), "data")


class FullSkySynchrotron(gaussianfg.Synchrotron):
    """Match up Synchrotron amplitudes to those found in La Porta et al. 2008,
    for galactic latitudes abs(b) > 5 degrees"""
    A = 6.6e-3
    beta = 2.8
    nu_0 = 408.0
    l_0 = 100.0


class FullSkyPolarisedSynchrotron(gaussianfg.Synchrotron):
    """Polarised Synchrotron amplitudes. Use same spectral shape,
    but with polarisation fraction=0.5, and a reduced correlation
    length (zeta=0.64), estimated from Faraday rotation=2pi, with RM=16.7 from
    http://adsabs.harvard.edu/abs/2009ApJ...702.1230T"""

    A = 1.65e-3
    beta = 2.8
    nu_0 = 408.0
    l_0 = 100.0
    zeta = 0.04


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

    Attributes
    ----------
    spectral_map : one of ['gsm', 'md', 'gd']    
        Specify which spectral index map to use. `gsm' uses a GSM derived map,
        this was the old behaviour. `md' uses the synchrotron index map
        derived by Miville-Deschenes et al. 2008, and is  now the default.
        `gd' uses the map from Giardino et al. 2002.
    nside

    Methods
    -------
    getsky
    getpolsky
    """

    # Spectral index map to use. Values are:
    # gsm (the old method)
    spectral_map = 'md'

    def __init__(self):

        self._load_data()

        vm = map_variance(healpy.smoothing(self._haslam, sigma=np.radians(0.5)), 16)
        self._amp_map = healpy.smoothing(healpy.ud_grade(vm**0.5, 512), sigma=np.radians(2.0))


    def _load_data(self):
        print "Loading data for galaxy simulation."

        _data_file = join(_datadir, "skydata.npz")

        f = np.load(_data_file)
        self._haslam = f['haslam']

        self._sp_ind = { 'gsm' : f['spectral_gsm'],
                         'md' : f['spectral_md'],
                         'gd' : f['spectral_gd'] }

        self._faraday = f['faraday']

        # Upgrade the map resolution to the same as the Healpix map (nside=512).
        #self._sp_ind = healpy.smoothing(healpy.ud_grade(self._sp_ind, 512), sigma=np.radians(1.0))


    def getsky(self, debug=False, celestial=True):
        """Create a realisation of the *unpolarised* sky.

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

        ## Construct map of random fluctuations
        #cla = skysim.clarray(syn.angular_powerspectrum, lmax, efreq, zwidth=(self.nu_pixels[1] - self.nu_pixels[0]))
        cla = skysim.clarray(syn.angular_powerspectrum, lmax, efreq, zromb=0)
        fg = skysim.mkfullsky(cla, self.nside)

        ## Find the smoothed fluctuations on each scale
        sub408 = healpy.smoothing(fg[0], fwhm=np.radians(1.0))
        sub1420 = healpy.smoothing(fg[1], fwhm=np.radians(5.8))
    
        ## Make a multifrequency map constrained to look like the smoothed maps
        ## depending on the spectral_map apply constraints at upper and lower frequency (GSM), 
        ## or just at Haslam map frequency
        if self.spectral_map == 'gsm':
            fgs = skysim.mkconstrained(cla, [(0, sub408), (1, sub1420)], self.nside)
        else:
            fgs = skysim.mkconstrained(cla, [(0, sub408)], self.nside)

        # Construct maps of appropriate resolution
        sc = healpy.ud_grade(self._sp_ind[self.spectral_map], self.nside)
        am = healpy.ud_grade(self._amp_map, self.nside)

        ## Bump up the variance of the fluctuations according to the variance
        #  map
        mv = healpy.smoothing(map_variance(healpy.smoothing(fg[0], sigma=np.radians(0.5)), 16)**0.5, sigma=np.radians(2.0)).mean()

        ## Construct the fluctuations map
        fgt = (am / mv) * (fg - fgs)

        ## Get the smooth, large scale emission from Haslam+spectralmap
        fgsmooth = haslam[np.newaxis, :] * ((efreq / 408.0)[:, np.newaxis]**sc)

        tanh_lin = lambda x: np.where(x < 0, np.tanh(x), x)

        fg2 = (fgsmooth * (1.0 + tanh_lin(fgt / fgsmooth)))[2:]

        ## Co-ordinate transform if required
        if celestial:
            fg2 = hputil.coord_g2c(fg2)

        if debug:
            return fg2, fg, fgs, fgt, fgsmooth

        return fg2



    def getpolsky(self, debug=False, celestial=True):
        """Create a realisation of the *polarised* sky.

        Parameters
        ----------
        debug : boolean, optional
            Return intermediate products for debugging. Default is False.
        celestial : boolean, optional
            Maps are returned in celestial co-ordinates.

        Returns
        -------
        skymap : np.ndarray[freq, pol, pixel]
        """

        # Load and smooth the Faraday emission
        sigma_phi = healpy.ud_grade(healpy.smoothing(np.abs(self._faraday), fwhm=np.radians(10.0)), self.nside)

        # Create a map of the correlation length in phi
        xiphi = 1.0

        lmax = 3*self.nside - 1
        la = np.arange(lmax+1)

        # The angular powerspectrum of polarisation fluctuations
        def angular(l):
            l[np.where(l == 0)] = 1.0e16
            return (l / 100.0)**-2.8

        dphi = 1.0
        maxphi = 500.0
        nphi = 2 * int(maxphi / dphi)
        phifreq = np.fft.fftfreq(nphi, d=(1.0 / (dphi * nphi)))

        ## Make a realisation of c(n, l2)

        print "Generating random field."
        w = (np.random.standard_normal((nphi, lmax+1, 2*lmax+1)) + 1.0J * np.random.standard_normal((nphi, lmax+1, 2*lmax+1))) / 2**0.5
        w *= angular(la[np.newaxis, :, np.newaxis])**0.5

        map1 = np.zeros((12*self.nside**2, nphi), dtype=np.complex128)
        print "SHTing to maps"
        for i in range(nphi):
            map1[:, i] = hputil.sphtrans_inv_complex(w[i], self.nside)

        print "FFTing to phi field"
        map2 = np.fft.fft(map1, axis=1)

        if not debug:
            del map1, w

        pcfreq = np.fft.fftfreq(nphi, d=dphi)

        map2 *= np.exp(-2 * (np.pi * xiphi * pcfreq[np.newaxis, :])**2)

        map3 = np.fft.ifft(map2, axis=1)
        map3 /= (2.0 * map3.std())

        if not debug:
            del map2

        w = np.exp(-0.25 * (phifreq[np.newaxis, :] / sigma_phi[:, np.newaxis])**2)

        # Calculate the normalisation explicitly (required when Faraday depth
        # is small, as grid is too large).
        w /= w.sum(axis=1)[:, np.newaxis]

        print "Applying phi weighting"
        map3 *= w

        if not debug:
            del w

        def ptrans(phi, freq, dfreq):

            dx = dfreq / freq

            alpha = 2.0 * phi * 3e2**2 / freq**2

            return (np.exp(1.0J * alpha) * np.sinc(alpha * dx / np.pi))

        fa = self.nu_pixels
        df = np.median(np.diff(fa))

        pta = ptrans(phifreq[:, np.newaxis], fa[np.newaxis, :], df) / dphi

        print "Transforming to freq"
        map4 = np.dot(map3, pta)

        if not debug:
            del map3

        print "Rescaling freq"
        map4a = np.abs(map4)
        map4 = map4 * np.tanh(map4a) / map4a

        del map4a

        map5 = np.zeros((self.nu_num, 4, 12 * self.nside**2), dtype=np.float64)


        print "Scaling by T"
        # Unflatten the intensity by multiplying by the unpolarised realisation
        map5[:, 0] = self.getsky(celestial=False)
        map5[:, 1] = map4.real.T
        map5[:, 2] = map4.imag.T
        map5[:, 1:3] *= map5[:, 0, np.newaxis, :]

        if not debug:
            del map4

        print "Rotating"
        # Rotate to celestial co-ordinates if required.
        if celestial:
            map5 = hputil.coord_g2c(map5)

        if debug:
            return map1, map2, map3, map4, w, sigma_phi
        else:
            return map5
