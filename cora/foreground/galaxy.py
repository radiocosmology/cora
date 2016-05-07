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
import healpy

from cora.core import maps, skysim
from cora.util import hputil
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



def chunk_var(a):
    """A variance routine that breaks the calculation up into chunks to save
    memory.

    Parameters
    ----------
    a : np.ndarray
        Array to take the variance of.

    Returns
    -------
    var : float
    """
    nchunks = min(30, a.size)

    # Does not eat memory.
    mean = a.mean()
    splits = np.array_split(a.ravel(), nchunks)

    t = 0.0
    for sec in splits:
        x = sec - mean
        x2 = np.sum(np.abs(x)**2)
        t += x2

    return t / a.size




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

    _dphi = 1.0
    _maxphi = 500.0

    def __init__(self):

        self._load_data()

        vm = map_variance(healpy.smoothing(self._haslam, sigma=np.radians(0.5), verbose=False), 16)
        self._amp_map = healpy.smoothing(healpy.ud_grade(vm**0.5, 512), sigma=np.radians(2.0), verbose=False)


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
        sub408 = healpy.smoothing(fg[0], fwhm=np.radians(1.0), verbose=False)
        sub1420 = healpy.smoothing(fg[1], fwhm=np.radians(5.8), verbose=False)

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
        vm = healpy.smoothing(fg[0], sigma=np.radians(0.5), verbose=False)
        vm = healpy.smoothing(map_variance(vm, 16)**0.5, sigma=np.radians(2.0), verbose=False)
        mv = vm.mean()

        ## Construct the fluctuations map
        fgt = (am / mv) * (fg - fgs)

        ## Get the smooth, large scale emission from Haslam+spectralmap
        fgsmooth = haslam[np.newaxis, :] * ((efreq / 408.0)[:, np.newaxis]**sc)

        # Rescale to ensure output is always positive
        tanh_lin = lambda x: np.where(x < 0, np.tanh(x), x)
        fg2 = (fgsmooth * (1.0 + tanh_lin(fgt / fgsmooth)))[2:]

        ## Co-ordinate transform if required
        if celestial:
            fg2 = hputil.coord_g2c(fg2)

        if debug:
            return fg2, fg, fgs, fgt, fgsmooth, am, mv

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

        Notes
        -----
        This routine tries to make a decent simulation of the galactic
        polarised emission.
        """

        # Load and smooth the Faraday rotation map to get an estimate for the
        # width of the distribution
        sigma_phi = healpy.ud_grade(healpy.smoothing(np.abs(self._faraday), fwhm=np.radians(10.0), verbose=False), self.nside)

        # Set the correlation length in phi
        xiphi = 1.0

        lmax = 3*self.nside - 1
        la = np.arange(lmax+1)

        # The angular powerspectrum of polarisation fluctuations, we will use
        # this to generate the base fluctuations
        def angular(l):
            l[np.where(l == 0)] = 1.0e16
            return (l / 100.0)**-2.8

        # Define a grid in phi that we will use to model the Faraday emission.
        dphi = self._dphi
        maxphi = self._maxphi
        nphi = 2 * int(maxphi / dphi)
        phifreq = np.fft.fftfreq(nphi, d=(1.0 / (dphi * nphi)))

        # Create the weights required to turn random variables into alms
        ps_weight = (angular(la[:, np.newaxis]) / 2.0)**0.5

        # Generate random maps in the Fourier conjugate of phi. This is
        # equivalent to generating random uncorrelated phi maps and FFting
        # into the conjugate.
        map2 = np.zeros((12*self.nside**2, nphi), dtype=np.complex128)
        print "SHTing to give random maps"
        for i in range(nphi):
            w = np.random.standard_normal((lmax+1, 2*lmax+1, 2)).view(np.complex128)[..., 0]
            w *= ps_weight
            map2[:, i] = hputil.sphtrans_inv_complex(w, self.nside)

        # Weight the conj-phi direction to give the phi correlation structure.
        pcfreq = np.fft.fftfreq(nphi, d=dphi)
        map2 *= np.exp(-2 * (np.pi * xiphi * pcfreq[np.newaxis, :])**2)

        # We need to FFT back into phi, but as scipy does not have an inplace
        # transform, we can do this in blocks, replacing as we go.
        chunksize = self.nside**2
        nchunk = 12

        for ci in range(nchunk):
            si = ci * chunksize
            ei = (ci + 1) * chunksize

            map2[si:ei] = np.fft.ifft(map2[si:ei], axis=1)

        # numpy's var routine is extremely memory inefficient. Use a crappy
        # chunking one.
        map2 /= (2.0 * chunk_var(map2)**0.5)

        w = np.exp(-0.25 * (phifreq[np.newaxis, :] / sigma_phi[:, np.newaxis])**2)

        # Calculate the normalisation explicitly (required when Faraday depth
        # is small, as grid is too large).
        w /= w.sum(axis=1)[:, np.newaxis]

        print "Applying phi weighting"
        map2 *= w

        if not debug:
            del w

        def ptrans(phi, freq, dfreq):

            dx = dfreq / freq

            alpha = 2.0 * phi * 3e2**2 / freq**2

            return (np.exp(1.0J * alpha) * np.sinc(alpha * dx / np.pi))
            #return np.exp(1.0J * alpha)

        fa = self.nu_pixels
        df = np.median(np.diff(fa))

        pta = ptrans(phifreq[:, np.newaxis], fa[np.newaxis, :], df) / dphi

        print "Transforming to freq"
        map4 = np.dot(map2, pta)

        if not debug:
            del map2

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
            return map2, map4, w, sigma_phi, pta, map5
        else:
            return map5
