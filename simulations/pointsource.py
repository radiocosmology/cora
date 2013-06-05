"""Simulating extra-galactic point sources."""

from os.path import join, dirname

import numpy as np
import numpy.random as rnd

from scipy.optimize import newton

import healpy

from cosmoutils import units

import poisson as ps

from maps import *

import foregroundsck



def faraday_rotate(polmap, rm_map, frequencies):
    """Faraday rotate a set of sky maps.

    Parameters
    ----------
    polmap : np.ndarray[freq, pol, pixel]
        The maps of the sky (assumed to be packed as T, Q, U and optionally V.
    rm_map : np.ndarray[pixel]
        The rotation measure across the sky (in radians / m^2).
    frequencies : np.ndarray[freq]
        The frequencies in the map in MHz.

    Returns
    -------
    rot_map : np.ndarray[freq, pol, pixel]
        The Faraday rotated map.
    """
    
    qu_complex = polmap[:, 1] + 1.0J * polmap[:, 2]

    wv = 1e-6 * units.c / frequencies

    faraday = np.exp(-2.0J * wv[:, np.newaxis] * rm_map[np.newaxis, :])

    qu_complex = qu_complex * faraday

    polmap[:, 1] = qu_complex.real
    polmap[:, 2] = qu_complex.imag

    return polmap


class PointSourceModel(Map3d):
    r"""Represents a population of astrophysical point sources.

    This is the base class for modelling a population of point
    sources. This assumes they are described by a source count
    function, and a spectral function which may depend on the flux. To
    create a model, a source_count and spectral_realisation must be
    implemented.

    Attributes
    ----------
    flux_min : float
        The lower flux limit of sources to include. Defaults to 1 mJy.
    flux_max : {float, None}
        The upper flux limit of sources to include. If `None` then
        include all sources (with a high probability).
    faraday : boolean
        Whether to Faraday rotate polarisation maps (default is True).
    sigma_pol_frac : scalar
        The standard deviation of the polarisation fraction of sources.
        Default is 0.03. See http://adsabs.harvard.edu/abs/2004A&A...415..549R
    """

    flux_min = 1e-4
    flux_max = None

    faraday = True

    sigma_pol_frac = 0.03


    def __init__(self):

        _data_file = join(dirname(__file__), 'data', "skydata.npz")
        
        with np.load(_data_file) as f:
            self._faraday = f['faraday']


    def source_count(self, flux):
        r"""The expected number of sources per unit flux (Jy) per steradian.

        This is an abstract method that must be implemented in an actual model.

        Parameters
        ----------
        flux : float
            The strength of the source in Jy.

        Returns
        -------
        value : float
            The differential source count at `flux`.
        
        Notes
        -----
        """
        pass

    def spectral_realisation(self, flux, frequencies):
        r"""Generate a frequency distribution for a source of given `flux`.
        
        This is an abstract method that must be implemented in an
        actual model. Must be able to broadcast if `flux`, and
        `frequencies` are numpy arrays.

        Parameters
        ----------
        flux : float
            The strength of the source in Jy.
        frequencies : ndarray
            The frequencies to calculate the spectral distribution at.

        Returns
        -------
        fluxes : ndarray
            The flux at each of the `frequencies` given.
        """
        pass
        
    def generate_population(self, area):
        r"""Create a set of point sources.

        Parameters
        ----------
        area : float
            The area the population is contained within (in sq degrees).
            
        Returns
        -------
        sources : ndarray
            The fluxes of the sources in the population.
        """


        flux_max = self.flux_max

        # If we don't have a maximum flux set, set one by calculating
        # the flux at which there is only a very small probability of
        # there being a brighter source.
        #
        # For a power law with dN/dS \propto S^(1-\beta), this is the
        # flux at which the probability of a source P(>S_max) < 0.05/
        # \beta
        if(flux_max == None):
            ratelog = lambda s: (s*area*self.source_count(s) - 5e-2)
            flux_max = newton(ratelog, self.flux_min)
            print "Using maximum flux: %e Jy" % flux_max

        # Generate realisation by creating a rate and treating like an
        # inhomogenous Poisson process.
        #rate = lambda s: area*self.source_count(self.flux_min+s)
        #fluxes = self.flux_min + ps.inhomogeneous_process_approx(flux_max-self.flux_min, rate)
        rate = lambda s: self.flux_min*np.exp(s)*area*self.source_count(self.flux_min*np.exp(s))
        fluxes = self.flux_min * np.exp(ps.inhomogeneous_process_approx(np.log(flux_max/self.flux_min), rate))

        return fluxes


    def getfield(self, catalogue = False):
        r"""Create a simulated cube of point sources.

        Create a pixelised realisation of the sources.

        Parameters
        ----------
        catalogue : boolean, optional
            if true return the population catalogue.

        Returns
        -------
        cube : ndarray
            An array of dimensions (`numf`, `numx` `numy`)
        """

        c = np.zeros(self._num_array())

        fluxes = self.generate_population(np.radians(self.x_width) * np.radians(self.y_width))

        freq = self.nu_pixels
        
        sr = self.spectral_realisation(fluxes[:,np.newaxis], freq[np.newaxis,:])
        
        for i in xrange(sr.shape[0]):
            # Pick random pixel
            x = int(rnd.rand() * self.x_num)
            y = int(rnd.rand() * self.y_num)

            c[:,x,y] += sr[i,:]

        if not catalogue:
            return c
        else:
            return c, fluxes


    def getsky(self):
        """Simulate a map of point sources.

        Returns
        -------
        sky : ndarray [nfreq, npix]
            Map of the brightness temperature on the sky (in K).
        """

        if self.flux_min < 0.1:
            print "This is going to take a long time. Try raising the flux limit."

        npix = 12*self.nside**2

        sky = np.zeros((self.nu_num, npix), dtype=np.float64)

        pxarea = 4*np.pi / npix
        
        fluxes = self.generate_population(4*np.pi)

        freq = self.nu_pixels
        
        sr = self.spectral_realisation(fluxes[:,np.newaxis], freq[np.newaxis,:])

        for i in xrange(sr.shape[0]):
            # Pick random pixel
            ix = int(rnd.rand() * npix)

            sky[:, ix] += sr[i,:]

        # Convert flux map in Jy to brightness temperature map in K.
        sky = sky * 1e-26 * units.c**2 / (2 * units.k_B * self.nu_pixels[:, np.newaxis]**2 * 1e12 * pxarea)
        return sky


    def getpolsky(self):
        """Simulate polarised point sources.
        """

        sky_I = self.getsky()

        sky_pol = np.zeros((sky_I.shape[0], 4, sky_I.shape[1]), dtype=sky_I.dtype)

        q_frac = self.sigma_pol_frac * np.random.standard_normal(sky_I.shape[1])[np.newaxis, :]
        u_frac = self.sigma_pol_frac * np.random.standard_normal(sky_I.shape[1])[np.newaxis, :]     

        sky_pol[:, 0] = sky_I
        sky_pol[:, 1] = sky_I * q_frac
        sky_pol[:, 2] = sky_I * u_frac

        if self.faraday:
            faraday_rotate(sky_pol, healpy.ud_grade(self._faraday, self.nside), self.nu_pixels)

        return sky_pol



class PowerLawModel(PointSourceModel):
    r"""A simple point source model.
    
    Use a power-law luminosity function, and power-law spectral
    distribution with a single spectral index drawn from a Gaussian.

    Attributes
    ----------
    source_index : scalar
        The spectral index of the luminosity function.
    source_pivot : scalar
        The pivot of the luminosity function (in Jy).
    source_amplitude : scalar
        The amplitude of the luminosity function (number of sources /
        Jy / deg^2 at the pivot).
    spectral_mean : scalar
        The mean index of the spectral distribution
    spectral_width : scalar
        Width of distribution spectral indices.
    spectral_pivot : scalar
        Frequency of the pivot point (in Mhz). This is the frequency
        that the flux is defined at.

    Notes
    ----- 
    Default source count parameters based loosely on the results of
    the 6C survey [1]_

    References
    ----------
    .. [1] Hales et al. 1988.
    """

    source_index = 2.5
    source_pivot = 1.0
    source_amplitude = 2.396e3

    spectral_mean = -0.7
    spectral_width = 0.1

    spectral_pivot = 151.0

    def source_count(self, flux):
        r"""Power law luminosity function."""

        return self.source_amplitude * (flux / self.source_pivot)**(-self.source_index)

    def spectral_realisation(self, flux, freq):
        r"""Power-law spectral function with Gaussian distributed index."""

        ind = self.spectral_mean + self.spectral_width * rnd.standard_normal(flux.shape)

        return flux * (freq / self.spectral_pivot)**ind


class DiMatteo(PointSourceModel):
    r"""Double power-law point source model
    
    Uses the results of Di Mattero et al. [1]_
    
    Attributes
    ----------
    gamma1, gamma2 : scalar
        The two power law indices.
    S_0 : scalar
        The pivot of the source count function (in Jy).
    k1 : scalar
        The amplitude of the luminosity function (number of sources /
        Jy / deg^2 at the pivot).
    spectral_mean : scalar
        The mean index of the spectral distribution
    spectral_width : scalar
        Width of distribution spectral indices.
    spectral_pivot : scalar
        Frequency of the pivot point (in Mhz). This is the frequency
        that the flux is defined at.

    Notes
    -----
    Based on [1]_ and clarification in [2]_ (footnote 6). In this
    `S_0` is both the pivot and normalising flux, which means that k1
    is rescaled by a factor of 0.88**-1.75.i

    References
    ----------
    .. [1] Di Matteo et al. 2002 (http://arxiv.org/abs/astro-ph/0109241)
    .. [2] Santos et al. 2005 (http://arxiv.org/abs/astro-ph/0408515)
    """

    gamma1 = 1.75
    gamma2 = 2.51
    S_0 = 0.88
    k1 = 1.52

    spectral_mean = -0.7
    spectral_width = 0.1

    spectral_pivot = 151.0

    def source_count(self, flux):
        r"""Power law luminosity function."""
        
        s = flux / self.S_0
        
        return self.k1 / (s**self.gamma1 + s**self.gamma2)

    def spectral_realisation(self, flux, freq):
        r"""Power-law spectral function with Gaussian distributed index."""

        ind = self.spectral_mean + self.spectral_width * rnd.standard_normal(flux.shape)

        return flux * (freq / self.spectral_pivot)**ind



class CombinedPointSources(foregroundsck.PointSources, DiMatteo):
    """Combined class for efficiently generating full sky point source maps.

    For S < S_{cut} = 0.1 Jy use a Gaussian approximation to generate a map,
    and for S > S_{cut} generate a synthetic population. Amplitude of
    foregroundsck.PointSources class rescaled for a maximum flux of 0.1.
    """ 
    A = 3.55e-5
    nu_0 = 408.0
    l_0 = 100.0

    flux_min = 0.1
    #flux_max = 100.0

    def getsky(self):

        sky = foregroundsck.PointSources.getsky(self) + DiMatteo.getsky(self)

        return sky

    def getpolsky(self):

        # Force calling the DiMatteo getpolsky
        return DiMatteo.getpolsky(self)



## Test program run when executing module.
if __name__=='__main__':
    
    p = DiMatteo()
    r = p.generate_population(1.0)

    c = p.getfield(catalogue = True)
    
