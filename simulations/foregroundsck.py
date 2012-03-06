"""Simulate foregrounds in the style of Santos, Cooray and Knox [1]_.

These are foregrounds that can be represented with a covariance
function that seperates with angle and frequency. Th

See Also
--------
foregroundmap

References
----------
.. [1] http://arxiv.org/abs/astro-ph/0408515

"""

import numpy as np

from utils import cubicspline as cs

from foregroundmap import *




class ForegroundSCK(ForegroundMap):
    r"""Base class for SCK style foregrounds.

    Need to set the four attributes `A`, `alpha`, `beta` and
    `zeta`. This class also allows calculation of the angular
    correlation function.

    See Also
    --------
    Synchrotron
    ExtraGalacticFreeFree
    GalacticFreeFree
    PointSources
    """

    nu_0 = 130.0

    _cf_int = None

    def angular_powerspectrum(self, larray):
        psarray =  self.A*(1e-3*larray)**(-self.beta)

        if isinstance(larray, np.ndarray):
            psarray[np.where(larray == 0)] = 0.0

        return psarray

    #def angular_powerspectrum(self, larray):
    #    return self.angular_ps((larray**2).sum(axis=2)**0.5)

    def frequency_covariance(self, nu1, nu2):
        return (self.frequency_variance(nu1) * self.frequency_variance(nu2))**0.5 * self.frequency_correlation(nu1, nu2)

    def frequency_variance(self, nu):
        r"""Variance on a single frequency slice."""
        return (nu/self.nu_0)**(-2*self.alpha)

    def frequency_correlation(self, nu1, nu2):
        r"""Correlation between two frequency slices."""
        return np.exp( - np.log(nu1/nu2)**2 / (2*self.zeta**2))

    def frequency_correlation_dlog(self, dlognu):
        r"""Correlation between two slices as a function of the seperation of the
        log of the two frequencies.

        This is useful because it is stationary in the log of the
        frequency and makes certain calculations easier.

        Parameters
        ----------
        dlognu : array_like
            The seperation between the log of the two frequencies
            (log(nu1) - log(nu2)).

        Returns
        -------
        acf: array_like
        """
        return np.exp( - (dlognu**2) / (2*self.zeta**2))



    def angular_correlation(self, tarray):
        r"""The 2-point angular correlation function.

        This will tabulate across the range [0, pi] on the first run,
        and will subsequently interpolate all calls.

        Parameters
        ----------
        tarray : array_like
            The angular seperations (in radians) to calculate at.

        Returns
        -------
        acf : array_like
        """

        if not self._cf_int:
            from scipy.special import lpn

            @np.vectorize
            def cf(theta):
                nmax = 10000
                nmin = 1
                larr = np.arange(nmin,nmax+1).astype(np.float64)
                pl = lpn(nmax, np.cos(theta))[0][nmin:]
                
                return ((2*larr+1.0)*pl*self.angular_powerspectrum(larr)).sum() / (4*np.pi)

            tarr = np.linspace(0, np.pi, 1000)
            cfarr = cf(tarr)
            self._cf_int = cs.Interpolater(tarr, cfarr)

        return self._cf_int(tarray)
            

        

class Synchrotron(ForegroundSCK):
    A = 700.0
    alpha = 2.80
    beta = 2.4
    zeta = 4.0

class ExtraGalacticFreeFree(ForegroundSCK):
    A = 0.014
    alpha = 2.10
    beta = 1.0
    zeta = 35.0

class GalacticFreeFree(ForegroundSCK):
    A = 0.088
    alpha = 2.15
    beta = 3.0
    zeta = 35.0

class PointSources(ForegroundSCK):
    A = 57.0
    alpha = 2.07
    beta = 1.1
    zeta = 1.0

