"""
Gaussian foreground simultaion

Simulate gaussian foregrounds, in particular those in the style of Santos, Cooray and Knox [1]_.
These are foregrounds that can be represented with a covariance
function that seperates with angle and frequency.

References
----------
 .. [1] http://arxiv.org/abs/astro-ph/0408515
"""


import numpy as np

from cora.core import maps, gaussianfield
from cora.util import nputil
from cora.util import cubicspline as cs


class ForegroundMap(maps.Sky3d):
    r"""Simulate foregrounds with a separable angular and frequency
    covariance.

    Used to simulate foregrounds that can be modeled by angular
    covariances of the form:
    .. math:: C_l(\nu,\nu') = A_l B(\nu, \nu')
    """

    _weight_gen = False

    def angular_ps(self, l):
        r"""The angular function A_l. Must be a vectorized function
        taking either np.ndarrays or scalars.
        """
        pass

    def frequency_covariance(self, nu1, nu2):
        pass

    def angular_powerspectrum(self, l, nu1, nu2):
        return self.angular_ps(l) * self.frequency_covariance(nu1, nu2)

    def generate_weight(self, regen=False):
        r"""Pregenerate the k weights array.

        Parameters
        ----------
        regen : boolean, optional
            If True, force regeneration of the weights, to be used if
            parameters have been changed,
        """

        if self._weight_gen and not regen:
            return

        f1, f2 = np.meshgrid(self.nu_pixels, self.nu_pixels)

        ch = self.frequency_covariance(f1, f2)

        self._freq_weight, self._num_corr_freq = nputil.matrix_root_manynull(ch)

        rf = gaussianfield.RandomFieldA2.like_map(self)

        ## Construct a lambda function to evalutate the array of
        ## k-vectors.
        rf.powerspectrum = lambda karray: self.angular_ps(
            (karray**2).sum(axis=2) ** 0.5
        )

        self._ang_field = rf
        self._weight_gen = True

    def getfield(self):

        self.generate_weight()

        aff = np.fft.rfftn(self._ang_field.getfield())

        s2 = (self._num_corr_freq,) + aff.shape

        norm = np.tensordot(
            self._freq_weight, np.random.standard_normal(s2), axes=(1, 0)
        )

        return np.fft.irfft(np.fft.ifft(norm * aff[np.newaxis, :, :], axis=1), axis=2)


class ForegroundSCK(ForegroundMap):
    r"""Base class for SCK style foregrounds.

    Need to set the four attributes `A`, `alpha`, `beta` and
    `zeta`. This class also allows calculation of the angular
    correlation function. The units are the temperature in K.

    See Also
    --------
    Synchrotron
    ExtraGalacticFreeFree
    GalacticFreeFree
    PointSources
    """

    nu_0 = 130.0
    l_0 = 1000.0

    _cf_int = None

    def angular_ps(self, larray):

        if isinstance(larray, np.ndarray):
            mask0 = np.where(larray == 0)
            larray[mask0] = 1.0

        psarray = self.A * (larray / self.l_0) ** (-self.beta)

        if isinstance(larray, np.ndarray):
            psarray[mask0] = 0.0

        return psarray

    def frequency_covariance(self, nu1, nu2):
        return (
            self.frequency_variance(nu1) * self.frequency_variance(nu2)
        ) ** 0.5 * self.frequency_correlation(nu1, nu2)

    def frequency_variance(self, nu):
        r"""Variance on a single frequency slice."""
        return (nu / self.nu_0) ** (-2 * self.alpha)

    def frequency_correlation(self, nu1, nu2):
        r"""Correlation between two frequency slices."""
        return np.exp(-0.5 * (np.log(nu1 / nu2) / self.zeta) ** 2)

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
        return np.exp(-(dlognu**2) / (2 * self.zeta**2))

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
                larr = np.arange(nmin, nmax + 1).astype(np.float64)
                pl = lpn(nmax, np.cos(theta))[0][nmin:]

                return ((2 * larr + 1.0) * pl * self.angular_ps(larr)).sum() / (
                    4 * np.pi
                )

            tarr = np.linspace(0, np.pi, 1000)
            cfarr = cf(tarr)
            self._cf_int = cs.Interpolater(tarr, cfarr)

        return self._cf_int(tarray)


class Synchrotron(ForegroundSCK):
    A = 7.00e-4
    alpha = 2.80
    beta = 2.4
    zeta = 4.0


class ExtraGalacticFreeFree(ForegroundSCK):
    A = 1.40e-8
    alpha = 2.10
    beta = 1.0
    zeta = 35.0


class GalacticFreeFree(ForegroundSCK):
    A = 8.80e-8
    alpha = 2.15
    beta = 3.0
    zeta = 35.0


class PointSources(ForegroundSCK):
    A = 5.70e-5
    alpha = 2.07
    beta = 1.1
    zeta = 1.0
