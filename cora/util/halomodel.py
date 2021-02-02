# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from ..signal import corr
import numpy as np
import scipy.integrate as integrate


class HaloModel(object):

    # TODO: move these to util/units.py ?
    dc = 1.686  # Critical density of colapse
    solar_mass = 1.98847e30

    def __init__(
        self,
        redshift=0.0,
        r_scales=None,
        gf=None,
        window="tophat",
        collapse_method="ellipsoidal",
        ps=None,
        cosmology=None,
    ):

        # Redshift to compute hm at.
        self.redshift = redshift

        if r_scales is None:
            # To get as low as M=10^3 solar masses
            r_scales = np.logspace(np.log10(2e-3), np.log10(15), 250)
        self.rs = r_scales

        self._set_mh1_interp()

        if cosmology is None:
            cosmology = corr.Cosmology()

        if gf is None:
            from .cosmology import growth_factor as gf
        # Ensure growth factor is normalized to z=0.
        self.growth_factor = lambda z: gf(z) / gf(0.0)

        if ps is None:
            from . import cubicspline as cs
            from os.path import join, dirname

            ps_redshift = 1.5  # Redshift of the power sectrum to be loaded
            psfile = join(dirname(corr.__file__), "data/ps_z1.5.dat")
            cs1 = cs.LogInterpolater.fromfile(psfile)
            self.ps = lambda k: cs1(k) * (gf(0.0) / gf(ps_redshift)) ** 2
        else:
            self.ps = ps

        h = cosmology.H0 / 100.0
        H0_1s = cosmology.H0 / (corr.units.kilo_parsec)  # H0 in 1/s (SI)
        rho_bar = (
            3.0 * H0_1s ** 2 / (8.0 * np.pi * corr.units.G) * cosmology.omega_m
        )  # in kg/m^3 (G comes in SI)
        rho_bar = (
            rho_bar * (corr.units.kilo_parsec * 1e3) ** 3
        )  # in kg/Mpc^3 (G comes in SI)
        self.rho_bar = (
            rho_bar / h ** 2 / self.solar_mass
        )  # in Mo h^-1 / (h^-1 Mpc)^3 (Mo is solar mass)
        self.collapse_method = collapse_method

        if window == "gauss":
            # Using Gaussian window function
            # Prefactor for the volume
            self.gammaf = (2.0 * np.pi) ** (1.5)

            # Window functio in k-space
            self.Wk = lambda kR: np.exp(-0.5 * kR ** 2)

        elif window == "sharpk":
            # Using sharp-k window function
            # Prefactor for the volume
            self.gammaf = 6.0 * np.pi ** 2

            # Window functio in k-space
            self.Wk = lambda kR: np.array(kR <= 1.0).astype(float)

        elif window == "tophat":
            # Using tophat window function
            # Prefactor for the volume
            self.gammaf = 4.0 * np.pi / 3.0

            # Window functio in k-space
            self.Wk = lambda kR: 3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR ** 3

    @property
    def rs(self):
        """Array of scales R"""
        return self._rs

    @rs.setter
    def rs(self, value):
        """Setter for scales R"""
        # Ensure rs is array-like
        if not isinstance(value, (np.ndarray, list)):
            value = np.array([value])
        self._rs = value
        self.rs_up_to_date = False

    @property
    def redshift(self):
        """"""
        return self._z

    @redshift.setter
    def redshift(self, value):
        """"""
        self._z = value
        self.z_up_to_date = False

    def _update_rs(self):
        """"""
        # Variance at each scale
        self._s2s = np.array([self._sigma2(r) for r in self.rs])
        # Volumes associated with the scales in rs
        self._vs = self.gammaf * self.rs ** 3
        # Masses associated with the scales in rs
        self._ms = self.rho_bar * self._vs
        # Mark as updated
        self.rs_up_to_date = True
        # Need to update redshift dependent quantities also
        self._update_z()

    def _update_z(self):
        """"""
        self._dct = self.dc / self.growth_factor(self.redshift)
        self._nus = self._dct / self._s2s ** 0.5
        self._nu2s = self._dct ** 2 / self._s2s
        # Mark as updated
        self.z_up_to_date = True

    def check_update(self):
        """"""
        if not self.rs_up_to_date:
            # Updates z-dependent quantities as well
            self._update_rs()
        elif not self.z_up_to_date:
            self._update_z()

    def _sigma2(self, r, k_int_lim=None):
        """From MBW eq.(7.44)
        R is a scalar here"""

        def integrand(k):
            return self.ps(k) * self.Wk(k * r) ** 2 * k ** 2

        # Determine integration limit if not given.
        if k_int_lim is None:
            k_int_lim = [0.0, 3.0 / r]
        # integrate.quad returns a tupple (value, error)
        return integrate.quad(integrand, k_int_lim[0], k_int_lim[1])[0] / (
            2.0 * np.pi ** 2
        )

    def _fps(self, nu, nu2):
        """ From MBW eq.(7.47) """
        return (2.0 / np.pi) ** 0.5 * nu * np.exp(-0.5 * nu2)

    def f(self):
        """From MBW eq.(7.47) and eq.(7.67)
        or from arXiv:astro-ph/9907024 eq.(6)"""
        self.check_update()

        if self.collapse_method == "ellipsoidal":
            A = 0.322
            q = 0.3
            a = 0.707
            nutilde = a ** 0.5 * self._nus
            nutilde2 = a * self._nu2s
            return A * (1.0 + 1.0 / nutilde2 ** q) * self._fps(nutilde, nutilde2)
        elif self.collapse_method == "spherical":
            return self._fps(self._nus, self._nu2s)

    def n(self):
        """ From MBW eq.(7.46) """
        self.check_update()
        return (
            self.rho_bar
            / self._ms ** 2
            * self.f()
            * abs(np.gradient(np.log(self._nus), np.log(self._ms)))
        )

    def lbias(self):
        """Lagrangian halo bias.
        From arXiv:astro-ph/9907024"""
        self.check_update()
        a = 0.707
        b = 0.5
        c = 0.6
        nutilde2 = a * self._nu2s
        #         return 1./self.dct(z) * ( nutilde2
        return (
            1.0
            / self.dc
            * (
                nutilde2
                + b * nutilde2 ** (1.0 - c)
                - (nutilde2 ** c / a ** 0.5)
                / (nutilde2 ** c + b * (1.0 - c) * (1.0 - c * 0.5))
            )
        )

    def _set_mh1_interp(self):
        """Halo HI mass function - the average HI mass
        hosted by a halo of mass M at redshift z.
        Taken from arXiv:1804.09180 eq.(13).
        (all masses in h^-1 M_solar)"""

        import scipy.interpolate as interpolate

        kind = "quadratic"

        zs = np.array([0, 1, 2, 3, 4, 5])
        M0pts = np.array([4.3, 1.5, 1.3, 0.29, 0.14, 0.19]) * 1e10
        Mminpts = np.array([200.0, 60.0, 36.0, 6.7, 2.1, 2.0]) * 1e10
        alphapts = np.array([0.24, 0.53, 0.6, 0.76, 0.79, 0.74])

        # TODO: use the errors to introduce stochastic variability?
        M0err = np.array([1.1, 0.7, 0.6, 0.2, 0.1, 0.12]) * 1e10
        Mminerr = np.array([60.0, 29.0, 16.0, 4.0, 1.3, 1.2]) * 1e10
        alphaerr = np.array([0.05, 0.06, 0.05, 0.05, 0.04, 0.04])

        self.M0 = interpolate.interp1d(zs, M0pts, kind=kind)
        self.Mmin = interpolate.interp1d(zs, Mminpts, kind=kind)
        self.alpha = interpolate.interp1d(zs, alphapts, kind=kind)

    def mh1(self, M, z=None):
        """Halo HI mass function - the average HI mass
        hosted by a halo of mass M at redshift z.
        Taken from arXiv:1804.09180 eq.(13).
        (all masses in h^-1 M_solar)"""

        if z is None:
            z = self.redshift

        return (
            self.M0(z)
            * (M / self.Mmin(z)) ** self.alpha(z)
            * np.exp(-((self.Mmin(z) / M) ** 0.35))
        )

    def lbias_h1(self):
        """ """
        # Check that object is up to date
        self.check_update()

        dndm = self.n()
        mh1 = self.mh1(self._ms)
        rhoh1 = np.sum(dndm * mh1 * np.gradient(self._ms))
        return 1.0 / rhoh1 * np.sum(dndm * mh1 * self.lbias() * np.gradient(self._ms))
