"""
Cosmology routines: A module for various cosmological calculations.

The bulk of the work is within the class :py:class:`Cosmology` which stores a
cosmology and can calculate quantities like distance measures.
"""
from dataclasses import dataclass, asdict

import numpy as np

# Import integration routines
from scipy import integrate as si

# Package imports
from cora.util import units as u


@dataclass
class Cosmology(object):
    """Class to store a cosmology, and compute measures.

    Defines a cosmology and allows calculation of a few simple
    quantities (notably distance measures and lookback time).

    Default params from Planck+WP+highL+BAO (http://arxiv.org/pdf/1303.5076v1.pdf).

    Attributes
    ----------
    units : {'astro', 'cosmo', 'si'}
        The unit system to return quantities in. SI is self
        explanatory, `astro` has distances in Mpc and time in Myr, and
        `cosmo` has distances in Mpc/h and time in Myr. Defaults to 'cosmo'.
    omega_b : scalar
        Baryon fraction relative to the critical density.
    omega_c : scalar
        Dark matter fraction relative to the critical density.
    omega_l : scalar
        Dark energy fraction relative to the critical density (assumes
        a cosmological constant).
    omega_g : scalar
        Fraction of electromagnetic radiation relative to the critical density.
    omega_n : scalar
        Fraction of massless neutrinos relative to the critical density.
    H0 : scalar
        The Hubble constant in km/s / Mpc
    w_0, w_a : scalars
        Dark energy parameters.

    omega_k : scalar, readonly
        The curvature as a fraction of the critical density.
    omega_m : scalar, readonly
        The total matter density as a fraction of the critical density.
    omega_r : scalar, readonly
        The total radiation density as a fraction of the critical density.

    """

    # The unit type to use
    # TODO: switch to Literal["cosmo", "astro", "si"] on Python 3.8
    units: str = "cosmo"

    # Standard density parameters
    omega_b: float = 0.0483
    omega_c: float = 0.2589
    omega_l: float = 0.6914

    # Density parameters more relevant for the early Universe
    omega_g: float = 0.0
    omega_n: float = 0.0

    # H_0 given in km/s / Mpc
    H0: float = 67.77

    # Dark energy parameters
    w_0: float = -1.0
    w_a: float = 0.0

    @property
    def omega_m(self) -> float:
        return self.omega_b + self.omega_c

    @property
    def omega_r(self) -> float:
        return self.omega_g + self.omega_n

    @property
    def omega_k(self) -> float:
        return 1.0 - (
            self.omega_l + self.omega_b + self.omega_c + self.omega_g + self.omega_n
        )

    @classmethod
    def from_physical(
        cls,
        ombh2: float = 0.022161,
        omch2: float = 0.11889,
        H0: float = 67.77,
        omkh2: float = 0.0,
        t0=2.726,
        nnu=3.046,
    ) -> "Cosmology":
        r"""Initialise a new cosmology from the physical parameters.

        This uses the CMB relevant parameterization that is commonly
        used. The dark energy density is calculated from the given
        parameters.

        Parameters
        ----------
        ombh2 : scalar, optional
            The relative baryon density times h^2 (h = H_0 / 100).
        omch2 : scalar, optional
            The fractional dark matter density times h^2 (h = H_0 / 100).
        H0 : scalar, optional
            The Hubble constant
        omkh2 : scalar, optional
            The curvature fraction times h^2.
        t0 : scalar, optional
            The CMB temperature (used to calculate the radiation density).
        nnu : scalar, optional
            The number of massless neutrinos. Used to set the neutrino density.

        Returns
        -------
        cosmo : instance of Cosmology
        """
        h = H0 / 100.0
        H_si = H0 * 1000.0 / u.mega_parsec
        rhoc = 3.0 * H_si ** 2 * u.c_sl ** 2 / (8.0 * np.pi * u.G_n)
        rhorad = u.a_rad * t0 ** 4
        rhonu = nnu * rhorad * 7.0 / 8.0 * (4.0 / 11.0) ** (4.0 / 3.0)

        omega_b = ombh2 / h ** 2
        omega_c = omch2 / h ** 2
        omega_g = rhorad / rhoc
        omega_n = rhonu / rhoc
        omega_l = 1.0 - (omkh2 + ombh2 + omch2) / h ** 2 - (omega_g + omega_n)

        return cls(
            omega_b=omega_b,
            omega_c=omega_c,
            omega_l=omega_l,
            omega_g=omega_g,
            omega_n=omega_n,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def H(self, z: float = 0.0):
        """The Hubble parameter at redshift z.

        Return the Hubble parameter in SI units (s^-1), regardless of
        the value of `self.units`.

        Parameters
        ----------
        z : scalar, optional
            The redshift to calculate the Hubble parameter
            at. Defaults to z = 0.

        Returns
        -------
        H : scalar
            The Hubble parameter.
        """

        H = (
            self.H0
            * (
                self.omega_r * (1 + z) ** 4
                + self.omega_m * (1 + z) ** 3
                + self.omega_k * (1 + z) ** 2
                + self.omega_l
                * (1 + z) ** (3 * (1 + self.w_0 + self.w_a))
                * np.exp(-3 * self.w_a * z / (1 + z))
            )
            ** 0.5
        )

        # Convert to SI
        return H * 1000.0 / u.mega_parsec

    def comoving_distance(self, z):
        r"""The comoving distance to redshift z.

        This routine is vectorized.

        Parameters
        ----------
        z : array_like
            The redshift(s) to calculate at.

        Returns
        -------
        dist : array_like
            The comoving distance to each redshift.
        """

        # Calculate the integrand.
        def f(z1):
            return u.c_sl / self.H(z1)

        return _intf_0_z(f, z) / self._unit_distance

    def proper_distance(self, z):
        r"""The proper distance to an event at redshift z.

        The proper distance can be ill defined. In this case we mean
        the comoving transverse separation between two events at the
        same redshift divided by their angular separation. This
        routine is vectorized.

        Parameters
        ----------
        z : array_like
            The redshift(s) to calculate at.

        Returns
        -------
        dist : array_like
            The proper distance to each redshift.
        """

        x = self.comoving_distance(z)

        om_k = self.omega_k

        dhi = np.sqrt(np.fabs(om_k)) * self.H() / u.c_sl * self._unit_distance

        if om_k < 0.0:
            x = np.sin(x * dhi) / dhi
        elif om_k > 0.0:
            x = np.sinh(x * dhi) / dhi

        return x

    def angular_distance(self, z):
        r"""The angular diameter distance to redshift z.

        Not to be confused with the `proper_distance`. This is the
        *physical* transverse separation between two events at the
        same redshift divided by their angular separation. This
        routine is vectorized.

        Parameters
        ----------
        z : array_like
            The redshift(s) to calculate at.

        Returns
        -------
        dist : array_like
            The angular diameter distance to each redshift.
        """

        return self.proper_distance(z) / (1 + z)

    def luminosity_distance(self, z):
        r"""The luminosity distance to redshift z. This
        routine is vectorized.

        Parameters
        ----------
        z : array_like
            The redshift(s) to calculate at.

        Returns
        -------
        dist : array_like
            The luminosity distance to each redshift.
        """
        return self.proper_distance(z) * (1 + z)

    def lookback_time(self, z):
        r"""The lookback time out to redshift z.

        Parameters
        ----------
        z : array_like
            The redshift(s) to calculate at.

        Returns
        -------
        time : array_like
            The lookback time to each redshift.
        """

        # Calculate the integrand.
        def f(z1):
            return 1.0 / (self.H(z1) * (1 + z1))

        return _intf_0_z(f, z) / self._unit_time

    @property
    def _unit_distance(self):
        # Select the appropriate distance unit
        if self.units == "astro":
            return u.mega_parsec
        elif self.units == "cosmo":
            return u.mega_parsec / (self.H0 / 100.0)
        elif self.units == "si":
            return 1.0

        raise Exception("Units not known")

    @property
    def _unit_time(self):
        # Select the appropriate time unit
        if self.units == "astro":
            return u.mega_year
        elif self.units == "cosmo":
            return u.mega_year
        elif self.units == "si":
            return 1.0

        raise Exception("Units not known")


def _intf_0_z(f, z):
    # Use an ODE integrator to carry out integrals up to certain redshifts.
    # The reason to use an ODE integrator is that this makes it quick to
    # evaluate at a large vector of redshifts

    # Wrap and unwrap in case we are operating on a scalar
    if not isinstance(z, np.ndarray):
        z = np.array([z])
        return _intf_0_z(f, z)[0]

    # Write the y' function for the ODE solver
    def _yp(y, z):
        return f(z)

    # Create the output array
    x = np.zeros_like(z)

    # Figure out the indices required to sort the redshifts into order
    sort_ind = np.argsort(z, axis=None)

    # Produce the redshift array required
    za = np.insert(z.ravel()[sort_ind], 0, 0)

    # Use the ODE integrator and write into the output array
    x.ravel()[sort_ind] = si.odeint(_yp, 0.0, za)[1:, 0]

    return x


def growth_factor(z, c=None):
    r"""Approximation for the matter growth factor.

    Uses a Pade approximation.

    Parameters
    ----------
    z : array_like
        Redshift to calculate at.

    Returns
    -------
    growth_factor : array_like

    Notes
    -----
    See [1]_.

    References
    ----------
    .. [1] http://arxiv.org/abs/1012.2671
    """

    if c is None:
        c = Cosmology()

    x = ((1.0 / c.omega_m) - 1.0) / (1.0 + z) ** 3

    num = 1.0 + 1.175 * x + 0.3064 * x ** 2 + 0.005355 * x ** 3
    den = 1.0 + 1.857 * x + 1.021 * x ** 2 + 0.1530 * x ** 3

    d = (1.0 + x) ** 0.5 / (1.0 + z) * num / den

    return d


def growth_rate(z, c):
    r"""Approximation for the matter growth rate.

    From explicit differentiation of the Pade approximation for
    the growth factor.

    Parameters
    ----------
    z : array_like
        Redshift to calculate at.

    Returns
    -------
    growth_factor : array_like

    Notes
    -----
    See [2]_.

    References
    ----------
    .. [2] http://arxiv.org/abs/1012.2671
    """

    if c is None:
        c = Cosmology()

    x = ((1.0 / c.omega_m) - 1.0) / (1.0 + z) ** 3

    dnum = 3.0 * x * (1.175 + 0.6127 * x + 0.01607 * x ** 2)
    dden = 3.0 * x * (1.857 + 2.042 * x + 0.4590 * x ** 2)

    num = 1.0 + 1.175 * x + 0.3064 * x ** 2 + 0.005355 * x ** 3
    den = 1.0 + 1.857 * x + 1.021 * x ** 2 + 0.1530 * x ** 3

    f = 1.0 + 1.5 * x / (1.0 + x) + dnum / num - dden / den

    return f


def sound_horizon(c=None):
    if c is None:
        c = Cosmology()

    h = c.H0 / 100.0

    # Calculate the sound horizon in Mpc
    s = (
        44.5
        * np.log(9.83 / (c.omega_m * h ** 2))
        / (1.0 + 10.0 * (c.omega_b * h ** 2) ** 0.75) ** 0.5
    )

    return s


def ps_nowiggle(kh, z=0.0, c=None):

    if c is None:
        c = Cosmology()

    h = c.H0 / 100.0

    # Change units in Mpc^{-1} not h Mpc^{-1}
    k = kh * h

    omh2 = c.omega_m * h ** 2
    rb = c.omega_b / c.omega_m
    alpha = (
        1.0 - 0.328 * np.log(431.0 * omh2) * rb + 0.38 * np.log(22.3 * omh2) * rb ** 2
    )

    s = sound_horizon(c)

    gamma = c.omega_m * h * (alpha + (1 - alpha) / (1 + (0.43 * k * s) ** 4))

    tcmb_27 = 2.726 / 2.7

    q = k * tcmb_27 ** 2 / (gamma * h)

    l0 = np.log(2 * np.exp(1.0) + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)

    t = l0 / (l0 + c0 * q ** 2)

    ns = 0.9611
    nbar = ns - 1.0

    A_s = 2.214e-9
    k0 = 0.05
    pkp = A_s * (k / k0) ** nbar

    # deltah = 1.94e-5 * c.omega_m**(-0.785 - 0.05 * np.log(c.omega_m)) * np.exp(-0.95 * nbar - 0.169 * nbar**2)

    d2k = (
        4.0
        / 25
        * (u.c_sl * k / (1000.0 * c.H0)) ** 4
        * t ** 2
        * pkp
        / c.omega_m ** 2
        * growth_factor(z, c) ** 2
    )

    # d2k = deltah**2 * (u.c_sl * k / (1000.0 * c.H0))**(3 + ns) * t**2

    pk = d2k * 2 * np.pi ** 2 / kh ** 3  # Change to normal PS

    return pk
