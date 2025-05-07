"""Implementations of models for cosmological quantities."""

from typing import Callable, Optional, Union, List

import numpy as np

from caput import units

from .lssutil import FloatArrayLike
from ..util.cosmology import Cosmology


class PolyModelSet:
    """Define a set of related models defined by polynomial functions.

    To define the models use a `_model` attribute. This must be a dict with the name of
    the model as they key and the value is a tuple `(x0, [coeff, ...], [power, ...])`.
    The model is evaluated as sum_i(coeff_i * (x - x0)**power_i). T
    """

    default_model = None

    @classmethod
    def get(
        cls, model: Optional[str] = None
    ) -> Callable[[FloatArrayLike], FloatArrayLike]:
        """Return a function to evaluate the given model.

        Parameters
        ----------
        model
            Name of the model to return.

        Returns
        -------
        func
            Function that evaluates the model. This is a vector function that works for
            both scalars and array.
        """
        model = cls._validate_model(model)

        def f(x):
            return cls.evaluate(x, model=model)

        return f

    def __class_getitem__(
        cls, model: str
    ) -> Callable[[FloatArrayLike], FloatArrayLike]:
        """Return a function that evaluates the given model."""
        return cls.get(model)

    @classmethod
    def evaluate(cls, x: FloatArrayLike, model: Optional[str] = None) -> FloatArrayLike:
        """Evaluate the given model at the specified locations.

        Parameters
        ----------
        x
            The locations to evaluate at.
        model
            Name of the model to use.

        Returns
        -------
        f
            The values of the model at `x`.
        """
        model = cls._validate_model(model)
        mparams = cls._models[model]
        return cls.evaluate_poly(x, *mparams)

    @classmethod
    def _validate_model(cls, model: Union[str, None]) -> str:
        """Validate the model argument"""

        if model is None:
            if cls.default_model is None:
                raise ValueError(f"No model provided and no default specified.")
            model = cls.default_model

        if model not in cls._models:
            raise ValueError(f'Model "{model}" not known.')

        return model

    @staticmethod
    def evaluate_poly(x, x0, coeffs, powers=None):

        if powers is None:
            pc_iter = enumerate(coeffs)
        else:
            pc_iter = zip(powers, coeffs)

        return np.sum([c * (x - x0) ** p for p, c in pc_iter], axis=0)

    @classmethod
    def models(cls) -> List[str]:
        """The names of the available models."""
        return list(cls._models.keys())


class bias(PolyModelSet):
    """Models for the tracer and HI bias as a function of redshift.

    .. warning::
        This is the Lagrangian bias. Add one to get the Eulerian bias.

    Notes
    -----
    This includes three explicit models for the BOSS/eBOSS tracers: quasars, LRGs and
    ELGs.

    For the quasars we use the modelling of Laurent et al. 2017 [1]_, which fits a
    quadratic function of redshift to the measured bias from subsamples of the eBOSS
    quasar data (equations 5.2 and 5.3).

    For the LRGs there are bias estimates across a range of redshifts within Zhai et al.
    2017 [2]_ combining BOSS and eBOSS data along with a HOD scheme for modelling the
    bias. They don't give an explicit fit to the data, so here we construct a quadratic
    approximation to their BOSS+eBOSS model within the top panel of Figure 12 (red
    dotted line).

    For the ELGs there do not seem to be any estimates of the redshift dependence of the
    bias estimates. We combine the single estimate from de Mattia et al. 2020 [3]_ (b_E
    = 1.5 at z_eff = 0.85) along with the redshift dependence derived from simulations
    in Merson et al. 2019 [4]_ (db/dz ~ 0.7).

    It also includes a model for the HI bias. This is a 5th order polynomial fit to the
    data from [5]_ which is itself a fit to the simulations in [6]_ and [7]_.

    References
    ----------
    .. [1] https://arxiv.org/abs/1705.04718
    .. [2] https://arxiv.org/abs/1607.05383
    .. [3] https://arxiv.org/abs/2007.09008
    .. [4] https://arxiv.org/abs/1903.02030
    .. [5] https://github.com/slosar/PUMANoise/blob/b17b30ec84c6e55d8/castorina.py
    .. [6] https://arxiv.org/abs/1609.05157
    .. [7] https://arxiv.org/abs/1804.09180

    Examples
    --------
    >>> from cora.signal.lssmodels import bias
    >>> bias["HI"](1.0)
    0.489
    >>> bias.evaluate(np.array([0.5, 1.5]), model="eboss_qso")
    array([0.195495, 1.309695])
    """

    _models = {
        "eboss_qso": (1.55, [1.38, 1.42, 0.278]),
        "eboss_lrg": (0.40, [1.03, 0.862, 0.131]),
        "eboss_elg": (0.85, [0.5, 0.7]),
        "HI": (1.0, [0.489, 0.460, -0.118, 0.0678, -0.0128, 0.0009]),
    }


class omega_HI(PolyModelSet):
    """Models for the neutral hydrogen fraction.

    Notes
    -----
    We use two models for the evolution of Omega_HI. A fit from Crighton et al. 2015
    [1]_ (`Crighton2015`) and a fit from the SKA WG whitepaper [2]_ (`SKA`) as well as
    allowing a constant in redshift value to be set (`uniform`), with a value
    taken from Switzer et al. 2013 [3]_ assuming :math:`b_{HI}=1`.

    References
    ----------
    .. [1]: https://arxiv.org/abs/1506.02037
    .. [2]: https://arxiv.org/abs/1811.02743
    .. [3]: https://arxiv.org/abs/1304.3712
    """

    _models = {
        "Crighton2015": (-1.0, [4e-4], [0.6]),
        "SKA": (0.0, [4.8e-4, 3.9e-4, -6.5e-5]),
        "uniform": (0.0, [0.6e-3]),
    }

    default_model = "Crighton2015"


class sigma_P(PolyModelSet):
    """Models for the virial velocity scale in Mpc/h.

    Notes
    -----
    There are builtin models for three tracers: Quasars, LRGs, and ELGs. For these a
    measurement from actual data has been used to provide an overall normalisation and a
    satellite-HOD-weighted average over `sigma_v(M)` has been used to provide a redshift
    dependence (the output of this calculation is renormalised by the measurement).

    For LRGs we use a measurement from Gil-Marin et al. 2021 [1]_. The baseline model
    uses a HOD from Alam et al. 2020 [8]_, and the alt model from Zhai et al. 2017 [2]_.

    For ELGs we use a measurement from de Mattia et al. 2021 [3]_. The baseline model
    uses the HMQ HOD from Alam et al. 2020 [8]_ and alt model the `HOD-3` HOD from Avila
    et al. 2020 [4]_.

    For QSOs we use a measurement from Zarrouk et al. 2018 [5]_ that attempts to
    separate the physical FoG damping from the damping caused by redshift errors. Their
    damping is modelled by a Gaussian, and we have converted their stated damping scale
    into one that is roughly equivalent for a double Lorentzian. We have taken the
    average of the values from their 3-multipole and 3-wedge HOD values. The baseline
    HOD is also from Alam et al. [8]_, and the alt model uses the `4KDE+15eBOSS` QSO HOD
    from Eftekharzadeh et al. 2020 [6]_.

    For HI we use the S+B LP model from Sarkar & Bharadwaj 2019 [7]_, with a
    :math:`\sqrt{2}` factor to approximately account for the fact they model with a
    single, not squared, Lorentzian. This model seems to be roughly in the middle of
    predictions made in the literature.

    References
    ----------
    .. [1] https://arxiv.org/abs/2007.08994
    .. [2] https://arxiv.org/abs/1607.05383
    .. [3] https://arxiv.org/abs/2007.09008
    .. [4] https://arxiv.org/abs/2007.09012
    .. [5] https://arxiv.org/abs/1801.03062
    .. [6] https://arxiv.org/abs/1812.05760
    .. [7] https://arxiv.org/abs/1906.07032
    .. [8] https://arxiv.org/abs/1910.05095
    """

    _models = {
        "HI": (1.0, [1.930, -1.479, 0.814]),
        "LRG": (0.70, [3.642, 0.019, -0.194]),
        "ELG": (0.85, [2.787, -0.774, 0.083]),
        "QSO": (1.48, [1.119, -0.138, -0.058]),
        "LRGalt": (0.70, [3.642, -0.469, -0.183]),
        "ELGalt": (0.85, [2.787, -0.780, 0.078]),
        "QSOalt": (1.48, [1.119, -0.007, -0.117]),
    }


def mean_21cm_temperature(
    c: Cosmology, z: FloatArrayLike, omega_HI: FloatArrayLike
) -> FloatArrayLike:
    """Mean 21cm brightness temperature at a given redshift.

    Temperature is in K.

    Parameters
    ----------
    c
        The Cosmology to use.
    z
        Redshift to calculate at.
    omega_HI
        The HI fraction at each redshift.

    Returns
    -------
    T_b
        Brightness temperature at each redshift.
    """
    # The prefactor T0 below is independent of cosmology (consisting only of fundamental
    # constants and the A_10 Einstein coefficient).
    #
    # This constant correspond to the 0.39 mK prefactor used in the GBT analyses and the
    # former default in cora:
    # T0 = 181.9e-3
    # however, the previous method incorporated a factor of `h` into itself, which
    # meant the cosmology was applied inconsistently.
    #
    # We now use the value below, which is calculated using recent values of A_10 from
    # http://articles.adsabs.harvard.edu/pdf/1994ApJ...423..522G
    T0 = 191.06e-3

    h = c.H0 / 100.0

    T_b = T0 * (c.H(0) / c.H(z)) * (1 + z) ** 2 * h * omega_HI
    return T_b


def log_M_HI_g_to_n_eff(
    log_M_HI_g: float,
    c: Cosmology,
    z: FloatArrayLike,
    model: Optional[str] = None,
) -> FloatArrayLike:
    """Calculate the effective number density for correlated shot noise.

    Parameters
    ----------
    log_M_HI_g
        The average HI mass per tracer galaxy. Specified as the log-10 of the mass in
        solar mass units.
    c
        A cosmology object.
    z
        The redshift to calculate at.
    model
        The model for omega_HI.

    Returns
    -------
    n_eff
        The effective number density in (Mpc/h)^-3.
    """
    h = c.H0 / 100
    H0_SI = c.H(0)  # H_0 in s^-1
    omHI = omega_HI.evaluate(z, model=model)
    M_HI_g = (10**log_M_HI_g) * units.solar_mass  # in kg

    # First compute M_HI / rho_HI(z) in SI units (m^3)
    n_eff = (3.0 * omHI * H0_SI**2) / (8 * np.pi * units.G * M_HI_g)
    # Then convert to h^-3 Mpc^3
    n_eff *= units.mega_parsec**3 / h**3

    return n_eff
