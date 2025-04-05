from typing import Optional, Callable

import numpy as np
from scipy.interpolate import CubicSpline

from caput import memh5
from cora.util.cosmology import Cosmology
from cora.util import units, cubicspline as cs

from draco.core import containers
from draco.core.containers import CosmologyContainer

from ..util.nputil import FloatArrayLike


# Types of interpolation that can be used
_INTERP_TYPES = ["linear", "log", "sinh", "linear_scipy", "sinh_scipy"]


class InterpolatedFunction(memh5.BasicCont):
    """A container for interpolated 1D functions.

    This is intended to allow saving to disk of functions which are expensive to
    generate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We also want to make this happens when an object is created directly and not
        # just via `from_file`
        self._finish_setup()

    def _finish_setup(self):
        # Create the function cache dict
        self._function_cache = {}

    def get_function(
        self,
        name: str,
        interp_type: str = None,
    ) -> Callable[[FloatArrayLike], FloatArrayLike]:
        """Get the named function.

        Parameters
        ----------
        name
            The name of the function to return.
        interp_type
            Type of interpolation. If supplied, override value stored in container.
            Default: None.

        Returns
        -------
        function
        """

        # If name is in function cache, generate and store interpolator if not
        # already in cache
        if name in self._function_cache:

            if interp_type is None:
                interp_type = self[name].attrs["type"]

            if interp_type not in self._function_cache[name]:
                self._function_cache[name][interp_type] = self._make_interpolator(
                    name, interp_type
                )

        # If name is not in cache, check that name is valid, then generate and
        # store interpolator
        else:

            if name not in self:
                raise ValueError(f"Function {name} unknown.")

            self._function_cache[name] = {}

            if interp_type is None:
                interp_type = self[name].attrs["type"]

            self._function_cache[name][interp_type] = self._make_interpolator(
                name, interp_type
            )

        return self._function_cache[name][interp_type]

    def _make_interpolator(
        self, name: str, interp_type: str
    ) -> Callable[[FloatArrayLike], FloatArrayLike]:

        dset = self[name]

        if len(dset.attrs["axis"]) != 1:
            raise RuntimeError("Can only return a single value.")

        # Get the abscissa
        axis = dset.attrs["axis"][0]
        x = self.index_map[axis]

        # Get the ordinate
        f = dset[:]

        data = np.dstack([x, f])[0]

        if interp_type == "linear":
            # Cubic spline in linear-linear space
            func = cs.Interpolater(data)

        elif interp_type == "log":
            # Cubic spline in log-log space
            func = cs.LogInterpolater(data)

        elif interp_type == "sinh":
            # Cubic spline in sinh-transformed space (effectively log-space
            # for values greater than the threshold and linear-space for values
            # less than the threshold)
            x_t = dset.attrs["x_t"]
            f_t = dset.attrs["f_t"]

            func = cs.SinhInterpolater(data, x_t, f_t)

        elif interp_type == "linear_scipy":
            # Scipy cubic spline in linear-linear space.
            # Use natural boundary conditions to match the behavior
            # of cs.Interpolater
            func = CubicSpline(data[:, 0], data[:, 1], bc_type="natural")

        elif interp_type == "sinh_scipy":
            # Scipy cubic spline in sinh-transformed space
            x_t = dset.attrs["x_t"]
            f_t = dset.attrs["f_t"]

            _spline = CubicSpline(
                np.arcsinh(data[:, 0] / x_t),
                np.arcsinh(data[:, 1] / f_t),
                bc_type="natural",
            )

            def _func(x):
                return f_t * np.sinh(_spline(np.arcsinh(x / x_t)))

            func = _func

        else:
            # Unrecognized interpolation type
            raise RuntimeError(f"Unrecognized interpolation type: {interp_type}")

        return func

    def add_function(
        self, name: str, x: np.ndarray, f: np.ndarray, type: str = "linear", **kwargs
    ):
        """Add a function to the container.

        Parameters
        ----------
        name
            The name of the function to add. This is used to retrieve it later using
            `get_function`.
        x
            The abscissa.
        f
            The ordinate.
        type
            The type of the interpolation. Valid options: "linear", "log", "sinh",
            "linear_scipy", "sinh_scipy". If sinh-type interpolation, kwargs accepts
            additional keys for the `x_t` and `f_t` parameters.
            See `cubicspline.pyx` for details. Default: "linear".
        """

        if name in self:
            raise ValueError(f"Function {name} already exists.")

        xname = f"x_{name}"
        self.create_index_map(xname, x)

        dset = self.create_dataset(name, data=f)
        dset.attrs["axis"] = [xname]
        dset.attrs["type"] = type

        # Copy over any kwargs containing extra info for the interpolation
        for key, val in kwargs.items():
            dset.attrs[key] = val


class FZXContainer(CosmologyContainer):
    """Container with a comoving radial axis.

    This can be specified either directly with a grid in comoving distance, or in
    redshift, or 21 cm line frequency. One of these will be considered as defining
    the primary axis, and implicitly defines the others. `freq` is the highest
    priority, followed by `redshift` and finally the comoving distance `chi`.

    Parameters
    ----------
    freq
        The radial axis given as 21cm line frequencies.
    redshift
        The radial axis given as a redshift.
    chi
        The radial axis given as a comoving distance in Mpc/h.
    """

    # Chi is the only required axis, and must be used for any datasets
    _axes = ("chi",)

    def __init__(
        self,
        freq: Optional[np.ndarray] = None,
        redshift: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):

        # Insert the Cosmological parameters
        cosmology = Cosmology(**CosmologyContainer._resolve_args(**kwargs))

        # If none of the high priority radial axes are set directly, see if any exist
        # in an axes_from object
        if freq is None and redshift is None and "axes_from" in kwargs:
            if "freq" in kwargs["axes_from"].index_map:
                freq = kwargs["axes_from"].index_map["freq"]
            elif "redshift" in kwargs["axes_from"].index_map:
                redshift = kwargs["axes_from"].index_map["redshift"]

        # Go through the high priority axes, and if present generate the lower priority
        # ones
        if freq is not None:
            redshift = units.nu21 / freq - 1.0

        if redshift is not None:
            kwargs["chi"] = cosmology.comoving_distance(redshift)

        super().__init__(*args, **kwargs)

        # Create the additional radial axes (if present) and determine the primary
        # radial axis.
        # NOTE: this must be done *after* the call to `super().__init__(...)` such that
        # the container internals are defined
        radial_axis = "chi"
        if redshift is not None:
            self.create_index_map("redshift", redshift)
            radial_axis = "redshift"
        if freq is not None:
            self.create_index_map("freq", freq)
            radial_axis = "freq"

        # Set the cosmology and radial axis attributes
        self.attrs["primary_radial_axis"] = radial_axis

    @property
    def chi(self):
        """The comoving distance to each radial slice in Mpc / h."""
        return self.index_map["chi"]

    @property
    def redshift(self):
        """The redshift for each radial slice."""
        # TODO: derive this one
        if "redshift" not in self.index_map:
            raise RuntimeError("Container does not have a redshift axis.")
        return self.index_map["redshift"]

    @property
    def freq(self):
        """The 21cm line frequency for each radial slice."""
        # TODO: maybe derive this one
        if "freq" not in self.index_map:
            raise RuntimeError("Container does not have a 21cm frequency axis.")
        return self.index_map["freq"]


class MatterPowerSpectrum(CosmologyContainer, InterpolatedFunction):
    """A container to hold a matter power spectrum.

    This object can evaluate a power spectrum at specified wavenumbers (in h / Mpc
    units) and redshifts.

    Parameters
    ----------
    k
        Wavenumbers the power spectrum samples are at (in h / Mpc).
    ps
        Power spectrum samples.
    ps_redshift
        The redshift the samples are calculated at. Default is z=0.
    """

    def __init__(
        self,
        k: FloatArrayLike,
        ps: FloatArrayLike,
        *args,
        ps_redshift: float = 0.0,
        **kwargs,
    ):

        # Initialise the base classes (which sets the cosmology etc)
        super().__init__(*args, **kwargs)

        # This shouldn't be necessary, but due to a bug in `draco` where ContainerBase
        # does not correctly call its superconstructor we need to do this explicitly
        self._finish_setup()

        # Add the interpolated function
        self.add_function("powerspectrum", k, ps, type="log")
        self.attrs["ps_redshift"] = ps_redshift

    def powerspectrum(
        self, k: FloatArrayLike, z: FloatArrayLike = 0.0
    ) -> FloatArrayLike:
        """Calculate the power spectrum at given wavenumber and redshift.

        Parameters
        ----------
        k
            The wavenumber (in h / Mpc) to get the power spectrum at.
        z : optional
            The redshift to calculate the power spectrum at (default z=0).

        Returns
        -------
        ps
            The power spectrum.
        """
        c = self.cosmology
        Dratio = c.growth_factor(z) / c.growth_factor(self._ps_redshift)
        return self.get_function("powerspectrum")(k) * Dratio**2

    def powerspectrum_at_z(
        self, z: FloatArrayLike
    ) -> Callable[[FloatArrayLike], FloatArrayLike]:
        """Return a function which gives the power spectrum at fixed redshift.

        Parameters
        ----------
        z
            The redshift to fix the power spectrum at.

        Returns
        -------
        psfunc
            A function which calculates the power spectrum at given wavenumbers.
        """

        def _ps(k):
            return self.powerspectrum(k, z)

        return _ps

    @property
    def _ps_redshift(self):
        return self.attrs["ps_redshift"]


class CorrelationFunction(CosmologyContainer, InterpolatedFunction):
    """A container to store correlation functions."""

    # TODO: at the moment this has no special functionality, but should eventually
    # provide specific access to the correlation functions as well as redshift scaling

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # This whole constructor shouldn't be necessary, but due to a bug in `draco`
        # where ContainerBase does not correctly call its superconstructor we need to do
        # this explicitly
        self._finish_setup()


class MultiFrequencyAngularPowerSpectrum(FZXContainer):
    """Container for holding C_ell(chi,chi')."""

    _axes = ("ell",)

    def __init__(
        self,
        lmax: float,
        *args,
        **kwargs,
    ):
        # Set ell axis to span from ell=0 to lmax
        kwargs["ell"] = lmax + 1
        super().__init__(*args, **kwargs)

    _dataset_spec = {
        "Cl_phi_phi": {
            "axes": ["ell", "chi", "chi"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ell",
        },
        "Cl_phi_delta": {
            "axes": ["ell", "chi", "chi"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ell",
        },
        "Cl_delta_delta": {
            "axes": ["ell", "chi", "chi"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "ell",
        },
    }

    @property
    def Cl_phi_phi(self):
        """Phi-phi angular power spectrum."""
        return self.datasets["Cl_phi_phi"]

    @property
    def Cl_phi_delta(self):
        """Phi-delta angular power spectrum."""
        return self.datasets["Cl_phi_delta"]

    @property
    def Cl_delta_delta(self):
        """Delta-delta angular power spectrum."""
        return self.datasets["Cl_delta_delta"]

    @property
    def ell(self):
        """Ell values for stored angular power spectra."""
        return self.index_map["ell"]


class InitialLSS(FZXContainer, containers.HealpixContainer):
    """Container for holding initial LSS fields used for simulation.

    These fields are all implicitly the linear fields at redshift z=0.
    """

    _dataset_spec = {
        "delta": {
            "axes": ["chi", "pixel"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "chi",
        },
        "phi": {
            "axes": ["chi", "pixel"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "chi",
        },
    }

    @property
    def delta(self):
        """The linear density field at the initial redshift."""
        return self.datasets["delta"]

    @property
    def phi(self):
        r"""The potential field at the initial redshift.

        This is not the actual gravitational potential, but the Lagrangian potential
        defined by:

        .. math:: \nabla^2 \phi = - \delta

        This is related to the linear gravitational potential :math:`\phi_G` by:

        .. math:: \phi = \frac{3}{3 \mathcal{H}^2} \phi_G
        """
        return self.datasets["phi"]


class BiasedLSS(FZXContainer, containers.HealpixContainer):
    """A biased large scale structure field.

    Parameters
    ----------
    lightcone : bool, optional
        Is the field on the lightcone. If not set, default to True.
    fixed_redshift : float, optional
        If not on the lightcone what is the fixed redshift.
    *args, **kwargs
        Passed through to the superclasses.
    """

    _dataset_spec = {
        "delta": {
            "axes": ["chi", "pixel"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "chi",
        }
    }

    def __init__(
        self,
        *args,
        lightcone: Optional[bool] = None,
        fixed_redshift: Optional[float] = None,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        # Set lightcone taking into account it might have been set already by an
        # `attrs_from` argument to the super constructor
        if lightcone is not None:
            self.attrs["lightcone"] = lightcone
        elif "lightcone" not in self.attrs:
            self.attrs["lightcone"] = True

        if fixed_redshift is not None:
            self.attrs["fixed_redshift"] = fixed_redshift

    @property
    def lightcone(self) -> bool:
        """Is the field on the lightcone or at fixed redshift."""
        return bool(self.attrs["lightcone"])

    @property
    def fixed_redshift(self) -> Optional[float]:
        """The fixed redshift of the field."""
        if "fixed_redshift" in self.attrs:
            return float(self.attrs["fixed_redshift"])
        return None

    @property
    def delta(self) -> np.ndarray:
        r"""The biased field.

        As standard this is interpreted as a density contrast, i.e. the field is

        .. math:: \delta = (\rho - \bar{\rho}) / \bar{\rho}

        Returns
        -------
        delta
            The biased field as a [redshift, pixel] array.
        """
        return self.datasets["delta"]
