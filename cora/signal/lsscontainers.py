from typing import Optional, Union, Callable

import numpy as np

from caput import memh5
from cora.util.cosmology import Cosmology
from cora.util import units, cubicspline as cs

from draco.core import containers

from ..util.nputil import FloatArrayLike


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

    def get_function(self, name: str) -> Callable[[FloatArrayLike], FloatArrayLike]:
        """Get the named function.

        Parameters
        ----------
        name
            The name of the function to return.

        Returns
        -------
        function
        """

        # Return immediately from cache if available
        if name not in self._function_cache:

            # Check if the underlying data is actually present
            if name not in self:
                raise ValueError(f"Function {name} unknown.")

            dset = self[name]

            if len(dset.attrs["axis"]) != 1:
                raise RuntimeError("Can only return a single value.")

            # Get the abscissa
            axis = dset.attrs["axis"][0]
            x = self.index_map[axis]

            # Get the ordinate
            f = dset[:]

            interpolation_type = dset.attrs["type"]

            data = np.dstack([x, f])[0]

            if interpolation_type == "linear":
                self._function_cache[name] = cs.Interpolater(data)
            elif interpolation_type == "log":
                self._function_cache[name] = cs.LogInterpolater(data)
            elif interpolation_type == "sinh":

                x_t = dset.attrs["x_t"]
                f_t = dset.attrs["f_t"]

                self._function_cache[name] = cs.SinhInterpolater(data, x_t, f_t)
            else:  # Unrecognized interpolation type
                raise RuntimeError(
                    f"Unrecognized interpolation type {interpolation_type}"
                )

        return self._function_cache[name]

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
            The type of the interpolation. Valid options are "linear", "log" or
            "sinh".If the latter kwargs accepts additional keys for the `x_t` and
            `f_t` parameters. See `sinh_interpolate` for details. By default
            use "linear" interpolation.
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


class FZXContainer(containers.ContainerBase):
    """Container with a comoving radial axis.

    This can be specified either directly with a grid in comoving distance, or in
    redshift, or 21 cm line frequency. One of these will be considered as defining
    the primary axis, and implicitly defines the others. `freq` is the highest
    priority, followed by `redshift` and finally the comoving distance `chi`.

    Parameters
    ----------
    cosmology
        A dictionary of params supplied to `Cosmology.init_physical` to create a
        `Cosmology` object.
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
        cosmology: Union[Cosmology, dict, None] = None,
        freq: Optional[np.ndarray] = None,
        redshift: Optional[np.ndarray] = None,
        *args,
        **kwargs,
    ):

        # Insert the Cosmological parameters
        if cosmology is None:
            if "attrs_from" in kwargs and "cosmology" in kwargs["attrs_from"].attrs:
                cosmology = kwargs["attrs_from"].attrs["cosmology"]
            else:
                raise ValueError("A cosmology must be supplied.")
        elif not isinstance(cosmology, (Cosmology, dict)):
            raise TypeError("cosmology argument must be a Cosmology instance.")

        if isinstance(cosmology, dict):
            cosmology = Cosmology(**cosmology)

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
        self.attrs["cosmology"] = cosmology.to_dict()

    _cosmology_instance = None

    @property
    def cosmology(self):
        """The background cosmology."""

        if self._cosmology_instance is None:
            self._cosmology_instance = Cosmology(**self.attrs["cosmology"])

        return self._cosmology_instance

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
