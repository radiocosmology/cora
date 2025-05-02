"""Distributed containers for simulation pipelines.

Containers
==========
- :py:class:
"""

from typing import ClassVar
import numpy as np

from caput.containers import ContainerBase, FreqContainer
from ..util.cosmology import Cosmology


class CosmologyContainer(ContainerBase):
    """A baseclass for a container that is referenced to a background Cosmology.

    Parameters
    ----------
    cosmology
        An explicit cosmology instance or dict representation. If not set, the cosmology
        *must* get set via `attrs_from`.
    """

    def __init__(self, cosmology: Cosmology | dict | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cosmo_dict = self._resolve_args(cosmology, **kwargs)
        self.attrs["cosmology"] = cosmo_dict

    @staticmethod
    def _resolve_args(
        cosmology: Cosmology | dict | None = None,
        attrs_from: ContainerBase | None = None,
        **kwargs,
    ):
        """Try and extract a Cosmology dict representation from the parameters.

        Useful as subclasses sometimes need access *before* the full class is setup.
        """
        # Insert the Cosmological parameters
        if cosmology is None:
            if attrs_from is not None and "cosmology" in attrs_from.attrs:
                cosmology = attrs_from.attrs["cosmology"]
            else:
                raise ValueError("A cosmology must be supplied.")
        elif not isinstance(cosmology, (Cosmology | dict)):
            raise TypeError("cosmology argument must be a Cosmology instance.")

        if isinstance(cosmology, Cosmology):
            cosmology = cosmology.to_dict()

        return cosmology

    _cosmology_instance = None

    @property
    def cosmology(self):
        """The background cosmology."""
        if self._cosmology_instance is None:
            self._cosmology_instance = Cosmology(**self.attrs["cosmology"])

        return self._cosmology_instance


class HealpixContainer(ContainerBase):
    """Base class container for holding Healpix map data.

    Parameters
    ----------
    nside : int
        The nside of the Healpix maps.
    """

    _axes = ("pixel",)

    def __init__(self, nside=None, *args, **kwargs):
        # Set up axes from passed arguments
        if nside is not None:
            kwargs["pixel"] = 12 * nside**2

        super().__init__(*args, **kwargs)

    @property
    def nside(self):
        """Get the nside of the map."""
        return int((len(self.index_map["pixel"]) // 12) ** 0.5)


class Map(FreqContainer, HealpixContainer):
    """Container for holding multi-frequency sky maps.

    The maps are packed in format `[freq, pol, pixel]` where the polarisations
    are Stokes I, Q, U and V, and the pixel dimension stores a Healpix map.

    Parameters
    ----------
    nside : int
        The nside of the Healpix maps.
    polarisation : bool, optional
        If `True` all Stokes parameters are stored, if `False` only Stokes I is
        stored.
    """

    _axes = ("pol",)

    _dataset_spec: ClassVar = {
        "map": {
            "axes": ["freq", "pol", "pixel"],
            "dtype": np.float64,
            "initialise": True,
            "distributed": True,
            "distributed_axis": "freq",
        }
    }

    def __init__(self, polarisation=True, *args, **kwargs):
        # Set up axes from passed arguments
        kwargs["pol"] = (
            np.array(["I", "Q", "U", "V"]) if polarisation else np.array(["I"])
        )

        super().__init__(*args, **kwargs)

    @property
    def map(self):
        """Get the map dataset."""
        return self.datasets["map"]
