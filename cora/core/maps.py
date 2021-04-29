# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import os
import numpy as np

import healpy as hp

from cora.util import units

from cora.core import skysim

from caput import mpiutil, pipeline, config, mpiarray
from draco.core import containers, task, io


class Map2d(object):
    r"""A 2-d sky map.

    Attributes
    ----------
    x_width, y_width : float
        Angular size along each axis (in degrees).
    x_num, y_num : int
        Number of pixels along each angular axis.
    nside : int
        Resolution of Healpix map (must be power of 2).
    """
    x_width = 5.0
    y_width = 5.0

    x_num = 128
    y_num = 128

    _nside = 128

    @classmethod
    def like_map(cls, mapobj, *args, **kwargs):
        r"""Create a Map2d (or subclassed) object the same shape as a given object."""

        c = cls(*args, **kwargs)
        c.x_width = mapobj.x_width
        c.y_width = mapobj.y_width
        c.x_num = mapobj.x_num
        c.y_num = mapobj.y_num
        c._nside = mapobj._nside

        return c

    def _width_array(self):
        return np.array([self.x_width, self.y_width], dtype=np.float64) * units.degree

    def _num_array(self):
        return np.array([self.x_num, self.y_num], dtype=np.int)

    @property
    def x_pixels(self):
        return (np.arange(self.x_num) + 0.5) * (self.x_width / self.x_num)

    @property
    def y_pixels(self):
        return (np.arange(self.y_num) + 0.5) * (self.y_width / self.y_num)

    @property
    def nside(self):
        """The resolution of the Healpix map."""
        return self._nside

    @nside.setter
    def nside(self, value):
        ns = int(value)
        lns = np.log2(ns)

        if int(lns) != lns or lns < 0:
            raise Exception("Not a valid value of nside.")

        self._nside = ns


class Map3d(Map2d):
    r"""A 3-d sky map.

    Attributes
    ----------
    x_width, y_width : float
        Angular size along each axis (in degrees).
    nu_upper, nu_lower : float
        Range of frequencies (in Mhz).
    x_num, y_num : int
        Number of pixels along each angular axis.
    nu_num : int
        Number of frequency bins.
    nside : int
        Resolution of Healpix map (must be power of 2).
    freq_mode_chime : bool, optional
        If `True` then define frequency band like CHIME.

    Notes
    -----
    The normal frequency mode uses `nu_lower` and `nu_upper` to define the far
    edges of the band, with exactly `nu_num` channels between them. For example
    `nu_lower = 400`, `nu_upper = 404` and `nu_num = 2`, creates two channels
    centred at 401 and 403 MHz. The CHIME frequency mode (which comes from the
    way the PFB is performed) defines relative to the channel centres, with the
    first channel centred on `nu_lower`, and the next channel after the final
    included channel centred at `nu_upper`. For example, with the same settings
    as above, this frequency mode would have channels at 400 and 402 MHz (with
    the not included "next" channel at 404 MHz).
    """

    nu_lower = 500.0
    nu_upper = 900.0

    @classmethod
    def like_map(cls, mapobj, *args, **kwargs):
        r"""Create a Map3d (or subclassed) object the same shape as a given object."""
        # c = Map2d.like_map(cls, mapobj, *args, **kwargs)

        c = cls(*args, **kwargs)
        c.x_width = mapobj.x_width
        c.y_width = mapobj.y_width
        c.x_num = mapobj.x_num
        c.y_num = mapobj.y_num
        c._nside = mapobj._nside

        c.nu_upper = mapobj.nu_upper
        c.nu_lower = mapobj.nu_lower
        c.nu_num = mapobj.nu_num

        c._frequencies = mapobj._frequencies

        return c

    def _width_array(self):
        return np.array(
            [
                self.nu_upper - self.nu_lower,
                self.x_width * units.degree,
                self.y_width * units.degree,
            ],
            dtype=np.float64,
        )

    def _num_array(self):
        return np.array([self.nu_num, self.x_num, self.y_num], dtype=np.int)

    _frequencies = None

    @property
    def nu_num(self):
        return len(self.frequencies)

    _nu_num = 128

    @nu_num.setter
    def nu_num(self, num):
        self._nu_num = num

    @property
    def frequencies(self):
        """List of frequencies in the map."""

        if self._frequencies is not None:
            return self._frequencies
        else:
            return self.nu_lower + (np.arange(self._nu_num) + 0.5) * (
                (self.nu_upper - self.nu_lower) / self._nu_num
            )

    @frequencies.setter
    def frequencies(self, freq):
        """Set the frequencies in the map."""
        self._frequencies = freq

    # Alias for frequencies for supporting old code.
    nu_pixels = frequencies

    @classmethod
    def like_kiyo_map(cls, mapobj, *args, **kwargs):
        r"""Crease a Map3d (or subclassed) object based on one of kiyo's map
        objects
        """
        c = cls(*args, **kwargs)

        freq_axis = mapobj.get_axis("freq")
        ra_axis = mapobj.get_axis("ra")
        dec_axis = mapobj.get_axis("dec")

        ra_fact = np.cos(np.pi * mapobj.info["dec_centre"] / 180.0)
        c.x_width = (max(ra_axis) - min(ra_axis)) * ra_fact
        c.y_width = max(dec_axis) - min(dec_axis)
        (c.x_num, c.y_num) = (len(ra_axis), len(dec_axis))

        c.nu_lower = min(freq_axis) / 1.0e6
        c.nu_upper = max(freq_axis) / 1.0e6
        c.nu_num = len(freq_axis)

        print(
            "Map3D: %dx%d field (%fx%f deg) from nu=%f to nu=%f (%d bins)"
            % (c.x_num, c.y_num, c.x_width, c.y_width, c.nu_lower, c.nu_upper, c.nu_num)
        )

        return c


class Sky3d(Map3d):
    """Base class for generating maps of the spherical sky at multiple frequencies.

    Attributes
    ----------
    oversample : int
        Over sample the angular power spectrum in frequency/redshift and
        integrate over them to capture the effect of finite width frequency
        channels.
    """

    oversample = 3

    def angular_powerspectrum(self, l, nu1, nu2):
        """C_l(nu1, nu2) for the given map."""

        raise Exception("Not implemented in base class.")

    def mean_nu(self, freq):
        return np.zeros_like(freq)

    def getfield(self):
        raise Exception("Not implemented in base class.")

    def getsky(self):
        """Create a map of the unpolarised sky."""

        lmax = 3 * self.nside - 1
        cla = skysim.clarray(
            self.angular_powerspectrum, lmax, self.nu_pixels, zromb=self.oversample
        )

        return self.mean_nu(self.nu_pixels)[:, np.newaxis] + skysim.mkfullsky(
            cla, self.nside
        )

    def getpolsky(self):
        """Create a map of the fully polarised sky (Stokes, I, Q, U and V)."""
        sky_I = self.getsky()

        sky_IQU = np.zeros((sky_I.shape[0], 4, sky_I.shape[1]), dtype=sky_I.dtype)

        sky_IQU[:, 0] = sky_I

        return sky_IQU

    def getalms(self, lmax):

        cla = skysim.clarray(self.angular_powerspectrum, lmax, self.nu_pixels)

        return skysim.mkfullsky(cla, self.nside, alms=True)



class FLASKMapsToMapContainer(task.SingleTask):
    """Convert a set of single-frequency FLASK maps into a Map container.

    Attributes
    ----------
    input_map_prefix : str
        Prefix of filenames for input FLASK maps, including final "-".
    nside : int
        Nside of input maps.
    flask_field_num : int, optional
        FLASK field number for input maps (default: 1).
    use_mean_21cmT : bool, optional
        Multiply map by mean 21cm temperature as a function of z (default: False).
    map_prefactor : float, optional
        Constant prefactor to multiply maps by (default: 1).
    """

    input_map_prefix = config.Property(proptype=str)
    nside = config.Property(proptype=int)

    flask_field_num = config.Property(proptype=int, default=1)
    use_mean_21cmT = config.Property(proptype=int, default=False)
    map_prefactor = config.Property(proptype=float, default=1.)

    done = False

    def setup(self, bt):
        """Setup the simulation.

        Parameters
        ----------
        bt : ProductManager or BeamTransfer
            Beam Transfer maanger.
        """
        self.beamtransfer = io.get_beamtransfer(bt)
        self.telescope = io.get_telescope(bt)

    def process(self) -> containers.Map:
        """Read FLASK maps and assemble into Map container.

        Returns
        -------
        out_map : Map
            Output Map container.
        """

        # Form frequency map in appropriate format to feed into Map container
        freq = self.telescope.frequencies
        n_freq = len(freq)
        freqmap = np.zeros(n_freq, dtype=[("centre", np.float64), ("width", np.float64)])
        freqmap["centre"][:] = freq
        freqmap["width"][:] = np.abs(np.diff(freq)[0])

        # Make new map container and copy delta into Stokes I component
        m = containers.Map(
            freq=freqmap,
            polarisation=True,
            distributed=True,
            pixel=np.arange(hp.nside2npix(self.nside))
        )

        # Get indexing info for this rank
        loff = m.map.local_offset[0]
        lshape = m.map.local_shape[0]
        gshape = m.map.global_shape[0]

        # For each frequency on this rank, read FLASK map and insert into
        # Map container
        for fi_local in range(lshape):
            # print("Rank %d: fi_local, fi_global = %d, %d" % (mpiutil.rank, fi_local, loff+fi_local))

            in_file = (
                self.input_map_prefix
                + "f%dz%d.fits" % (self.flask_field_num, loff + fi_local + 1)
            )
            if not os.path.isfile(in_file):
                raise RuntimeError("File %s not found!" % in_file)

            m.map[fi_local, 0, :] = hp.read_map(in_file)
            self.log.debug(f"Rank %d: Read file %s" % (mpiutil.rank, in_file))
            
        # Multiply map by specific prefactor, if desired
        if self.map_prefactor != 1:
            self.log.info("Multiplying map by %g" % self.map_prefactor)
            m.map[:] *= self.map_prefactor

        # If desired, multiply by Tb(z)
        if self.use_mean_21cmT:
            from cora.signal.corr21cm import Corr21cm
            cr = Corr21cm()

            z_local = u.nu21 / freq[loff : loff+lshape] - 1
            m.map[:, 0] *= cr.T_b(z_local)[:, np.newaxis]

        self.done = True

        return m
