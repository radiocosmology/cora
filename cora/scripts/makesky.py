"""Command line script for making sky maps.
"""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import os

import click

import numpy as np


class ListOfType(click.ParamType):
    """Click option type that accepts a list of objects of a given type.

    Must be a type accepted by a Python literal eval.

    Parameters
    ----------
    name : str
        Name of click type.
    type_ : type object
        Type to accept.
    """

    def __init__(self, name, type_):
        self.name = name
        self.type = type_

    def convert(self, value, param, ctx):
        import ast

        try:
            l = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            self.fail('Could not parse "%s" into list.' % value)

        if not isinstance(l, list):
            self.fail('Could not parse "%s" into list.' % value)

        if not all([isinstance(x, self.type) for x in l]):
            self.fail('Not all values were of type "%s"' % repr(self.type))

        return l


class FreqState(object):
    """Process and store the frequency spec from the command line."""

    def __init__(self):

        # Set the CHIME band as the internal default
        self.freq = (800.0, 400.0, 1025)

        self.channel_range = None
        self.channel_list = None
        self.channel_bin = 1
        self.freq_mode = "centre"

    @property
    def frequencies(self):
        """The frequency centres in MHz."""
        return self._calculate()[0]

    @property
    def freq_width(self):
        """The frequency width in MHz."""
        return self._calculate()[1]

    def _calculate(self):
        """Calculate the frequencies from the parameters."""
        # Generate the set of frequency channels given the parameters

        sf, ef, nf = self.freq
        if self.freq_mode == "centre":
            df = abs(ef - sf) / nf
            frequencies = np.linspace(sf, ef, nf, endpoint=False)
        elif self.freq_mode == "centre_nyquist":
            df = abs((ef - sf) / (nf - 1))
            frequencies = np.linspace(sf, ef, nf, endpoint=True)
        else:
            df = (ef - sf) / nf
            frequencies = sf + df * (np.arange(nf) + 0.5)

        # Rebin frequencies if needed
        if self.channel_bin > 1:
            frequencies = frequencies.reshape(-1, self.channel_bin).mean(axis=1)
            df = df * self.channel_bin

        # Select a subset of channels if required
        if self.channel_list is not None:
            frequencies = frequencies[self.channel_list]
        elif self.channel_range is not None:
            frequencies = frequencies[self.channel_range[0] : self.channel_range[1]]

        return frequencies, df

    @classmethod
    def _set_attr(cls, ctx, param, value):
        state = ctx.ensure_object(cls)
        setattr(state, param.name, value)
        return value

    @classmethod
    def options(cls, f):

        FREQ = ListOfType("frequency list", int)

        options = [
            click.option(
                "--freq",
                help=(
                    "Define the frequency channelisation, give the start and stop "
                    "frequencies (in MHz) and the effective number of channels. "
                    "Default is for CHIME: FSTART=800.0, FSTOP=400.0, FNUM=1025"
                ),
                metavar="FSTART FSTOP FNUM",
                type=(float, float, int),
                default=(800.0, 400.0, 1024),
                expose_value=False,
                callback=cls._set_attr,
            ),
            click.option(
                "--channel-range",
                help="Select a range of frequency channels. Overriden by channel range.",
                type=(int, int),
                metavar="CSTART CSTOP",
                default=(None, None),
                expose_value=False,
                callback=cls._set_attr,
            ),
            click.option(
                "--channel-list",
                help="Select a list of frequency channels. Takes priority over channel range.",
                type=FREQ,
                metavar="CHANNEL LIST",
                default=None,
                expose_value=False,
                callback=cls._set_attr,
            ),
            click.option(
                "--channel-bin",
                help="If set, average over BIN channels. The binning is done before channel selection.",
                metavar="BIN",
                type=int,
                default=1,
                expose_value=False,
                callback=cls._set_attr,
            ),
            click.option(
                "--freq-mode",
                type=click.Choice(["centre", "centre_nyquist", "edge"]),
                default="centre",
                help=(
                    'Choose if FSTART and FSTOP are the edges of the band ("edge"), '
                    "or whether they are the central frequencies of the first and "
                    "last channel, in this case the last (nyquist) frequency can "
                    'either be skipped ("centre", default) or included '
                    '("centre_nyquist"). The behaviour of the "centre" mode '
                    "matches the output of the CASPER PFB-FIR block."
                ),
                expose_value=False,
                callback=cls._set_attr,
            ),
        ]

        handle = click.make_pass_decorator(cls, ensure=True)(f)

        for option in options:
            handle = option(handle)

        return handle


def map_options(f):
    """The set of options for generating a map."""
    options = [
        click.option(
            "--nside",
            help="Set the map resolution (default: 256)",
            metavar="NSIDE",
            default=256,
        ),
        click.option(
            "--pol",
            type=click.Choice(["full", "zero", "none"]),
            default="full",
            help="Pick polarisation mode. Full output, zero polarisation, or only return Stokes I (default: full).",
        ),
        click.option(
            "--filename",
            help="Output file [default=map.h5]",
            metavar="FILENAME",
            default="map.h5",
        ),
    ]

    handle = FreqState.options(f)

    for option in options:
        handle = option(handle)

    return handle


@click.group()
def cli():
    """Generate a map of the low frequency radio sky.

    To produce frequency bins compatible with the output of a CASPER based
    channelisation, for example in CHIME, set --freq-mode=centre and set the
    --freq parameter to the start and (missing) end frequencies, and then add
    one to the number of channels (for the missing Nyquist frequency).
    """
    pass


@cli.command()
@map_options
@click.option(
    "--maxflux",
    default=1e6,
    type=float,
    help="Maximum flux of point included point source (in Jy). Default is 1 MJy.",
)
def foreground(fstate, nside, pol, filename, maxflux):
    """Generate a full foreground sky map.

    The requested map must have more than two frequencies for this type.
    """

    if fstate.frequencies.shape[0] < 2:
        print("Number of frequencies must be more than two.")
        return

    from cora.foreground import galaxy, pointsource

    # Read in arguments.
    gal = galaxy.ConstrainedGalaxy()
    gal.nside = nside
    gal.frequencies = fstate.frequencies

    # Fetch galactic sky
    cs = gal.getpolsky() if pol == "full" else gal.getsky()

    # Fetch point source maps
    ps = pointsource.CombinedPointSources.like_map(gal)
    ps.flux_max = maxflux

    cs = cs + (ps.getpolsky() if pol == "full" else ps.getsky())

    # Save map
    write_map(filename, cs, gal.frequencies, fstate.freq_width, pol != "none")


@cli.command()
@map_options
@click.option("--spectral-index", default="md", type=click.Choice(["md", "gsm", "gd"]))
def galaxy(fstate, nside, pol, filename, spectral_index):
    """Generate a Milky way only foreground map.

    Use Haslam (extrapolated with a spatially varying spectral index) as a base,
    and then generate random spatial, spectral and polarisation fluctuations for
    unconstrained modes.

    The requested map must have more than two frequencies for this type.
    """

    if fstate.frequencies.shape[0] < 2:
        print("Number of frequencies must be more than two.")
        return

    from cora.foreground import galaxy

    # Read in arguments.
    gal = galaxy.ConstrainedGalaxy()
    gal.nside = nside
    gal.frequencies = fstate.frequencies
    gal.spectral_map = spectral_index

    # Fetch galactic sky
    cs = gal.getpolsky() if pol == "full" else gal.getsky()

    # Save map
    write_map(filename, cs, gal.frequencies, fstate.freq_width, pol != "none")


@cli.command()
@map_options
@click.option(
    "--maxflux",
    default=1e6,
    type=float,
    help="Maximum flux of point included point source (in Jy). Default is 1 MJy.",
)
def pointsource(fstate, nside, pol, filename, maxflux):
    """Generate a point source only foreground map.

    For S > 4 Jy (at 600 MHz) use real point sources, for dimmer sources (but S >
    0.1 Jy at 151 MHz) generate a synthetic population, for even dimmer sources
    use a Gaussian realisation of unresolved sources.
    """

    from cora.foreground import pointsource

    # Fetch point source maps
    ps = pointsource.CombinedPointSources()
    ps.nside = nside
    ps.frequencies = fstate.frequencies
    ps.flux_max = maxflux

    cs = ps.getpolsky() if pol == "full" else ps.getsky()

    # Save map
    write_map(filename, cs, ps.frequencies, fstate.freq_width, pol != "none")


@cli.command("21cm")
@map_options
@click.option(
    "--eor",
    is_flag=True,
    help="Use parameters more suitable for reionisation epoch (rather than intensity mapping).",
)
@click.option(
    "--oversample",
    type=int,
    help="Oversample in redshift by 2**oversample_z + 1 to approximate finite width bins.",
)
@click.option(
    "--za",
    is_flag=True,
    help="Use the Zeldovich approximation (instead of a simple Gaussian field). Ignored if EoR.",
)
@click.option(
    "--seed",
    type=int,
    help='Set the seed for the matter realization random generator.'
)
@click.option(
    "--bias",
    type=float,
    default = 1.0,
    help="If a number > 0 apply this constant bias (default is 1). If 0 is given, use the halo model to compute a redshift dependent bias."
)
def _21cm(fstate, nside, pol, filename, eor, oversample, za, seed, bias):
    """Generate a Gaussian simulation of the unresolved 21cm background.
    """

    from cora.signal import corr21cm
    from cora.util import nputil

    # Read in arguments.
    if eor:
        cr = corr21cm.EoR21cm()
    else:
        if za:
            cr = corr21cm.Corr21cmZA(bias=bias)
        else:
            cr = corr21cm.Corr21cm(bias=bias)

    cr.nside = nside
    cr.frequencies = fstate.frequencies
    cr.oversample = oversample if oversample is not None else 3

    if ((pol == "full") and za):
        msg = "Option pol = 'full' is not implemented for ZA case. Please use pol = ['zero', 'none']"
        raise NotImplementedError(msg)

    # If seed is given, re-seed the numpy random generator.
    if seed is not None:
        np.random.seed(seed)

    # Generate signal realisation and save.
    sg_map = cr.getpolsky() if pol == "full" else cr.getsky()

    # Save map
    write_map(filename, sg_map, cr.frequencies, fstate.freq_width, pol != "none")


@cli.command("tracer")
@map_options
@click.option(
    "--oversample",
    type=int,
    help="Oversample in redshift by 2**oversample_z + 1 to approximate finite width bins.",
)
@click.option(
    "--za",
    is_flag=True,
    help="Use the Zeldovich approximation (instead of a simple Gaussian field). Ignored if EoR.",
)
@click.option(
    "--seed",
    type=int,
    help='Set the seed for the matter realization random generator.'
)
@click.option(
    "--ttype",
    type=str,
    default='none',
    help="The type of tracer to use. For now ony accepts 'qso' or 'none'"
)
def tracer(fstate, nside, pol, filename, oversample, za, seed, ttype):
    """Generate an unresolved Gaussian simulation of the distribution of the chosen tracer.
       For now it only does QSOs.
    """

    from cora.signal import corr21cm
    from cora.util import nputil

    if za:
        cr = corr21cm.CorrBiasedTracerZA(tracer_type=ttype)
    else:
        cr = corr21cm.CorrBiasedTracer(tracer_type=ttype)

    cr.nside = nside
    cr.frequencies = fstate.frequencies
    cr.oversample = oversample if oversample is not None else 3

    if ((pol == "full") and za):
        msg = "Option pol = 'full' is not implemented for ZA case. Please use pol = ['zero', 'none']"
        raise NotImplementedError(msg)

    # If seed is given, re-seed the numpy random generator.
    if seed is not None:
        np.random.seed(seed)

    # Generate signal realisation and save.
    sg_map = cr.getpolsky() if pol == "full" else cr.getsky()

    # Save map
    write_map(filename, sg_map, cr.frequencies, fstate.freq_width, pol != "none")


@cli.command()
@map_options
def gaussianfg(fstate, nside, pol, filename):
    """Generate a full-sky Gaussian random field for synchrotron emission.
    """

    import numpy as np

    from cora.core import skysim
    from cora.util import hputil
    from cora.foreground import galaxy

    fsyn = galaxy.FullSkySynchrotron()
    fpol = galaxy.FullSkyPolarisedSynchrotron()

    # Set frequency parameters
    fsyn.frequencies = fstate.frequencies
    nfreq = len(fsyn.frequencies)

    lmax = 3 * nside
    npol = 4 if pol == "full" else 1

    cv_fg = np.zeros((lmax + 1, npol, nfreq, npol, nfreq))

    cv_fg[:, 0, :, 0, :] = skysim.clarray(
        fsyn.angular_powerspectrum, lmax, fsyn.nu_pixels
    )

    if pol == "full":
        cv_fg[:, 1, :, 1, :] = skysim.clarray(
            fpol.angular_powerspectrum, lmax, fsyn.nu_pixels
        )
        cv_fg[:, 2, :, 2, :] = skysim.clarray(
            fpol.angular_powerspectrum, lmax, fsyn.nu_pixels
        )

    cv_fg = cv_fg.reshape(lmax + 1, npol * nfreq, npol * nfreq)

    alms = skysim.mkfullsky(cv_fg, nside, alms=True).reshape(
        npol, nfreq, lmax + 1, lmax + 1
    )
    alms = alms.transpose((1, 0, 2, 3))

    maps = hputil.sphtrans_inv_sky(alms, nside)
    write_map(filename, maps, fsyn.frequencies, fstate.freq_width, pol != "none")


@cli.command()
@map_options
@click.option("--ra", type=float, help="RA (in degrees) for source to add.", default=0)
@click.option("--dec", type=float, help="DEC (in degrees) of source to add.", default=0)
def singlesource(fstate, nside, pol, filename, ra, dec):
    """Generate a test map with a single source (amplitude I=1) at the given position.
    """
    import healpy

    nfreq = len(fstate.frequencies)
    npol = 4 if pol == "full" else 1

    map_ = np.zeros((nfreq, npol, 12 * nside ** 2), dtype=np.float64)

    map_[:, 0, healpy.ang2pix(nside, ra, dec, lonlat=True)] = 1.0

    # Save map
    write_map(filename, map_, fstate.frequencies, fstate.freq_width, pol != "none")


def write_map(filename, data, freq, fwidth=None, include_pol=True):
    # Write out the map into an HDF5 file.

    import h5py
    import numpy as np
    from future.utils import text_type

    # Make into 3D array
    if data.ndim == 3:
        polmap = np.array(["I", "Q", "U", "V"])
    else:
        if include_pol:
            data2 = np.zeros((data.shape[0], 4, data.shape[1]), dtype=data.dtype)
            data2[:, 0] = data
            data = data2
            polmap = np.array(["I", "Q", "U", "V"])
        else:
            data = data[:, np.newaxis, :]
            polmap = np.array(["I"])

    # Construct frequency index map
    freqmap = np.zeros(len(freq), dtype=[("centre", np.float64), ("width", np.float64)])
    freqmap["centre"][:] = freq
    freqmap["width"][:] = fwidth if fwidth is not None else np.abs(np.diff(freq)[0])

    # Open up file for writing
    with h5py.File(filename, "w") as f:
        f.attrs["__memh5_distributed_file"] = True

        dset = f.create_dataset("map", data=data)
        dt = h5py.special_dtype(vlen=text_type)
        dset.attrs["axis"] = np.array(["freq", "pol", "pixel"]).astype(dt)
        dset.attrs["__memh5_distributed_dset"] = True

        dset = f.create_dataset("index_map/freq", data=freqmap)
        dset.attrs["__memh5_distributed_dset"] = False
        dset = f.create_dataset("index_map/pol", data=polmap.astype(dt))
        dset.attrs["__memh5_distributed_dset"] = False
        dset = f.create_dataset("index_map/pixel", data=np.arange(data.shape[2]))
        dset.attrs["__memh5_distributed_dset"] = False
