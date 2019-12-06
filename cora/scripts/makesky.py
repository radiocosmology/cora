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

@click.group()
@click.option('--nside', help='Set the map resolution (default: 256)', metavar='NSIDE', default=256)
@click.option('--freq', help='Define the frequency channels (in MHz). Default is for CHIME: FSTART=800.0, FSTOP=400.0, FNUM=1025',
              metavar='FSTART FSTOP FNUM', type=(float, float, int), default=(800.0, 400.0, 1025))
@click.option('--channels', help='Select a range of frequency channels',
              type=(int, int), metavar='CSTART CSTOP', default=(None, None))
@click.option('--channel-bin', help='If set, average over BIN channels', metavar='BIN', type=int, default=1)
@click.option('--freq-mode', type=click.Choice(['centre', 'edge']), default='centre',
              help=('Choose whether FSTART and FSTOP are the very edges of the band, '+
                    'or whether they are the centre frequencies (default: centre).'))
@click.option('--pol', type=click.Choice(['full', 'zero', 'none']), default='full',
              help='Pick polarisation mode. Full output, zero polarisation, or only return Stokes I (default: full).')
@click.option('--filename', help='Output file [default=map.h5]', metavar='FILENAME', default='map.h5')
@click.pass_context
def cli(ctx, nside, freq, channels, channel_bin, freq_mode, pol, filename):
    """Generate a map of the low frequency radio sky.

    To produce frequency bins compatible with the output of a CASPER based
    channelisation, for example in CHIME, set --freq-mode=centre and set the
    --freq parameter to the start and (missing) end frequencies, and then add
    one to the number of channels (for the missing Nyquist frequency).
    """

    # Generate the set of frequency channels given the parameters
    if freq_mode == 'centre':
        df = abs((freq[1] - freq[0]) / (freq[2] - 1))
        frequencies = np.linspace(freq[0], freq[1], freq[2], endpoint=True)
    else:
        df = (freq[1] - freq[0]) / freq[2]
        frequencies = freq[0] + df * (np.arange(freq[2]) + 0.5)

    # Select a subset of channels if required
    if channels != (None, None):
        frequencies = frequencies[channels[0]:channels[1]]

    # Rebin frequencies if needed
    if channel_bin > 1:
        frequencies = frequencies.reshape(-1, channel_bin).mean(axis=1)
        df = df * channel_bin

    class t(object):
        pass

    ctx.obj = t()

    ctx.obj.freq = frequencies
    ctx.obj.freq_width = np.abs(df)

    ctx.obj.nside = nside
    ctx.obj.full_pol = (pol == 'full')
    ctx.obj.include_pol = (pol != 'none')
    ctx.obj.filename = filename

    if os.path.exists(filename):
        raise click.ClickException("Output file %s already exists" % filename)


@cli.command()
@click.option('--maxflux', default=1e6, type=float, help='Maximum flux of point included point source (in Jy). Default is 1 MJy.')
@click.pass_context
def foreground(ctx, maxflux):
    """Generate a full foreground sky map."""

    from cora.foreground import galaxy, pointsource

    # Read in arguments.
    gal = galaxy.ConstrainedGalaxy()
    gal.nside = ctx.obj.nside
    gal.frequencies = ctx.obj.freq

    # Fetch galactic sky
    cs = gal.getpolsky() if ctx.obj.full_pol else gal.getsky()

    # Fetch point source maps
    ps = pointsource.CombinedPointSources.like_map(gal)
    ps.flux_max = maxflux

    cs = cs + (ps.getpolsky() if ctx.obj.full_pol else ps.getsky())

    # Save map
    write_map(ctx.obj.filename, cs, gal.frequencies, ctx.obj.freq_width, ctx.obj.include_pol)


@cli.command()
@click.option('--spectral-index', default='md', type=click.Choice(['md', 'gsm', 'gd']))
@click.pass_context
def galaxy(ctx, spectral_index):
    """Generate a Milky way only foreground map.

    Use Haslam (extrapolated with a spatially varying spectral index) as a base,
    and then generate random spatial, spectral and polarisation fluctuations for
    unconstrained modes.
    """

    from cora.foreground import galaxy

    # Read in arguments.
    gal = galaxy.ConstrainedGalaxy()
    gal.nside = ctx.obj.nside
    gal.frequencies = ctx.obj.freq
    gal.spectral_map = spectral_index

    # Fetch galactic sky
    cs = gal.getpolsky() if ctx.obj.full_pol else gal.getsky()

    # Save map
    write_map(ctx.obj.filename, cs, gal.frequencies, ctx.obj.freq_width, ctx.obj.include_pol)


@cli.command()
@click.option('--maxflux', default=1e6, type=float, help='Maximum flux of point included point source (in Jy). Default is 1 MJy.')
@click.pass_context
def pointsource(ctx, maxflux):
    """Generate a point source only foreground map.

    For S > 4 Jy (at 600 MHz) use real point sources, for dimmer sources (but S >
    0.1 Jy at 151 MHz) generate a synthetic population, for even dimmer sources
    use a Gaussian realisation of unresolved sources.
    """

    from cora.foreground import pointsource

    # Fetch point source maps
    ps = pointsource.CombinedPointSources()
    ps.nside = ctx.obj.nside
    ps.frequencies = ctx.obj.freq
    ps.flux_max = maxflux

    cs = ps.getpolsky() if ctx.obj.full_pol else ps.getsky()

    # Save map
    write_map(ctx.obj.filename, cs, ps.frequencies, ctx.obj.freq_width, ctx.obj.include_pol)


@cli.command('21cm')
@click.option('--eor', is_flag=True, help='Use parameters more suitable for reionisation epoch (rather than intensity mapping).')
@click.option('--oversample', type=int, help='Oversample in redshift by 2**oversample_z + 1 to capute finite width bins.')
@click.pass_context
def _21cm(ctx, eor, oversample):
    """Generate a Gaussian simulation of the unresolved 21cm background.
    """

    from cora.signal import corr21cm

    # Read in arguments.
    if eor:
        cr = corr21cm.EoR21cm()
    else:
        cr = corr21cm.Corr21cm()

    cr.nside = ctx.obj.nside
    cr.frequencies = ctx.obj.freq
    cr.oversample = oversample if oversample is not None else 3

    # Generate signal realisation and save.
    sg_map = cr.getpolsky() if ctx.obj.full_pol else cr.getsky()

    # Save map
    write_map(ctx.obj.filename, sg_map, cr.frequencies, ctx.obj.freq_width, ctx.obj.include_pol)


# TODO: Need to implement the options to pass nside_factor and ndiv_radial
@cli.command('21cm_qso_za')
@click.option('--oversample', type=int, help='Oversample in redshift by 2**oversample_z + 1 to capute finite width bins.')
@click.pass_context
def _21cm_qso_za(ctx, oversample):
    """Generate a simulation of the unresolved 21cm background
    and the quasars overdensities using the Zeldovich approximation.
    """

    from cora.signal import corr21cm
    from datetime import datetime
    from cora.util import nputil

    t1 = datetime.now()

    cr = corr21cm.Corr21cmZA()
    crq = corr21cm.CorrQuasarZA()

    cr.nside = ctx.obj.nside
    cr.frequencies = ctx.obj.freq
    cr.oversample = oversample if oversample is not None else 0
    crq.nside = ctx.obj.nside
    crq.frequencies = ctx.obj.freq
    crq.oversample = oversample if oversample is not None else 0

    lmax = 3 * cr.nside - 1
    gaussvars_list = [nputil.complex_std_normal((len(cr.freqs_full), l + 1))
                      for l in range(lmax+1)]

    # Generate signal realization and save
    sg_map = cr.getsky(gaussvars_list=gaussvars_list)
    qs_map = crq.getsky(gaussvars_list=gaussvars_list)

    if ctx.obj.full_pol:
        print("ZA code does not support polarizations. Writting zeros for Stokes Q, U, V")

    # Save map
    write_map('21cm_'+ctx.obj.filename, sg_map, cr.frequencies, ctx.obj.freq_width, ctx.obj.include_pol)
    write_map('qso_'+ctx.obj.filename, qs_map, cr.frequencies, ctx.obj.freq_width, ctx.obj.include_pol)

    t2 = datetime.now()
    with open('runtime.dat','w') as fl:
        fl.write('Total runtime: {0}'.format(t2-t1))


# TODO: Need to implement the options to pass nside_factor and ndiv_radial
@cli.command('21cm_za')
@click.option('--oversample', type=int, help='Oversample in redshift by 2**oversample_z + 1 to capute finite width bins.')
@click.pass_context
def _21cm_za(ctx, oversample):
    """Generate a simulation of the unresolved 21cm background
    using the Zeldovich approximation.
    """

    from cora.signal import corr21cm
    from datetime import datetime

    t1 = datetime.now()

    cr = corr21cm.Corr21cmZA()

    cr.nside = ctx.obj.nside
    cr.frequencies = ctx.obj.freq
    cr.oversample = oversample if oversample is not None else 0

    # Generate signal realization and save
    sg_map = cr.getsky()

    if ctx.obj.full_pol:
        print("ZA code does not support polarizations. Writting zeros for Stokes Q, U, V")

    # Save map
    write_map(ctx.obj.filename, sg_map, cr.frequencies, ctx.obj.freq_width, ctx.obj.include_pol)

    t2 = datetime.now()
    with open('runtime.dat','w') as fl:
        fl.write('Total runtime: {0}'.format(t2-t1))


@cli.command()
@click.pass_context
def gaussianfg(ctx):
    """Generate a full-sky Gaussian random field for synchrotron emission.
    """

    import numpy as np

    from cora.core import skysim
    from cora.util import hputil
    from cora.foreground import galaxy

    fsyn = galaxy.FullSkySynchrotron()
    fpol = galaxy.FullSkyPolarisedSynchrotron()

    # Set frequency parameters
    fsyn.frequencies = ctx.obj.freq
    nfreq = len(fsyn.frequencies)

    nside = ctx.obj.nside
    lmax = 3 * nside
    npol = 4 if ctx.obj.full_pol else 1

    cv_fg = np.zeros((lmax+1, npol, nfreq, npol, nfreq))

    cv_fg[:, 0, :, 0, :] = skysim.clarray(fsyn.angular_powerspectrum, lmax, fsyn.nu_pixels)

    if ctx.obj.full_pol:
        cv_fg[:, 1, :, 1, :] = skysim.clarray(fpol.angular_powerspectrum, lmax, fsyn.nu_pixels)
        cv_fg[:, 2, :, 2, :] = skysim.clarray(fpol.angular_powerspectrum, lmax, fsyn.nu_pixels)

    cv_fg = cv_fg.reshape(lmax+1, npol*nfreq, npol*nfreq)

    alms = skysim.mkfullsky(cv_fg, nside, alms=True).reshape(npol, nfreq, lmax+1, lmax+1)
    alms = alms.transpose((1, 0, 2, 3))

    maps = hputil.sphtrans_inv_sky(alms, nside)
    write_map(ctx.obj.filename, maps, fsyn.frequencies, ctx.obj.freq_width, ctx.obj.include_pol)


def write_map(filename, data, freq, fwidth=None, include_pol=True):
    # Write out the map into an HDF5 file.

    import h5py
    import numpy as np
    from future.utils import text_type

    # Make into 3D array
    if data.ndim == 3:
        polmap = np.array(['I', 'Q', 'U', 'V'])
    else:
        if include_pol:
            data2 = np.zeros((data.shape[0], 4, data.shape[1]), dtype=data.dtype)
            data2[:, 0] = data
            data = data2
            polmap = np.array(['I', 'Q', 'U', 'V'])
        else:
            data = data[:, np.newaxis, :]
            polmap = np.array(['I'])

    # Construct frequency index map
    freqmap = np.zeros(len(freq), dtype=[('centre', np.float64), ('width', np.float64)])
    freqmap['centre'][:] = freq
    freqmap['width'][:] = fwidth if fwidth is not None else np.abs(np.diff(freq)[0])

    # Open up file for writing
    with h5py.File(filename, "w") as f:
        f.attrs['__memh5_distributed_file'] = True

        dset = f.create_dataset('map', data=data)
        dt = h5py.special_dtype(vlen=text_type)
        dset.attrs['axis'] = np.array(['freq', 'pol', 'pixel']).astype(dt)
        dset.attrs['__memh5_distributed_dset'] = True

        dset = f.create_dataset('index_map/freq', data=freqmap)
        dset.attrs['__memh5_distributed_dset'] = False
        dset = f.create_dataset('index_map/pol', data=polmap.astype(dt))
        dset.attrs['__memh5_distributed_dset'] = False
        dset = f.create_dataset('index_map/pixel', data=np.arange(data.shape[2]))
        dset.attrs['__memh5_distributed_dset'] = False
