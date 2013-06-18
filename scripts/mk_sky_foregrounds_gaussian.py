import argparse

import numpy as np

import h5py

from cosmoutils import hputil

from simulations import galaxy, skysim

# Read in arguments.
parser = argparse.ArgumentParser(description="Generate a simulated foreground sky (galaxy + point sources).")
parser.add_argument('nside', help='Resolution of the map (given as a Healpix NSIDE)', type=int)
parser.add_argument('freq_lower', help='Lowest frequency channel.', type=float)
parser.add_argument('freq_upper', help='Highest frequency channel.', type=float)
parser.add_argument('nfreq', help='Number of frequency channels.', type=int)
parser.add_argument('mapname', help='Name of the file to save into.')
#parser.add_argument('--pol', help='Polarised or not.', action='store_true')

args = parser.parse_args()


fsyn = galaxy.FullSkySynchrotron()
fpol = galaxy.FullSkyPolarisedSynchrotron()    

nfreq = args.nfreq
df = (args.freq_upper - args.freq_lower) / nfreq
frequencies = np.linspace(args.freq_lower, args.freq_upper, nfreq) + 0.5*df

lmax = 3 * args.nside
npol = 4

cv_fg = np.zeros((lmax+1, npol, nfreq, npol, nfreq))

cv_fg[:, 0, :, 0, :] = skysim.clarray(fsyn.angular_powerspectrum, lmax, frequencies)
cv_fg[:, 1, :, 1, :] = skysim.clarray(fpol.angular_powerspectrum, lmax, frequencies)
cv_fg[:, 2, :, 2, :] = skysim.clarray(fpol.angular_powerspectrum, lmax, frequencies)

cv_fg = cv_fg.reshape(lmax+1, npol*nfreq, npol*nfreq)

alms = skysim.mkfullsky(cv_fg, args.nside, alms=True).reshape(npol, nfreq, lmax+1, lmax+1)
alms = alms.transpose((1, 0, 2, 3))

maps = hputil.sphtrans_inv_sky(alms, args.nside)

# Save map
f = h5py.File(args.mapname, 'w')
f.create_dataset("map", data=maps)
f.close()
