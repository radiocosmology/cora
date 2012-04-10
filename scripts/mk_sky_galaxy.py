import sys
import argparse

import numpy as np
import h5py

from simulations import galaxy

# Read in arguments.
parser = argparse.ArgumentParser(description="Generate a simulated galactic synchrotron sky.")
parser.add_argument('nside', help='Resolution of the map (given as a Healpix NSIDE)', type=int)
parser.add_argument('freq_lower', help='Lowest frequency channel.', type=float)
parser.add_argument('freq_upper', help='Highest frequency channel.', type=float)
parser.add_argument('nfreq', help='Number of frequency channels.', type=int)
parser.add_argument('mapname', help='Name of the file to save into.')
args = parser.parse_args()

# Read in arguments.
gal = galaxy.ConstrainedGalaxy()
gal.nside = args.nside
gal.nu_lower = args.freq_lower
gal.nu_upper = args.freq_upper
gal.nu_num = args.nfreq

# Fetch and save sky.
cs = gal.getsky()
f = h5py.File(args.mapname, 'w')
f.create_dataset("map", data=cs)
f.close()
