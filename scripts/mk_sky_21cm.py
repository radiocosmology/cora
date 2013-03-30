import argparse

import h5py

from simulations import corr21cm


# Read in arguments.
parser = argparse.ArgumentParser(description="Generate a simulated 21cm sky.")
parser.add_argument('nside', help='Resolution of the map (given as a Healpix NSIDE)', type=int)
parser.add_argument('freq_lower', help='Lowest frequency channel.', type=float)
parser.add_argument('freq_upper', help='Highest frequency channel.', type=float)
parser.add_argument('nfreq', help='Number of frequency channels.', type=int)
parser.add_argument('mapname', help='Name of the file to save into.')
parser.add_argument('--pol', help='Polarised or not.', action='store_true')
args = parser.parse_args()

# Read in arguments.
cr = corr21cm.Corr21cm()
cr.nside = args.nside
cr.nu_lower = args.freq_lower
cr.nu_upper = args.freq_upper
cr.nu_num = args.nfreq

# Generate signal realisation and save.
sg_map = cr.getpolsky() if args.pol else cr.getsky()
f = h5py.File(args.mapname, 'w')
f.create_dataset("map", data=sg_map)
f.close()
