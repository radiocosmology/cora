import argparse

import h5py

from simulations import galaxy, pointsource

# Read in arguments.
parser = argparse.ArgumentParser(description="Generate a simulated foreground sky (galaxy + point sources).")
parser.add_argument('nside', help='Resolution of the map (given as a Healpix NSIDE)', type=int)
parser.add_argument('freq_lower', help='Lowest frequency channel.', type=float)
parser.add_argument('freq_upper', help='Highest frequency channel.', type=float)
parser.add_argument('nfreq', help='Number of frequency channels.', type=int)
parser.add_argument('mapname', help='Name of the file to save into.')
parser.add_argument('--pol', help='Polarised or not.', action='store_true')

args = parser.parse_args()

# Read in arguments.
gal = galaxy.ConstrainedGalaxy()
gal.nside = args.nside
gal.nu_lower = args.freq_lower
gal.nu_upper = args.freq_upper
gal.nu_num = args.nfreq

# Fetch galactic sky
cs = gal.getpolsky() if args.pol else gal.getsky()

# Fetch point source maps
ps = pointsource.CombinedPointSources.like_map(gal)
cs = cs + (ps.getpolsky() if args.pol else ps.getsky())

# Save map
f = h5py.File(args.mapname, 'w')
f.create_dataset("map", data=cs)
f.close()
