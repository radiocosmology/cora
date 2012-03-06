
""" A set of useful constants and conversions in Astronomy and Cosmology. """

import math

#### Astrophysical units (in SI) ####
# Solar masses in kg
solar_mass = 1.98892e30

# Prefixes
nano = 1e-9
micro = 1e-6
milli = 1e-3
kilo = 1e3
mega = 1e6
giga = 1e9

# Parsecs in m
parsec = 3.08568025e16
kilo_parsec = kilo * parsec
mega_parsec = mega * parsec
giga_parsec = giga * parsec 

# Time lengths in seconds.
second = 1.0
minute = 60.0
hour = 60.0 * minute
day = 24.0 * hour
year = 365.25 * day
kilo_year = kilo * year
mega_year = mega * year
giga_year = giga * year

t_sidereal = 23.9344696 * hour # Sidereal day

#### Physical constants (in SI) ####
# Gravitational constant G kg^-1 m^3 s^-2
G = 6.6742e-11
# In case of clashes
G_n = G

# Speed of light in m/s
c = 2.99792458e8
# In case of clashes.
c_sl = c

# Stefan-Boltzmann (in W m^{-2} K^{-4})
stefan_boltzmann = 5.6705e-8

# Radiation constant (in J m^{-3} K^{-4})
a_rad = 4*stefan_boltzmann / c

# 21cm transition frequency (in MHz)
nu21 = 1420.40575177


#### Angular units (in radians) ####
degree     = 2 * math.pi / 360
arc_minute = 2 * math.pi / (60 * 360)
arc_second = 2 * math.pi / (60 * 60 * 360)




