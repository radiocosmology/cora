"""A set of useful constants and conversions in Astronomy and Cosmology."""

import math

# Astrophysical units (in SI)

#: Solar masses in kg
solar_mass = 1.98892e30

#: Nano prefix
nano = 1e-9

#: Micro prefix
micro = 1e-6

#: Milli prefix
milli = 1e-3

#: Kilo prefix
kilo = 1e3

#: Mega prefix
mega = 1e6

#: Giga prefix
giga = 1e9

#: Parsecs in m
parsec = 3.08568025e16

#: Kilo Parsecs in m
kilo_parsec = kilo * parsec

#: Mega Parsecs in m
mega_parsec = mega * parsec

#: Giga Parsecs in m
giga_parsec = giga * parsec

# Time lengths in seconds.

#: One second in seconds
second = 1.0

#: One minute in seconds
minute = 60.0

#: One hour in seconds
hour = 60.0 * minute

#: One day in seconds
day = 24.0 * hour

#: One year in seconds
year = 365.25 * day

#: One kilo year in seconds
kilo_year = kilo * year

#: One mega year in seconds
mega_year = mega * year

#: One giga year in seconds
giga_year = giga * year

#: One sidereal day in seconds
t_sidereal = 23.9344696 * hour

# Physical constants (in SI)

#: Gravitational constant :math:`G kg^-1 m^3 s^-2`.
G = 6.6742e-11

#: Gravitational constant :math:`G kg^-1 m^3 s^-2`. (In case of clashes)
G_n = G

#: Speed of light in m/s
c = 2.99792458e8

#: Speed of light in m/s (In case of clashes.)
c_sl = c

#: Stefan-Boltzmann (in W m^{-2} K^{-4})
stefan_boltzmann = 5.6705e-8

#: Radiation constant (in J m^{-3} K^{-4})
a_rad = 4 * stefan_boltzmann / c

#: 21cm transition frequency (in MHz)
nu21 = 1420.40575177

#: Boltzmann constant
k_B = 1.3806503e-23


# Angular units (in radians)

#: One degree in radians
degree = 2 * math.pi / 360

#: One arc minute in radians
arc_minute = 2 * math.pi / (60 * 360)

#: One arc second in radians
arc_second = 2 * math.pi / (60 * 60 * 360)
