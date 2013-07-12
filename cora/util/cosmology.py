""" Class for storing a cosmology, and calculating various distance measures. """

# Global imports
import math
import numpy as np

# Import integration routines
from scipy import integrate as si

# Package imports
from cora.util.units import *


class Cosmology(object):
    r""" Class to store a cosmology, and compute measures.
    
    Defines a cosmology and allows calculation of a few simple
    quantities (notably distance measures and lookback time).

    Attributes
    ----------
    units : {'astro', 'cosmo', 'si'}
        The unit system to return quantities in. SI is self
        explanatory, `astro` has distances in Mpc and time in Myr, and
        `cosmo` has distances in Mpc/h and time in Myr. Defaults to 'cosmo'.
    omega_b : scalar
        Baryon fraction relative to the critical density.
    omega_c : scalar
        Dark matter fraction relative to the critical density.
    omega_l : scalar
        Dark energy fraction relative to the critical density (assumes
        a cosmological constant).
    omega_g : scalar
        Fraction of electromagnetic radiation relative to the critical density.
    omega_g : scalar
        Fraction of massless neutrinos relative to the critical density.
    H0 : scalar
        The Hubble constant in km/s / Mpc
    w_0, w_a : scalars
        Dark energy parameters.

    omega_k : scalar, readonly
        The curvature as a fraction of the critical density.
    omega_m : scalar, readonly
        The total matter density as a fraction of the critical density.
    omega_r : scalar, readonly
        The total radiation density as a fraction of the critical density.

    """

    units = 'cosmo'

    omega_b = 0.0458
    omega_c = 0.229
    omega_l = 0.7252

    omega_g = 0.0
    omega_n = 0.0

    # H_0 given in km/s / Mpc
    H0 = 70.2

    # Dark energy parameters
    w_0 = -1.0
    w_a =  0.0

    @property
    def omega_m(self):
        return self.omega_b + self.omega_c

    @property
    def omega_r(self):
        return self.omega_g + self.omega_n

    @property
    def omega_k(self):
        return 1.0 - (self.omega_l + self.omega_b + self.omega_c + self.omega_g + self.omega_n)

    

    @staticmethod
    def init_physical(ombh2 = 0.02255, omch2 = 0.1126, H0 = 70.2, omkh2 = 0.0, t0 = 2.726, nnu = 3.046):
        r"""Initialise a new cosmology from the physical parameters.

        This uses the CMB relevant parameterisation that is commonly
        used. The dark energy density is calculated from the given
        parameters.

        Parameters
        ----------
        ombh2 : scalar, optional
            The relative baryon density times h^2 (h = H_0 / 100).
        omch2 : scalar, optional
            The fractional dark matter density times h^2 (h = H_0 / 100).
        H0 : scalar, optional
            The Hubble constant
        omkh2 : scalar, optional
            The curvature fraction times h^2.
        t0 : scalar, optional
            The CMB temperature (used to calculate the radiation density).
        nnu : scalar, optional
            The number of massless neutrinos. Used to set the neutrino density.

        Returns
        -------
        cosmo : instance of Cosmology
        """
        h = H0 / 100.0

        c = Cosmology()

        c.omega_b = ombh2 / h**2
        c.omega_c = omch2 / h**2
        c.H0 = H0

        rhoc = 3.0 * c.H()**2 * c_sl**2/ (8.0 * math.pi * G_n)
        rhorad = a_rad * t0**4
        c.omega_g = rhorad / rhoc

        rhonu = nnu * rhorad * 7.0 / 8.0 * (4.0 / 11.0)**(4.0 / 3.0)
        c.omega_n = rhonu / rhoc

        c.omega_l = 1.0 - (omkh2 + ombh2 + omch2) / h**2 - (c.omega_g + c.omega_n)


        return c


    def H(self, z=0.0):
        """The Hubble parameter at redshift z.

        Return the Hubble parameter in SI units (s^-1), regardless of
        the value of `self.units`.

        Parameters
        ----------
        z : scalar, optional
            The redshift to calculate the Hubble parameter
            at. Defaults to z = 0.

        Returns
        -------
        H : scalar
            The Hubble parameter.
        """
        
        H  = self.H0 * (self.omega_r * (1 + z)**4 +  self.omega_m * (1 + z)**3
                        + self.omega_k * (1 + z)**2
                        + self.omega_l * (1 + z)**(3*(1+self.w_0 + self.w_a)) * np.exp(-3*self.w_a * z / (1 + z)))**0.5
                
        # Convert to SI
        return H * 1000.0 / mega_parsec
        
    def comoving_distance(self, z):
        r"""The comoving distance to redshift z.

        This routine is vectorized.

        Parameters
        ----------
        z : array_like
            The redshift(s) to calculate at.
        
        Returns
        -------
        dist : array_like
            The comoving distance to each redshift.
        """

        # Calculate the integrand.
        def f(z1):
            return c_sl / self.H(z1)

        return _intfz(f, 0.0, z) / self._unit_distance


    def proper_distance(self, z):
        r"""The proper distance to an event at redshift z.
        
        The proper distance can be ill defined. In this case we mean
        the comoving transverse separation between two events at the
        same redshift divided by their angular separation. This
        routine is vectorized.

        Parameters
        ----------
        z : array_like
            The redshift(s) to calculate at.
        
        Returns
        -------
        dist : array_like
            The proper distance to each redshift.
        """

        x = self.comoving_distance(z)

        om_k = self.omega_k

        dhi = math.sqrt(math.fabs(om_k)) * self.H() / c_sl * self._unit_distance

        if(om_k < 0.0):
            x = np.sin(x * dhi)  / dhi
        elif(om_k > 0.0):
            x = np.sinh(x * dhi) / dhi

        return x

    def angular_distance(self, z):
        r"""The angular diameter distance to redshift z.

        Not to be confused with the `proper_distance`. This is the
        *physical* transverse separation between two events at the
        same redshift divided by their angular separation. This
        routine is vectorized.

        Parameters
        ----------
        z : array_like
            The redshift(s) to calculate at.
        
        Returns
        -------
        dist : array_like
            The angular diameter distance to each redshift.
        """

        
        return self.proper_distance(z) / (1 + z)


    def luminosity_distance(self, z):
        r""" The luminosity distance to redshift z. This
        routine is vectorized.

        Parameters
        ----------
        z : array_like
            The redshift(s) to calculate at.
        
        Returns
        -------
        dist : array_like
            The luminosity distance to each redshift.
        """
        return self.proper_distance(z) * (1 + z)


    def lookback_time(self, z):
        r""" The lookback time out to redshift z.

        Parameters
        ----------
        z : array_like
            The redshift(s) to calculate at.
        
        Returns
        -------
        time : array_like
            The lookback time to each redshift.
        """

        # Calculate the integrand.
        def f(z1):
            return 1.0 / (self.H(z1) * (1 + z1))

        return _intfz(f, 0.0, z) / self._unit_time

    @property
    def _unit_distance(self):
        # Select the appropriate distance unit
        if self.units == 'astro':
            return mega_parsec
        elif self.units == 'cosmo':
            return mega_parsec / (self.H0 / 100.0)
        elif self.units == 'si':
            return 1.0

        raise Exception("Units not known")

    @property
    def _unit_time(self):
        # Select the appropriate time unit
        if self.units == 'astro':
            return mega_year
        elif self.units == 'cosmo':
            return mega_year
        elif self.units == 'si':
            return 1.0

        raise Exception("Units not known")
        

@np.vectorize
def _intfz(f, a, b):
    # A wrapper function to allow vectorizing integrals, cuts such
    # that integrals to very high-z converge (by integrating in log z
    # instead).

    def _int(f, a, b):
        return si.romberg(f, a, b, rtol=1e-5, tol=1e-10, vec_func=True)
    
    
    cut = 1e2
    if a < cut and b > cut:
        return _int(f, a, cut) + _int(lambda lz: np.exp(lz) * f(np.exp(lz)), np.log(cut), np.log(b))
    else:
        return _int(f, a, b)
