from ..signal import corr
import numpy as np
import scipy.integrate as integrate


class haloModel(object):

    # TODO: move these to util/units.py ?
    dc = 1.686  # Critical density of colapse
    solar_mass = 1.98847E30
    # TODO: move this to init?
    redshift = 0.

    def __init__(self, gf=None, window='tophat', collapse_method='ellipsoidal', ps=None, cosmology=None):

        if cosmology is None:
            cosmology = corr.Cosmology()

        if gf is None:
            from .cosmology import growth_factor as gf
            self.gf = gf

        if ps is None:
            from . import cubicspline as cs
            from os.path import join, dirname
            ps_redshift = 1.5  # Redshift of the power sectrum to be loaded
            psfile = join(dirname(corr.__file__), "data/ps_z1.5.dat")
            # psfile = join(dirname(corr.__file__),"data/ps_z1.5_planck.dat")
            cs1 = cs.LogInterpolater.fromfile(psfile)
            self.ps = lambda k: cs1(k) / self.growth_factor(ps_redshift)**2

        h = cosmology.H0/100.
        H0_1s = cosmology.H0/(corr.units.kilo_parsec)  # H0 in 1/s (SI)
        rho_bar = 3.*H0_1s**2/(8.*np.pi*corr.units.G)*cosmology.omega_m  # in kg/m^3 (G comes in SI)
        rho_bar = rho_bar*(corr.units.kilo_parsec*1E3)**3  # in kg/Mpc^3 (G comes in SI)
        self.rho_bar = rho_bar/h**2/self.solar_mass  # in Mo h^-1 / (h^-1 Mpc)^3 (Mo is solar mass)
        self.collapse_method = collapse_method

        if window == 'gauss':
            # Using Gaussian window function
            # Prefactor for the volume
            self.gammaf = (2.*np.pi)**(1.5)

            # Window functio in k-space
            self.Wk = lambda kR: np.exp(-0.5*kR**2)

        elif window == 'sharpk':
            # Using sharp-k window function
            # Prefactor for the volume
            self.gammaf = 6.*np.pi**2

            # Window functio in k-space
            self.Wk = lambda kR: np.array(kR <= 1.).astype(float)

        elif window == 'tophat':
            # Using tophat window function
            # Prefactor for the volume
            self.gammaf = 4.*np.pi/3.

            # Window functio in k-space
            self.Wk = lambda kR: 3.*(np.sin(kR)-kR*np.cos(kR))/kR**3

    def growth_factor(self, z=None):
        """TODO: There are three definition of this function (growth_factor())
        in CORA: in corr.py, corr21cm.py and cosmology.py."""
        if z is None:
            z = self.redshift
        
        # return 1./(1. + z)
        return self.gf(z)/self.gf(0.)

    def dct(self, z=None):
        if z is None:
            z = self.redshift
        return self.dc/self.growth_factor(z)

    def V(self, R):
        """From MBW discussion after eq.(6.35)"""
        return self.gammaf*R**3

    def M(self, R):
        """From MBW eq.(6.36)"""
        return self.rho_bar * self.V(R)

    def sigma2(self, R, k_int_lim=None):
        """From MBW eq.(7.44)
        R is a scalar here"""
        
        def integrand(k):
            return self.ps(k) * self.Wk(k*R)**2 * k**2
        
        # Determine integration limit if not given.
        if k_int_lim is None:
            k_int_lim = [0., 3./R]
        # integrate.quad returns a tupple (value, error)
        return integrate.quad(integrand, k_int_lim[0], k_int_lim[1])[0]/(2.*np.pi**2)

    def s2_array(self, R):
        if not isinstance(R, (np.ndarray, list)):
            R = np.array([R])
        return np.array([self.sigma2(r) for r in R])

    def nu(self, s2, z=None):
        if z is None:
            z = self.redshift
        return self.dct(z) / s2**0.5

    def nu2(self, s2, z=None):
        if z is None:
            z = self.redshift
        return self.dct(z)**2 / s2

    def fps(self, nu, nu2):
        """ From MBW eq.(7.47) """
        return (2./np.pi)**0.5*nu*np.exp(-0.5*nu2)

    def f(self, R, z=None):
        """ From MBW eq.(7.47) and eq.(7.67)
        or from arXiv:astro-ph/9907024 eq.(6)"""
        if z is None:
            z = self.redshift
        s2 = self.s2_array(R)
        nus = self.nu(s2, z)
        nu2s = self.nu2(s2, z)
        if self.collapse_method == 'ellipsoidal':
            A = 0.322
            q = 0.3
            a = 0.707
            nutilde = a**0.5*nus
            nutilde2 = a*nu2s
            return A * (1.+1./nutilde2**q) * self.fps(nutilde, nutilde2)
        elif self.collapse_method == 'spherical':
            return self.fps(nus, nu2s)

    def n(self, R, z=None):
        """ From MBW eq.(7.46) """
        if z is None:
            z = self.redshift
        Ms = self.M(R)
        s2 = self.s2_array(R)
        nus = self.nu(s2, z)
        return self.rho_bar / Ms**2 * self.f(R, z) * abs(np.gradient(np.log(nus), np.log(Ms)))
    
    def lbias(self, R, z=None):
        """ Lagrangian halo bias.
        From arXiv:astro-ph/9907024"""
        if z is None:
            z = self.redshift
        s2 = self.s2_array(R)
        nu2s = self.nu2(s2, z)
        a = 0.707
        b = 0.5
        c = 0.6
        nutilde2 = a*nu2s
#         return 1./self.dct(z) * ( nutilde2
        return 1./self.dc * (nutilde2
                             + b*nutilde2**(1.-c)
                             - (nutilde2**c / a**0.5)/(nutilde2**c + b*(1.-c)*(1.-c*0.5)))