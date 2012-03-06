
import numpy as np
import math

from utils import fftutil, units

from maps import *



class RandomField(object):
    r"""Generate a realisation a n-dimensional gaussian field.

    An appropriate `powerspectrum` function must be implemented to use
    this class.
    
    Parameters
    ----------
    npix : list of ints
        Number of pixels along each axis.
    wsize : list of floats, optional
        Width along each axis (in arbitrary units). If not given, each
        pixel has length one (i.e. wsize = npix).
    """
    
    
    ## Internal attributes
    _kweightgen = False 
    _n = None
    _w = None
    
    def __init__(self, npix = None, wsize = None):
        
        self._n = np.array(npix) if npix != None else npix
        self._w = np.array(wsize) if wsize != None else self._n
        
        
    def _check_input(self):
        if self._n == None or self._w == None:
            raise Exception("Either self._n or self._w has not been set.")

        if len(self._n) != len(self._w):
            raise Exception("Width array must be the same length as number of pixels.")
        
        if not ( (self._n > 0).all() and (self._w > 0).all() ):
            raise Exception("Array elements must be positive.")
        


    def powerspectrum(self, karray):
        r"""Get the power spectrum value at each wavevector.
        
        Parameters
        ----------
        karray : ndarray
            An array filled with k-vectors. The last axis gives the
            components of each vector, the rest of the array can be of
            any shape.

        Returns
        -------
        psarray : ndarray
            An array containing the power spectrum values for each
            vector in `karray'. Must be the same shape as `karray'
            with its last axis removed.

        Notes
        -----
        Frequencies are in terms of wavenumber. That is the positive
        frequencies for one axis are:
        0, 2*math.pi / self._w[0], 2 * 2*math.pi / self._w[0], 3 * 2*math.pi
        / self._w[0]....

        """
        raise Exception("Abstract method: need to override.")


    def generate_kweight(self, regen = False):
        r"""Pregenerate the k weights array.

        Parameters
        ==========
        regen : boolean, optional
            If True, force regeneration of the weights, to be used if
            parameters have been changed,
        """

        # Perform some sanity checks
        self._check_input()

        # If already generated return array now.
        if(self._kweightgen == True and not regen):
            return
        
        spacing = self._w / self._n
        
        kvec = fftutil.rfftfreqn(self._n, spacing / (2*math.pi))
        
        self._kweight = self.powerspectrum(kvec)**0.5 * self._n.prod() / (2.0 * self._w.prod() )**0.5

        if not np.isfinite(self._kweight.flat[0]):
            self._kweight.flat[0] = 0.0

        self._kweightgen = True
        

    def getfield(self):
        r"""Generate a new realisation of the 3d field.

        Returns
        -------
        field : ndarray
            A realisation of the power spectrum. Shape is self._n.
        """

        self.generate_kweight()
        s = self._kweight.shape
        
        # Fill array
        f = np.random.standard_normal(s) + 1.0J * np.random.standard_normal(s)
        f *= self._kweight
        
        r = np.fft.irfftn(f)
        return r
           

    

class RandomFieldA2F(RandomField, Map3d):
    r"""Generate a realisation of a 3d gaussian field.
    
    This routine is to generate simulated 3d observations. It is
    tailored for situations where there are two angular dimensions,
    and the third is frequency. The `powerspectrum` function must be
    implemented to use this class.
    """


    def generate_kweight(self, *args):
        r"""Pregenerate the power spectrum values."""

        self._n = self._num_array()
        self._w = self._width_array()
                
        RandomField.generate_kweight(self, *args)



class RandomFieldA2(RandomField, Map2d):
    r"""Generate a realisation of a 2d gaussian field.
    
    This routine is to generate simulated 2d observations. It is
    tailored for situations where there are two angular
    dimensions. The `powerspectrum` function must be implemented to
    use this class.
    """

    def generate_kweight(self, *args):
        r"""Pregenerate the power spectrum values."""

        self._n = self._num_array()
        self._w = self._width_array()
        
        RandomField.generate_kweight(self, *args)



class Cmb(RandomFieldA2):
    r"""Simulate a patch of the CMB."""

    def __init__(self, psfile = None, cambnorm = True):
        """ Initialise the CMB field class. """

        from utils.cubicspline import LogInterpolater
                
        if(psfile == None):
            from os.path import dirname, join
            psfile = join(dirname(__file__), 'ps_cmb2.dat')
        if(cambnorm):
            a = np.loadtxt(psfile)
            l = a[:,0]
            tt = (2*math.pi) *a[:,1] / (l*(l+1.0))
            self._powerspectrum_int = LogInterpolater(np.vstack((l,tt)).T)
        else:
            self._powerspectrum_int = LogInterpolater.fromfile(psfile)

            

    def powerspectrum(self, karray):
        return np.vectorize(self._powerspectrum_int.value)((karray**2).sum(axis=2)**0.5)



class TestF(RandomFieldA2F):
    
    def powerspectrum(self, karray):

        return (np.exp(-0.5*(karray[...,0] / (2*math.pi / 250.0))**2) * 
                np.exp(-0.5*(karray[...,1:3]**2).sum(axis=3) / (2*math.pi / (1.0*units.degree))**2))

