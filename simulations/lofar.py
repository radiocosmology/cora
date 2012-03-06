r"""LOFAR foreground model.

Implements the foreground simulation method of Jelic et al [1]_. At
the moment only the Synchrotron is implemented.

References
----------
.. [1] http://arxiv.org/abs/0804.1130
"""
import numpy as np

from gaussianfield import RandomField
from maps import *



class _LofarGDSE_3D(RandomField):

    delta = -4.0

    def powerspectrum(self, karray):
        r"""Power law power spectrum"""

        ps = (karray**2).sum(axis=3)**(self.delta / 2.0)
        ps[0,0,0] = 0.0

        return ps

class LofarGDSE(Map3d):
    r"""LOFAR synchrotron model.

    Generates a 3d section of the galaxy with a independent power law
    emission at each point (seperate amplitude and spectral index),
    and integrates over the third axis, to produce a 3d
    angle-angle-frequency map.
    """
    nu_0 = 325.0
    
    correlated = False

    A_amp = 20
    A_std = A_amp * 0.02
    
    beta_mean = -2.55
    beta_std = 0.1

    alpha = -2.7
    
    def getfield(self):
        r"""Lofar synchrotron."""

        numz = int((self.x_num + self.y_num) / 2)

        # Set up 3D field generator
        npix = [self.x_num, self.y_num, numz]
        wsize = [5.0 / self.x_width, 5.0 / self.y_width, 1.0]
        lf = _LofarGDSE_3D(npix = npix, wsize = wsize)
        lf.delta = self.alpha
        
        A = lf.getfield()
        beta = A if self.correlated else lf.getfield()

        A = ((1.0*self.A_amp) / numz) + A * (self.A_std / A.sum(axis = 2).std())
        beta = self.beta_mean + beta * (self.beta_std / beta.std())

        freq = self.nu_pixels / self.nu_0

        Tb = np.zeros(self._num_array())

        for i in range(self.nu_num):
            Tb[i,:,:] = (A * freq[i]**beta).sum(axis=2)

        return Tb


