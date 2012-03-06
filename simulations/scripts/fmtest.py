## Test Foreground simulation via Lofar or SCK methods


import numpy as np

from simulations import foregroundsck, lofar

tf = foregroundsck.Synchrotron()

freq = tf.nu_pixels

cv = tf.frequency_covariance(*np.meshgrid(freq, freq))

f1 = tf.getfield()

fdt = f1 * ((freq/tf.nu_0)**2.80)[:,np.newaxis,np.newaxis]

lf = lofar.LofarGDSE()

f2 = lf.getfield()
