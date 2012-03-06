#==================================================================================================
import math
import numpy as np

#==================================================================================================
def romberg(f, a, b, eps = 1e-10, nmax = 20, nmin = 2):

    r = np.zeros((nmax,nmax))

    h = b-a
    n = 1
    r[0,0] = 0.5 * h * (f(a) + f(b))

    for i in range(1,nmax):
        n *= 2
        h = h / 2.0
        
        # Create array of x-values (spaced between existing points)
        xa = a + (2.0 * np.arange(1, n/2 +1) - 1.0) * h
        # Create array of function values from x-coords
        fa = f(xa)
        # Sum up
        t = fa.sum()

        # Set up first entry at this level (equivalent to trapezoidal)
        r[i,0] = 0.5 * r[i-1,0] + t*h
        
        # Extrapolate along row
        for j in range(1,i+1):
            r[i,j] = r[i,j-1] + 1.0 / (4**j - 1) * (r[i,j-1] - r[i-1,j-1])

        # Test relative tolerance if we've done the minimum number of
        # iterations
        if(i > nmin and np.abs(1.0 - r[i-1,j-1] / r[i,j]) < eps):
            return r[i,j]
    
    # If max number of iterations reached return the final
    # extrapolation
    return r[nmax-1,nmax-1]

#==================================================================================================
