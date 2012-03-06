import _chebyshev
import _patterson
import _romberg

def chebyshev(f, a, b, epsrel = 1e-6, epsabs = 1e-16, args =(), vectorized = True, fulloutput = False):
    """Integrate using a Chebyshev scheme.
    
    Parameters
    ----------
    f : function f(x, *args)
        The function to integrate. Optionally takes a set of extra
        arguments.
    a, b : scalar
        Integration limits.
    epsrel, epsabs : scalar, optional
        Relative and absolute integration tolerances.
    args : tuple/list, optional
        Optional arguments to pass to the integrand.
    vectorized : boolean, optional
        Is the integrand function vectorized (i.e. can evaluate a
        numpy array of abscissae). Assumed by default, leads to
        massive speedup if so.
    fulloutput : boolean, optional
        Whether to return more output info

    Returns
    -------
    i : scalar
        Result of integration.
    neval : integer
        Number of integrand evaluations (only returned if
        `fulloutput`).
    """

    if vectorized:
        r = _chebyshev.chebyshev_vec(f, a, b, epsrel=epsrel, epsabs=epsabs, args=args)
    else:
        r = _chebyshev.chebyshev(f, a, b, epsrel=epsrel, epsabs=epsabs, args=args)

    if r[2] != 0:
        raise Exception("Chebyshev integration not converged properly.")

    if fulloutput:
        return r
    else:
        return r[0]


def patterson(f, a, b, epsrel = 1e-6, epsabs = 1e-16, args =(), fulloutput = False):
    """Integrate using a Patterson scheme. Integrand must be vectorized.
    
    Parameters
    ----------
    f : function f(x, *args)
        The function to integrate. Optionally takes a set of extra
        arguments.
    a, b : scalar
        Integration limits.
    epsrel, epsabs : scalar, optional
        Relative and absolute integration tolerances.
    args : tuple/list, optional
        Optional arguments to pass to the integrand.
    fulloutput : boolean, optional
        Whether to return more output info

    Returns
    -------
    i : scalar
        Result of integration.
    neval : integer
        Number of integrand evaluations (only returned if
        `fulloutput`).
    """

    r = _patterson.Integrate_Patterson(f, a, b, eps=epsrel, abs=epsabs, args=args)
    
    if r[2] != 0:
        raise Exception("Patterson integration not converged properly.")

    if fulloutput:
        return r
    else:
        return r[0]


def romberg(f, a, b, epsrel = 1e-6, epsabs = 1e-16, args =()):
    """Integrate using a Romberg scheme. Integrand must be vectorized.
    
    Parameters
    ----------
    f : function f(x, *args)
        The function to integrate. Optionally takes a set of extra
        arguments.
    a, b : scalar
        Integration limits.
    epsrel, epsabs : scalar, optional
        Relative and absolute integration tolerances.
    args : tuple/list, optional
        Optional arguments to pass to the integrand.

    Returns
    -------
    i : scalar
        Result of integration.
    """

    if args:
        f = lambda x: f(x, *args)

    return _romberg.romberg(f, a, b, eps=epsrel)




