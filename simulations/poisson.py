
import numpy as np
import numpy.random as rnd
from scipy.optimize import fminbound
from scipy.integrate import quad, cumtrapz


from utils import cubicspline as cs


def homogeneous_process(t, rate):
    r"""Generate a realisation of a Possion process.

    Parameters
    ----------
    t : scalar
        The time length to realise. The events generated are between time 0 and `t`.
    rate : scalar
        The average rate of events.

    Returns
    -------
    events : ndarry
        An array containing the times of the events.
    """

    # Generates the intervals between events from an exponential
    # distribution (until total is time is greater than t) and then
    # perform cumulative sum to find event times. Generates in blocks
    # for efficiency.
    n = int(1.2*rate*t + 1)

    iv = rnd.exponential(1.0/rate, n)

    n = int(0.4*rate*t + 1)
    while (iv.sum() < t):
        ivt = rnd.exponential(1.0/rate, n)
        iv = np.concatenate((iv,ivt))

    ts = np.cumsum(iv)
    maxi = np.searchsorted(ts, [t])
    return ts[:maxi]

    

def test_hpp():
    r"""Test the homogenous Poisson process.

    Realise a poisson process, bin time intervals to count number of
    events in each bin, and then compare with a Poisson distribution. Test
    can not be reliably automated, results are returned to be checked
    by eye.

    Returns
    -------
    r1 : ndarray
        Number of events in an interval
    h2 : ndarray
        Number of intervals containing a specific number of events
        (corresponding to r1)
    p2 : ndarray
        Prediction for h2 from Poisson distribution
    """

    from scipy.misc import factorial

    pp = homogeneous_process(5000.0, 1.0)

    h1 = np.histogram(pp, bins=5000, range=(0,5000))[0]
    h2 = np.histogram(h1, bins=10, range=(0,10))[0]

    r1 = np.arange(0,11)
    p2 = 5000.0 / factorial(r1) * np.exp(-1.0)

    return (r1, h2, p2)




def inhomogeneous_process(t, rate):
    r"""Create a realisation of an inhomogenous (i.e. variable rate)
    Poisson process.

    Parameters
    ----------
    t : scalar
        The time length to realise. The events generated are between time 0 and `t`.
        
    rate : function
        A function which returns the event rate at a given time.
 
    Returns
    -------
    events : ndarry
        An array containing the times of the events.
    """

    # General scheme is to generate a homogenous process with the
    # highest rate, and then thin by accepting each event with
    # probability P(accept event i) = rate(t_i) / max(rate). This is
    # performed by _inhomogenous_process_wk.
    #
    # To increase efficiency for a highly variable rate, we divide the
    # total interval into blocks, and perform this scheme on each,
    # before combining.

    def _inhomogeneous_process_wk(t, rate):
        # Work routine that implements the basic inhomogeneous
        # algorithm described above. This is called repeatedly to do
        # the blocking.
        t_rmax = fminbound(lambda x: -rate(x), 0.0, t)
        rmax = rate(t_rmax)
        
        ut = homogeneous_process(t, rmax)

        if(ut.shape[0] == 0):
            return ut
        
        da = rnd.rand(ut.shape[0])
        
        ra = np.vectorize(rate)(ut)
        
        return ut[np.where(da < ra / rmax)]
    
    # For the moment simply split into intervals equal in time.
    nbin = 500
    iv = np.array([], dtype=np.float64)
    for i in range(nbin):
        tmin = i*t/(1.0*nbin)
        dt = t/(1.0*nbin)
        
        drate = lambda tr: rate(tr+tmin)

        ut = tmin + _inhomogeneous_process_wk(dt, drate)
        iv = np.concatenate((iv, ut))
    return iv


def test_ipp():
    r"""Test the inhomgeneous Poisson process.

    Generate an inhomogenous process, divide the time into bins, and
    calculate the events in each, to compare against the original rate
    function. This test is difficult to automate, so the results must
    be checked by eye.

    Returns
    -------
    r1 : ndarray
        Time of each bin.
    h1 : ndarray
        Average rate of events in each time bin.
    p2 : ndarray
        Input rate of events at each time.
    """

    def rate(t):
        return 30.0*(t / 2500.0 - 1.0)**2 + 100.0
    pp = inhomogeneous_process(5000.0, rate)

    h1 = np.histogram(pp, bins=50, range=(0,5000))[0] / 100.0

    r1 = np.arange(0,50)*100.0 + 50.0
    p2 = np.vectorize(rate)(r1)

    return (r1, h1, p2)




def inhomogeneous_process_approx(t, rate):
    r"""Create a realisation of an inhomogenous (i.e. variable rate)
    Poisson process.

    Uses an approximate method (but fast) based on generating a
    Cumulative Distribution Function from a set of samples, and the
    inverting this to generate the random deviates. The total number
    of events is calculated by drawing it from a Poisson distribution.

    Parameters
    ----------
    t : scalar
        The time length to realise. The events generated are between time 0 and `t`.
        
    rate : function
        A function which returns the event rate at a given time.
 
    Returns
    -------
    events : ndarry
        An array containing the times of the events.
    """

    # Work out average number of events in interval, and then draw the
    # actual number from a Poisson distribution
    av = quad(rate, 0.0, t)[0]
    total = np.random.poisson(av)

    # Sample from the rate function, and calculate the cumulative
    # distribution function
    ts = np.linspace(0.0, t, 10000)
    rs = rate(ts)

    cumr = np.zeros_like(ts)
    cumr[1:] = cumtrapz(ts, rs)
    cumr /= cumr[-1]

    # Interpolate to generate the inverse CDF and use this to generate
    # the random deviates.
    csint = cs.Interpolater(cumr, ts)

    return csint(np.random.rand(total))
