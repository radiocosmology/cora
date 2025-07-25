"""Test caput.foreground.poisson"""

import numpy as np

from cora.foreground.poisson import inhomogeneous_process_approx


def test_inhomogeneous_process_approx():
    """Test inhomogeneous_process_approx"""

    # the most boringest rate function
    rate = lambda s: 1000 * (5 - s)

    result = inhomogeneous_process_approx(5, rate)

    # Mean should be approximately 1.666
    mean = result.mean()
    assert mean > 1.6
    assert mean < 1.75

    # stdev should be approximately 1.2
    stdev = result.std()
    assert stdev > 1.1
    assert stdev < 1.3
