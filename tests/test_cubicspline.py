import numpy as np
import pytest
from pytest import approx

from cora.util import cubicspline


def test_usage():
    """Test wrong usage."""

    # too few points
    x = np.arange(3)
    y = np.asarray([0, 10, 20], dtype=np.float64)
    with pytest.raises(cubicspline.InterpolationException):
        cubicspline.Interpolater(x, y)

    # x longer than y
    x = np.arange(4)
    with pytest.raises(cubicspline.InterpolationException):
        cubicspline.Interpolater(x, y)

    # invalid values
    y[1] = np.inf
    with pytest.raises(cubicspline.InterpolationException):
        cubicspline.Interpolater(x, y)
    y[1] = np.nan
    with pytest.raises(cubicspline.InterpolationException):
        cubicspline.Interpolater(x, y)


@pytest.mark.parametrize(
    "interpolater", [(cubicspline.Interpolater), (cubicspline.LogInterpolater)]
)
def test_constant(interpolater):
    """Interpolate and extrapolate f(x) = 1."""
    x = np.arange(1, 8)
    y = np.ones(7)
    data = np.dstack((x, y))[0]
    p = interpolater(data)

    # two different ways to do the same thing:
    assert (p(np.asarray([0.025, 1, 2.5, 4, 5.55, 7.01, 19])) == 1).all()
    if interpolater == cubicspline.LogInterpolater:
        assert (
            p.value_log_array(np.asarray([0.0125, 1, 2.5, 4, 5.55, 7.01, 19])) == 1
        ).all()
    else:
        assert (
            p.value_array(np.asarray([-0.025, 1, 2.5, 4, 5.55, 7.01, 19])) == 1
        ).all()


@pytest.mark.parametrize(
    "interpolater", [(cubicspline.Interpolater), (cubicspline.LogInterpolater)]
)
def test_linear(interpolater):
    """Interpolate f(x) = 10x."""
    x = np.arange(1, 5)
    y = np.asarray([10, 20, 30, 40])
    data = np.dstack((x, y))[0]
    p = interpolater(data)

    if interpolater == cubicspline.Interpolater:
        assert p(-1) == approx(-10)
        assert p(0) == approx(0)

    assert p.value(0.5) == approx(5)
    assert p(1) == approx(10)
    assert p(1.75) == approx(17.5)
    assert p(2) == approx(20)
    assert p(2.2) == approx(22)
    assert p(3) == approx(30)
    assert p(4) == approx(40)


@pytest.mark.parametrize(
    "interpolater", [(cubicspline.Interpolater), (cubicspline.LogInterpolater)]
)
def test_random(interpolater):
    """Interpolate random points."""
    x = np.arange(1, 5)
    y = np.asarray([1.67, 1.99, 0.465, 0.234])
    data = np.dstack((x, y))[0]
    p = interpolater(data)

    for i, x_ in enumerate(x):
        assert p(x_) == pytest.approx(y[i], rel=1e-13)


def test_polynomial():
    """Interpolate f(x) = 1 + 2x + 3x**2."""
    f = np.polynomial.polynomial.Polynomial((1, 2, 3))
    x = np.arange(0, 1000, 0.01)
    y = np.array([f(i) for i in x])
    interpolater = cubicspline.Interpolater(x, y)

    for x_ in np.asarray([0, 1, 0.0998, 456, 666.666, 998.501, 999.98, 99.98999]):
        assert abs(interpolater(x_) - f(x_)) < 1e-7
    t = interpolater.test(0, 10, 100)
    for x_, y_ in t:
        assert abs(y_ - f(x_)) < 1e-7


def test_polynomial_edge():
    """
    Interpolate f(x) = 1 + 2x + 3x**2.

    Tests values close to the lower edge of the interpolation function.

    This test currently fails.
    """
    f = np.polynomial.polynomial.Polynomial((1, 2, 3))
    x = np.arange(0, 1000, 0.01)
    y = np.array([f(i) for i in x])
    interpolater = cubicspline.Interpolater(x, y)

    # To estimate an error boundary for this test, use cubicspline from scipy w/o
    # enforcing boundary conditions:
    # import scipy
    # ## same type as cora
    # scinterpolater = scipy.interpolate.CubicSpline(x, f(x), bc_type="natural")
    error_scipy = np.asarray([1.46e-05, 4.21e-06, 1.71e-06])

    for x_, err in zip(np.asarray([0.00101, 0.01111, 0.0001]), error_scipy):
        assert abs(interpolater(x_) - f(x_)) <= err
