import numpy as np

from cora.signal import corr21cm
from cora.foreground import galaxy


def test_corr_signal():
    """Test that the signal power spectrum is being calculated correctly.

    Correct here is referenced to a specific version believed to have no errors.
    """

    cr = corr21cm.Corr21cm()

    aps1 = cr.angular_powerspectrum(np.arange(1000), 800.0, 800.0)
    assert len(aps1) == 1000
    assert np.allclose(
        aps1.sum(), 1.5963772205823096e-09, rtol=1e-7
    )  # Calculated for commit 02f4d1cd3f402d

    fa = np.linspace(400.0, 800.0, 64)
    aps2 = cr.angular_powerspectrum(
        np.arange(1000)[:, None, None], fa[None, :, None], fa[None, None, :]
    )
    assert aps2.shape == (1000, 64, 64)

    # Calculated for commit 02f4d1cd3f402d
    v1 = 8.986790805379046e-13  # l=400, fi=40, fj=40
    v2 = 1.1939298801340165e-18  # l=200, fi=10, fj=40
    assert np.allclose(aps2[400, 40, 40], v1, rtol=1e-7)
    assert np.allclose(aps2[200, 10, 40], v2, rtol=1e-7)


def test_corr_foreground():
    """Test that the foreground power spectrum is being calculated correctly.

    Correct here is referenced to a specific version believed to have no errors.
    """
    cr = galaxy.FullSkySynchrotron()

    aps1 = cr.angular_powerspectrum(np.arange(1000), 800.0, 800.0)
    assert len(aps1) == 1000
    assert np.allclose(
        aps1.sum(), 75.47681191093129, rtol=1e-7
    )  # Calculated for commit 02f4d1cd3f402d

    fa = np.linspace(400.0, 800.0, 64)
    aps2 = cr.angular_powerspectrum(
        np.arange(1000)[:, None, None], fa[None, :, None], fa[None, None, :]
    )
    assert aps2.shape == (1000, 64, 64)

    # Calculated for commit 02f4d1cd3f402d
    v1 = 9.690708728692975e-06  # l=400, fi=40, fj=40
    v2 = 0.00017630767166797886  # l=200, fi=10, fj=40
    assert np.allclose(aps2[400, 40, 40], v1, rtol=1e-7)
    assert np.allclose(aps2[200, 10, 40], v2, rtol=1e-7)
