# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import numpy as np

from cora.foreground import galaxy, pointsource
from cora.signal import corr21cm

nside = 32
fa = np.linspace(400.0, 500.0, 32)


def test_galaxy():

    cr = galaxy.ConstrainedGalaxy()
    cr.nside = nside
    cr.frequencies = fa

    unpol = cr.getsky()
    assert unpol.shape == (32, 12 * nside**2)

    pol = cr.getpolsky()
    assert pol.shape == (32, 4, 12 * nside**2)

    # Check fluctuation amplitude for unpol
    ustd = unpol.std(axis=-1)
    assert (ustd > 10.0).all()
    assert (ustd < 50.0).all()

    # Check fluctuation amplitude for pol
    pstd = pol.std(axis=-1)
    assert (pstd[:, 0] > 10.0).all()
    assert (pstd[:, 0] < 50.0).all()
    assert (pol.std(axis=-1)[:, 1:3] > 0.1).all()
    assert (pol.std(axis=-1)[:, 1:3] < 3.0).all()
    assert (pol.std(axis=-1)[:, 3] == 0.0).all()

def test_pointsource():

    cr = pointsource.CombinedPointSources()
    cr.nside = nside
    cr.frequencies = fa

    unpol = cr.getsky()
    assert unpol.shape == (32, 12 * nside**2)

    pol = cr.getpolsky()
    assert pol.shape == (32, 4, 12 * nside**2)

    # Check fluctuation amplitude for unpol
    ustd = unpol.std(axis=-1)
    assert (ustd > 3.0).all()
    assert (ustd < 15.0).all()

    # Check fluctuation amplitude for pol
    pstd = pol.std(axis=-1)
    assert (pstd[:, 0] > 3.0).all()
    assert (pstd[:, 0] < 15.0).all()
    assert (pol.std(axis=-1)[:, 1:3] > 0.01).all()
    assert (pol.std(axis=-1)[:, 1:3] < 0.05).all()
    assert (pol.std(axis=-1)[:, 3] == 0.0).all()
