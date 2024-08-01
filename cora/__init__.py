"""Cosmology in the Radio Band

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    core
    foreground
    scripts
    signal
    util
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cora")
except PackageNotFoundError:
    # package is not installed
    pass

del version, PackageNotFoundError
