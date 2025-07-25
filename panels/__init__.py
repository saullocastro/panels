r"""
====================================================================
Semi-analytical models for plates, shells and panels (:mod:`panels`)
====================================================================

Models for plates, shells, stiffened panels, single or multi-domain are
available in this package.

.. currentmodule:: panels

.. automodule:: panels.shell
    :members:

.. automodule:: panels.plot_shell
    :members:

"""
import ctypes

from .version import __version__
from .shell import Shell

if ctypes.sizeof(ctypes.c_long) == 8:
    # here the C long will correspond to np.int64
    INT = np.int64
else:
    # here the C long will correspond to np.int32
    INT = np.int32

DOUBLE = np.float64
