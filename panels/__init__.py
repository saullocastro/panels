r"""
====================================================================
Semi-analytical models for plates, shells and panels (:mod:`panels`)
====================================================================

Models for plates, shells, stiffened panels, single or multi-domain are
available in this package.

.. currentmodule:: panels

.. automodule:: panels.shell
    :members:

.. automodule:: panels.shell_fext
    :members:

.. automodule:: panels.plot_shell
    :members:

"""
import ctypes

import numpy as np

from .version import __version__


if ctypes.sizeof(ctypes.c_long) == 8:
    # here the C long will correspond to np.int64
    INT = np.int64
else:
    # here the C long will correspond to np.int32
    INT = np.int32

DOUBLE = np.float64
