r"""
====================================================================
Semi-analytical models for plates, shells and panels (:mod:`panels`)
====================================================================

.. currentmodule:: panels

Ritz models for plates, shells and stiffened panels, single or multi-domain,
using Bardell's hierarchical functions as approximation functions:

- :class:`.Shell`: plates and cylindrical shells, see :mod:`panels.models`
- :class:`.MultiDomain`: assemblies of shell domains connected by penalties,
  see :mod:`panels.multidomain.connections`
- :class:`.BladeStiff1D` and :class:`.BladeStiff2D`: blade stiffeners
- :class:`.StiffPanelBay`: stiffened panel bays

The structural matrices and force vectors of these classes are solved with
`structsolve <https://github.com/saullocastro/structsolve>`_.

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
