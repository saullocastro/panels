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

These classes are saved to and loaded from zip files with JSON and NumPy
arrays, see :mod:`panels.json_io`.

The structural matrices and force vectors of these classes are solved with
`structsolve <https://github.com/saullocastro/structsolve>`_.

Notes
-----
The models are semi-analytical: in each domain the displacement field is a
Ritz series of products of Bardell's hierarchical functions along `x` and
`y`, whose coefficients, the Ritz constants, are the degrees of freedom of
the structural matrices, see :class:`.Shell` and :mod:`panels.bardell`. The
domains of an assembly share no nodes; their compatibility is imposed
along the connections, as in the semi-analytical models of stiffened
panels of [1]_.

References
----------
.. [1] Castro, S. G. P., and Donadon, M. V., "Assembly of Semi-Analytical
   models to Address Linear Buckling and Vibration of Stiffened Composite
   Panels with Debonding Defect", Composite Structures, 2017.
   https://doi.org/10.1016/j.compstruct.2016.10.026

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
