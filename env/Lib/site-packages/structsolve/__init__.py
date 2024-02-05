r"""
===============================================
Structural Analysis Solver (:mod:`structsolve`)
===============================================

.. currentmodule:: structsolve

.. autoclass:: Analysis
    :members:

.. autofunction:: freq

.. autofunction:: lb

.. autofunction:: static

"""
from __future__ import absolute_import

from .analysis import Analysis
from .freq import freq
from .linear_buckling import lb
from .static import solve, static
