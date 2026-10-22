r"""
================================================
CLPT (:mod:`panels.stiffener.models`)
================================================

.. currentmodule:: panels.stiffener.models

Compiled kernels of the stiffener models based on the classical laminated
plate theory with the Donnell kinematics: ``bladestiff1d_clt_donnell``, used
by :class:`.BladeStiff1D`, and ``bladestiff2d_clt_donnell``, used by
:class:`.BladeStiff2D`. They are selected through
:mod:`panels.stiffener.modelDB`.

"""
from . import bladestiff1d_clt_donnell
from . import bladestiff2d_clt_donnell
