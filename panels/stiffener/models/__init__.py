r"""
================================================
CLPT (:mod:`panels.stiffener.models`)
================================================

.. currentmodule:: panels.stiffener.models

Compiled kernels of the stiffener models based on the classical laminated
plate theory: ``bladestiff1d_clt_donnell``, used by :class:`.BladeStiff1D`,
and ``bladestiff2d_clt_donnell`` and ``bladestiff2d_clt_sanders``, used by
:class:`.BladeStiff2D` on a skin with the Donnell kinematics (flat or
cylindrical) and on a cylindrical skin with the Sanders kinematics. They are
selected through :mod:`panels.stiffener.modelDB`.

"""
from . import bladestiff1d_clt_donnell
from . import bladestiff2d_clt_donnell
from . import bladestiff2d_clt_sanders
