r"""
==============================================================================
Stiffened Panel Bay (:mod:`panels.stiffpanelbay`)
==============================================================================

.. currentmodule:: panels.stiffpanelbay

Main features:

- possibility to use many panels with different properties. In such case the
  panels are separated by their `y` (circumferential) coordinate. Usually
  there is a stiffener positioned at the `y` coordinate between two panels.
- possibility to use stiffeners (blade) modeled with 1D or 2D formulation.

.. autoclass:: StiffPanelBay
    :members:

"""
from .stiffpanelbay import (load, StiffPanelBay)
