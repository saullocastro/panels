"""
===================================================================
Panel built of plates, shells and stiffeners (:mod:`panels.panel`)
===================================================================

.. currentmodule:: panels.panel

.. autoclass:: Panel
    :members:


"""
from __future__ import absolute_import

from . panel import Panel
from . tstiff2d_1stiff_freq import *
from . tstiff2d_1stiff_compression import *
from . tstiff2d_1stiff_flutter import *
from . cylinder import *
from . cylinder_blade_stiffened import *
