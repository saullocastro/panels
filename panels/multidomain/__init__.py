"""
======================================================================
Multi-Domain semi-analytical method (:mod:`panels.multidomain`)
======================================================================

.. currentmodule:: panels.multidomain

Assemblies of several :class:`.Shell` domains connected along edges or over
areas. The package imports the class :class:`.MultiDomain` and the function
:func:`.load` of :mod:`panels.multidomain.multidomain`, and the functions
that build and analyse example assemblies of
:mod:`panels.multidomain.tstiff2d_1stiff_freq`,
:mod:`panels.multidomain.tstiff2d_1stiff_compression`,
:mod:`panels.multidomain.tstiff2d_1stiff_flutter`,
:mod:`panels.multidomain.cylinder` and
:mod:`panels.multidomain.cylinder_blade_stiffened`.

.. autoclass:: MultiDomain
    :members:


"""
from . multidomain import MultiDomain, load
from . tstiff2d_1stiff_freq import *
from . tstiff2d_1stiff_compression import *
from . tstiff2d_1stiff_flutter import *
from . cylinder import *
from . cylinder_blade_stiffened import *
