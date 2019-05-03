r"""
================================================================================
Define structural matrices for each semi-analytical model (:mod:`panels.models`)
================================================================================

.. currentmodule:: panels.models

The modules herein contained are identified as follows:

    TYPE_THEORY_EQUATION_FIELDFUNCTION_SUFIX

If one of these is not present in the module name, consider not applicable.


TYPE refers to:

- cylshell - Cylindrical shells
- coneshell - Conical shells
- plate - Flat plates

THEORY refers to:

- clpt - Classical laminated plate theory
- fsdt - First-order shear deformation theory

EQUATION refers to which type of nonlinear equation is being used:

- donnell - kinematic equations using Donnell's equations

FIELDFUNCTION refers to the type of shape function used for the field
approximation:

- bardell - Rodrigues version of Legendre polynomials, largely applied by
  Bardell

SUFIX used to indicate additional information

- field - module used to calculate field variables
- num - stiffness matrices integrated numerically


"""
from __future__ import absolute_import

from . import clpt_bardell_field
from . import coneshell_clpt_donnell_bardell_num
from . import cylshell_clpt_donnell_bardell_num
from . import plate_clpt_donnell_bardell_num
