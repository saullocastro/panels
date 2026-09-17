r"""
================================================================================
Define structural matrices for each semi-analytical model (:mod:`panels.models`)
================================================================================

.. currentmodule:: panels.models

The modules herein contained are identified as follows:

    TYPE_THEORY_EQUATION_SUFIX

If one of these is not present in the module name, consider not applicable.
All models use the Rodrigues version of Legendre hierarchic polynomials,
largely applied by Bardell, as approximation functions.


TYPE refers to:

- cylshell - Cylindrical shells
- plate - Flat plates

THEORY refers to:

- clpt - Classical laminated plate theory
- fsdt - First-order shear deformation theory

EQUATION refers to which type of nonlinear equation is being used:

- donnell - kinematic equations using Donnell's equations

SUFIX used to indicate additional information

- field - module used to calculate field variables
- num - stiffness matrices integrated numerically


"""
from . import clpt_field
from . import cylshell_clpt_donnell, cylshell_clpt_donnell_num
from . import plate_clpt_donnell, plate_clpt_donnell_num
