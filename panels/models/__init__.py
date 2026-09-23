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
- tsdt - Third-order shear deformation theory of Reddy

EQUATION refers to which type of nonlinear equation is being used:

- donnell - kinematic equations using Donnell's equations, which for flat
  plates are the von Karman equations
- sanders - kinematic equations using Sanders-Koiter's equations, which,
  differently from Donnell's, give zero strains for any rigid-body motion of
  a cylindrical shell

SUFIX used to indicate additional information

- field - module used to calculate field variables, ``clpt_field`` for the
  models with 3 degrees of freedom ``u, v, w``, and ``fsdt_tsdt_field`` for
  the models with 5 degrees of freedom ``u, v, w, phix, phiy``
- num - stiffness matrices integrated numerically


Available models
----------------

The available models, which are selected with ``Shell.model``, are:

- ``'plate_clpt_donnell'``: flat plates, classical laminated plate theory
  (CLPT) with von Karman non-linear kinematics, 3 DOFs ``u, v, w``.

- ``'cylshell_clpt_donnell'``: cylindrical shells, CLPT with the
  Donnell-Mushtari-Vlasov (shallow shell) kinematics, 3 DOFs ``u, v, w``.

- ``'cylshell_clpt_sanders'``: cylindrical shells, CLPT with the
  Sanders-Koiter kinematics (Sanders 1959, 1963), 3 DOFs ``u, v, w``. This is
  the default model of a :class:`.Shell` with a radius ``r``. With
  `y = r \theta` the arc length and `\phi_x = -w_{,x}`, `\phi_y = -w_{,y} +
  v/r` the rotations of the normal:

  .. math::

      \varepsilon_{xx} = u_{,x} + \frac{1}{2} \phi_x^2 \qquad
      \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \phi_y^2 \qquad
      \gamma_{xy} = u_{,y} + v_{,x} + \phi_x \phi_y

      \kappa_{xx} = -w_{,xx} \qquad
      \kappa_{yy} = -w_{,yy} + \frac{v_{,y}}{r} \qquad
      \kappa_{xy} = -2 w_{,xy} + \frac{3}{2} \frac{v_{,x}}{r}
                    - \frac{1}{2} \frac{u_{,y}}{r}

  Differently from the Donnell kinematics, all strains vanish for any
  rigid-body motion, which matters for long shells and for deformations of
  long wavelength, e.g. low circumferential wave numbers. The non-linear
  terms do not include the rotation about the normal, such that the
  geometric stiffness of the axial load is the same as for the Donnell
  kinematics. The rotation `\phi_y` also enters the mass matrix and the
  point constraints of the rotation about `x`.

- ``'plate_fsdt_donnell'``: flat plates, first-order shear deformation
  theory (FSDT, Reissner-Mindlin) with von Karman non-linear kinematics, 5
  DOFs ``u, v, w, phix, phiy``. The transverse shear stiffness is controlled
  by ``Shell.fsdt_shear_correction``, with `k = 5/6` by default.

- ``'plate_tsdt_donnell'``: flat plates, third-order shear deformation
  theory of Reddy (1984) with von Karman non-linear kinematics, 5 DOFs ``u,
  v, w, phix, phiy``. No shear correction is needed, the transverse shear
  stresses vanish at `z = \pm h/2`, which assumes the reference surface at
  mid-thickness, i.e. no laminate offset.

- ``'cylshell_fsdt_donnell'``, ``'cylshell_tsdt_donnell'``: cylindrical
  shells, FSDT and TSDT with the Donnell kinematics, 5 DOFs ``u, v, w, phix,
  phiy``, i.e. the plate models with the hoop strain `w/r` and the curvature
  term of the piston theory.

- ``'cylshell_fsdt_sanders'``, ``'cylshell_tsdt_sanders'``: cylindrical
  shells, FSDT and TSDT with the Sanders-Koiter kinematics, 5 DOFs ``u, v,
  w, phix, phiy``. The rotation of the normal about `x` is `\phi_y + v/r`,
  which enters the displacement field, the changes of curvature and the
  mass matrix, the twist includes the rotation about the normal, and the
  non-linear terms use `\beta_y = w_{,y} - v/r`, such that all strains
  vanish for any small rigid-body motion. No Sanders-Koiter term appears in
  the third-order terms of the TSDT, and the transverse shear strains are
  those of the Donnell kinematics. For these models the fields ``phiy`` of
  :func:`.fsdt_tsdt_field.fuvw` and :func:`.fsdt_tsdt_field.fg` are the
  rotation of the normal `\phi_y + v/r`.

The kinematic equations of each model, from which the integrands of the
matrices are generated, are in the ``theory/shells`` folder of the
repository. The models with 5 DOFs are supported by :class:`.MultiDomain`
with the connections of :mod:`panels.multidomain.connections.kCsdt`, but not
by :class:`.StiffPanelBay`, whose stiffeners assume the 3 DOFs of the CLPT.

"""
from . import clpt_field
from . import cylshell_clpt_donnell, cylshell_clpt_donnell_num
from . import cylshell_clpt_sanders, cylshell_clpt_sanders_num
from . import plate_clpt_donnell, plate_clpt_donnell_num
from . import fsdt_tsdt_field
from . import plate_fsdt_donnell, plate_fsdt_donnell_num
from . import plate_tsdt_donnell, plate_tsdt_donnell_num
from . import cylshell_fsdt_donnell, cylshell_fsdt_donnell_num
from . import cylshell_fsdt_sanders, cylshell_fsdt_sanders_num
from . import cylshell_tsdt_donnell, cylshell_tsdt_donnell_num
from . import cylshell_tsdt_sanders, cylshell_tsdt_sanders_num
