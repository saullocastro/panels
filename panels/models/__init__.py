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
- sanders - kinematic equations using Sanders-Koiter's equations

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
  Sanders-Koiter kinematics, 3 DOFs ``u, v, w``. This is the default model
  of a :class:`.Shell` with a radius ``r``.

- ``'plate_fsdt_donnell'``: flat plates, first-order shear deformation
  theory (FSDT, Reissner-Mindlin) with von Karman non-linear kinematics, 5
  DOFs ``u, v, w, phix, phiy``. The transverse shear stiffness is controlled
  by ``Shell.fsdt_shear_correction``, ``'rohwer'`` by default.

- ``'plate_tsdt_donnell'``: flat plates, third-order shear deformation
  theory of Reddy with von Karman non-linear kinematics, 5 DOFs ``u, v, w,
  phix, phiy``, without shear correction.

- ``'cylshell_fsdt_donnell'``, ``'cylshell_tsdt_donnell'``: cylindrical
  shells, FSDT and TSDT with the Donnell kinematics, 5 DOFs ``u, v, w, phix,
  phiy``.

- ``'cylshell_fsdt_sanders'``, ``'cylshell_tsdt_sanders'``: cylindrical
  shells, FSDT and TSDT with the Sanders-Koiter kinematics, 5 DOFs ``u, v,
  w, phix, phiy``. For these models the fields ``phiy`` of
  :func:`.fsdt_tsdt_field.fuvw` and :func:`.fsdt_tsdt_field.fg` are the
  rotation of the normal ``phiy + v/r``.

The models with 5 DOFs are supported by :class:`.MultiDomain` with the
connections of :mod:`panels.multidomain.connections.kCsdt`, but not by
:class:`.StiffPanelBay`, whose stiffeners assume the 3 DOFs of the CLPT.

Notes
-----
For the cylinders, `y = r \theta` is the arc length along the
circumference, `w` is positive outwards, and the shallow shell assumption
`r + z \approx r` is adopted; for the plates `1/r = 0`. The models with 3
DOFs have the strains of the CLPT:

- Donnell kinematics, ``'plate_clpt_donnell'`` and
  ``'cylshell_clpt_donnell'``, which for the plates are the von Kármán
  strains:

  .. math::

      \varepsilon_{xx} = u_{,x} + \frac{1}{2} w_{,x}^2 \qquad
      \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} w_{,y}^2 \qquad
      \gamma_{xy} = u_{,y} + v_{,x} + w_{,x} w_{,y}

      \kappa_{xx} = -w_{,xx} \qquad
      \kappa_{yy} = -w_{,yy} \qquad
      \kappa_{xy} = -2 w_{,xy}

- Sanders-Koiter kinematics, ``'cylshell_clpt_sanders'``, Sanders (1959,
  1963) [1]_, [2]_. With `y = r \theta` the arc length and `\phi_x =
  -w_{,x}`, `\phi_y = -w_{,y} + v/r` the rotations of the normal:

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

The models with 5 DOFs have the rotations `\phi_x` and `\phi_y` as
independent fields, with the displacement field and the strains of
:mod:`panels.models.fsdt_tsdt_field`. The FSDT uses the shear corrected
transverse shear stiffness, with the method of Rohwer (1988) [4]_ by
default, see :class:`.Shell`. The TSDT of Reddy (1984) [3]_ needs no shear
correction, the transverse shear stresses vanishing at the faces of the
laminate, `z = d \pm h/2` with the offset `d`: the cubic term is written
about the mid-surface, `-c_1 [(z - d)^3 + d^3] (\phi_x + w_{,x})`, which
keeps `u` the displacement of the reference surface, see
:meth:`.Shell._get_lam_ABD`.
The cylinders with the Donnell kinematics are the plate models with the
hoop strain `w/r` and the curvature term of the piston theory, see
:meth:`.Shell.calc_kA`. With the Sanders-Koiter kinematics, the rotation of
the normal about `x` is `\phi_y + v/r`, which enters the displacement field,
the changes of curvature and the mass matrix, the twist includes the
rotation about the normal, and the non-linear terms use `\beta_y = w_{,y} -
v/r`, such that all strains vanish for any small rigid-body motion. No
Sanders-Koiter term appears in the third-order terms of the TSDT, and the
transverse shear strains are those of the Donnell kinematics.

The integrands of the matrices of each model are derived symbolically from
these kinematics. The modules without suffix integrate them in closed form
over the whole domain, with the integrals of Bardell's functions of
:mod:`panels.bardell`, which requires a constant constitutive matrix and
matrices that do not depend on the Ritz constants; the modules ``num``
apply the Gauss-Legendre quadrature, which also gives the non-linear
matrices, the internal force vector, a stiffness varying over the domain
and partial integration domains, see :meth:`.Shell.calc_kC`.

The mass matrices integrate the kinetic energy of the displacement field
through the thickness, with `z` measured from the reference surface and the
laminate spanning `-h/2 + d \le z \le h/2 + d`, `d` being the offset
``Shell.offset``, the convention of :func:`composites.laminated_plate` also
used by the stiffness matrices. The density enters only through its
moments `I_n = \int \rho z^n dz`, ``intrho``, ``intrhoz``, ...,
``intrhoz6`` of :class:`composites.Laminate`, `n = 0, 1, 2` for the CLPT
and the FSDT, and for the TSDT the moments about the mid-surface, `n = 0,
1, 2, 3, 4, 6`, see :meth:`.Shell._get_lam_mid`, which account exactly for
the density of each ply. The kernel ``fkM(shell, size, row0, col0)`` reads
them from the laminate, and ``fkM_num(shell, I_input, size, row0, col0, nx,
ny)`` receives them, with the thickness `h` of the TSDT, as one set for the
whole domain or one set per integration point, see :meth:`.Shell.calc_kM`.

References
----------
.. [1] Sanders, J. L., "An improved first-approximation theory for thin
   shells", NASA TR R-24, 1959.
   TODO(author): confirm this source (the title is added to the citation
   of tests/tests_shell/test_fsdt_tsdt_cylinders.py)
.. [2] Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
   Applied Mathematics, 21(1), 21-36, 1963.
   TODO(author): confirm this source
.. [3] Reddy, J. N., "A simple higher-order theory for laminated composite
   plates", J. Appl. Mech., 51, 745-752, 1984.
.. [4] Rohwer, K., "Improved transverse shear stiffness for layered finite
   elements", DFVLR-FB 88-32, 1988.
   TODO(author): confirm this source (cited in full in the composites
   package, not in panels)

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
