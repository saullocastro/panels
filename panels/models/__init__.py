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
  plates are the von Kármán equations
- sanders - kinematic equations using Sanders-Koiter's equations
- gl - Green-Lagrange membrane strains, complete in the displacement
  gradients
- tl - total Lagrangian: the Green-Lagrange membrane strains and the
  changes of curvature measured with the deformed normal (CLPT) or with the
  second-order director (FSDT)

SUFIX used to indicate additional information

- field - module used to calculate field variables, ``clpt_field`` for the
  models with 3 degrees of freedom ``u, v, w``, and ``fsdt_tsdt_field`` for
  the models with 5 degrees of freedom ``u, v, w, phix, phiy``
- num - stiffness matrices integrated numerically


The available models, which are selected with ``Shell.model``, are:

- ``'plate_clpt_donnell'``: flat plates, classical laminated plate theory
  (CLPT) with von Kármán non-linear kinematics, 3 DOFs ``u, v, w``.

- ``'plate_clpt_gl'``: flat plates, CLPT with the Green-Lagrange membrane
  strains, which vanish for any rigid rotation, and the changes of
  curvature of ``'plate_clpt_donnell'``, 3 DOFs ``u, v, w``. For assemblies
  of plates with large rotations, e.g. a wing box.

- ``'plate_clpt_tl'``: flat plates, total-Lagrangian Kirchhoff-Love: the
  Green-Lagrange membrane strains of ``'plate_clpt_gl'`` and the changes of
  curvature of the deformed normal, all invariant under rigid rotations,
  3 DOFs ``u, v, w``.

- ``'annular_clpt_donnell'``: annular sectors of flat plates, CLPT with
  the von Kármán strains in polar coordinates, 3 DOFs ``u, v, w``. ``a`` is
  the radial width, ``b`` the arc length of the outer edge and ``r`` the
  outer radius; `x = r - \rho` runs from the outer edge inwards and `y = r
  \theta` along the outer edge. The web of a ring frame is an application,
  with ``a`` the height of the web; it connects to a cylinder with
  ``'BFxcte'``, see :mod:`panels.models.annular_clpt_donnell_num`.

- ``'annular_clpt_gl'``: as ``'annular_clpt_donnell'``, with the complete
  Green-Lagrange membrane strains in polar coordinates, invariant under
  rigid rotations, 3 DOFs ``u, v, w``.

- ``'annular_clpt_tl'``: as ``'annular_clpt_gl'``, with the changes of
  curvature of the deformed normal, all strains invariant under rigid
  rotations, 3 DOFs ``u, v, w``.

- ``'annular_fsdt_donnell'``: annular sectors of flat plates, FSDT with the
  von Kármán strains in polar coordinates, 5 DOFs ``u, v, w, phix, phiy``,
  with the geometry and the local axes of ``'annular_clpt_donnell'``, see
  :mod:`panels.models.annular_fsdt_donnell_num`.

- ``'annular_fsdt_gl'``: as ``'annular_fsdt_donnell'``, with the complete
  Green-Lagrange membrane strains in polar coordinates, 5 DOFs ``u, v, w,
  phix, phiy``.

- ``'annular_fsdt_tl'``: as ``'annular_fsdt_gl'``, total-Lagrangian with the
  second-order director of ``'plate_fsdt_tl'``, 5 DOFs ``u, v, w, phix,
  phiy``.

- ``'annular_tsdt_donnell'``, ``'annular_tsdt_gl'``: annular sectors of
  flat plates, Reddy's third-order shear deformation theory (TSDT) with the
  von Kármán strains and with the Green-Lagrange membrane strains in polar
  coordinates, 5 DOFs ``u, v, w, phix, phiy``, the geometry of
  ``'annular_clpt_donnell'``, see
  :mod:`panels.models.annular_tsdt_donnell_num`.

- ``'annular_tsdt_tl'``: as ``'annular_tsdt_gl'``, total-Lagrangian with the
  second-order director of ``'annular_fsdt_tl'`` and the third-order term
  built from the director and the normal of the deformed plane, 5 DOFs
  ``u, v, w, phix, phiy``, see :mod:`panels.models.annular_tsdt_tl_num`.

- ``'cylshell_clpt_donnell'``: cylindrical shells, CLPT with the
  Donnell-Mushtari-Vlasov (shallow shell) kinematics, 3 DOFs ``u, v, w``.

- ``'cylshell_clpt_sanders'``: cylindrical shells, CLPT with the
  Sanders-Koiter kinematics, 3 DOFs ``u, v, w``. This is the default model
  of a :class:`.Shell` with a radius ``r``.

- ``'cylshell_clpt_gl'``: cylindrical shells, CLPT with the Green-Lagrange
  membrane strains, which vanish for any rigid motion, and the changes of
  curvature of ``'cylshell_clpt_sanders'``, 3 DOFs ``u, v, w``. For the
  bending of long cylinders as beams (column buckling of tubes, low
  circumferential wave numbers) and for curved panels loaded in their own
  plane.

- ``'cylshell_clpt_tl'``: cylindrical shells, total-Lagrangian
  Kirchhoff-Love: the Green-Lagrange membrane strains of
  ``'cylshell_clpt_gl'`` and the changes of curvature of the deformed
  normal, whose linear part is that of ``'cylshell_clpt_sanders'``, all
  invariant under rigid rotations, 3 DOFs ``u, v, w``. For curved panels
  and cylinders bent through large rotations, e.g. a panel whose cap
  inverts under a point load.

- ``'plate_fsdt_donnell'``: flat plates, first-order shear deformation
  theory (FSDT, Reissner-Mindlin) with von Kármán non-linear kinematics, 5
  DOFs ``u, v, w, phix, phiy``. The transverse shear stiffness is controlled
  by ``Shell.fsdt_shear_correction``, ``'rohwer'`` by default.

- ``'plate_tsdt_donnell'``: flat plates, third-order shear deformation
  theory of Reddy with von Kármán non-linear kinematics, 5 DOFs ``u, v, w,
  phix, phiy``, without shear correction.

- ``'cylshell_fsdt_donnell'``, ``'cylshell_tsdt_donnell'``: cylindrical
  shells, FSDT and TSDT with the Donnell kinematics, 5 DOFs ``u, v, w, phix,
  phiy``.

- ``'cylshell_fsdt_sanders'``, ``'cylshell_tsdt_sanders'``: cylindrical
  shells, FSDT and TSDT with the Sanders-Koiter kinematics, 5 DOFs ``u, v,
  w, phix, phiy``. For these models the fields ``phiy`` of
  :func:`.fsdt_tsdt_field.fuvw` and :func:`.fsdt_tsdt_field.fg` are the
  rotation of the normal ``phiy + v/r``.

- ``'plate_fsdt_gl'``, ``'plate_tsdt_gl'``: flat plates, FSDT and TSDT with
  the Green-Lagrange membrane strains of ``'plate_clpt_gl'`` and the linear
  bending and transverse shear strains of ``'plate_fsdt_donnell'`` and
  ``'plate_tsdt_donnell'``, 5 DOFs ``u, v, w, phix, phiy``. A finite rigid
  rotation, with constant ``phix = -w,x`` and ``phiy = -w,y``, stores no
  energy.

- ``'cylshell_fsdt_gl'``, ``'cylshell_tsdt_gl'``: cylindrical shells, FSDT
  and TSDT with the Green-Lagrange membrane strains of
  ``'cylshell_clpt_gl'`` and the linear bending and transverse shear strains
  of ``'cylshell_fsdt_sanders'`` and ``'cylshell_tsdt_sanders'``, 5 DOFs
  ``u, v, w, phix, phiy``; the fields ``phiy`` of
  :func:`.fsdt_tsdt_field.fuvw` and :func:`.fsdt_tsdt_field.fg` are the
  rotation of the normal ``phiy + v/r``, as for the Sanders models.

- ``'plate_fsdt_tl'``, ``'cylshell_fsdt_tl'``: flat plates and cylindrical
  shells, total-Lagrangian FSDT: the Green-Lagrange membrane strains of
  ``'plate_fsdt_gl'`` and ``'cylshell_fsdt_gl'``, and the curvatures and
  transverse shear strains of the second-order director of the moderate
  rotation theory [6]_, [7]_, cubic in the derivatives of the displacements
  and rotations, 5 DOFs ``u, v, w, phix, phiy``; the linear part is that of
  ``'plate_fsdt_donnell'`` and ``'cylshell_fsdt_sanders'``, and the fields
  ``phiy`` of the cylinder are the rotation of the normal ``phiy + v/r``.

- ``'plate_tsdt_tl'``, ``'cylshell_tsdt_tl'``: flat plates and cylindrical
  shells, total-Lagrangian TSDT: the strains of ``'plate_fsdt_tl'`` and
  ``'cylshell_fsdt_tl'`` and third-order strains built with the
  second-order director and the normal of the deformed mid-surface, 5 DOFs
  ``u, v, w, phix, phiy``; the linear part is that of
  ``'plate_tsdt_donnell'`` and ``'cylshell_tsdt_sanders'``, and the
  transverse shear strains vanish at the faces for any deformation.

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

- Green-Lagrange membrane strains, ``'plate_clpt_gl'``, with the changes
  of curvature above:

  .. math::

      \varepsilon_{xx} = u_{,x} + \frac{1}{2} (u_{,x}^2 + v_{,x}^2 + w_{,x}^2) \qquad
      \varepsilon_{yy} = v_{,y} + \frac{1}{2} (u_{,y}^2 + v_{,y}^2 + w_{,y}^2)

      \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} + w_{,x} w_{,y}

  The membrane strains vanish for any rigid rotation, about the normal as
  well as about an in-plane axis, such that plates joined at an angle, whose
  out-of-plane rotations are in-plane rotations of their neighbours, measure
  the same stretch. The geometric stiffness has ``u-u`` and ``v-v`` blocks
  besides the ``w-w`` block, and the linear kernels are those of
  ``'plate_clpt_donnell'``. The changes of curvature are measured in the
  fixed frame of the plate, which limits the large rotations to those
  carried by the membrane strains.

- Total-Lagrangian Kirchhoff-Love, ``'plate_clpt_tl'``: the membrane strains
  above and the changes of curvature of the deformed mid-surface
  `\boldsymbol{X} = (x, y, 0) + (u, v, w)`, with its area vector
  `\boldsymbol{a} = \boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y}`:

  .. math::

      \kappa_{xx} = -\boldsymbol{a} \cdot \boldsymbol{u}_{,xx} \qquad
      \kappa_{yy} = -\boldsymbol{a} \cdot \boldsymbol{u}_{,yy} \qquad
      \kappa_{xy} = -2 \boldsymbol{a} \cdot \boldsymbol{u}_{,xy}

  cubic in the displacements, `|\boldsymbol{a}| = 1` for rigid motions and
  `1 + \varepsilon_{xx} + \varepsilon_{yy}` to first order otherwise. A
  bending deformation keeps its energy through any rigid rotation, which
  ``'plate_clpt_gl'`` multiplies by `\cos^2\theta`. The default
  integration is exact, with `3p` points along a direction instead of `2p +
  1`, see :func:`panels.shell.gauss_points`, about twice the cost of
  ``'plate_clpt_gl'``; ``nx = gauss_points(m)``, ``ny = gauss_points(n)``
  give the points of ``'plate_clpt_gl'``, not exact but with the same
  results in practice at about half the cost. The analytical kernels are those of ``'plate_clpt_gl'``.

- Sanders-Koiter kinematics, ``'cylshell_clpt_sanders'``, Sanders (1963)
  [1]_. With `y = r \theta` the arc length and `\phi_x =
  -w_{,x}`, `\phi_y = -w_{,y} + v/r` the rotations of the normal:

  .. math::

      \varepsilon_{xx} = u_{,x} + \frac{1}{2} \phi_x^2 \qquad
      \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \phi_y^2 \qquad
      \gamma_{xy} = u_{,y} + v_{,x} + \phi_x \phi_y

      \kappa_{xx} = -w_{,xx} \qquad
      \kappa_{yy} = -w_{,yy} + \frac{v_{,y}}{r} \qquad
      \kappa_{xy} = -2 w_{,xy} + \frac{3}{2} \frac{v_{,x}}{r}
                    - \frac{1}{2} \frac{u_{,y}}{r}

  The changes of curvature are those of the Sanders-Koiter theory as
  written by Amabili (2008) [4]_, Sec. 1.3.8, Eqs. (1.136)-(1.138).
  Differently from the Donnell kinematics, all strains vanish for any
  rigid-body motion, which matters for long shells and for deformations of
  long wavelength, e.g. low circumferential wave numbers. The non-linear
  terms do not include the rotation about the normal, `\omega_3 =
  \frac{1}{2}(v_{,x} - u_{,y})`, such that the geometric stiffness of the
  axial load is the same as for the Donnell kinematics. This is a
  simplification of the Sanders-Koiter non-linear theory for small strains
  and moderately small rotations [1]_, whose mid-surface strains contain,
  in addition, the term `\frac{1}{2}\omega_3^2 = \frac{1}{8}(v_{,x} -
  u_{,y})^2` in both `\varepsilon_{xx}` and `\varepsilon_{yy}` [4]_, Sec.
  1.3.8, Eqs. (1.133)-(1.134); it amounts to assuming that the rotation
  about the normal is small compared with the other two rotations. The
  rotation `\phi_y` also enters the mass matrix and the point constraints
  of the rotation about `x`.

- Green-Lagrange membrane strains of the cylinder, ``'cylshell_clpt_gl'``,
  with the changes of curvature of ``'cylshell_clpt_sanders'``: the
  complete strains `(\boldsymbol{F}^\top \boldsymbol{F} -
  \boldsymbol{I})/2` of the mid-surface, `\boldsymbol{F}` having the
  columns `(1 + u_{,x}, v_{,x}, w_{,x})` and `(u_{,y}, 1 + v_{,y} + w/r,
  w_{,y} - v/r)` in the local basis:

  .. math::

      \varepsilon_{xx} = u_{,x} + \frac{1}{2} (u_{,x}^2 + v_{,x}^2 + w_{,x}^2)

      \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \left(u_{,y}^2
      + \left(v_{,y} + \frac{w}{r}\right)^2 + \phi_y^2\right)

      \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} \left(v_{,y} +
      \frac{w}{r}\right) + \phi_x \phi_y

  the membrane strains of the Flügge-Lur'e-Byrne non-linear theory as
  given by Amabili (2008) [4]_, Sec. 1.3.6, Eqs. (1.118)-(1.120). They vanish for any rigid motion, of any
  angle, and contain the rotation about the normal that
  ``'cylshell_clpt_sanders'`` omits: under an axial load `N_{xx}` the
  geometric stiffness is `N_{xx} (u_{,x}^2 + v_{,x}^2 + w_{,x}^2)` instead
  of `N_{xx} w_{,x}^2`. For the bending of a long tube as a beam, `w =
  W(x) \cos\theta`, `v = -W(x) \sin\theta`, the term `v_{,x}^2` doubles the
  work of the axial load, such that ``'cylshell_clpt_sanders'`` gives about
  twice the Euler load and ``'cylshell_clpt_gl'`` the Euler load. For
  shell-like modes the difference decreases as the circumferential waves
  shorten: for axially compressed panels with `R/h = 100` and `L/R = 2`
  supported by shear diaphragms, ``'cylshell_clpt_gl'`` is 4.2 % below
  ``'cylshell_clpt_sanders'`` for a panel of 180 degrees and 1.0 % below
  for a panel of 40 degrees. The linear kernels are those of ``'cylshell_clpt_sanders'``, and the changes
  of curvature are linear, measured in the fixed local basis, exact for
  small rigid rotations only.

- Total-Lagrangian Kirchhoff-Love cylinder, ``'cylshell_clpt_tl'``: the
  membrane strains of ``'cylshell_clpt_gl'`` and the changes of curvature
  of the deformed mid-surface `\boldsymbol{X}`, with its area vector
  `\boldsymbol{a} = \boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y}`:

  .. math::

      \kappa_{xx} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,xx} \qquad
      \kappa_{yy} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,yy} - \frac{1}{r}
      - \frac{\varepsilon_{xx} + 2 \varepsilon_{yy}}{r} \qquad
      \kappa_{xy} = -2 \boldsymbol{a} \cdot \boldsymbol{X}_{,xy} -
      \frac{\gamma_{xy}}{2 r}

  with, in the local basis, `\boldsymbol{X}_{,xx} = (u_{,xx}, v_{,xx},
  w_{,xx})`, `\boldsymbol{X}_{,xy} = (u_{,xy}, v_{,xy} + w_{,x}/r, w_{,xy}
  - v_{,x}/r)` and `\boldsymbol{X}_{,yy} = (u_{,yy}, v_{,yy} + 2 w_{,y}/r -
  v/r^2, w_{,yy} - 2 v_{,y}/r - w/r^2 - 1/r)`. The first terms are the
  change of the second fundamental form `\boldsymbol{n} \cdot
  \boldsymbol{X}_{,\alpha\beta}` [5]_, Sec. 3.16, p. 182, times the area
  stretch `|\boldsymbol{a}| = 1 + \varepsilon_{xx} + \varepsilon_{yy} +
  \ldots`. Its linear part differs from Koiter's linear changes of
  curvature [5]_, Eq. (3.16.8), p. 183, Eq. (3.17.3), p. 202, by
  `\varepsilon_{yy}/r` in `\kappa_{yy}` and `\gamma_{xy}/(2r)` in
  `\kappa_{xy}`; the terms `-\varepsilon_{yy}/r` and `-\gamma_{xy}/(2r)`
  remove them and make the linear part that of
  ``'cylshell_clpt_sanders'``, a change of the order of the membrane strain
  divided by the radius, which leaves the accuracy of the classical theory
  of thin shells unchanged [5]_, p. 202. All six strains
  are invariant under any rigid rotation of any deformed state, and cubic
  in the displacements. Up to the quadratic terms, the curvatures differ
  from the non-linear changes of curvature of Novozhilov's theory as given
  by Amabili [4]_, Sec. 1.3.7, Eqs. (1.130)-(1.132), by terms of the order
  of the membrane strain divided by the radius or times the curvature, and
  by terms of the rotation about the normal, which those equations do not
  treat exactly. The analytical kernels are those of
  ``'cylshell_clpt_gl'``; the default integration is exact, with `3p + 1`
  points along each direction, see :func:`panels.shell.gauss_points`.

The models with 5 DOFs have the rotations `\phi_x` and `\phi_y` as
independent fields, with the displacement field and the strains of
:mod:`panels.models.fsdt_tsdt_field`. The FSDT uses the shear corrected
transverse shear stiffness of Rohwer (1988) [3]_, Sec. 3.4, Eq. (43), by
default, see :class:`.Shell`. The TSDT of Reddy (1984) [2]_ needs no shear
correction, the transverse shear stresses vanishing at the faces of the
laminate, `z = d \pm h/2` with the offset `d`: the cubic term is written
about the mid-surface, `-c_1 [(z - d)^3 + d^3] (\phi_x + w_{,x})`, which
keeps `u` the displacement of the reference surface, see
:meth:`.Shell._get_lam_ABD`. The conditions `p(0) = 0` and `p'(d \pm h/2)
= -1` on the cubic `p(z)` of `u(z) = u + z \phi_x + p(z)(\phi_x +
w_{,x})` leave one free parameter, which only selects where the rotation
variable is measured: `\phi_x` is the rotation at the mid-surface, as in
Reddy's field for `d = 0`, since at the reference surface the rotation is
not an independent variable when `z = 0` is a face of the laminate,
`d = \pm h/2`. Panels with different offsets are connected at their common
reference surface, with the rotations `\theta_x = \phi_x - (4 d^2/h^2)
(\phi_x + w_{,x})` at `z = 0`, see
:mod:`panels.multidomain.connections.kCsdt`.
The cylinders with the Donnell kinematics are the plate models with the
hoop strain `w/r` and the curvature term of the piston theory, see
:meth:`.Shell.calc_kA`. With the Sanders-Koiter kinematics, the rotation of
the normal about `x` is `\phi_y + v/r`, which enters the displacement field,
the changes of curvature and the mass matrix, the twist includes the
rotation about the normal, and the non-linear terms use `\beta_y = -w_{,y} +
v/r`, such that all strains vanish for any small rigid-body motion. No
Sanders-Koiter term appears in the third-order terms of the TSDT, and the
transverse shear strains are those of the Donnell kinematics.

The total-Lagrangian FSDT models, ``'plate_fsdt_tl'`` and
``'cylshell_fsdt_tl'``, take the displacement field of the moderate
rotation theory of Schmidt and Reddy [6]_, Eq. (16), linear through the
thickness, with the transverse normal strain of Palmerio et al. [7]_, Eq.
(20), set to zero: the deformed position of the point at `z` is
`\mathbf{X} + z \mathbf{d}`, with the second-order director `\mathbf{d}
= (\psi_x, \psi_y, 1 - (\psi_x^2 + \psi_y^2)/2)`, `\psi_x = \phi_x`,
`\psi_y = \phi_y + v/r`, in the local basis. The curvatures and transverse
shear strains are those of :func:`.fsdt_tsdt_field.is_tl`, the
coefficients of `z` and the transverse components of the Green-Lagrange
strains of this field, with the terms `-\varepsilon_{yy}/r` and
`-\gamma_{xy}/(2 r)` of ``'cylshell_clpt_tl'`` that make the linear part
that of ``'cylshell_fsdt_sanders'``; to the order `\epsilon^3` of the
moderate rotation theory (strains of order `\epsilon^2`, rotations of the
normal of order `\epsilon`) they are the strains of [7]_, Eq. (20).

The total-Lagrangian TSDT models, ``'plate_tsdt_tl'`` and
``'cylshell_tsdt_tl'``, write the cubic term of Reddy's field [2]_ with the
director and the normal of the deformed mid-surface, not normalized,
`\mathbf{N} = \mathbf{X}_{,x} \times \mathbf{X}_{,y}`: the deformed
position of the point at `z` is `\mathbf{X} + z \mathbf{d} + z^3
\mathbf{q}`, `\mathbf{q} = -c_1 (\mathbf{d} - \mathbf{N})`, whose linear
part is Reddy's field. The third-order strains, those of
:func:`.fsdt_tsdt_field.is_tl`, are the coefficients of `z^3` of the
in-plane and of `z^2` of the transverse shear Green-Lagrange strains of
this field, with, for the cylinder, the term `-c_1 (\varepsilon_{xx} +
\varepsilon_{yy})/r` that makes the linear part that of
``'cylshell_tsdt_sanders'``; the transverse shear strain of the second
order is `-3 c_1` times that of the director, so that the faces stay free
of transverse shear for any deformation.

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

The annular sector, ``'annular_clpt_donnell'``, has the von Kármán
strains of circular plates [4]_, Sec. 1.4.2, Eqs. (1.175)-(1.180), with
`x = r - \rho`, `y = r \theta`, `s = r/\rho` and the local axes
`\boldsymbol{e}_x = -\boldsymbol{e}_\rho`, `\boldsymbol{e}_y =
\boldsymbol{e}_\theta`:

.. math::

    \varepsilon_{yy} = s v_{,y} - \frac{u}{\rho} + \frac{1}{2} s^2 w_{,y}^2
    \qquad
    \gamma_{xy} = s u_{,y} + v_{,x} + \frac{v}{\rho} + s w_{,x} w_{,y}

    \kappa_{yy} = \frac{w_{,x}}{\rho} - s^2 w_{,yy} \qquad
    \kappa_{xy} = -2 s \left(w_{,xy} + \frac{w_{,y}}{\rho}\right)

and `\varepsilon_{xx}`, `\kappa_{xx}` of the flat plate. The coefficients
in `1/\rho` make the integrands rational in `x`, and the matrices of the
whole domain are also integrated numerically, see
:mod:`panels.models.annular_clpt_donnell_num`. ``'annular_clpt_gl'`` has the
Green-Lagrange membrane strains of [4]_, Sec. 1.4.1, Eqs. (1.166)-(1.168),
and ``'annular_clpt_tl'`` also the changes of curvature of the deformed
normal, see :mod:`panels.models.annular_clpt_gl_num` and
:mod:`panels.models.annular_clpt_tl_num`. The offset ``Shell.offset``
enters the three through ``Shell.ABD`` and the moments of the density.

The annular sectors of the first-order shear deformation theory,
``'annular_fsdt_donnell'``, have the membrane strains of
``'annular_clpt_donnell'`` and the changes of curvature and transverse
shear strains of the rotations `\phi_x`, `\phi_y` in polar coordinates
[8]_, Eqs. (2) and (6):

.. math::

    \kappa_{yy} = s \phi_{y,y} - \frac{\phi_x}{\rho} \qquad
    \kappa_{xy} = s \phi_{x,y} + \phi_{y,x} + \frac{\phi_y}{\rho} \qquad
    \gamma_{yz} = \phi_y + s w_{,y}

and `\kappa_{xx}`, `\gamma_{xz}` of the flat plate. ``'annular_fsdt_gl'``
has the Green-Lagrange membrane strains of ``'annular_clpt_gl'``, and
``'annular_fsdt_tl'`` the strains of the second-order director of
``'plate_fsdt_tl'`` [6]_, [7]_ in polar coordinates, see
:mod:`panels.models.annular_fsdt_donnell_num`,
:mod:`panels.models.annular_fsdt_gl_num` and
:mod:`panels.models.annular_fsdt_tl_num`.

``'annular_tsdt_donnell'`` and ``'annular_tsdt_gl'`` have the displacement
field of the TSDT [2]_ written in the local axes of the annular sector,
with the derivative `s w_{,y}` of `w` along the arc in place of `w_{,y}`.
The plane being flat, the in-plane strains at `z` are the linear membrane
strains in polar coordinates of the in-plane displacements `u(z)`, `v(z)`,
plus the non-linear membrane terms of ``'annular_clpt_donnell'`` or
``'annular_clpt_gl'``; the offset enters through ``Shell.ABD`` of the TSDT
and the moments of the density about the mid-surface, see
:mod:`panels.models.annular_tsdt_donnell_num` and
:mod:`panels.models.annular_tsdt_gl_num`.

``'annular_tsdt_tl'`` writes the TSDT field for finite rotations, the
deformed position of a point being `\boldsymbol{X} + z \boldsymbol{d} +
p(z) \boldsymbol{q}`, `p(z) = (z - d)^3 + d^3`, with the second-order
director `\boldsymbol{d}` of ``'annular_fsdt_tl'`` [6]_, [7]_ and
`\boldsymbol{q} = -c_1 (\boldsymbol{d} - \boldsymbol{N})`, `\boldsymbol{N}`
the normal of the deformed plane, not normalized, whose linear part is the
vector `(\phi_x + w_{,x}, \phi_y + s w_{,y})` of the TSDT; the strains are
the coefficients of `1`, `z`, `p(z)` of the Green-Lagrange strains and the
transverse shear strains vanish at both faces for any deformation, see
:mod:`panels.models.annular_tsdt_tl_num`.

References
----------
.. [1] Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
   Applied Mathematics, 21(1), 21-36, 1963.
   https://doi.org/10.1090/qam/147023
.. [2] Reddy, J. N., "A simple higher-order theory for laminated composite
   plates", J. Appl. Mech., 51, 745-752, 1984.
.. [3] Rohwer, K., "Improved transverse shear stiffnesses for layered
   finite elements", DFVLR-FB 88-32, Deutsche Forschungs- und
   Versuchsanstalt für Luft- und Raumfahrt (DFVLR), Institut für
   Strukturmechanik, Braunschweig, 1988.
.. [4] Amabili, M., "Nonlinear Vibrations and Stability of Shells and
   Plates", Cambridge University Press, 2008.
   https://doi.org/10.1017/CBO9780511619694
.. [5] van der Heijden, A. M. A. (ed.), "W. T. Koiter's Elastic Stability
   of Solids and Structures", Cambridge University Press, 2008.
   https://doi.org/10.1017/CBO9780511546174
.. [6] Schmidt, R., and Reddy, J. N., "A Refined Small Strain and Moderate
   Rotation Theory of Elastic Anisotropic Shells", Journal of Applied
   Mechanics, Vol. 55, No. 3, 1988, pp. 611-617.
   https://doi.org/10.1115/1.3125837
.. [7] Palmerio, A. F., Reddy, J. N., and Schmidt, R., "On a moderate
   rotation theory of laminated anisotropic shells - Part 1. Theory",
   International Journal of Non-Linear Mechanics, Vol. 25, No. 6, 1990,
   pp. 687-700. https://doi.org/10.1016/0020-7462(90)90007-V
.. [8] Bagheri, H., Kiani, Y., and Eslami, M. R., "Buckling of moderately
   thick annular plates subjected to torque", Archive of Mechanical
   Engineering, Vol. 66, No. 2, 2019, pp. 209-227.
   https://doi.org/10.24425/ame.2019.128445

"""
from . import clpt_field
from . import cylshell_clpt_donnell, cylshell_clpt_donnell_num
from . import cylshell_clpt_sanders, cylshell_clpt_sanders_num
from . import cylshell_clpt_gl, cylshell_clpt_gl_num
from . import cylshell_clpt_tl_num
from . import plate_clpt_donnell, plate_clpt_donnell_num
from . import plate_clpt_gl, plate_clpt_gl_num
from . import plate_clpt_tl_num
from . import annular_clpt_donnell_num, annular_clpt_donnell
from . import annular_clpt_gl_num, annular_clpt_gl
from . import annular_clpt_tl_num, annular_clpt_tl
from . import annular_fsdt_donnell_num, annular_fsdt_donnell
from . import annular_fsdt_gl_num, annular_fsdt_gl
from . import annular_fsdt_tl_num, annular_fsdt_tl
from . import annular_tsdt_donnell_num, annular_tsdt_donnell
from . import annular_tsdt_gl_num, annular_tsdt_gl
from . import annular_tsdt_tl_num, annular_tsdt_tl
from . import fsdt_tsdt_field
from . import plate_fsdt_donnell, plate_fsdt_donnell_num
from . import plate_tsdt_donnell, plate_tsdt_donnell_num
from . import cylshell_fsdt_donnell, cylshell_fsdt_donnell_num
from . import cylshell_fsdt_sanders, cylshell_fsdt_sanders_num
from . import cylshell_tsdt_donnell, cylshell_tsdt_donnell_num
from . import cylshell_tsdt_sanders, cylshell_tsdt_sanders_num
from . import plate_fsdt_gl, plate_fsdt_gl_num
from . import plate_tsdt_gl, plate_tsdt_gl_num
from . import cylshell_fsdt_gl, cylshell_fsdt_gl_num
from . import plate_fsdt_tl_num
from . import cylshell_fsdt_tl_num
from . import plate_tsdt_tl_num
from . import cylshell_tsdt_tl_num
from . import cylshell_tsdt_gl, cylshell_tsdt_gl_num
