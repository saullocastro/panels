.. _follower_pressure:

Follower pressure loads: theory
===============================

A pressure created by a gas or a liquid acts normal to the current surface
and on its current area. With :meth:`.Shell.add_pressure_load` and
``follower=True`` the pressure `p` follows the deformed mid-surface of the
shell, as opposed to the default dead load, ``follower=False``, which keeps
the direction and the area of the undeformed mid-surface. The pressure
`p(x, y)` is a function of the undeformed coordinates, i.e. a *body-attached
follower load* in the classification of Schweizerhof and Ramm
[schweizerhof1984Pressure]_. It is the load of the hydrostatic buckling of
rings and cylinders, see e.g. [timoshenko1961Stability]_,
[brush1975Buckling]_ and [simitses1976Stability]_, and of the elasticity
solutions of thick orthotropic and sandwich cylinders under external
pressure [kardomateas1993Orthotropic]_, [kardomateas1995Orthotropic]_,
[kardomateas2003Sandwich]_, [han2004SandwichCylinder]_,
[kardomateas2005Sandwich]_, in which the pressure "remains normal to the
deflected surface during the buckling process".

This section derives the force vector of the follower pressure, its
contribution to the internal force vector and its load stiffness matrix
:meth:`.Shell.calc_kCfollower`, and discusses when this matrix is symmetric.
The kernels ``calc_fext_follower`` and ``fkCfollower_num`` of every model
are generated with SymPy from this derivation, see
:ref:`follower_pressure_implementation`.


Virtual work of the pressure
----------------------------

Let `x` and `y` be the coordinates of the undeformed mid-surface, `y` being
the arc length for the cylindrical shells, and `\{e_x, e_y, e_z\}` the local
basis of the undeformed mid-surface, with `e_z` the normal, positive outwards
for the cylinders of radius `r`, such that `e_{y,y} = -e_z/r` and
`e_{z,y} = e_y/r`. The displacement of the mid-surface is
`\boldsymbol{u} = u e_x + v e_y + w e_z`, and the position of the deformed
mid-surface is `\boldsymbol{X} = \boldsymbol{X}_0 + \boldsymbol{u}`. A
pressure `p`, positive along `+e_z` as the force ``fz`` of
:meth:`.Shell.add_point_load` (an external
pressure on a cylinder is negative), acting normal to the deformed
mid-surface and on its deformed area, does the virtual work:

.. math::
    :label: fp-virtual-work

    \delta W_p = \int_{A} p \, \boldsymbol{n} \cdot \delta\boldsymbol{u}
                 \, dA = \int_{A_0} p \, \boldsymbol{a} \cdot
                 \delta\boldsymbol{u} \, dx \, dy,
    \qquad
    \boldsymbol{a} = \boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y}

where `\boldsymbol{a}` is the area vector of the deformed mid-surface per
unit undeformed area, `\boldsymbol{n} \, dA = \boldsymbol{a} \, dx \, dy`.
For the models
based on the shear deformation theories, FSDT and TSDT, the pressure acts on
the mid-surface, such that only `u`, `v` and `w` enter Eq.
:eq:`fp-virtual-work`, never the rotations `\phi_x` and `\phi_y`.

In the local basis:

.. math::
    :label: fp-tangents

    \boldsymbol{X}_{,x} = \begin{Bmatrix} 1 + u_{,x} \\ v_{,x} \\
                          w_{,x} \end{Bmatrix},
    \qquad
    \boldsymbol{X}_{,y} = \begin{Bmatrix} u_{,y} \\
                          1 + v_{,y} + k_w w \\
                          w_{,y} - k_v v \end{Bmatrix}

with the factors of the kinematics of each model:

========================== ========= =========
kinematics                  `k_w`     `k_v`
========================== ========= =========
plate                       `0`       `0`
cylinder, Donnell           `1/r`     `0`
cylinder, Sanders           `1/r`     `1/r`
========================== ========= =========

For the plates and the Sanders cylinders Eq. :eq:`fp-tangents` is the exact
geometry, and `w_{,y} - v/r` is the rotation of the Sanders kinematics,
`\phi_y = -w_{,y} + v/r`, see :mod:`panels.models`. The Donnell cylinders
neglect `v/r` in the rotation, as their strains do. The area vector
`\boldsymbol{a} = \boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y}` is
bilinear in the displacement gradients:

.. math::
    :label: fp-area-vector

    \boldsymbol{a} = \boldsymbol{a}_0 + \boldsymbol{a}_1 + \boldsymbol{a}_2,
    \qquad
    \boldsymbol{a}_0 = \begin{Bmatrix} 0 \\ 0 \\ 1 \end{Bmatrix},
    \qquad
    \boldsymbol{a}_1 = \begin{Bmatrix} -w_{,x} \\ -w_{,y} + k_v v \\
                   u_{,x} + v_{,y} + k_w w \end{Bmatrix},

.. math::
    :label: fp-area-vector-2

    \boldsymbol{a}_2 = \begin{Bmatrix}
        v_{,x} (w_{,y} - k_v v) - w_{,x} (v_{,y} + k_w w) \\
        w_{,x} u_{,y} - u_{,x} (w_{,y} - k_v v) \\
        u_{,x} (v_{,y} + k_w w) - v_{,x} u_{,y}
    \end{Bmatrix}

The dead part `\boldsymbol{a}_0` is the dead load. The first-order part
`\boldsymbol{a}_1` contains the rotation of the normal, `(-w_{,x}, -w_{,y} +
k_v v)`, and the change of area, `u_{,x} + v_{,y} + k_w w`, which is the sum
of the linear membrane strains `\varepsilon_{xx} + \varepsilon_{yy}`.


Truncation consistent with the strains
--------------------------------------

The strains of all models are of moderate rotations, von Kármán, Donnell or
Sanders, i.e. with the in-plane gradients `u_{,x} \sim \varepsilon` and the
rotations `w_{,x} \sim \varepsilon^{1/2}`. The virtual displacements are
ordered alike, `\delta u \sim \varepsilon^{1/2} \delta w`, such that the
integrand of Eq. :eq:`fp-virtual-work`, `p (a_z \delta w + a_x \delta u +
a_y \delta v)`, has:

- from `\boldsymbol{a}_0 + \boldsymbol{a}_1`: the terms `p \, \delta w` and
  `p \, O(\varepsilon) \, \delta w`;
- from `\boldsymbol{a}_2`: the terms `p \, O(\varepsilon^2) \, \delta w`, of
  the order of the terms neglected by the strains themselves.

The consistent truncation is therefore the first-order area vector,
``follower=True`` or ``'linear'``:

.. math::
    :label: fp-truncation

    \boldsymbol{a} \approx \boldsymbol{a}_0 + \boldsymbol{a}_1

With ``follower='quadratic'`` the complete area vector of Eq.
:eq:`fp-area-vector` is used instead, which is useful to check the
sensitivity of a result to the truncation.


Force vector
------------

With the approximation of the displacements by the Ritz constants
`\{\bar{c}\}`, see :ref:`theory_func_bardell`,

.. math::
    :label: fp-ritz

    u = \boldsymbol{S}^u \{\bar{c}\}, \qquad v = \boldsymbol{S}^v
    \{\bar{c}\}, \qquad w = \boldsymbol{S}^w \{\bar{c}\}

where `\boldsymbol{S}^u`, `\boldsymbol{S}^v` and `\boldsymbol{S}^w` are the
rows of the approximation functions of each field, null at the positions of
the other degrees of freedom, Eq. :eq:`fp-virtual-work` gives the force
vector of the follower pressure:

.. math::
    :label: fp-force

    \{F_p(\bar{c})\} = \int_{A_0} p \left( (\boldsymbol{S}^u)^\top a_x +
    (\boldsymbol{S}^v)^\top a_y + (\boldsymbol{S}^w)^\top a_z \right) dx \,
    dy

where `\boldsymbol{a}` is evaluated at `\{\bar{c}\}` with Eq.
:eq:`fp-truncation`, or Eq. :eq:`fp-area-vector` for ``'quadratic'``. In
the undeformed state, `\boldsymbol{a} = \boldsymbol{a}_0` and the follower
load equals the dead load:

.. math::
    :label: fp-force-reference

    \{F_p(0)\} = \int_{A_0} p \, (\boldsymbol{S}^w)^\top \, dx \, dy

which is the reference force vector returned by :meth:`.Shell.calc_fext`,
see :func:`.shell_fext`. For the first-order area vector `\{F_p\}` is affine
in `\{\bar{c}\}`:

.. math::
    :label: fp-force-affine

    \{F_p(\bar{c})\} = \{F_p(0)\} - [K_F] \{\bar{c}\}

with the load stiffness matrix `[K_F]` of Eq. :eq:`fp-jacobian`.


Load stiffness matrix
---------------------

The load stiffness matrix, returned by :meth:`.Shell.calc_kCfollower`, is
the Jacobian of the force vector with the sign of a stiffness, such that
all the stiffness matrices are additive, see Eq. :eq:`fp-tangent`:

.. math::
    :label: fp-jacobian

    [K_F] = - \frac{\partial \{F_p\}}{\partial \{\bar{c}\}}, \qquad
    K_{F_{ij}} = - \int_{A_0} p \, \boldsymbol{S}_i \cdot \frac{\partial
    \boldsymbol{a}}{\partial \bar{c}_j} \, dx \, dy

with `\boldsymbol{S}_i = (S^u_i, S^v_i, S^w_i)` the functions of the degree
of freedom `i` and, from `\boldsymbol{a} = \boldsymbol{X}_{,x} \times
\boldsymbol{X}_{,y}`:

.. math::
    :label: fp-da-dc

    \frac{\partial \boldsymbol{a}}{\partial \bar{c}_j} =
        \frac{\partial \boldsymbol{X}_{,x}}{\partial \bar{c}_j} \times
        \boldsymbol{X}_{,y} + \boldsymbol{X}_{,x} \times \frac{\partial
        \boldsymbol{X}_{,y}}{\partial \bar{c}_j}

where `\partial \boldsymbol{X}_{,x}/\partial \bar{c}_j = (S^u_{j,x},
S^v_{j,x}, S^w_{j,x})` and `\partial \boldsymbol{X}_{,y}/\partial \bar{c}_j
= (S^u_{j,y}, S^v_{j,y} + k_w S^w_j, S^w_{j,y} - k_v S^v_j)`. For the
first-order area vector, `\boldsymbol{X}_{,x}` and `\boldsymbol{X}_{,y}`
are replaced by `e_x` and `e_y` in Eq. :eq:`fp-da-dc`, `[K_F]` does not
depend on `\{\bar{c}\}`, and its blocks between the fields of the rows
(`\delta u, \delta v, \delta w`) and of the columns (`u, v, w`) are:

.. math::
    :label: fp-blocks

    [K_F] = \int_{A_0} p \begin{bmatrix}
        0 & 0 & (\boldsymbol{S}^u)^\top \boldsymbol{S}^w_{,x} \\
        0 & -k_v (\boldsymbol{S}^v)^\top \boldsymbol{S}^v &
            (\boldsymbol{S}^v)^\top \boldsymbol{S}^w_{,y} \\
        -(\boldsymbol{S}^w)^\top \boldsymbol{S}^u_{,x} &
            -(\boldsymbol{S}^w)^\top \boldsymbol{S}^v_{,y} &
            -k_w (\boldsymbol{S}^w)^\top \boldsymbol{S}^w
    \end{bmatrix} dx \, dy

For an external pressure on a cylinder with Sanders' kinematics, `p < 0`,
the block `-k_v p (\boldsymbol{S}^v)^\top \boldsymbol{S}^v` of `[K_F]` is
positive definite, the stiffening that cancels the destabilizing hoop
prestress of the rigid rotation, see :ref:`follower_pressure_ring`.


Internal force vector, residual and tangent stiffness
-----------------------------------------------------

The pressures of the loads added with ``cte=False`` are multiplied by the
load factor `\lambda`, ``inc``, and those with ``cte=True`` are constant.
Writing `\{F_p(\bar{c})\}` for the sum over the follower loads with their
multipliers, the residual of the equilibrium is:

.. math::
    :label: fp-residual

    \{R(\bar{c}, \lambda)\} = \{F_{ext}(\lambda)\} + \left(
    \{F_p(\bar{c})\} - \{F_p(0)\} \right) - \{F_{int}^e(\bar{c})\}

where `\{F_{ext}(\lambda)\}` is :meth:`.Shell.calc_fext`, which includes the
reference part `\{F_p(0)\}` of the follower loads, and
`\{F_{int}^e(\bar{c})\}` is the internal force vector of the strains.
:meth:`.Shell.calc_fint` returns the internal force vector with the
configuration-dependent part of the follower loads:

.. math::
    :label: fp-fint

    \{F_{int}(\bar{c}, \lambda)\} = \{F_{int}^e(\bar{c})\} - \left(
    \{F_p(\bar{c})\} - \{F_p(0)\} \right)

such that `\{R\} = \{F_{ext}(\lambda)\} - \{F_{int}(\bar{c}, \lambda)\}`,
the form used by :class:`structsolve.Analysis`. The tangent stiffness
matrix, :meth:`.Shell.calc_kT`, is its exact Jacobian:

.. math::
    :label: fp-tangent

    [K_T] = \frac{\partial \{F_{int}\}}{\partial \{\bar{c}\}}
          = [K_{C0}] + [K_{CNL}] + [K_{GNL}] + [K_G(N_0 + N_L)] + [K_F]

and the derivative of the residual with respect to the load factor, used by
the arc-length methods, is the load vector of the current configuration,
``calc_fext(inc=1., c=c)`` for the loads proportional to `\lambda`:

.. math::
    :label: fp-dr-dlambda

    \frac{\partial \{R\}}{\partial \lambda} = \{F_{ext}(1)\} +
    \{F_p(\bar{c})\} - \{F_p(0)\}

From version 0.5.0, ``structsolve`` passes the load factor to the callables
that accept the keyword argument ``inc`` and uses Eq. :eq:`fp-dr-dlambda` in
the arc-length methods.


Linear analyses
---------------

**Linear static analysis.** A geometrically linear structure under a
follower load with the first-order area vector is in equilibrium when
`[K_{C0}]\{\bar{c}\} = \{F_p(\bar{c})\}`, i.e., from Eq.
:eq:`fp-force-affine`:

.. math::
    :label: fp-linear-static

    \left( [K_{C0}] + [K_F] \right) \{\bar{c}\} = \{F_p(0)\}

an unsymmetric linear system in general, solved by
``structsolve.Analysis.static(NLgeom=False)`` when ``calc_fext`` accepts the
configuration ``c``.

**Linear buckling, static criterion.** With the pre-buckling state
`\{\hat{c}\}` of the reference load, the geometric stiffness
`[K_G(\hat{c})]` and the load stiffness both scale with the load factor:

.. math::
    :label: fp-lb

    \left( [K_{C0}] + \lambda \left( [K_G(\hat{c})] + [K_F] \right)
    \right) \{\bar{c}\} = \{0\}

solved by ``structsolve.lb(kC0, kG + kCfollower)``, which uses the solvers
of unsymmetric matrices when needed.

**Kinetic criterion.** A non-conservative system may lose stability by
flutter, which Eq. :eq:`fp-lb` does not detect: the eigenvalues `\lambda^2
= -\omega^2` of `([K_T] + \lambda^2 [M])\{\bar{c}\} = \{0\}`, with the load
stiffness in `[K_T]`, are real below the critical load and become complex
conjugate pairs at a flutter load, ``structsolve.freq``.


Symmetry of the load stiffness matrix
-------------------------------------

For the first-order area vector, the block of the rows `\delta u` and of the
columns `w` of Eq. :eq:`fp-blocks` is `(\boldsymbol{S}^u)^\top
\boldsymbol{S}^w_{,x}`. A symmetric matrix would have its transpose,
`(\boldsymbol{S}^w_{,x})^\top \boldsymbol{S}^u`, as the block of the rows
`\delta w` and of the columns `u`, which is instead
`-(\boldsymbol{S}^w)^\top \boldsymbol{S}^u_{,x}`, and similarly for the
coupling of `v` and `w`. The difference between the first block and the
transpose of the second one is a derivative,

.. math::
    :label: fp-skew-block

    (\boldsymbol{S}^u)^\top \boldsymbol{S}^w_{,x} +
    (\boldsymbol{S}^u_{,x})^\top \boldsymbol{S}^w = \left(
    (\boldsymbol{S}^u)^\top \boldsymbol{S}^w \right)_{,x}

whereas the blocks of the curvature, `-k_v (\boldsymbol{S}^v)^\top
\boldsymbol{S}^v` and `-k_w (\boldsymbol{S}^w)^\top \boldsymbol{S}^w`, are
symmetric. Integrating by parts over the loaded region `A_p`, whose contour
`\partial A_p` has the outward normal `\{n_x, n_y\}`, the skew part of the
load stiffness matrix of Eq. :eq:`fp-blocks` is:

.. math::
    :label: fp-skew

    \begin{aligned}
    [K_F] - [K_F]^\top = {} &
    \oint_{\partial A_p} p \left[ \left( (\boldsymbol{S}^u)^\top
    \boldsymbol{S}^w - (\boldsymbol{S}^w)^\top \boldsymbol{S}^u \right) n_x
    + \left( (\boldsymbol{S}^v)^\top \boldsymbol{S}^w -
    (\boldsymbol{S}^w)^\top \boldsymbol{S}^v \right) n_y \right] ds \\
    & - \int_{A_p} \left[ p_{,x} \left( (\boldsymbol{S}^u)^\top
    \boldsymbol{S}^w - (\boldsymbol{S}^w)^\top \boldsymbol{S}^u \right) +
    p_{,y} \left( (\boldsymbol{S}^v)^\top \boldsymbol{S}^w -
    (\boldsymbol{S}^w)^\top \boldsymbol{S}^v \right) \right] dx \, dy
    \end{aligned}

The curvature terms having cancelled, Eq. :eq:`fp-skew` holds for the
plates and for the Donnell and Sanders cylinders alike. Hence, `[K_F]` is
symmetric, and the follower pressure conservative, if and only if:

- `p` is uniform over the loaded region, such that the area integral
  vanishes; and
- at each point of the contour of the loaded region, either `w = 0` or the
  in-plane displacement normal to the contour is zero, i.e. `u = 0` on the
  edges `x = const` and `v = 0` on the edges `y = const`, such that the
  contour integral vanishes.

For a load over the whole domain, the second condition is met on the edge
`x = 0` with the flag ``x1w = 0`` or ``x1u = 0`` of :class:`.Shell`, see
:ref:`theory_func_bardell`, and likewise on the other edges. These
conditions are met by simply supported and clamped edges, by symmetry
planes, and by closed surfaces such as complete rings and cylinders, for
which no boundary exists; in these cases the load potential is the pressure
times the change of the enclosed volume. Conversely, `[K_F]`
is unsymmetric for free or partially restrained edges, for a pressure
applied on a patch whose edges lie inside the shell, and for a pressure that
varies over the surface. This agrees with [hibbitt1979Follower]_, p. 938,
for whom a uniform pressure on a fully enclosed volume, or on a surface
whose boundaries have prescribed displacements, gives a symmetric load
stiffness, unlike free edges and a non-uniform pressure, p. 939, and, for
body-attached loads, with Table 2 of [schweizerhof1984Pressure]_, whose
boundary terms vanish under the conditions of their Sec. 3.2.1, Eqs.
(32)-(35), e.g. when at least two displacement components are prescribed
at the edge, or when the edge is restrained normal to a continuous
surface, such as the undeformed surface of the shell, `w = 0`.

A closed cylinder of ``panels`` is an assembly of domains, see
:func:`.create_cylinder`, whose load stiffness matrices are each
unsymmetric: the contour integrals along a connected edge cancel only when
the displacements of the two sides are equal, which holds for the matrices
reduced by the null-space method, symmetric to round-off, but not for the
assembled matrix of the penalty method, which remains unsymmetric. When it
is symmetric the load is conservative, with the potential

.. math::
    :label: fp-potential

    \Pi_p = - \lambda \left( \{F_p(0)\}^\top \{\bar{c}\} - \frac{1}{2}
            \{\bar{c}\}^\top [K_F] \{\bar{c}\} \right)

For the quadratic area vector `[K_F]` depends on `\{\bar{c}\}` and is not
symmetric in general. A hydrostatic head, a pressure that depends on the
current position, would add a further term, not implemented.


Load surface
------------

A pressure `p` on the surface `z = z_p` of a cylinder, e.g. the outer face
`z_p = h/2` of a thick sandwich, has the resultant `p (r + z_p)` per unit
length of mid-surface arc. With ``add_pressure_load(..., zp=zp)`` it is
applied as

.. math::
    :label: fp-zp

    p \left( 1 + \frac{z_p}{r} \right)

per unit mid-surface area, with the direction and the change of area of the
mid-surface, as in a shallow shell, see :func:`.pressure_factor`. The terms
of relative order `z_p \kappa`, `\kappa` the change of curvature, in the
change of area and the moment of the in-plane components about the
mid-surface are neglected; for a ring mode `n` they are of relative order
`n^2 z_p / r` in the load stiffness. Plates are not affected.


.. _follower_pressure_ring:

Ring under uniform pressure
---------------------------

For a ring in plane strain, mode `w = W \cos(n y/r)`, `v = V \sin(n y/r)`,
membrane prestress `N_{yy} = p r` and the CLPT strains of the kernels, the
second variation of the total potential is

.. math::
    :label: fp-ring

    U_2 = \frac{1}{2} \oint \left( A \varepsilon_{yy}^2 + D \kappa_{yy}^2 +
          N_{yy} \beta_y^2 \right) dy - \frac{1}{2} \oint p \, \boldsymbol{u}
          \cdot \boldsymbol{a}_1(\boldsymbol{u}) \, dy

with `\beta_y = w_{,y} - k_v v`. In the inextensional limit its singularity
gives `p_{cr} r^3/D`, derived with SymPy:

======================= =============== =============== ==================
kinematics               follower         dead             centrally
                         (hydrostatic)    (constant        directed
                                          direction)
======================= =============== =============== ==================
Sanders                  `-3`             `-4`             `-9/2`
Donnell                  `-16/5`          `-4`             `-64/15`
======================= =============== =============== ==================

The Sanders values are the classical ones, see [timoshenko1961Stability]_
and [brush1975Buckling]_. For a ring, with `D` replaced by the bending
stiffness `EI` of the ring, they are the analytical values `3`, `4` and
`4.5` that Table 3 of [schweizerhof1984Pressure]_ lists as reference for
their finite element results. The follower value is also the hydrostatic
critical pressure `3D/r^3` of the long cylinders, Eq. (66) of
[nasa2020SP8007]_. With the FSDT and the transverse shear stiffness `S`, the
follower load gives `p_{cr} = 3D/(r^3 (1 + 4D/(S r^2)))`, the shell formula
of [han2004SandwichCylinder]_.

With Sanders' kinematics and a dead pressure, the rigid rotation `v =
const`, `w = 0`, strain free, has the energy `\oint N_{yy} (v/r)^2 dy < 0`
under an external pressure, a spurious mode at a vanishing critical load
whenever `v` is free at an edge. The follower load stiffness adds `-\oint p
\, v^2/r \, dy`, which cancels it exactly.


.. _follower_pressure_implementation:

Implementation and verification
-------------------------------

- Eqs. :eq:`fp-area-vector`, :eq:`fp-force` and :eq:`fp-jacobian` are
  derived with SymPy in the notation of the kernels, which also verifies the
  symmetry identity Eq. :eq:`fp-skew` and the ring closed forms, and writes
  the kernels ``calc_fext_follower`` (Eq. :eq:`fp-force`) and
  ``fkCfollower_num`` (Eq. :eq:`fp-jacobian`, assembled in full, not as an
  upper triangle) of the modules ``panels.models.<model>_num``, integrated
  with ``nx*ny`` Gauss-Legendre points over each loaded patch, the same
  points as Eq. :eq:`fp-force-reference` in :func:`.shell_fext`.
- ``tests/tests_shell/test_follower_pressure.py``: Eq.
  :eq:`fp-force-reference`, Taylor tests of Eq. :eq:`fp-force` against Eq.
  :eq:`fp-jacobian`, finite-difference Jacobian of Eq. :eq:`fp-fint` against
  Eq. :eq:`fp-tangent`, quadratic convergence of Newton-Raphson and Riks,
  the symmetry conditions of Eq. :eq:`fp-skew` and Eq.
  :eq:`fp-linear-static`.
- ``tests/multidomain/test_follower_pressure_multidomain.py``: the
  follower loads in a :class:`.MultiDomain` assembly, and the symmetry of
  the load stiffness of a closed cylinder reduced by the null-space method.
- ``tests/tests_shell/test_follower_pressure_validation.py`` and the
  notebooks of :ref:`ex_follower_pressure`: the ring closed forms, finite
  cylinders, [nasa2020SP8007]_, [kardomateas1993Orthotropic]_,
  [han2004SandwichCylinder]_, [kardomateas2003Sandwich]_ and
  [schweizerhof1984Pressure]_.
