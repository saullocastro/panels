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
The derivation is implemented with sympy in
``theory/shells/follower_pressure/follower_pressure.py``, which also
generates the kernels ``calc_fext_follower`` and ``fkCfollower_num`` of every
model, see :ref:`follower_pressure_implementation`.


Virtual work of the pressure
----------------------------

Let `x` and `y` be the coordinates of the undeformed mid-surface, `y` being
the arc length for the cylindrical shells, and `\{e_x, e_y, e_z\}` the local
basis of the undeformed mid-surface, with `e_z` the normal, positive outwards
for the cylinders of radius `r`, such that `e_{y,y} = -e_z/r` and
`e_{z,y} = e_y/r`. The displacement of the mid-surface is `\mathbf{u} = u
e_x + v e_y + w e_z`, and the position of the deformed mid-surface is
`\mathbf{X} = \mathbf{X}_0 + \mathbf{u}`. A pressure `p`, positive along
`+e_z` as the force ``fz`` of :meth:`.Shell.add_point_load` (an external
pressure on a cylinder is negative), acting normal to the deformed
mid-surface and on its deformed area, does the virtual work:

.. math::
    :label: fp-virtual-work

    \delta W_p = \int_{A} p \, \mathbf{n} \cdot \delta\mathbf{u} \, dA
               = \int_{A_0} p \, \mathbf{a} \cdot \delta\mathbf{u}
                 \, dx \, dy,
    \qquad
    \mathbf{a} = \mathbf{X}_{,x} \times \mathbf{X}_{,y}

where `\mathbf{a}` is the area vector of the deformed mid-surface per unit
undeformed area, `\mathbf{n} \, dA = \mathbf{a} \, dx \, dy`. For the models
based on the shear deformation theories, FSDT and TSDT, the pressure acts on
the mid-surface, such that only `u`, `v` and `w` enter Eq.
:eq:`fp-virtual-work`, never the rotations `\phi_x` and `\phi_y`.

In the local basis:

.. math::
    :label: fp-tangents

    \mathbf{X}_{,x} = \mathbf{A} = \begin{Bmatrix} 1 + u_{,x} \\ v_{,x} \\
                                   w_{,x} \end{Bmatrix},
    \qquad
    \mathbf{X}_{,y} = \mathbf{B} = \begin{Bmatrix} u_{,y} \\
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
`\phi_y = -w_{,y} + v/r`, see ``theory/shells/cylshell_clpt_sanders`` and
``theory/shells/fsdt_tsdt``. The Donnell cylinders neglect `v/r`
in the rotation, as their strains do. The area vector `\mathbf{a} =
\mathbf{A} \times \mathbf{B}` is bilinear in the displacement gradients:

.. math::
    :label: fp-area-vector

    \mathbf{a} = \mathbf{a}_0 + \mathbf{a}_1 + \mathbf{a}_2,
    \qquad
    \mathbf{a}_0 = \begin{Bmatrix} 0 \\ 0 \\ 1 \end{Bmatrix},
    \qquad
    \mathbf{a}_1 = \begin{Bmatrix} -w_{,x} \\ -w_{,y} + k_v v \\
                   u_{,x} + v_{,y} + k_w w \end{Bmatrix},

.. math::
    :label: fp-area-vector-2

    \mathbf{a}_2 = \begin{Bmatrix}
        v_{,x} (w_{,y} - k_v v) - w_{,x} (v_{,y} + k_w w) \\
        w_{,x} u_{,y} - u_{,x} (w_{,y} - k_v v) \\
        u_{,x} (v_{,y} + k_w w) - v_{,x} u_{,y}
    \end{Bmatrix}

The dead part `\mathbf{a}_0` is the dead load. The first-order part
`\mathbf{a}_1` contains the rotation of the normal, `(-w_{,x}, -w_{,y} + k_v
v)`, and the change of area, `u_{,x} + v_{,y} + k_w w`, which is the sum of
the linear membrane strains `\varepsilon_{xx} + \varepsilon_{yy}`.


Truncation consistent with the strains
--------------------------------------

The strains of all models are of moderate rotations, von Kármán, Donnell or
Sanders, i.e. with the in-plane gradients `u_{,x} \sim \varepsilon` and the
rotations `w_{,x} \sim \varepsilon^{1/2}`. The virtual displacements are
ordered alike, `\delta u \sim \varepsilon^{1/2} \delta w`, such that the
integrand of Eq. :eq:`fp-virtual-work`, `p (a_z \delta w + a_x \delta u +
a_y \delta v)`, has:

- from `\mathbf{a}_0 + \mathbf{a}_1`: the terms `p \, \delta w` and `p \,
  O(\varepsilon) \, \delta w`;
- from `\mathbf{a}_2`: the terms `p \, O(\varepsilon^2) \, \delta w`, of the
  order of the terms neglected by the strains themselves.

The consistent truncation is therefore the first-order area vector,
``follower=True`` or ``'linear'``:

.. math::
    :label: fp-truncation

    \mathbf{a} \approx \mathbf{a}_0 + \mathbf{a}_1

With ``follower='quadratic'`` the complete area vector of Eq.
:eq:`fp-area-vector` is used instead, which is useful to check the
sensitivity of a result to the truncation.


Force vector
------------

With the approximation of the displacements by the Ritz constants `\{c\}`,
see :ref:`theory_func_bardell`,

.. math::
    :label: fp-ritz

    u = \{N_u\}^T \{c\}, \qquad v = \{N_v\}^T \{c\}, \qquad
    w = \{N_w\}^T \{c\}

where `\{N_u\}`, `\{N_v\}` and `\{N_w\}` are the vectors of the approximation
functions of each field, null at the positions of the other degrees of
freedom, Eq. :eq:`fp-virtual-work` gives the force vector of the follower
pressure:

.. math::
    :label: fp-force

    \{F_p(c)\} = \int_{A_0} p \left( \{N_u\} a_x + \{N_v\} a_y + \{N_w\} a_z
                 \right) dx \, dy

where `\mathbf{a}` is evaluated at `\{c\}` with Eq. :eq:`fp-truncation`, or
Eq. :eq:`fp-area-vector` for ``'quadratic'``. In the undeformed state,
`\mathbf{a} = \mathbf{a}_0` and the follower load equals the dead load:

.. math::
    :label: fp-force-reference

    \{F_p(0)\} = \int_{A_0} p \{N_w\} \, dx \, dy

which is the reference force vector returned by :meth:`.Shell.calc_fext`,
see :func:`.shell_fext`. For the first-order area vector `\{F_p\}` is affine
in `\{c\}`:

.. math::
    :label: fp-force-affine

    \{F_p(c)\} = \{F_p(0)\} + [K_p] \{c\}


Load stiffness matrix
---------------------

The Jacobian of the force vector, the load stiffness, is:

.. math::
    :label: fp-jacobian

    [K_p] = \frac{\partial \{F_p\}}{\partial \{c\}}, \qquad
    K_{p_{ij}} = \int_{A_0} p \, \mathbf{N}_i \cdot
                 \frac{\partial \mathbf{a}}{\partial c_j} \, dx \, dy

with `\mathbf{N}_i = (N_{u_i}, N_{v_i}, N_{w_i})` and, from `\mathbf{a} =
\mathbf{A} \times \mathbf{B}`:

.. math::
    :label: fp-da-dc

    \frac{\partial \mathbf{a}}{\partial c_j} =
        \frac{\partial \mathbf{A}}{\partial c_j} \times \mathbf{B}
        + \mathbf{A} \times \frac{\partial \mathbf{B}}{\partial c_j}

where `\partial \mathbf{A}/\partial c_j = (N_{u_j,x}, N_{v_j,x},
N_{w_j,x})` and `\partial \mathbf{B}/\partial c_j = (N_{u_j,y}, N_{v_j,y} +
k_w N_{w_j}, N_{w_j,y} - k_v N_{v_j})`. For the first-order area vector,
`\mathbf{A}` and `\mathbf{B}` are replaced by `\mathbf{A}_0 = e_x` and
`\mathbf{B}_0 = e_y` in Eq. :eq:`fp-da-dc`, `[K_p]` does not depend on
`\{c\}`, and its blocks between the fields of the rows (`\delta u, \delta v,
\delta w`) and of the columns (`u, v, w`) are:

.. math::
    :label: fp-blocks

    [K_p] = \int_{A_0} p \begin{bmatrix}
        0 & 0 & -\{N_u\}\{N_{w,x}\}^T \\
        0 & k_v \{N_v\}\{N_v\}^T & -\{N_v\}\{N_{w,y}\}^T \\
        \{N_w\}\{N_{u,x}\}^T & \{N_w\}\{N_{v,y}\}^T &
            k_w \{N_w\}\{N_w\}^T
    \end{bmatrix} dx \, dy

The matrix returned by :meth:`.Shell.calc_kCfollower` is the contribution to
the tangent stiffness matrix:

.. math::
    :label: fp-kcfollower

    [K_{C_{follower}}] = - \frac{\partial \{F_p\}}{\partial \{c\}} = -[K_p]

such that all the stiffness matrices are additive, see Eq.
:eq:`fp-tangent`. For an external pressure on a cylinder with Sanders' kinematics, `p < 0`,
the block `-k_v p \{N_v\}\{N_v\}^T` of `[K_{C_{follower}}]` is positive
definite, the stiffening that cancels the destabilizing hoop prestress of the
rigid rotation, see :ref:`follower_pressure_ring`.


Internal force vector, residual and tangent stiffness
-----------------------------------------------------

The pressures of the loads added with ``cte=False`` are multiplied by the
load factor `\lambda`, ``inc``, and those with ``cte=True`` are constant.
Writing `\{F_p(c)\}` for the sum over the follower loads with their
multipliers, the residual of the equilibrium is:

.. math::
    :label: fp-residual

    \{R(c, \lambda)\} = \{F_{ext}(\lambda)\} + \left( \{F_p(c)\} -
                        \{F_p(0)\} \right) - \{F_{int}^e(c)\}

where `\{F_{ext}(\lambda)\}` is :meth:`.Shell.calc_fext`, which includes the
reference part `\{F_p(0)\}` of the follower loads, and `\{F_{int}^e(c)\}` is
the internal force vector of the strains. :meth:`.Shell.calc_fint` returns
the internal force vector with the configuration-dependent part of the
follower loads:

.. math::
    :label: fp-fint

    \{F_{int}(c, \lambda)\} = \{F_{int}^e(c)\} - \left( \{F_p(c)\} -
                              \{F_p(0)\} \right)

such that `\{R\} = \{F_{ext}(\lambda)\} - \{F_{int}(c, \lambda)\}`, the form
used by :class:`structsolve.Analysis`. The tangent stiffness matrix,
:meth:`.Shell.calc_kT`, is its exact Jacobian:

.. math::
    :label: fp-tangent

    [K_T] = \frac{\partial \{F_{int}\}}{\partial \{c\}}
          = [K_0] + [K_{0L}] + [K_{L0}] + [K_{LL}] + [K_{G_{NL}}]
            + [K_G(N_0 + N_L)] + [K_{C_{follower}}]

and the derivative of the residual with respect to the load factor, used by
the arc-length methods, is the load vector of the current configuration,
``calc_fext(inc=1., c=c)`` for the loads proportional to `\lambda`:

.. math::
    :label: fp-dr-dlambda

    \frac{\partial \{R\}}{\partial \lambda} = \{F_{ext}(1)\} + \{F_p(c)\} -
                                             \{F_p(0)\}

From version 0.5.0, ``structsolve`` passes the load factor to the callables
that accept the keyword argument ``inc`` and uses Eq. :eq:`fp-dr-dlambda` in
the arc-length methods.


Linear analyses
---------------

**Linear static analysis.** A geometrically linear structure under a
follower load with the first-order area vector is in equilibrium when
`[K_0]\{c\} = \{F_p(c)\}`, i.e., from Eqs. :eq:`fp-force-affine` and
:eq:`fp-kcfollower`:

.. math::
    :label: fp-linear-static

    \left( [K_0] + [K_{C_{follower}}] \right) \{c\} = \{F_p(0)\}

an unsymmetric linear system in general, solved by
``structsolve.Analysis.static(NLgeom=False)`` when ``calc_fext`` accepts the
configuration ``c``.

**Linear buckling, static criterion.** With the prestate `\{c_0\}` of the
reference load, the geometric stiffness `[K_G(c_0)]` and the load stiffness
both scale with the load factor:

.. math::
    :label: fp-lb

    \left( [K_0] + \lambda \left( [K_G(c_0)] + [K_{C_{follower}}] \right)
    \right) \{c\} = \{0\}

solved by ``structsolve.lb(kC0, kG + kCfollower)``, which uses the solvers
of unsymmetric matrices when needed.

**Kinetic criterion.** A non-conservative system may lose stability by
flutter, which Eq. :eq:`fp-lb` does not detect: the eigenvalues
`\lambda^2 = -\omega^2` of `([K_T] + \lambda^2 [M])\{c\} = \{0\}`, with the
load stiffness in `[K_T]`, are real below the critical load and become
complex conjugate pairs at a flutter load, ``structsolve.freq``.


Symmetry of the load stiffness matrix
-------------------------------------

For the first-order area vector, `[K_p]` is the matrix of the bilinear form

.. math::
    :label: fp-bilinear

    P(\delta, \Delta) = \int_{A_0} p \, \delta\mathbf{u} \cdot
                        \mathbf{a}_1(\Delta\mathbf{u}) \, dx \, dy

and integrating by parts, the skew part is (verified symbolically by
``check_symmetry`` of ``follower_pressure.py``):

.. math::
    :label: fp-skew

    P(\delta, \Delta) - P(\Delta, \delta) =
    - \oint_{\partial A_p} p \left[ (\delta u \, \Delta w - \delta w \,
      \Delta u) n_x + (\delta v \, \Delta w - \delta w \, \Delta v) n_y
      \right] ds
    + \int_{A_p} \left[ p_{,x} (\delta u \, \Delta w - \delta w \, \Delta u)
      + p_{,y} (\delta v \, \Delta w - \delta w \, \Delta v) \right]
      dx \, dy

over the loaded region `A_p`, independently of `k_w` and `k_v`. Therefore
`[K_{C_{follower}}]` is symmetric if and only if `p` is uniform over the
whole integration domain and, on each edge, either `w = 0` or the
displacement normal to the edge is zero, `u` on the edges `x = const` and
`v` on the edges `y = const`. This is the case of simply supported or
clamped panels, of symmetry planes and of closed rings. Free or partially
restrained edges, patches, whose edges are inside the shell, and a
non-uniform `p(x, y)` give an unsymmetric load stiffness, a non-conservative
load. This is Table 2 of [schweizerhof1984Pressure]_ for body-attached
loads, see also [hibbitt1979Follower]_. When it is symmetric the load is
conservative, with the potential

.. math::
    :label: fp-potential

    \Pi_p = - \lambda \left( \{F_p(0)\}^T \{c\} + \frac{1}{2} \{c\}^T [K_p]
            \{c\} \right)

For the quadratic area vector `[K_p]` depends on `\{c\}` and is not
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
          N_{yy} \beta_y^2 \right) dy - \frac{1}{2} \oint p \, \mathbf{u}
          \cdot \mathbf{a}_1(\mathbf{u}) \, dy

with `\beta_y = w_{,y} - k_v v`. In the inextensional limit its singularity
gives `p_{cr} r^3/D` (``ring_buckling`` in ``follower_pressure.py``):

======================= =============== =============== ==================
kinematics               follower         dead             centrally
                         (hydrostatic)    (constant        directed
                                          direction)
======================= =============== =============== ==================
Sanders                  `-3`             `-4`             `-9/2`
Donnell                  `-16/5`          `-4`             `-64/15`
======================= =============== =============== ==================

The Sanders values are the classical ones, see [timoshenko1961Stability]_,
[brush1975Buckling]_ and Table 3 of [schweizerhof1984Pressure]_, and the
follower value is the hydrostatic
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

- ``theory/shells/follower_pressure/follower_pressure.py``: the sympy
  derivation of Eqs. :eq:`fp-area-vector`, :eq:`fp-force` and
  :eq:`fp-jacobian` in the notation of the kernels, the symmetry identity
  Eq. :eq:`fp-skew` and the ring closed forms; ``write_pyx.py`` writes the
  kernels ``calc_fext_follower`` (Eq. :eq:`fp-force`) and
  ``fkCfollower_num`` (Eq. :eq:`fp-kcfollower`, assembled in full, not as an
  upper triangle) into every ``panels/models/<model>_num.pyx``, integrated
  with ``nx*ny`` Gauss-Legendre points over each loaded patch, the same
  points as Eq. :eq:`fp-force-reference` in :func:`.shell_fext`.
- ``tests/tests_shell/test_follower_pressure.py``: Eq.
  :eq:`fp-force-reference`, Taylor tests of Eq. :eq:`fp-force` against Eq.
  :eq:`fp-jacobian`, finite-difference Jacobian of Eq. :eq:`fp-fint` against
  Eq. :eq:`fp-tangent`, quadratic convergence of Newton-Raphson and Riks,
  the symmetry conditions of Eq. :eq:`fp-skew` and Eq.
  :eq:`fp-linear-static`.
- ``tests/tests_shell/test_follower_pressure_validation.py`` and the
  notebooks of :ref:`ex_follower_pressure`: the ring closed forms, finite
  cylinders, [nasa2020SP8007]_, [kardomateas1993Orthotropic]_,
  [han2004SandwichCylinder]_, [kardomateas2003Sandwich]_ and
  [schweizerhof1984Pressure]_.
