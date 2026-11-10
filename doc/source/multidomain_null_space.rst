.. _null_space:

Exact connections with the null-space method
============================================

The domains of a :class:`.MultiDomain` assembly can be connected with
penalty stiffnesses, ``conn_method='penalty'``, see
:meth:`.MultiDomain.get_kC_conn` and [castro2017Multidomain]_. The penalty
method adds, for each quantity `q_k`
that must be continuous across a connection, the energy:

.. math::
    :label: ns-penalty

    U_k = \frac{1}{2} k_k \int \left( q_k^{(1)} - q_k^{(2)} \right)^2

where `k_k` is the translation or rotation penalty constant, ``'kt'`` or
``'kr'``, and the integral is along the connected edges, or over the area of
the connection ``'SB'``. The continuity is then only approximate. Low penalty
constants make the assembly too flexible, and high ones make the stiffness
matrix ill-conditioned. The default constants of :func:`.calc_kt_kr` are a
compromise, and they can still lead to errors of a few percent, see
:ref:`null_space_results`.

With ``conn_method='null-space'``, the default, the connections are imposed
exactly, as with penalty constants going to infinity, without any penalty
constant::

    md = MultiDomain(panels, conn)  # conn_method='null-space'

and the penalty method is selected with::

    md = MultiDomain(panels, conn, conn_method='penalty')

The connections that can be imposed with the null-space method are given in
:data:`.nullspace.NULL_SPACE_FUNCS`: ``'SSxcte'``, ``'SSycte'``,
``'BFxcte'``, ``'BFycte'``, ``'BFxycte'`` and ``'SB'``, for all the models, see
:ref:`null_space_quantities`. The damaged connection ``'SB_TSL'`` represents
the physical stiffness of a cohesive interface, see :ref:`cohesive_zone`,
and it keeps its penalty stiffness also with ``conn_method='null-space'``.


Theory
------

The quantities `q_k` of Eq. :eq:`ns-penalty` are linear combinations of the
Ritz constants `\boldsymbol{c}` of the two domains. Since Bardell's functions are
polynomials, see :ref:`theory_func_bardell`, the difference `q_k^{(1)} -
q_k^{(2)}` is a polynomial along the connection, of the natural coordinate
`\xi` or `\eta` shared by the two domains. It vanishes everywhere when it
vanishes at as many points as the number of its coefficients, and the
connections become the homogeneous linear constraints:

.. math::
    :label: ns-constraints

    \boldsymbol{B} \boldsymbol{c} = \boldsymbol{0}

where each row of `\boldsymbol{B}` is `q_k^{(1)} - q_k^{(2)}` at one Gauss-Legendre
point along the connection, or over the area of ``'SB'``, see
:func:`.nullspace.constraint_matrix`. The first four functions of Bardell
are cubic and the function `i \ge 4` has the order `i`, so that
`\max(m_1, m_2, 4)` points are used along `x` and `\max(n_1, n_2, 4)` along
`y`. The zero energy of Eq. :eq:`ns-penalty` and the constraints of
Eq. :eq:`ns-constraints` are equivalent: the vectors with zero penalty energy
are exactly those that satisfy the constraints.

Why not Lagrange multipliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The constraints could be imposed with Lagrange multipliers `\boldsymbol{\lambda}`,
which add one unknown per constraint and give the system:

.. math::

    \begin{bmatrix} \boldsymbol{K} & \boldsymbol{B}^\top \\ \boldsymbol{B} & \boldsymbol{0} \end{bmatrix}
    \begin{Bmatrix} \boldsymbol{c} \\ \boldsymbol{\lambda} \end{Bmatrix} =
    \begin{Bmatrix} \boldsymbol{F} \\ \boldsymbol{0} \end{Bmatrix}

This matrix is indefinite. For the linear buckling and frequency analyses
the geometric stiffness matrix and the mass matrix have zero rows and
columns for the multipliers, giving infinite eigenvalues that the
shift-and-invert eigenvalue solvers of ``structsolve`` do not handle. Instead
of adding unknowns, the null-space method removes them, giving the same
solution.

Elimination of the constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the vectors that satisfy Eq. :eq:`ns-constraints` are written with a
basis `\boldsymbol{Z}` of the null space of `\boldsymbol{B}`, `\boldsymbol{B} \boldsymbol{Z} = \boldsymbol{0}`:

.. math::
    :label: ns-Z

    \boldsymbol{c} = \boldsymbol{Z} \boldsymbol{c}_r

where `\boldsymbol{c}_r` are the independent Ritz constants. Substituting
Eq. :eq:`ns-Z` into the total potential energy gives the reduced system:

.. math::
    :label: ns-reduced

    \boldsymbol{K}_r = \boldsymbol{Z}^\top \boldsymbol{K} \boldsymbol{Z}, \qquad \boldsymbol{K}_{G r} = \boldsymbol{Z}^\top \boldsymbol{K}_G \boldsymbol{Z}, \qquad
    \boldsymbol{M}_r = \boldsymbol{Z}^\top \boldsymbol{M} \boldsymbol{Z}, \qquad \boldsymbol{F}_r = \boldsymbol{Z}^\top \boldsymbol{F}

where `\boldsymbol{K}`, `\boldsymbol{K}_G`, `\boldsymbol{M}` and `\boldsymbol{F}` are the matrices and vectors of the
domains, which do not contain the connections imposed exactly. The reduced
stiffness matrix is symmetric and positive definite whenever the connected
assembly has no rigid-body motion, like the stiffness matrix of the penalty
method, and the static, linear buckling, frequency and non-linear solvers of
``structsolve`` are used unchanged. The constraints are linear in `\boldsymbol{c}`,
therefore `\boldsymbol{Z}` does not change during a non-linear analysis:

.. math::

    \boldsymbol{F}_{int, r} = \boldsymbol{Z}^\top \boldsymbol{F}_{int}(\boldsymbol{Z} \boldsymbol{c}_r), \qquad
    \boldsymbol{K}_{T r} = \boldsymbol{Z}^\top \boldsymbol{K}_T(\boldsymbol{Z} \boldsymbol{c}_r) \boldsymbol{Z}

`\boldsymbol{Z}` is built by :func:`.nullspace.null_space_basis`, one connection at a
time:

1. the rows of `\boldsymbol{B}` of the connection are normalized to a unit norm and
   written in the independent Ritz constants of the connections already
   processed, `\boldsymbol{B} \boldsymbol{Z}`;
2. the rank of the constraints is revealed with a QR decomposition with
   column pivoting, `\boldsymbol{B} \boldsymbol{Z} \boldsymbol{P} = \boldsymbol{Q} \boldsymbol{U}`, with the permutation `\boldsymbol{P}`,
   the orthogonal factor `\boldsymbol{Q}` and the upper triangular factor
   `\boldsymbol{U}`, and each pivot larger than the tolerance ``tol``
   gives one independent constraint;
3. the Ritz constants of the pivot columns are eliminated, `\boldsymbol{c}_e =
   -\boldsymbol{U}_{11}^{-1} \boldsymbol{U}_{12} \boldsymbol{c}_k`, where `\boldsymbol{c}_k` are the
   remaining constants, and `\boldsymbol{Z}` is updated.

Redundant constraints are found in step 2 and dropped. They happen for
instance at a corner shared by four domains, around a closed cylinder, or
where a boundary condition flag already removes the term that a constraint
would remove. Bardell's functions of order 5 and higher vanish together with
their slope at the ends of the domain, `\xi = \pm 1`, so the constraints of
an edge involve only the first four functions across it, and `\boldsymbol{Z}` has very
few non-zero terms: the elimination is similar to the assembly of the
hierarchical elements of the p-version of the finite element method. A
connection along an interior line, e.g. ``'BFycte'`` at the middle of the
base, involves all the functions across the line.

The terms removed by the boundary condition flags are zero rows and columns
of the matrices of the domains, as with the penalty method, and remain in
`\boldsymbol{c}_r` with zero rows and columns in the reduced matrices.

`\boldsymbol{Z}` is stored in ``MultiDomain.Z`` and reused while the connections, the
number of terms and the boundary condition flags do not change, see
:meth:`.MultiDomain.get_Z`.


.. _null_space_quantities:

Quantities imposed
------------------

The quantities imposed by each connection are those of the corresponding
penalty kernels, with ``p1`` and ``p2`` the first and second panels of the
connection. For ``'SB'``, the two panels share the axes `x, y, z`, `z` being
normal to their mid-surfaces, ``p1`` is the panel on the positive side of the
interface along `z` and ``p2`` the one on the negative side, e.g. a stiffener
base on the side of positive `z` of a skin is ``p1`` and the skin ``p2``,
see :meth:`.MultiDomain.get_kC_conn`:

.. list-table::
    :header-rows: 1
    :widths: 12 44 44

    * - Connection
      - Classical laminated plate theory (3 DOFs)
      - Shear deformation theories (5 DOFs)
    * - ``'SSxcte'``
      - `u, v, w, w_{,x}`, see ``kCSSxcte.pyx``
      - `u, v, w, \phi_x, \Phi_y`, and `w_{,x}` for the TSDT, see
        :func:`.fkCSSxcte_sdt`
    * - ``'SSycte'``
      - `u, v, w, w_{,y}`, see ``kCSSycte.pyx``
      - `u, v, w, \phi_x, \Phi_y`, and `w_{,y}` for the TSDT, see
        :func:`.fkCSSycte_sdt`
    * - ``'BFycte'``
      - `u_1 = u_2`, `v_1 = w_2`, `w_1 = -v_2`, `w_{1,y} - v_1/r_1 =
        w_{2,y} - v_2/r_2`, see :mod:`panels.multidomain.connections.kCBFycte`
      - `u_1 = u_2`, `v_1 = w_2`, `w_1 = -v_2`, `\Phi_{y1} = \Phi_{y2}`
    * - ``'BFxcte'``
      - `u_1 = w_2`, `v_1 = v_2`, `w_1 = -u_2`, `w_{1,x} = w_{2,x}`, see
        :mod:`panels.multidomain.connections.kCBFxcte`
      - `u_1 = w_2`, `v_1 = v_2`, `w_1 = -u_2`, `\phi_{x1} = \phi_{x2}`
    * - ``'BFxycte'``
      - `u_1 = -\cos\theta \, v_2 + \sin\theta \, w_2`, `v_1 = u_2`, `w_1 =
        \sin\theta \, v_2 + \cos\theta \, w_2`, `-w_{1,x} = w_{2,y} -
        v_2/r_2`, with the angle `\theta` of the key ``'angle'``, `u_1 =
        w_2`, `w_1 = v_2` for the default 90 degrees, see
        :mod:`panels.multidomain.connections.kCBFxycte`
      - the same displacements, `\phi_{x1} = -\Phi_{y2}`
    * - ``'SB'``
      - `u_1(z_1) = u_2(z_2)`, `v_1(z_1) = v_2(z_2)`, `w_1 = w_2` at the
        interface, `z_1 = d_1 - h_1/2` of ``p1`` and `z_2 = d_2 + h_2/2` of
        ``p2``, `d_1`, `d_2` being the offsets, with `u(z) = u - z w_{,x}`
        and `v(z) = (1 + z/r) v - z w_{,y}` of each panel, see
        :func:`.fkCSB`; the rotations are not tied and the key ``'kr'`` is
        not used
      - `u, v, w` at the interface, `z = d_1 - h_1/2` of ``p1`` and `z = d_2
        + h_2/2` of ``p2``, see :func:`.fkCSB`, and `\phi_x, \Phi_y`
        (`\theta_x, \theta_y` of :eq:`tsdt-offset-theta` for the TSDT with
        an offset) only when the key ``'kr'`` of the connection is given and
        not zero

where `\Phi_y = \phi_y + v/r` with the Sanders-Koiter kinematics and `\Phi_y
= \phi_y` otherwise, and the terms `v/r` and `z/r` of the classical
laminated plate theory are present only with the Sanders-Koiter kinematics.
The quantities of ``'SB'`` are the same for all the models, the
displacements of the faces in contact, see
:func:`.kCsdt._sb_components`. The keys
``'kt'`` and ``'kr'`` of the connections are not used, apart from ``'kr'``
of ``'SB'`` with the shear deformation theories, which tells whether the
rotations are connected. The rotations of the classical laminated plate
theory, `-w_{,x}` and `-w_{,y} + v/r`, follow from `w`, already tied by `w_1
= w_2`: tying them too adds redundant constraints, or, with the
Sanders-Koiter kinematics and distinct reference surfaces, the constraint
`(z_1 - z_2) w_{,y} = 0`, i.e. `w_{,y} = 0` over the whole area, see
:func:`.kCsdt._sb_components`.


Usage
-----

The matrices and vectors computed by :class:`.MultiDomain` have the size of
all the Ritz constants and do not contain the connections imposed exactly:
solving them directly would leave the domains disconnected. They are reduced
with :meth:`.MultiDomain.reduce` before solving, and the solutions are expanded to
all the Ritz constants with :meth:`.MultiDomain.expand`, which are used by
the post-processing methods, e.g. :meth:`.MultiDomain.uvw`,
:meth:`.MultiDomain.stress` and :meth:`.MultiDomain.plot`. With
``conn_method='penalty'``, `\boldsymbol{Z}` is the identity matrix, so that the same
code runs with both methods.

Linear static analysis::

    from structsolve import solve

    md = MultiDomain(panels, conn)
    kC = md.calc_kC()
    fext = md.calc_fext()
    c = md.expand(solve(md.reduce(kC), md.reduce(fext)))

Linear buckling analysis with the pre-buckling state of a static analysis::

    from structsolve import lb, static

    kC = md.reduce(md.calc_kC())
    incs, cs = static(kC, md.reduce(md.calc_fext()))
    c = md.expand(cs[0])
    kG = md.reduce(md.calc_kG(c=c))
    eigvals, eigvecs = lb(kC, kG)
    eigvecs = md.expand(eigvecs)

Frequency analysis::

    from structsolve import freq

    eigvals, eigvecs = freq(md.reduce(md.calc_kC()), md.reduce(md.calc_kM()))
    eigvecs = md.expand(eigvecs)

Geometrically non-linear analysis, with the functions of the reduced problem
given by :meth:`.MultiDomain.get_reduced_functions`::

    from structsolve import Analysis

    an = Analysis(*md.get_reduced_functions())
    an.static(NLgeom=True)
    cs = [md.expand(c_r) for c_r in an.cs]

The functions :func:`.create_cylinder`,
:func:`.create_cylinder_blade_stiffened`,
:func:`.create_cylinder_z_stiffened`, the analyses of
:mod:`panels.multidomain.cylinder` and
:mod:`panels.multidomain.cylinder_blade_stiffened`, and
:func:`.tstiff2d_1stiff_freq`, :func:`.tstiff2d_1stiff_compression`,
:func:`.tstiff2d_1stiff_flutter` and :func:`.create_wingbox` take the
argument ``conn_method``, ``'null-space'`` by default.


.. _null_space_results:

Comparison with the penalty method
----------------------------------

The penalty energy of Eq. :eq:`ns-penalty` is non-negative: the penalty
method is more flexible than the exact connections, its stiffness and its
natural frequencies are lower, and its error decreases with the inverse of
the penalty constants. The condition number of the stiffness matrix grows
with the penalty constants, whereas the reduced stiffness matrix
`\boldsymbol{Z}^\top \boldsymbol{K} \boldsymbol{Z}` of the null-space method
does not depend on any constant. With the default constants of
:func:`.calc_kt_kr` the error of the penalty method is of the order of 1% for
thin plates and reaches several percent for thick plates and stiffened
shells, whereas the null-space method reproduces a single domain, or a single
laminate with ``'SB'``, to the convergence of the approximation.


.. _null_space_pd:

Prescribed displacements imposed exactly
----------------------------------------

The prescribed displacements of :meth:`.Shell.add_point_pd`,
:meth:`.Shell.add_distr_pd_fixed_x` and :meth:`.Shell.add_distr_pd_fixed_y`
are imposed by default with the penalty method: their forces `k u_p` enter
the external force vector and the penalty stiffnesses, e.g. of
:func:`.fkCpd`, :func:`.fkCld_xcte` and :func:`.fkCld_ycte`, for the models
with 3 and 5 degrees of freedom per term, must be added to the stiffness
matrix by the user, symmetrized with
:func:`structsolve.sparseutils.finalize_symmetric_matrix`. For a point
`(x_p, y_p)` with the prescribed values `q_p` and a line `x = x_l` with
the prescribed functions `q_l(y)`, `q = u, v, w`, and with
`\boldsymbol{S}^q` the rows of the approximation functions of `q`, the
penalty energies `\frac{1}{2} k_q (\boldsymbol{S}^q \boldsymbol{c} - q_p)^2`
and `\frac{1}{2} \int_0^b k_q (\boldsymbol{S}^q \boldsymbol{c} - q_l)^2 dy`
give the stiffness matrices and force vectors

.. math::
    :label: ns-pd-penalty

    \boldsymbol{K}_p = \sum_{q} k_q \left. (\boldsymbol{S}^q)^\top
    \boldsymbol{S}^q \right|_{(x_p, y_p)}, \qquad
    \boldsymbol{F}_p = \sum_{q} k_q q_p \left. (\boldsymbol{S}^q)^\top
    \right|_{(x_p, y_p)}

    \boldsymbol{K}_l = \sum_{q} k_q \int_0^b \left. (\boldsymbol{S}^q)^\top
    \boldsymbol{S}^q \right|_{x = x_l} dy, \qquad
    \boldsymbol{F}_l = \sum_{q} k_q \int_0^b q_l(y) \left.
    (\boldsymbol{S}^q)^\top \right|_{x = x_l} dy

of :func:`.fkCpd` and :func:`.fkCld_xcte` (:func:`.fkCld_ycte` along `y =
y_l`), the forces being added by :meth:`.MultiDomain.calc_fext`; for a
thickness coordinate ``z`` different from zero, the rows of `u` and `v`
are those of `u(z)` and `v(z)`. The prescribed values are reached with an
error of order `1/k`.

With ``pd_method='null-space'`` they are
imposed exactly, as inhomogeneous linear constraints eliminated together
with the connections, without any penalty constant::

    md = MultiDomain(panels, conn, pd_method='null-space')

A single panel is wrapped in a :class:`.MultiDomain` without connections,
``MultiDomain([shell], conn=[], pd_method='null-space')``. Both settings are
independent: the connections may be imposed with either ``conn_method``.
The default ``pd_method='penalty'`` keeps the previous behaviour.

The same methods of :class:`.Shell` define the prescribed displacements,
the in-plane ones at the reference surface or at the thickness coordinate
of the keyword ``z``, e.g. at the faces of a face-clamped edge, see
:ref:`boundary_conditions_face_clamp`.
With ``pd_method='null-space'`` the values of the penalty constants are not
used: a component is prescribed when its penalty constant is neither
``None`` nor zero and its value is given, the same rule as for the penalty
method, where a null penalty constant prescribes nothing. The rotations
of the normal at the reference surface `z = 0` can also be prescribed:
`-w_{,x}` and `-w_{,y}` for the classical laminated plate theory, and for
the shear deformation theories `\theta_x` and `\theta_y` of Eq.
:eq:`tsdt-offset-theta`, the rotations also tied by the connections, which
are the fields `\phi_x` and `\phi_y` for the FSDT and for the TSDT without
offset; the rotation about `x` includes `v/r` with the Sanders-Koiter
kinematics. With an offset `d`, the TSDT rotations `\phi_x` and `\phi_y`
are those at the mid-surface, `z = d`, and the prescribed rotations at `z =
0` include the transverse shear, `\theta_x = \phi_x - (4 d^2/h^2)(\phi_x +
w_{,x})`, see :ref:`tsdt_offset_prescribed_rotations`. The rotations are
prescribed only with ``pd_method='null-space'``, through the keyword
arguments ``kphix``,
``phixp``, ``kphiy`` and ``phiyp`` of :meth:`.Shell.add_point_pd`, or
``kphix``, ``funcphix``, ``kphiy`` and ``funcphiy`` of the distributed ones.
For example, a uniform end-shortening `\delta` of a plate, incremented with
the load factor, with the lateral contraction free::

    p.add_distr_pd_fixed_x(0., ku=1., funcu=lambda y: 0.)
    p.add_distr_pd_fixed_x(p.a, ku=1., funcu=lambda y: -delta, cte=False)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)  # v = 0 at a corner

Theory
~~~~~~

The connections give the homogeneous constraints `\boldsymbol{B}_h \boldsymbol{c} = \boldsymbol{0}` of
Eq. :eq:`ns-constraints`, and the prescribed displacements the
inhomogeneous constraints:

.. math::
    :label: ns-pd

    \boldsymbol{B}_d \boldsymbol{c} = \bar{\boldsymbol{d}}_0 + \lambda \bar{\boldsymbol{d}}_1

where `\bar{\boldsymbol{d}}_0` collects the constant prescribed values, ``cte=True``, and
`\bar{\boldsymbol{d}}_1` those scaled by the load factor `\lambda`, ``cte=False``. Each row
of `\boldsymbol{B}_d` is the prescribed component at a point, or at a Gauss-Legendre
point along a prescribed line, with ``max(n, 4)`` points along `y` for the
lines ``x = cte`` and ``max(m, 4)`` along `x` for the lines ``y = cte``, see
:func:`.nullspace.pd_constraint_matrix`. As for the connections, the
component along the line is a polynomial with that many coefficients: the
constraint holds along the whole line when the prescribed function is a
polynomial of degree up to ``max(n, 4) - 1``, e.g. a uniform or a linear
end-shortening; otherwise it holds at the Gauss-Legendre points, the
prescribed function being interpolated by the approximation functions.

With `\boldsymbol{B} = [\boldsymbol{B}_h; \boldsymbol{B}_d]`, every admissible configuration is

.. math::
    :label: ns-pd-c

    \boldsymbol{c} = \boldsymbol{Z} \boldsymbol{c}_r + \boldsymbol{c}_{p0} + \lambda \boldsymbol{c}_{p1}

where `\boldsymbol{Z}` spans the null space of `\boldsymbol{B}` and `\boldsymbol{c}_{p0}`, `\boldsymbol{c}_{p1}`
are particular solutions, `\boldsymbol{B} \boldsymbol{c}_{p0} = \{\boldsymbol{0}; \bar{\boldsymbol{d}}_0\}` and `\boldsymbol{B}
\boldsymbol{c}_{p1} = \{\boldsymbol{0}; \bar{\boldsymbol{d}}_1\}`, see :meth:`.MultiDomain.get_cp`. They are
obtained in the same elimination as `\boldsymbol{Z}`, see
:class:`.nullspace.Elimination`: the constraints are eliminated one block
at a time, the connections first, each block written in the independent
Ritz constants left by the previous ones, `\boldsymbol{B} (\boldsymbol{Z} \boldsymbol{c}_r + \boldsymbol{c}_p) =
\boldsymbol{b}`, with the right-hand side `\boldsymbol{b}` of the block, and
decomposed with a QR decomposition with column pivoting, which
gives both the new `\boldsymbol{Z}` and the correction of `\boldsymbol{c}_p` for the
right-hand side `\boldsymbol{b} - \boldsymbol{B} \boldsymbol{c}_p`. The data of the decompositions are
kept, such that `\boldsymbol{c}_{p0}` and `\boldsymbol{c}_{p1}` are computed again without
any decomposition when only the prescribed values change, e.g. in a loop
over the end-shortening; `\boldsymbol{Z}` changes only with the positions and the
components prescribed.

Substituting Eq. :eq:`ns-pd-c` into the residual of the unconstrained
system, `\boldsymbol{R}(\boldsymbol{c}, \lambda) = \boldsymbol{F}_{int}(\boldsymbol{c}, \lambda) - \lambda \boldsymbol{F}_{ext}`,
and projecting it with `\boldsymbol{Z}^\top` gives the reduced problem solved by
``structsolve``:

.. math::
    :label: ns-pd-reduced

    \boldsymbol{F}_{int, r}(\boldsymbol{c}_r, \lambda) &= \boldsymbol{Z}^\top \boldsymbol{F}_{int}(\boldsymbol{c}) \\
    \boldsymbol{K}_{T, r} &= \boldsymbol{Z}^\top \boldsymbol{K}_T(\boldsymbol{c}) \boldsymbol{Z} = \frac{\partial \boldsymbol{R}_r}{\partial \boldsymbol{c}_r} \\
    -\frac{\partial \boldsymbol{R}_r}{\partial \lambda} &= \boldsymbol{Z}^\top \left(\boldsymbol{F}_{ext} -
        \boldsymbol{K}_T(\boldsymbol{c}) \boldsymbol{c}_{p1}\right)

`\boldsymbol{K}_{T, r}` is the exact Jacobian, and `-\partial \boldsymbol{R}_r/\partial \lambda`
depends on the configuration and on the load factor. ``structsolve``
passes the load factor to ``calc_fext`` when it declares the keyword
argument ``lbd``, and uses `-\partial \boldsymbol{R}_r/\partial \lambda` in the predictors
and in the constraint of the arc-length methods, and `\lambda \partial
\boldsymbol{R}_r/\partial \lambda` as a reference of the convergence criteria,
Crisfield's, the default, and the relative one,
because under a prescribed displacement alone both `\boldsymbol{Z}^\top \boldsymbol{F}_{ext}` and
`\boldsymbol{Z}^\top \boldsymbol{F}_{int}` vanish at equilibrium, see
:meth:`.MultiDomain.get_reduced_functions`.

The linear static analysis solves

.. math::

    \boldsymbol{K}_r \boldsymbol{c}_r = \boldsymbol{Z}^\top \left(\lambda \boldsymbol{F}_{ext} - \boldsymbol{K} (\boldsymbol{c}_{p0} +
    \lambda \boldsymbol{c}_{p1})\right)

see :meth:`.MultiDomain.linear_static`, the loads of ``calc_fext(inc=1.)``
being scaled by `\lambda` as in the analyses of ``structsolve``. The linear
buckling analysis about the pre-buckling state `\hat{\boldsymbol{c}}(\lambda) =
\hat{\boldsymbol{c}}_0 + \lambda \hat{\boldsymbol{c}}_1`, with `\hat{\boldsymbol{c}}_0` due to the
constant prescribed displacements and `\hat{\boldsymbol{c}}_1` due to the load and
the incremented ones, solves `(\boldsymbol{K}_r + \boldsymbol{K}_{G, r}(\hat{\boldsymbol{c}}_0) + \lambda
\boldsymbol{K}_{G, r}(\hat{\boldsymbol{c}}_1)) \boldsymbol{\phi}_r = \boldsymbol{0}`, see
:meth:`.MultiDomain.linear_buckling`; the constant stress resultants
``Nxx``, ``Nyy`` and ``Nxy`` of the panels are part of the constant state,
in `\boldsymbol{K}_{G, r}(\hat{\boldsymbol{c}}_0)` only. The modes satisfy the
homogeneous constraints, `\boldsymbol{\phi} = \boldsymbol{Z} \boldsymbol{\phi}_r`: under displacement
control the prescribed components of the modes are zero, whereas under the
equivalent load they are free. For a flat plate the modes of the linear
buckling analysis are out of plane and both critical loads are equal; for a
cylinder, whose modes couple the membrane and bending displacements, the
critical end-shortening is higher than the end-shortening of the critical
axial load with ``x2u=1``.

Reactions
~~~~~~~~~

With `\boldsymbol{r}(\boldsymbol{c}, \lambda) = \boldsymbol{F}_{int}(\boldsymbol{c}) - \lambda \boldsymbol{F}_{ext}` the residual
of the unconstrained system, the force applied by the supports on the
structure, the generalized reaction work-conjugate to the set `j` of
prescribed displacements, the constant or the incremented ones, is

.. math::
    :label: ns-pd-reaction

    P_j = \boldsymbol{c}_{pj}^\top \boldsymbol{r}(\boldsymbol{c}, \lambda)

see :meth:`.MultiDomain.reaction_pd`. It does not depend on the choice of
the particular solution, since `\boldsymbol{Z}^\top \boldsymbol{r} = \boldsymbol{0}` at equilibrium.
`P_j \delta` is the work of the reactions for the prescribed values
multiplied by `\delta`: for a uniform end-shortening `\delta` of an edge
with the total axial reaction `N`, `P_1 = N \delta`, and `P_1 = \partial
U/\partial \lambda` is the force conjugate to the load factor of a
displacement-controlled analysis.

The resultant of the reactions along selected prescribed lines or points,
in the direction `u`, `v` or `w` of the panels, is `F_u = \boldsymbol{c}_u^\top \boldsymbol{r}`,
with `\boldsymbol{c}_u` a solution of the constraints for a unit translation `u = 1`
of the selected prescriptions that prescribe `u`, all the other prescribed
values being zero, see :meth:`.MultiDomain.reaction_resultant`. A line whose
ends are shared with a prescription that is not selected, directly or
through a connection, has no resultant of its own, the force at the shared
corner being undetermined, and an error is raised: for instance, the total
axial force at the end of a cylinder made of several panels is the
resultant of all the lines of that end. The sign is that of the force
applied by the support on the structure, `F_u = -N` at the shortened end.

Redundant and inconsistent prescriptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As for the connections, a constraint that is a linear combination of the
previous ones is redundant and dropped, e.g. at the corner where a
prescribed edge meets a connection or another prescribed edge, or on a term
already removed by a boundary condition flag. A redundant constraint is
dropped only when its right-hand side is consistent, i.e. satisfied by the
particular solution within ``rhs_tol`` of :meth:`.MultiDomain.get_Z`,
`10^{-8}` by default, relative to the largest prescribed value. Otherwise
:meth:`.MultiDomain.get_cp` raises a ``ValueError`` that names the
prescription, the panel, the boundary condition flags that restrain it, and
the connections and prescriptions imposed before it on the same or
connected panels:

- a prescribed edge meeting another one with a different value at a
  corner, or across a connection;
- a non-zero value on a term removed by a boundary condition flag, e.g. `u
  = \delta` along ``x = a`` with ``x2u = 0``: the message tells to set the
  flag to 1, whereas `u = 0` is consistent and dropped;
- the constant and the incremented sets must each be consistent by
  themselves, since the constraints hold for every `\lambda`: the same line
  cannot have a constant and an incremented prescription of the same
  component;
- a function that is not a polynomial, prescribed along a line whose end is
  restrained by a flag, is consistent only within the interpolation error,
  which may exceed ``rhs_tol`` with few terms.

API
~~~

- ``MultiDomain(..., pd_method='null-space')``
- :meth:`.MultiDomain.get_Z` and :meth:`.MultiDomain.get_cp`: `\boldsymbol{Z}`,
  `\boldsymbol{c}_{p0}` and `\boldsymbol{c}_{p1}`, also in the attributes ``Z``, ``c_p0``
  and ``c_p1``
- :meth:`.MultiDomain.expand`: ``expand(c_r, lbd)`` returns the
  configuration of Eq. :eq:`ns-pd-c`, ``expand(c_r)`` returns `\boldsymbol{Z}
  \boldsymbol{c}_r`, e.g. for the modes
- :meth:`.MultiDomain.get_reduced_functions`: the callables of
  ``structsolve.Analysis`` for the non-linear analyses, Newton-Raphson and
  the arc-length methods. The linear
  static analysis of ``Analysis.static(NLgeom=False)`` cannot contain the
  constant prescribed displacements and raises an error, see
  :meth:`.MultiDomain.linear_static`
- :meth:`.MultiDomain.linear_static` and :meth:`.MultiDomain.linear_buckling`
- :meth:`.MultiDomain.reaction_pd` and :meth:`.MultiDomain.reaction_resultant`
- :meth:`.MultiDomain.calc_fext` excludes the forces of the prescribed
  displacements with ``pd_method='null-space'``

Displacement control
~~~~~~~~~~~~~~~~~~~~

Under the uniform end-shortening `\delta` of Eq. :eq:`bc-end-shortening`,
the membrane patch test is passed exactly, for one domain or for connected
domains: the fields are `u = -\delta x/a` and `v = \nu \delta y/a` to the
round-off, and the reactions are `P_1 = N \delta`, Eq. :eq:`ns-pd-reaction`,
and `F_u = -N` at the shortened end, with `N = E h b \delta/a`.

Beyond the critical end-shortening `\delta_{cr}` of the linear buckling
analysis, the post-buckling path is traced under displacement control with
the reduced functions of :meth:`.MultiDomain.get_reduced_functions`, the
end-shortening being incremented, ``cte=False``, and a small lateral load
acting as an imperfection. Every converged increment of the Newton-Raphson
and arc-length methods of ``structsolve`` satisfies `\|\boldsymbol{Z}^\top
\boldsymbol{r}\| \le` ``relTOL`` `\|\lambda \, \partial
\boldsymbol{R}_r/\partial \lambda\|`. The arc length of ``structsolve`` is
scaled by the linear solution, which has no out-of-plane displacement under
an end-shortening: the post-buckling path is long in these units, and
``maxArcLength`` may need to be increased.

.. _null_space_pd_penalty:

Penalty vs null-space displacement control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The penalty method imposes a prescribed displacement `u_p` with the energy
`\frac{1}{2} k \int (u - u_p)^2`, i.e. a spring of stiffness `k` between the
structure and the support, see :func:`.fkCld_xcte` and :func:`.fkCpd`: the
structure is more flexible, the prescribed value is not reached, and the
error decreases with `1/k`, while the condition number of the stiffness
matrix grows with `k`. The null-space method has no penalty constant.

The reaction of the penalty method, `k \int (u_p - u)`, is the product of a
large constant and a small difference, and loses accuracy by cancellation
for the highest constants. In a geometrically non-linear analysis the
penalty forces `k u_p` are part of `\lambda \boldsymbol{F}_{ext}` and
`\boldsymbol{F}_{int}`, the references of the convergence criteria of
``structsolve``: the relative criterion measures them with the Euclidean
norm over all the degrees of freedom, which grows with `k`, Crisfield's
criterion with the norm `\|\boldsymbol{v}\|_{\boldsymbol{d}} = (\sum_j
v_j^2/|d_j|)^{1/2}` scaled by the diagonal `\boldsymbol{d}` of the linear
stiffness matrix, which contains the penalty stiffness, such that the scaled
norm of the penalty forces grows only with `\sqrt{k}`. Once the tolerance
dominates, the error of the penalty method stops decreasing with `1/k`,
earlier with the relative criterion. The null-space method has no such
trade-off, its reference being `\lambda \partial \boldsymbol{R}_r/\partial
\lambda`, see :ref:`null_space_pd`. Its `\partial \boldsymbol{R}_r/\partial
\lambda` requires the product `\boldsymbol{K}_T \boldsymbol{c}_{p1}`,
computed without assembling `\boldsymbol{K}_T`, see :ref:`tangent_product`,
at the cost of one evaluation of the internal force vector; the linear
analyses cost the same with both methods. With `k = 10^4 E h` the penalty
method is within `10^{-3}` of the null-space method for the linear and
non-linear analyses of a plate under end-shortening.

**When to use each method.** The null-space method is exact, needs no
penalty constant, keeps the conditioning of the model, and gives reactions
that do not depend on any constant; it is recommended for displacement
control, e.g. the end-shortening of panels and shells, and for linear
buckling under prescribed displacements. The penalty method is
the default of the prescribed displacements, ``pd_method='penalty'``, and it is needed when the prescribed
displacement represents a physical stiffness, e.g. a load cell or a loading
fixture, or for models that rely on the penalty forms, e.g. the cohesive zone
examples of :ref:`cohesive_zone`, whose reaction is given by
:meth:`.MultiDomain.reaction_line_pd_xcte`.


Limitations
-----------

A connection imposed exactly constrains the domains exactly as the penalty
method does with infinite penalty constants, which may constrain the
approximation more than the penalty method with the default constants does:

- When the two domains have a different number of terms along the
  connection, the terms of the trace of the richer domain that the other
  domain cannot represent are forced to vanish.

- The boundary condition flags at the ends of an edge of one domain are
  imposed on the other domain.

- With ``'SB'`` and the classical laminated plate theory, `u_1 - z_1 w_{,x}
  = u_2 - z_2 w_{,x}`, `z_1` and `z_2` being the faces in contact measured
  from the reference surface of each domain. When the reference surfaces of
  the two domains are apart, `z_1 \ne z_2`, e.g. without offsets,
  restraining the normal in-plane displacement of an edge in both domains,
  e.g. `u = 0` at `x = 0`, also restrains the rotation `w_{,x}` at that
  edge, which is clamped. With offsets that place both reference surfaces at
  the same height, `z_1 = z_2` and `u_1 = u_2`.
  The in-plane displacements of the bonded laminates must be restrained
  consistently with the single laminate that
  they represent, e.g. restraining only the tangential displacement of the
  edges.

The prescribed displacements of :meth:`.Shell.add_point_pd`,
:meth:`.Shell.add_distr_pd_fixed_x` and :meth:`.Shell.add_distr_pd_fixed_y`
remain penalty stiffnesses by default, and are imposed exactly with
``pd_method='null-space'``, see :ref:`null_space_pd`. The
traction-separation law of ``'SB_TSL'`` remains a penalty stiffness.

The rank of the constraints is found with the tolerance ``tol`` of
:meth:`.MultiDomain.get_Z`, ``1e-9`` by default, which applies to the
constraints normalized to a unit norm. The independent constraints have
pivots many orders of magnitude above the round-off errors of the redundant
ones, and a warning is issued when this gap is smaller than `10^4`.


API
---

The methods of :class:`.MultiDomain` for the null-space method are
:meth:`.MultiDomain.get_Z`, :meth:`.MultiDomain.reduce`,
:meth:`.MultiDomain.expand` and :meth:`.MultiDomain.get_reduced_functions`,
and for the prescribed displacements :meth:`.MultiDomain.get_cp`,
:meth:`.MultiDomain.linear_static`, :meth:`.MultiDomain.linear_buckling`,
:meth:`.MultiDomain.reaction_pd` and :meth:`.MultiDomain.reaction_resultant`,
see :ref:`null_space_pd`.

.. automodule:: panels.multidomain.connections.nullspace

.. autodata:: panels.multidomain.connections.nullspace.NULL_SPACE_FUNCS
    :no-value:

.. autofunction:: panels.multidomain.connections.nullspace.constraint_matrix

.. autofunction:: panels.multidomain.connections.nullspace.null_space_basis

.. autoclass:: panels.multidomain.connections.nullspace.Elimination
    :members:

.. autodata:: panels.multidomain.connections.nullspace.PD_COMPONENTS
    :no-value:

.. autofunction:: panels.multidomain.connections.nullspace.pd_constraint_matrix

.. autofunction:: panels.multidomain.connections.nullspace.pd_rows

.. autofunction:: panels.multidomain.connections.nullspace.pd_points

.. autofunction:: panels.multidomain.connections.nullspace.pd_blocks

.. autofunction:: panels.multidomain.connections.nullspace.pd_entries

.. autofunction:: panels.multidomain.connections.nullspace.pd_rhs
