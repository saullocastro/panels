.. _null_space:

Exact connections with the null-space method
============================================

The domains of a :class:`.MultiDomain` assembly were originally connected
with penalty stiffnesses, see :meth:`.MultiDomain.get_kC_conn` and
[castro2017Multidomain]_, still available with ``conn_method='penalty'``.
The penalty method adds, for each quantity `q_k`
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
``'BFxcte'``, ``'BFycte'`` and ``'SB'``, for all the models, see
:ref:`null_space_quantities`. The damaged connection ``'SB_TSL'`` represents
the physical stiffness of a cohesive interface, see :ref:`cohesive_zone`,
and it keeps its penalty stiffness also with ``conn_method='null-space'``.


Theory
------

The quantities `q_k` of Eq. :eq:`ns-penalty` are linear combinations of the
Ritz constants `\{c\}` of the two domains. Since Bardell's functions are
polynomials, see :ref:`theory_func_bardell`, the difference `q_k^{(1)} -
q_k^{(2)}` is a polynomial along the connection, of the natural coordinate
`\xi` or `\eta` shared by the two domains. It vanishes everywhere when it
vanishes at as many points as the number of its coefficients, and the
connections become the homogeneous linear constraints:

.. math::
    :label: ns-constraints

    [B] \{c\} = \{0\}

where each row of `[B]` is `q_k^{(1)} - q_k^{(2)}` at one Gauss-Legendre
point along the connection, or over the area of ``'SB'``, see
:func:`.nullspace.constraint_matrix`. The first four functions of Bardell
are cubic and the function `i \ge 4` has the order `i`, so that
`\max(m_1, m_2, 4)` points are used along `x` and `\max(n_1, n_2, 4)` along
`y`. The zero energy of Eq. :eq:`ns-penalty` and the constraints of
Eq. :eq:`ns-constraints` are equivalent: the vectors with zero penalty energy
are exactly those that satisfy the constraints, which is verified for every
connection and model in ``tests/multidomain/test_null_space.py``.

Why not Lagrange multipliers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The constraints could be imposed with Lagrange multipliers `\{\lambda\}`,
which add one unknown per constraint and give the system:

.. math::

    \begin{bmatrix} [K] & [B]^T \\ [B] & [0] \end{bmatrix}
    \begin{Bmatrix} \{c\} \\ \{\lambda\} \end{Bmatrix} =
    \begin{Bmatrix} \{f\} \\ \{0\} \end{Bmatrix}

This matrix is indefinite. For the linear buckling and frequency analyses
the geometric stiffness matrix and the mass matrix have zero rows and
columns for the multipliers, giving infinite eigenvalues that the
shift-and-invert eigenvalue solvers of ``structsolve`` do not handle. Instead
of adding unknowns, the null-space method removes them, giving the same
solution.

Elimination of the constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All the vectors that satisfy Eq. :eq:`ns-constraints` are written with a
basis `[T]` of the null space of `[B]`, `[B] [T] = [0]`:

.. math::
    :label: ns-T

    \{c\} = [T] \{c_r\}

where `\{c_r\}` are the independent Ritz constants. Substituting
Eq. :eq:`ns-T` into the total potential energy gives the reduced system:

.. math::
    :label: ns-reduced

    [K_r] = [T]^T [K] [T], \qquad [K_{G r}] = [T]^T [K_G] [T], \qquad
    [M_r] = [T]^T [M] [T], \qquad \{f_r\} = [T]^T \{f\}

where `[K]`, `[K_G]`, `[M]` and `\{f\}` are the matrices and vectors of the
domains, which do not contain the connections imposed exactly. The reduced
stiffness matrix is symmetric and positive definite whenever the connected
assembly has no rigid-body motion, like the stiffness matrix of the penalty
method, and the static, linear buckling, frequency and non-linear solvers of
``structsolve`` are used unchanged. The constraints are linear in `\{c\}`,
therefore `[T]` does not change during a non-linear analysis:

.. math::

    \{f_{int, r}\} = [T]^T \{f_{int}\}([T] \{c_r\}), \qquad
    [K_{T r}] = [T]^T [K_T]([T] \{c_r\}) [T]

`[T]` is built by :func:`.nullspace.null_space_basis`, one connection at a
time:

1. the rows of `[B]` of the connection are normalized to a unit norm and
   written in the independent Ritz constants of the connections already
   processed, `[B] [T]`;
2. the rank of the constraints is revealed with a QR decomposition with
   column pivoting, `[B] [T] [P] = [Q] [R]`, and each pivot larger than the
   tolerance ``tol`` gives one independent constraint;
3. the Ritz constants of the pivot columns are eliminated, `\{c_e\} =
   -[R_{11}]^{-1} [R_{12}] \{c_k\}`, where `\{c_k\}` are the remaining
   constants, and `[T]` is updated.

Redundant constraints are found in step 2 and dropped. They happen for
instance at a corner shared by four domains, around a closed cylinder, or
where a boundary condition flag already removes the term that a constraint
would remove. Bardell's functions of order 5 and higher vanish together with
their slope at the ends of the domain, `\xi = \pm 1`, so the constraints of
an edge involve only the first four functions across it, and `[T]` has very
few non-zero terms: the elimination is similar to the assembly of the
hierarchical elements of the p-version of the finite element method. A
connection along an interior line, e.g. ``'BFycte'`` at the middle of the
base, involves all the functions across the line.

The terms removed by the boundary condition flags are zero rows and columns
of the matrices of the domains, as with the penalty method, and remain in
`\{c_r\}` with zero rows and columns in the reduced matrices.

`[T]` is stored in ``MultiDomain.T`` and reused while the connections, the
number of terms and the boundary condition flags do not change, see
:meth:`.MultiDomain.get_T`.


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
``conn_method='penalty'``, `[T]` is the identity matrix, so that the same
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
:func:`.create_cylinder_blade_stiffened`, the analyses of
:mod:`panels.multidomain.cylinder` and
:mod:`panels.multidomain.cylinder_blade_stiffened`, and
:func:`.tstiff2d_1stiff_freq`, :func:`.tstiff2d_1stiff_compression` and
:func:`.tstiff2d_1stiff_flutter` take the argument ``conn_method``.


.. _null_space_results:

Comparison with the penalty method
----------------------------------

The results below are verified in ``tests/multidomain/test_null_space.py``.

The penalty method is more flexible than the exact connections, and its
error decreases with the inverse of the penalty constants. For a thin plate
divided in 2x2 domains, with the classical laminated plate theory, the error
of the first frequencies is 1.3% with the default constants of
:func:`.calc_kt_kr`, and it becomes 0.13%, 0.013% and 0.0013% with 10, 100 and
1000 times higher constants. The condition number of the stiffness matrix
grows with the penalty constants: for the 4-domain plate of
``tests/multidomain/test_conn_4panels_kt_kr.py``, the reduced stiffness matrix
of the null-space method has a condition number about 600 times lower than
the stiffness matrix of the penalty method with the default constants, and
`6 \times 10^4` times lower than with 100 times higher constants.

.. list-table::
    :header-rows: 1
    :widths: 40 20 20 20

    * - Case
      - Penalty, default constants
      - Penalty, higher constants
      - Null-space
    * - Frequencies of a thick plate divided in 2x2 domains, FSDT, error
        with respect to the single domain (Noor 1973)
      - `7.8 \times 10^{-2}`
      - `9.5 \times 10^{-5}` (1000 times)
      - `3 \times 10^{-12}`
    * - Laminates `[0/90]` over `[0/90]` with ``'SB'``, error of the
        frequencies with respect to `[0/90]_2`, CLPT
      -
      - `8 \times 10^{-6}` (`10^5` times)
      - `6 \times 10^{-13}`
    * - Same, large deflection with the von Karman kinematics
      -
      -
      - `10^{-15}`
    * - Blade-stiffened cylinder, buckling load with the pre-buckling state
        of a static analysis, `N_{xx}` (N/m)
      - -39993.8
      - -41694.9 (1000 times)
      - -41696.7
    * - Closed cylinder, buckling load `N_{xx}` (N/m)
      - -47056.0
      -
      - -47113.0
    * - T-stiffened panel, first frequency (rad/s)
      - 48.2733
      - 48.3858 (100 times)
      - 48.3875

The default penalty constants underestimate the buckling load of the
blade-stiffened cylinder by 4%, whereas the higher constants converge to the
result of the null-space method.


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
:func:`structsolve.sparseutils.finalize_symmetric_matrix`:

.. literalinclude:: ../../tests/multidomain/test_kCpd_kernels.py
    :pyobject: test_penalty_reaches_prescribed_values

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

The connections give the homogeneous constraints `[B_h] \{c\} = \{0\}` of
Eq. :eq:`ns-constraints`, and the prescribed displacements the
inhomogeneous constraints:

.. math::
    :label: ns-pd

    [B_d] \{c\} = \{d_0\} + \lambda \{d_1\}

where `\{d_0\}` collects the constant prescribed values, ``cte=True``, and
`\{d_1\}` those scaled by the load factor `\lambda`, ``cte=False``. Each row
of `[B_d]` is the prescribed component at a point, or at a Gauss-Legendre
point along a prescribed line, with ``max(n, 4)`` points along `y` for the
lines ``x = cte`` and ``max(m, 4)`` along `x` for the lines ``y = cte``, see
:func:`.nullspace.pd_constraint_matrix`. As for the connections, the
component along the line is a polynomial with that many coefficients: the
constraint holds along the whole line when the prescribed function is a
polynomial of degree up to ``max(n, 4) - 1``, e.g. a uniform or a linear
end-shortening; otherwise it holds at the Gauss-Legendre points, the
prescribed function being interpolated by the approximation functions.

With `[B] = [B_h; B_d]`, every admissible configuration is

.. math::
    :label: ns-pd-c

    \{c\} = [T] \{c_r\} + \{c_{p0}\} + \lambda \{c_{p1}\}

where `[T]` spans the null space of `[B]` and `\{c_{p0}\}`, `\{c_{p1}\}`
are particular solutions, `[B] \{c_{p0}\} = \{0; d_0\}` and `[B]
\{c_{p1}\} = \{0; d_1\}`, see :meth:`.MultiDomain.get_cp`. They are
obtained in the same elimination as `[T]`, see
:class:`.nullspace.Elimination`: the constraints are eliminated one block
at a time, the connections first, each block written in the independent
Ritz constants left by the previous ones, `[B] ([T] \{c_r\} + \{c_p\}) =
\{d\}`, and decomposed with a QR decomposition with column pivoting, which
gives both the new `[T]` and the correction of `\{c_p\}` for the
right-hand side `\{d\} - [B] \{c_p\}`. The data of the decompositions are
kept, such that `\{c_{p0}\}` and `\{c_{p1}\}` are computed again without
any decomposition when only the prescribed values change, e.g. in a loop
over the end-shortening; `[T]` changes only with the positions and the
components prescribed.

Substituting Eq. :eq:`ns-pd-c` into the residual of the unconstrained
system, `\{R(c, \lambda)\} = \lambda \{F_{ext}\} - \{F_{int}(c, \lambda)\}`,
and projecting it with `[T]^T` gives the reduced problem solved by
``structsolve``:

.. math::
    :label: ns-pd-reduced

    \{F_{int, r}(c_r, \lambda)\} &= [T]^T \{F_{int}(c)\} \\
    [K_{T, r}] &= [T]^T [K_T(c)] [T] = -\frac{\partial R_r}{\partial c_r} \\
    \frac{\partial R_r}{\partial \lambda} &= [T]^T \left(\{F_{ext}\} -
        [K_T(c)] \{c_{p1}\}\right)

`[K_{T, r}]` is the exact Jacobian, and `\partial R_r/\partial \lambda`
depends on the configuration and on the load factor. ``structsolve`` 0.7.0
passes the load factor to ``calc_fext`` when it declares the keyword
argument ``lbd``, and uses `\partial R_r/\partial \lambda` in the predictors
and in the constraint of the arc-length methods, and `\lambda \partial
R_r/\partial \lambda` as a reference of the convergence criteria, the
relative one and Crisfield's, the default since ``structsolve`` 0.9.0,
because under a prescribed displacement alone both `[T]^T \{F_{ext}\}` and
`[T]^T \{F_{int}\}` vanish at equilibrium, see
:meth:`.MultiDomain.get_reduced_functions`.

The linear static analysis solves

.. math::

    [K_r] \{c_r\} = [T]^T \left(\lambda \{F_{ext}\} - [K] (\{c_{p0}\} +
    \lambda \{c_{p1}\})\right)

see :meth:`.MultiDomain.linear_static`, the loads of ``calc_fext(inc=1.)``
being scaled by `\lambda` as in the analyses of ``structsolve``. The linear
buckling analysis about the pre-buckling state `\{c(\lambda)\} = \{c_0\} +
\lambda \{c_1\}`, with `\{c_0\}` due to the constant prescribed
displacements and `\{c_1\}` due to the load and the incremented ones,
solves `([K_r] + [K_{G, r}(c_0)] + \lambda [K_{G, r}(c_1)]) \{\phi_r\} =
\{0\}`, see :meth:`.MultiDomain.linear_buckling`; the constant stress
resultants ``Nxx``, ``Nyy`` and ``Nxy`` of the panels are part of the
constant state, in `[K_{G, r}(c_0)]` only. The modes satisfy the
homogeneous constraints, `\{\phi\} = [T] \{\phi_r\}`: under displacement
control the prescribed components of the modes are zero, whereas under the
equivalent load they are free. For a flat plate the modes of the linear
buckling analysis are out of plane and both critical loads are equal, for a
cylinder, whose modes couple the membrane and bending displacements, the
critical end-shortening of a cylinder of 5 panels is 3.8% higher than the
end-shortening of the critical axial load with ``x2u=1``.

Reactions
~~~~~~~~~

With `\{r(c, \lambda)\} = \{F_{int}(c)\} - \lambda \{F_{ext}\}` the residual
of the unconstrained system, the force applied by the supports on the
structure, the generalized reaction work-conjugate to the set `j` of
prescribed displacements, the constant or the incremented ones, is

.. math::
    :label: ns-pd-reaction

    P_j = \{c_{pj}\}^T \{r(c, \lambda)\}

see :meth:`.MultiDomain.reaction_pd`. It does not depend on the choice of
the particular solution, since `[T]^T \{r\} = \{0\}` at equilibrium.
`P_j \delta` is the work of the reactions for the prescribed values
multiplied by `\delta`: for a uniform end-shortening `\delta` of an edge
with the total axial reaction `N`, `P_1 = N \delta`, and `P_1 = \partial
U/\partial \lambda` is the force conjugate to the load factor of a
displacement-controlled analysis.

The resultant of the reactions along selected prescribed lines or points,
in the direction `u`, `v` or `w` of the panels, is `F_u = \{c_u\}^T \{r\}`,
with `\{c_u\}` a solution of the constraints for a unit translation `u = 1`
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
particular solution within ``rhs_tol`` of :meth:`.MultiDomain.get_T`,
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
- :meth:`.MultiDomain.get_T` and :meth:`.MultiDomain.get_cp`: `[T]`,
  `\{c_{p0}\}` and `\{c_{p1}\}`, also in the attributes ``T``, ``c_p0``
  and ``c_p1``
- :meth:`.MultiDomain.expand`: ``expand(c_r, lbd)`` returns the
  configuration of Eq. :eq:`ns-pd-c`, ``expand(c_r)`` returns `[T]
  \{c_r\}`, e.g. for the modes
- :meth:`.MultiDomain.get_reduced_functions`: the callables of
  ``structsolve.Analysis`` for the non-linear analyses, Newton-Raphson and
  the arc-length methods, requiring ``structsolve>=0.7.0``. The linear
  static analysis of ``Analysis.static(NLgeom=False)`` cannot contain the
  constant prescribed displacements and raises an error, see
  :meth:`.MultiDomain.linear_static`
- :meth:`.MultiDomain.linear_static` and :meth:`.MultiDomain.linear_buckling`
- :meth:`.MultiDomain.reaction_pd` and :meth:`.MultiDomain.reaction_resultant`
- :meth:`.MultiDomain.calc_fext` excludes the forces of the prescribed
  displacements with ``pd_method='null-space'``

Example
~~~~~~~

A simply supported plate under a uniform end-shortening, with the lateral
edges free in-plane: linear buckling, then the post-buckling path under
displacement control, with a small lateral load as imperfection::

    import numpy as np
    from structsolve import Analysis
    from panels.shell import Shell
    from panels.multidomain import MultiDomain

    a, b, h = 0.3, 0.2, 1e-3
    p = Shell(a=a, b=b, m=8, n=8, model='plate_clpt_donnell', stack=[0],
              plyt=h, laminaprop=(70e9, 0.3))
    for edge in ('x1', 'x2', 'y1', 'y2'):
        setattr(p, edge + 'u', 1.)  # u and v free at the edges, w = 0
        setattr(p, edge + 'v', 1.)
    md = MultiDomain([p], conn=[], pd_method='null-space')

    def prescribe(delta):
        p.clear_disps()
        p.add_distr_pd_fixed_x(0., ku=1., funcu=lambda y: 0.)
        p.add_distr_pd_fixed_x(a, ku=1., funcu=lambda y: -delta, cte=False)
        p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)

    # linear buckling: the eigenvalue is the critical end-shortening
    prescribe(1.)
    eigvals, modes = md.linear_buckling(num_eigvalues=3)
    delta_cr = eigvals[0]

    # post-buckling up to three times the critical end-shortening
    prescribe(3*delta_cr)
    p.add_point_load(a/2, b/2, 0., 0., 2.)
    an = Analysis(*md.get_reduced_functions())
    an.NL_method = 'arc_length_riks'
    an.maxArcLength = 100.
    an.static(NLgeom=True)
    cs = [md.expand(c_r, lbd) for c_r, lbd in zip(an.cs, an.increments)]
    N = [-md.reaction_resultant(c, [(p, 'x', a)], lbd)[0]
         for c, lbd in zip(cs, an.increments)]

The arc length of ``structsolve`` is scaled by the linear solution, which
has no out-of-plane displacement under an end-shortening, therefore the
post-buckling path is long in these units, and ``maxArcLength`` may need to
be increased. The tests of ``tests/multidomain/test_null_space_pd.py`` give
more examples, among them:

.. literalinclude:: ../../tests/multidomain/test_null_space_pd.py
    :pyobject: test_membrane_patch_test_isotropic

.. literalinclude:: ../../tests/multidomain/test_null_space_pd.py
    :pyobject: test_nonlinear_displacement_control

.. _null_space_pd_penalty:

Penalty vs null-space displacement control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The penalty method imposes a prescribed displacement `u_p` with the energy
`\frac{1}{2} k \int (u - u_p)^2`, i.e. a spring of stiffness `k` between the
structure and the support: the structure is more flexible, the prescribed
value is not reached, and the error decreases with `1/k`, while the
condition number of the stiffness matrix grows with `k`. The null-space
method has no penalty constant. The cases below were run with the models of
``tests/multidomain/test_null_space_pd.py``, the penalty stiffnesses being
those of :func:`.fkCld_xcte` and :func:`.fkCpd`; `E h` is the membrane
stiffness of the isotropic plate and `A_{11}` that of the laminate of the
cylinder.

**Accuracy and conditioning, linear static.** Plate under a uniform
end-shortening with the lateral contraction restrained along `y = 0, b`,
``m = n = 10``, 300 Ritz constants, 280 independent ones with the null-space
method. Errors of the Ritz constants and of the axial reaction, and condition
number of the stiffness matrix; the reduced stiffness matrix of the
null-space method has the condition number `1.4 \times 10^{11}`:

.. list-table::
    :header-rows: 1

    * - `k / (E h)`
      - `10`
      - `10^2`
      - `10^4`
      - `10^6`
      - `10^8`
      - `10^{10}`
    * - error of `\{c\}`
      - `1.6 \times 10^{-1}`
      - `2.0 \times 10^{-2}`
      - `2.0 \times 10^{-4}`
      - `2.0 \times 10^{-6}`
      - `2.0 \times 10^{-8}`
      - `3.2 \times 10^{-9}`
    * - error of the reaction
      - `-1.8 \times 10^{-1}`
      - `-2.2 \times 10^{-2}`
      - `-2.2 \times 10^{-4}`
      - `-2.2 \times 10^{-6}`
      - `4.4 \times 10^{-9}`
      - `-2.3 \times 10^{-6}`
    * - condition number
      - `2.5 \times 10^{12}`
      - `2.0 \times 10^{13}`
      - `2.0 \times 10^{15}`
      - `2.0 \times 10^{17}`
      - `2.0 \times 10^{19}`
      - `1.9 \times 10^{21}`

The reaction of the penalty method, `k \int (u_p - u)`, here integrated with
40 Gauss-Legendre points along `x = a`, is the product of a large constant
and a small difference, and loses accuracy by cancellation for the highest
constants: from `k/(E h) = 10^8` its error no longer decreases as `1/k`, and
depends on the round-off, e.g. on the number of integration points.

**Linear buckling.** Cylinder of 5 panels under a uniform end-shortening,
see ``test_cylinder_end_shortening``. The error of the critical
end-shortening of the penalty method is `1.4 \times 10^{-1}`, `1.3 \times
10^{-2}`, `1.3 \times 10^{-3}`, `1.3 \times 10^{-4}` and `1.3 \times
10^{-6}` for `k/A_{11} = 10, 10^2, 10^3, 10^4, 10^6`, from above, since the
penalty springs shorten the cylinder less than the prescribed value. Both
methods take about 0.5 s, from the creation of the model to the critical
end-shortening. With `k/A_{11} = 10^6` the sparse eigenvalue solver of
``structsolve`` 0.8.0 failed its verification of the residuals of the
eigenpairs, because of the conditioning, and ``structsolve.lb`` fell back to
the dense solver with a ``DenseFallbackWarning``. Since ``structsolve``
0.9.0 the relative residual of the verification is at least
``RESIDUAL_FLOOR = 1e-8`` times a scale of the norms of the matrices, and
the eigenpairs of the sparse solver are accepted.

**Non-linear analysis.** Post-buckling of the plate at twice the critical
end-shortening, ``m = n = 8``, with a lateral point load of 30 N at the
centre as imperfection, as in ``test_nonlinear_reaction_matches_penalty``,
Newton-Raphson with 8 increments. Error of the axial reaction of the penalty
method, relative to that of the null-space method, with Crisfield's
convergence criterion, ``Analysis.convergence = 'crisfield'``, the default of
``structsolve`` 0.9.0 with ``relTOL = 1e-9``, and with the relative
criterion, ``'relative'``, the only one up to ``structsolve`` 0.8.0, whose
default was ``relTOL = 1e-6``:

.. list-table::
    :header-rows: 1

    * - criterion, `k / (E h)`
      - `10^2`
      - `10^4`
      - `10^6`
      - `10^8`
    * - ``'crisfield'``, ``relTOL = 1e-9``
      - `-3.6 \times 10^{-2}`
      - `-3.8 \times 10^{-4}`
      - `-3.8 \times 10^{-6}`
      - `-5.6 \times 10^{-8}`
    * - ``'crisfield'``, ``relTOL = 1e-6``
      - `-3.6 \times 10^{-2}`
      - `-4.0 \times 10^{-4}`
      - `-1.9 \times 10^{-5}`
      - `-1.5 \times 10^{-5}`
    * - ``'relative'``, ``relTOL = 1e-6``
      - `-3.6 \times 10^{-2}`
      - `-5.5 \times 10^{-4}`
      - `-2.1 \times 10^{-1}`
      - `1.1`
    * - ``'relative'``, ``relTOL = 1e-10``
      - `-3.6 \times 10^{-2}`
      - `-3.8 \times 10^{-4}`
      - `-1.9 \times 10^{-5}`
      - `-1.7 \times 10^{-4}`

Both criteria compare the norm of the residual with the largest norm of the
references `\lambda \{F_{ext}\}`, `\{F_{int}\}` and, when they apply, the
external force vector of the increment of the arc-length methods and
`\lambda \partial R/\partial \lambda`, see ``structsolve.convergence``. For
the penalty method `\lambda \{F_{ext}\}` and `\{F_{int}\}` contain the
penalty forces `k u_p`. The relative criterion measures them with the
Euclidean norm over all the degrees of freedom, which grows with `k`: with
``relTOL = 1e-6`` the highest penalty constants give the largest errors,
21% and 110%, after 12 and 8 evaluations of the internal force vector in
the 8 increments, against 28 for `k/(E h) = 10^2`, and even with ``relTOL =
1e-10`` the error grows again from `k/(E h) = 10^6` to `10^8`. Crisfield's
criterion uses the norm `\|v\|_{\boldsymbol{d}} = (\sum_j v_j^2/|d_j|)^{1/2}`
over the free degrees of freedom, the entries `d_j` of the diagonal scaling
vector `\boldsymbol{d}` being the diagonal of the linear stiffness matrix
``calc_kC()``, which contains the penalty stiffness, such that the scaled
norm of the penalty forces grows with `\sqrt{k}`, by a factor of 10 for
each factor of 100 of `k` in this case: with ``relTOL = 1e-9`` the error
decreases as `1/k` up to `k/(E h) = 10^8`, whereas with ``relTOL = 1e-6`` it
stops decreasing at about `2 \times 10^{-5}`. The null-space method has no
such trade-off, its reference being `\lambda \partial R_r/\partial \lambda`,
see :ref:`null_space_pd`: its reaction is the same to `3 \times 10^{-10}`
with the four combinations of the table.

**Run time.** For the linear analyses both methods have the same cost: the
elimination of the constraints is a small fraction of the assembly. For the
non-linear analysis above the run time is dominated by the assembly of the
tangent stiffness matrix. With ``panels`` 0.14.0 and ``structsolve`` 0.9.0,
on the same machine, medians of 15 runs of ``Analysis.static``: 0.60 s with
the null-space method and Crisfield's criterion with ``relTOL = 1e-9``, and
0.50 s and 0.66 s with the relative criterion with ``relTOL = 1e-6`` and
``1e-10``, against 0.46 s, 0.24 s and 0.45 s with the penalty method and
`k/(E h) = 10^4`. With
Crisfield's criterion the null-space method factorizes the tangent stiffness
matrix 29 times, against 25 for the penalty method. Its `\partial
R_r/\partial \lambda` requires the product `[K_T] \{c_{p1}\}`, computed
without assembling `[K_T]`, see :ref:`tangent_product`, at the cost of one
evaluation of the internal force vector, and ``structsolve``, since 0.8.0,
evaluates it only together with the tangent stiffness matrix. With
``panels`` 0.12.0 before these two improvements and the tabulation of the
approximation functions, see :ref:`tabulation`, the same analyses with the
relative criterion and ``relTOL = 1e-6`` and ``1e-10`` took 12.6 s and
15.8 s against 4.4 s and 9.2 s.

**When to use each method.** The null-space method is exact, needs no
penalty constant, keeps the conditioning of the model, and gives reactions
that do not depend on any constant; it is recommended for displacement
control, e.g. the end-shortening of panels and shells, and for linear
buckling under prescribed displacements. The penalty method remains the
default for backward compatibility, and it is needed when the prescribed
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
  edges, as in ``test_skin_base_nonlinear_equals_single_laminate``.

The prescribed displacements of :meth:`.Shell.add_point_pd`,
:meth:`.Shell.add_distr_pd_fixed_x` and :meth:`.Shell.add_distr_pd_fixed_y`
remain penalty stiffnesses by default, and are imposed exactly with
``pd_method='null-space'``, see :ref:`null_space_pd`. The
traction-separation law of ``'SB_TSL'`` remains a penalty stiffness.

The rank of the constraints is found with the tolerance ``tol`` of
:meth:`.MultiDomain.get_T`, ``1e-9`` by default, which applies to the
constraints normalized to a unit norm. The independent constraints have
pivots many orders of magnitude above the round-off errors of the redundant
ones, and a warning is issued when this gap is smaller than `10^4`.


API
---

The methods of :class:`.MultiDomain` for the null-space method are
:meth:`.MultiDomain.get_T`, :meth:`.MultiDomain.reduce`,
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
