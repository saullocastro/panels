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
      - `u_1 + d_{sb} w_{1,x} = u_2`, `v_1 + d_{sb} w_{1,y} = v_2`, `w_1 =
        w_2`, with `d_{sb} = h_1/2 + h_2/2`, see ``kCSB.pyx``
      - `u, v, w` at the interface, `z = -h_1/2` of ``p1`` and `z = h_2/2`
        of ``p2``, see :func:`.fkCSB_sdt`, and `\phi_x, \Phi_y` only when
        the key ``'kr'`` of the connection is given and not zero

where `\Phi_y = \phi_y + v/r` with the Sanders-Koiter kinematics and `\Phi_y
= \phi_y` otherwise, and the terms `v/r` of the classical laminated plate
theory are present only with the Sanders-Koiter kinematics. The keys
``'kt'`` and ``'kr'`` of the connections are not used, apart from ``'kr'``
of ``'SB'`` with the shear deformation theories, which tells whether the
rotations are connected.


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

- With ``'SB'`` and the classical laminated plate theory, `u_1 = u_2 - d_{sb}
  w_{,x}`. Restraining the normal in-plane displacement of an edge in both
  domains, e.g. `u = 0` at `x = 0`, also restrains the rotation `w_{,x}` at
  that edge, which is clamped. The in-plane displacements of the bonded
  laminates must be restrained consistently with the single laminate that
  they represent, e.g. restraining only the tangential displacement of the
  edges, as in ``test_skin_base_nonlinear_equals_single_laminate``.

Only the connections of the domains are imposed exactly: the prescribed
displacements of :meth:`.Shell.add_point_pd`,
:meth:`.Shell.add_distr_pd_fixed_x` and :meth:`.Shell.add_distr_pd_fixed_y`,
and the traction-separation law of ``'SB_TSL'``, remain penalty stiffnesses.

The rank of the constraints is found with the tolerance ``tol`` of
:meth:`.MultiDomain.get_T`, ``1e-9`` by default, which applies to the
constraints normalized to a unit norm. The independent constraints have
pivots many orders of magnitude above the round-off errors of the redundant
ones, and a warning is issued when this gap is smaller than `10^4`.


API
---

The methods of :class:`.MultiDomain` for the null-space method are
:meth:`.MultiDomain.get_T`, :meth:`.MultiDomain.reduce`,
:meth:`.MultiDomain.expand` and :meth:`.MultiDomain.get_reduced_functions`.

.. automodule:: panels.multidomain.connections.nullspace

.. autodata:: panels.multidomain.connections.nullspace.NULL_SPACE_FUNCS
    :no-value:

.. autofunction:: panels.multidomain.connections.nullspace.constraint_matrix

.. autofunction:: panels.multidomain.connections.nullspace.null_space_basis
