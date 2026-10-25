.. _tangent_product:

Tangent stiffness matrix and its product by a vector
====================================================

This page documents how the internal force vector and its exact Jacobian,
the tangent stiffness matrix, are implemented:

- :ref:`tangent_clpt_donnell` derives `\boldsymbol{F}_{int}` and `\boldsymbol{K}_T` of the
  models ``plate_clpt_donnell`` and ``cylshell_clpt_donnell`` and relates
  every term to the function of the implementation that evaluates it,
  including the two integration paths, the grouping of the terms between
  :meth:`.Shell.calc_kC` and :meth:`.Shell.calc_kG`, and the multi-domain
  assembly;
- :ref:`tangent_kTv` derives the product `\boldsymbol{K}_T(\boldsymbol{c}) \boldsymbol{v}` for all the
  geometrically non-linear models, :meth:`.Shell.calc_kTv`, evaluated without
  assembling `\boldsymbol{K}_T`.

.. _tangent_clpt_donnell:

Tangent stiffness of the CLPT models with Donnell kinematics
------------------------------------------------------------

This section documents the internal force vector and the tangent stiffness
matrix as they are implemented in the models ``plate_clpt_donnell`` and
``cylshell_clpt_donnell``, both based on the classical laminated plate
theory (CLPT) with the Donnell-Mushtari-Vlasov (von Kármán) kinematics.
Each equation is related to the function of the implementation that
evaluates it, such that the reader can move between the equations and the
code in both directions. The objective is to show that each term of the
tangent stiffness matrix follows from the exact derivative of the internal
force vector,

.. math::
    :label: donnell-KT-definition

    \boldsymbol{K}_T(\boldsymbol{c}) = \frac{\partial
    \boldsymbol{F}_{int}}{\partial \boldsymbol{c}}

which is what delivers the quadratic convergence of the Newton-Raphson
iterations `\boldsymbol{K}_T(\boldsymbol{c}_k) \Delta
\boldsymbol{c} = \boldsymbol{F}_{ext} -
\boldsymbol{F}_{int}(\boldsymbol{c}_k)`, see
:doc:`ex_newton_raphson`. :ref:`tangent_kTv` uses the same notation, with
the membrane stresses `\tilde{\boldsymbol{N}}` of :eq:`donnell-KG`.

The tangent stiffness matrix is classically decomposed as

.. math::
    :label: donnell-KT-classical

    \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{0L} +
    \boldsymbol{K}_{L0} + \boldsymbol{K}_{LL} + \boldsymbol{K}_G(\boldsymbol{N})

with the constitutive matrices `\boldsymbol{K}_{C0}` to `\boldsymbol{K}_{LL}`
and the geometric stiffness `\boldsymbol{K}_G` of the membrane stress
`\boldsymbol{N}`, see :eq:`donnell-K-groups` and :eq:`donnell-KG`. The
implementation is organised around a three-way split:

.. math::
    :label: donnell-KT-split

    \boldsymbol{K}_T(\boldsymbol{c}) = \boldsymbol{K}_{C0} +
    \left( \boldsymbol{K}_{CNL}(\boldsymbol{c}) +
    \boldsymbol{K}_{GNL}(\boldsymbol{c}) \right) +
    \boldsymbol{K}_G(\boldsymbol{c})

with the non-linear constitutive stiffness matrix `\boldsymbol{K}_{CNL} =
\boldsymbol{K}_{0L} + \boldsymbol{K}_{L0} + \boldsymbol{K}_{LL}`, the
geometric stiffness `\boldsymbol{K}_{GNL}` of the membrane stress of the
non-linear strains, see :eq:`donnell-KGNL`, and `\boldsymbol{K}_G(\boldsymbol{c})`
that of the remaining membrane stress, see :eq:`donnell-grouping`. Its
relation to the grouping of the implementation, stated in a comment
of ``fkC_num`` in ``plate_clpt_donnell_num.pyx`` and in its cylindrical
twin, is established in :ref:`tangent_clpt_donnell_grouping` and tabulated
in :ref:`tangent_clpt_donnell_mapping`. The correspondence is not a
renaming only: the boundary between the constitutive and the geometric
contributions of :eq:`donnell-KT-split` sits at a different place from the
one of :eq:`donnell-KT-classical`, and
:ref:`tangent_clpt_donnell_discrepancies` says exactly where.

.. _tangent_clpt_donnell_paths:

The two implementation paths
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each model contributes two modules, registered in ``panels/modelDB.py``
under the keys ``matrices`` and ``matrices_num``:

- The **closed-form path**, ``matrices``, integrates the full domain
  analytically. The integrals of products of the approximation functions
  are evaluated in closed form, and the resulting matrices are constant
  with respect to `\boldsymbol{c}`. It provides ``fk0`` and
  ``fkG0`` of ``plate_clpt_donnell.pyx`` and ``cylshell_clpt_donnell.pyx``.
- The **numerically integrated path**, ``matrices_num``, applies a
  Gauss-Legendre rule over `n_x \times n_y` points, and is the only path
  that can honour a displacement state `\boldsymbol{c}`, a laminate
  stiffness varying over the domain, or a partial integration domain. It
  provides ``fkC_num``, ``fkG_num`` and ``calc_fint`` of
  ``plate_clpt_donnell_num.pyx`` and ``cylshell_clpt_donnell_num.pyx``.

The selection is made in :meth:`.Shell.calc_kC` and :meth:`.Shell.calc_kG`
by two flags, ``analytical_kC`` and ``analytical_kG``, initialised to
``True`` and cleared by the conditions of the following table.

.. list-table:: Conditions that force the numerically integrated path
    :header-rows: 1

    * - Condition
      - Affects
    * - :meth:`.Shell.is_partial_domain` is ``True``
      - ``kC`` and ``kG``
    * - ``c`` is given
      - ``kC`` and ``kG``
    * - the argument ``ABDnxny`` is given, or the attribute
        ``Shell.ABDnxny`` is set
      - ``kC``
    * - ``NLgeom=True``
      - ``kC`` and ``kG``
    * - ``c_cte`` is given
      - the ``kG`` block inside ``kC``

:meth:`.Shell.is_partial_domain` reports whether the integration limits
``x1, x2, y1, y2`` differ from the edges of the shell, `0`, `a` and `b`,
after :meth:`.Shell.integration_limits` has replaced a limit left as
``None`` by the corresponding edge and has validated `0 \le x_1 < x_2 \le
a` and `0 \le y_1 < y_2 \le b`. Two consequences are worth stating
explicitly:

- :meth:`.Shell.calc_kT` calls both :meth:`.Shell.calc_kC` and
  :meth:`.Shell.calc_kG` with ``NLgeom=True``. **The tangent stiffness
  matrix therefore always takes the numerically integrated path**, and the
  closed-form path never participates in a non-linear analysis.
- Conversely, the closed-form path is reached only for `\boldsymbol{K}_{C0}`
  and for `\boldsymbol{K}_G` evaluated at a uniform prescribed membrane
  stress state, i.e. for a linear static analysis and for the classical
  linear buckling problem `(\boldsymbol{K}_{C0} + \lambda \boldsymbol{K}_G)
  \boldsymbol{q} = \boldsymbol{0}`, over a full domain with a constant
  laminate.

Approximation functions
~~~~~~~~~~~~~~~~~~~~~~~

The three mid-surface displacements are approximated using the hierarchic
polynomials `P_i` of Bardell, see :ref:`theory_func_bardell`, evaluated by
``f``, ``fp`` and ``fpp`` of ``panels/core/include/bardell_functions.hpp``,
implemented in ``panels/core/src/bardell_functions.cpp``, with the Python
wrappers ``calc_f``, ``calc_fp`` and ``calc_fpp`` of
``panels/bardell.pyx``. Independent boundary flags are used for each
displacement component and for each direction:

.. math::
    :label: donnell-approx

    \begin{aligned}
        u(x,y) &= \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij}^u \, P_i^u(\xi) P_j^u(\eta) \\
        v(x,y) &= \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij}^v \, P_i^v(\xi) P_j^v(\eta) \\
        w(x,y) &= \sum_{i=1}^{m} \sum_{j=1}^{n} c_{ij}^w \, P_i^w(\xi) P_j^w(\eta)
    \end{aligned}

where the superscript in `P_i^u(\xi)` indicates that the flags
`\delta_{t1}`, `\delta_{r1}`, `\delta_{t2}`, `\delta_{r2}` of the first
four terms refer to the boundary conditions of `u` at `\xi = -1` and `\xi =
+1`, and similarly for `P_j^u(\eta)`, `P_i^v(\xi)`, and so forth. Only the
first four terms of ``vec_f`` in ``bardell_functions.cpp`` carry these
flags; the remaining terms vanish together with their first derivative at
both ends, which is what makes the set hierarchic. The implementation
provides 30 terms. The natural coordinates are:

.. math::

    \xi = \frac{2x}{a} - 1 \qquad \eta = \frac{2y}{b} - 1

where, for the cylindrical shell, `y = r\theta` is the circumferential arc
length and `b` the corresponding arc length of the panel. Derivatives with
respect to the physical coordinates follow from the chain rule:

.. math::

    \frac{\partial}{\partial x} = \frac{2}{a}\frac{\partial}{\partial \xi}
    \qquad \frac{\partial}{\partial y} = \frac{2}{b}\frac{\partial}{\partial
    \eta}

which is the origin of every factor ``(2/a)`` and ``(2/b)`` appearing in
the kernels, for instance in the strains of ``calc_fint``. In vector
notation:

.. math::

    u = \boldsymbol{S}^u \boldsymbol{c} \qquad v = \boldsymbol{S}^v
    \boldsymbol{c} \qquad w = \boldsymbol{S}^w \boldsymbol{c}

with the alternated ordering of the degrees of freedom:

.. math::

    \boldsymbol{c} = \left\{ c_{11}^u \quad c_{11}^v \quad c_{11}^w
    \quad c_{21}^u \quad c_{21}^v \quad c_{21}^w \quad \cdots \quad
    c_{mn}^u \quad c_{mn}^v \quad c_{mn}^w \right\}^\top

such that, using zero-based indices `i` and `j`, the coefficients
`c_{ij}^u`, `c_{ij}^v` and `c_{ij}^w` occupy the positions `3(jm+i)`,
`3(jm+i)+1` and `3(jm+i)+2` of `\boldsymbol{c}`. This is the mapping
``col = col0 + DOF*(j*m + i)`` with ``DOF = 3`` of the kernels, and the
size of `\boldsymbol{c}` is `3mn`, see :meth:`.Shell.get_size`. The
only non-zero components of `\boldsymbol{S}^u`, `\boldsymbol{S}^v` and
`\boldsymbol{S}^w` are:

.. math::

    S^u_{3(jm+i)} = P_i^u(\xi) P_j^u(\eta) \qquad
    S^v_{3(jm+i)+1} = P_i^v(\xi) P_j^v(\eta) \qquad
    S^w_{3(jm+i)+2} = P_i^w(\xi) P_j^w(\eta)

and, for instance:

.. math::

    S^w_{,x} = \frac{2}{a} P_i^{w\prime}(\xi) P_j^w(\eta) \qquad
    S^w_{,yy} = \frac{4}{b^2} P_i^w(\xi) P_j^{w\prime\prime}(\eta) \qquad
    S^w_{,xy} = \frac{4}{ab} P_i^{w\prime}(\xi) P_j^{w\prime}(\eta)

Throughout the kernels, the pair of indices `(i,j)` names the row term,
denoted `A`, and `(k,l)` names the column term, denoted `B`; the variables
``fAw``, ``gAweta``, ``fBwxi`` and so forth follow that convention, read
from the tables of the approximation functions evaluated once per
integration point, see :ref:`tabulation`. Only the upper triangle is
assembled, the lower one being recovered by ``finalize_symmetric_matrix``
of ``structsolve``, called by :meth:`.Shell.calc_kC` and
:meth:`.Shell.calc_kG` with ``finalize=True``, which is why every kernel
skips the entries with ``row > col``.

.. _tangent_clpt_donnell_kinematics:

Kinematics as implemented
~~~~~~~~~~~~~~~~~~~~~~~~~

The Donnell kinematics of a cylindrical shell neglect the terms of the
Sanders-Koiter kinematics that bring the in-plane displacements into the
rotations and the curvatures, see the model ``cylshell_clpt_sanders``, and
the plate kinematics of von Kármán are recovered by making `1/r = 0`:

.. math::
    :label: donnell-strains

    \boldsymbol{\varepsilon}^{(0)} = \left\{ \begin{matrix}
        u_{,x} + \frac{1}{2} w_{,x}^2 \\[2pt]
        v_{,y} + \frac{w}{r} + \frac{1}{2} w_{,y}^2 \\[2pt]
        u_{,y} + v_{,x} + w_{,x} w_{,y}
    \end{matrix} \right\}
    \qquad
    \boldsymbol{\varepsilon}^{(1)} = \left\{ \begin{matrix}
        -w_{,xx} \\[2pt] -w_{,yy} \\[2pt] -2 w_{,xy}
    \end{matrix} \right\}

:eq:`donnell-strains` is the strain state assembled term by term in
``calc_fint`` of ``cylshell_clpt_donnell_num.pyx``, where the hoop term
appears as ``1./r*cs[col+2]*fAw*gAw`` and the von Kármán contributions as
the products with ``wxi`` and ``weta``. The plate, ``calc_fint`` of
``plate_clpt_donnell_num.pyx``, differs from it in exactly one place, the
absence of that hoop term in ``eyy``; the five remaining strain components
are character for character identical. The same single-term difference
holds in ``fkG_num``.

.. _tangent_clpt_donnell_limit:

On presenting the plate as the limit of the cylinder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cylinder is the general case and the plate is its `1/r \to 0` limit,
but this statement must be read as a limit and not as a substitution, for
two reasons found in the code.

First, the plate is a *separate implementation*, not a specialisation
evaluated at a particular radius: ``panels/modelDB.py`` binds each model to
its own pair of modules, and no plate quantity is ever obtained by calling
a cylinder kernel.

Second, the limit is `r \to \infty` and not `r = 0`. The cylindrical
kernels contain `1/r` and `1/r^2` in denominators, for instance the term
``0.25*A22*a*b*fAwfBw*gAwgBw/(r*r)`` of ``fk0`` in
``cylshell_clpt_donnell.pyx`` and the corresponding terms of ``fkC_num``.
Evaluating them at `r = 0` produces infinities, which would only surface
later, as an assertion inside ``finalize_symmetric_matrix`` rejecting the
matrix. The radius is therefore validated before any kernel is called.
Each model declares in ``panels/modelDB.py`` whether it needs one, through
the key ``requires_r``, set for the cylinders and cleared for the plates.
``Shell._check_r()`` raises a ``ValueError`` naming the model when
``requires_r`` is set and `r` is ``None`` or `r \leq 0`, and it is called
by the methods that reach the kernels, e.g. :meth:`.Shell.calc_kC`,
:meth:`.Shell.calc_kG`, :meth:`.Shell.calc_kM`, :meth:`.Shell.calc_kA` and
:meth:`.Shell.calc_fint`. A shell declared with
``model='cylshell_clpt_donnell'`` and ``r=None`` is thus rejected with a
message pointing at ``plate_clpt_donnell``, rather than being evaluated at a
radius that is the opposite of the plate limit.

For the plate model, whose kernels never read ``r``, an unset radius is
normalised to zero, the value the remaining code compares against. There
`r = 0` is not a radius but a marker for the absence of one.

Numerically, the convergence of the cylinder towards the plate is first
order in `1/r`, as expected from :eq:`donnell-strains`, where the hoop term
enters linearly. Comparing the two implementations on the same
`\boldsymbol{c}`, with `a = 0.3` m, `b = 0.2` m, `m = n = 4` and an
unsymmetric laminate, gives the relative differences:

.. list-table:: Relative difference between the cylindrical and the plate
                kernels as the radius grows, at a fixed `\boldsymbol{c}`
    :header-rows: 1

    * - `r` [m]
      - `\|\boldsymbol{K}_C^{cyl} - \boldsymbol{K}_C^{plate}\| /
        \|\boldsymbol{K}_C^{plate}\|`
      - `\|\boldsymbol{F}_{int}^{cyl} - \boldsymbol{F}_{int}^{plate}\| /
        \|\boldsymbol{F}_{int}^{plate}\|`
    * - `10^{0}`
      - `3.58 \times 10^{-3}`
      - `2.26 \times 10^{-2}`
    * - `10^{2}`
      - `3.58 \times 10^{-5}`
      - `2.26 \times 10^{-4}`
    * - `10^{4}`
      - `3.58 \times 10^{-7}`
      - `2.26 \times 10^{-6}`
    * - `10^{6}`
      - `3.58 \times 10^{-9}`
      - `2.26 \times 10^{-8}`
    * - `10^{8}`
      - `3.58 \times 10^{-11}`
      - `2.26 \times 10^{-10}`

In the equations that follow, the cylinder is written as the general case
and the plate is recovered by dropping the terms carrying `1/r`, which is a
statement about the equations and not an instruction about the code.

Constitutive relation
^^^^^^^^^^^^^^^^^^^^^

The generalised strain and stress vectors are related by the laminate
stiffness:

.. math::
    :label: donnell-constitutive

    \hat{\boldsymbol{\varepsilon}} = \left\{ \begin{matrix}
    \boldsymbol{\varepsilon}^{(0)} \\ \boldsymbol{\varepsilon}^{(1)}
    \end{matrix} \right\}
    \qquad
    \hat{\boldsymbol{\sigma}} = \left\{ \begin{matrix} \boldsymbol{N} \\
    \boldsymbol{M} \end{matrix} \right\}
    = \begin{bmatrix} \boldsymbol{A} & \boldsymbol{B} \\ \boldsymbol{B} &
    \boldsymbol{D} \end{bmatrix}
    \left\{ \begin{matrix} \boldsymbol{\varepsilon}^{(0)} \\
    \boldsymbol{\varepsilon}^{(1)} \end{matrix} \right\}
    = \boldsymbol{C} \hat{\boldsymbol{\varepsilon}}

with `\boldsymbol{N} = \{N_{xx} \quad N_{yy} \quad N_{xy}\}^\top` and
`\boldsymbol{M} = \{M_{xx} \quad M_{yy} \quad M_{xy}\}^\top`.
:eq:`donnell-constitutive` is applied in ``calc_fint`` of both models. The
`6 \times 6` array `\boldsymbol{C}` is the argument named ``Finput`` of
every kernel, read from the attribute ``Shell.ABDnxny`` when it is set,
else from the read-only property :attr:`.Shell.ABD`, calculated by
``Shell._get_lam_ABD()`` from the laminate ``Shell.lam``. When ``Finput``
has the shape `(n_x, n_y, 6, 6)` it is interpreted as one constitutive
matrix per integration point, which is how variable stiffness laminates
enter, see :meth:`.Shell.integration_points`; a `(6, 6)` array is used for
the whole domain instead. Everything that follows holds pointwise, so a
`\boldsymbol{C}` varying over the domain changes nothing in the derivation.

Linear and non-linear differentiation operators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The linear part of :eq:`donnell-strains` is written as
`\hat{\boldsymbol{\varepsilon}}_L = \boldsymbol{B}_L \boldsymbol{c}`,
with:

.. math::
    :label: donnell-BL

    \boldsymbol{B}_L = \begin{bmatrix}
        \boldsymbol{S}^u_{,x} \\
        \boldsymbol{S}^v_{,y} + \frac{1}{r}\boldsymbol{S}^w \\
        \boldsymbol{S}^u_{,y} + \boldsymbol{S}^v_{,x} \\
        -\boldsymbol{S}^w_{,xx} \\
        -\boldsymbol{S}^w_{,yy} \\
        -2\boldsymbol{S}^w_{,xy}
    \end{bmatrix}

:eq:`donnell-BL` is exactly the set of six accumulators of the strains in
``fkG_num``, which is the one place in the code where the linear operator
appears on its own.

For the non-linear part, it is convenient to define the operator
`\boldsymbol{G}` that extracts the slopes of the deflection:

.. math::
    :label: donnell-G

    \boldsymbol{G} = \begin{bmatrix} \boldsymbol{G}_1 \\ \boldsymbol{G}_2
    \end{bmatrix} = \begin{bmatrix} \boldsymbol{S}^w_{,x} \\
    \boldsymbol{S}^w_{,y} \end{bmatrix}
    \qquad
    w_{,x} = \boldsymbol{G}_1 \boldsymbol{c}
    \qquad
    w_{,y} = \boldsymbol{G}_2 \boldsymbol{c}

:eq:`donnell-G` is the Donnell-Mushtari-Vlasov specialisation of the
rotation operators: the Sanders-Koiter kinematics add the terms
`\boldsymbol{S}^u/R_1` and `\boldsymbol{S}^v/R_2` of the in-plane
displacements, with the principal radii of curvature `R_1` and `R_2`. The
code confirms this: neither ``fkG_num`` nor ``fkG0`` contains
`\boldsymbol{S}^v`, and both allocate `1 \cdot m^2 n^2` entries instead of
the `9 \cdot m^2 n^2` of ``fkC_num`` (``fdim``), because only the `w`--`w`
block is populated. This holds for the cylinder as well as for the plate:
the `1/r` term present in `\boldsymbol{B}_L` has no counterpart in
`\boldsymbol{G}`.

In the kernels, the slopes are accumulated in the natural coordinates as
``wxi`` and ``weta``, in ``fkC_num`` and in ``calc_fint``, and converted as:

.. math::

    w_{,x} = \frac{2}{a}\,\texttt{wxi} \qquad w_{,y} = \frac{2}{b}\,\texttt{weta}

With :eq:`donnell-G`, the non-linear differentiation operator becomes:

.. math::
    :label: donnell-BNL

    \boldsymbol{B}_{NL}(\boldsymbol{c}) = \begin{bmatrix}
        w_{,x} \boldsymbol{G}_1 \\
        w_{,y} \boldsymbol{G}_2 \\
        w_{,y} \boldsymbol{G}_1 + w_{,x} \boldsymbol{G}_2 \\
        \boldsymbol{0} \\ \boldsymbol{0} \\ \boldsymbol{0}
    \end{bmatrix}

which gives the strain and its variation:

.. math::
    :label: donnell-strain-BNL

    \hat{\boldsymbol{\varepsilon}} = \left( \boldsymbol{B}_L + \frac{1}{2}\boldsymbol{B}_{NL} \right) \boldsymbol{c}
    \qquad
    \hat{\boldsymbol{\varepsilon}}_{NL} = \frac{1}{2}\boldsymbol{B}_{NL}\boldsymbol{c} = \left\{ \begin{matrix}
        \frac{1}{2} w_{,x}^2 \\[2pt] \frac{1}{2} w_{,y}^2 \\[2pt] w_{,x} w_{,y} \\[2pt] 0 \\ 0 \\ 0
    \end{matrix} \right\}
    \qquad
    \delta \hat{\boldsymbol{\varepsilon}} = (\boldsymbol{B}_L + \boldsymbol{B}_{NL}) \delta\boldsymbol{c}

The factor `\frac{1}{2}` of :eq:`donnell-strain-BNL` applies to the shear
row as well, and this is visible in the code as an asymmetry that is easy
to mistake for an error: in the strains of ``calc_fint`` the two normal
strains carry an explicit ``0.5`` whereas the shear strain does not. The
reason is that the shear row of `\boldsymbol{B}_{NL}` already contains two
terms, so `\frac{1}{2}\boldsymbol{B}_{NL}\boldsymbol{c}` evaluates to
`\frac{1}{2}(2 w_{,x} w_{,y}) = w_{,x}w_{,y}`, which is what the single
unscaled product in the code produces. Evaluating the non-linear shear
strain as `\boldsymbol{B}_{NL}\boldsymbol{c}` instead would give `2
w_{,x} w_{,y}`. That error does not show in `\boldsymbol{B}_{NL}` itself,
but it corrupts the stress state, and therefore the internal force vector
and the geometric stiffness matrix, leaving a tangent stiffness matrix that
is no longer the derivative of the internal force vector.

.. _tangent_clpt_donnell_fint:

Internal force vector
~~~~~~~~~~~~~~~~~~~~~

The internal force vector is the derivative of the strain energy, `U =
\frac{1}{2} \int_{\mathcal{A}} \hat{\boldsymbol{\varepsilon}}^\top
\boldsymbol{C} \hat{\boldsymbol{\varepsilon}} \, d\mathcal{A}`, integrated
over the mid-surface, with respect to `\boldsymbol{c}`:

.. math::
    :label: donnell-fint

    \boldsymbol{F}_{int}(\boldsymbol{c}) = \int_{\mathcal{A}} \left(
    \boldsymbol{B}_L + \boldsymbol{B}_{NL}(\boldsymbol{c})
    \right)^\top \hat{\boldsymbol{\sigma}}(\boldsymbol{c}) \,
    d\mathcal{A}

which gives, for the three degrees of freedom associated with the term
`(i,j)`:

.. math::
    :label: donnell-fint-rows

    \begin{aligned}
        F_{int}^{u} &= \int_{\mathcal{A}} \left( N_{xx} S^u_{,x} + N_{xy} S^u_{,y} \right) d\mathcal{A} \\
        F_{int}^{v} &= \int_{\mathcal{A}} \left( N_{yy} S^v_{,y} + N_{xy} S^v_{,x} \right) d\mathcal{A} \\
        F_{int}^{w} &= \int_{\mathcal{A}} \Big( N_{xx} w_{,x} S^w_{,x} + N_{yy} \left( \tfrac{1}{r} S^w + w_{,y} S^w_{,y} \right) + N_{xy} \left( w_{,y} S^w_{,x} + w_{,x} S^w_{,y} \right) \\
        &\qquad - M_{xx} S^w_{,xx} - M_{yy} S^w_{,yy} - 2 M_{xy} S^w_{,xy} \Big) d\mathcal{A}
    \end{aligned}

:eq:`donnell-fint-rows` is the update of ``fint`` in ``calc_fint`` of
``cylshell_clpt_donnell_num.pyx``. The hoop contribution `N_{yy} S^w / r`
appears there as ``Nyy*(fAw*gAw/r + ...)``, and the plate form, in
``plate_clpt_donnell_num.pyx``, is the same expression without it. Note
that `\boldsymbol{B}_L` contributes the hoop term and the three curvature
terms to the `w` row, whereas `\boldsymbol{B}_{NL}` contributes the three
products of membrane forces with slopes; the `u` and `v` rows receive no
contribution from `\boldsymbol{B}_{NL}`, which is why they contain only
`\boldsymbol{N}` and no slope.

Tangent stiffness matrix
~~~~~~~~~~~~~~~~~~~~~~~~

Differentiating :eq:`donnell-fint` with respect to `\boldsymbol{c}`,
:eq:`donnell-KT-definition`:

.. math::
    :label: donnell-KT

    \boldsymbol{K}_T = \frac{\partial \boldsymbol{F}_{int}}{\partial \boldsymbol{c}}
    = \underbrace{\int_{\mathcal{A}} (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top \boldsymbol{C} (\boldsymbol{B}_L + \boldsymbol{B}_{NL}) \, d\mathcal{A}}_{\text{from } \partial \hat{\boldsymbol{\sigma}} / \partial \boldsymbol{c}}
    + \underbrace{\int_{\mathcal{A}} \frac{\partial \boldsymbol{B}_{NL}^\top}{\partial \boldsymbol{c}} \hat{\boldsymbol{\sigma}} \, d\mathcal{A}}_{\text{from } \partial \boldsymbol{B}_{NL} / \partial \boldsymbol{c}}

The first integral uses `\partial \hat{\boldsymbol{\sigma}} / \partial
\boldsymbol{c} = \boldsymbol{C}\,(\boldsymbol{B}_L +
\boldsymbol{B}_{NL})`, which follows from :eq:`donnell-strain-BNL` together
with `\partial (\boldsymbol{B}_{NL}\boldsymbol{c}) / \partial
\boldsymbol{c} = 2\boldsymbol{B}_{NL}`, itself a consequence of
`\boldsymbol{B}_{NL}` being homogeneous of degree one in
`\boldsymbol{c}`. The second integral is evaluated by writing
`\boldsymbol{B}_{NL}^\top \hat{\boldsymbol{\sigma}}` explicitly with
:eq:`donnell-BNL`:

.. math::

    \boldsymbol{B}_{NL}^\top \hat{\boldsymbol{\sigma}} = \boldsymbol{G}_1^\top
    \left( N_{xx} w_{,x} + N_{xy} w_{,y} \right) + \boldsymbol{G}_2^\top
    \left( N_{xy} w_{,x} + N_{yy} w_{,y} \right)

and differentiating with respect to `\boldsymbol{c}` at a fixed stress
state, using :eq:`donnell-G`:

.. math::
    :label: donnell-KG

    \boldsymbol{K}_G(\boldsymbol{N}) = \int_{\mathcal{A}} \boldsymbol{G}^\top
    \tilde{\boldsymbol{N}} \boldsymbol{G} \, d\mathcal{A}
    \qquad
    \tilde{\boldsymbol{N}} = \begin{bmatrix} N_{xx} & N_{xy} \\ N_{xy} &
    N_{yy} \end{bmatrix}

:eq:`donnell-KG` is the geometric stiffness of the Donnell-Mushtari-Vlasov
kinematics, and it only has non-zero entries in the `w`--`w` blocks. For
the term pairs `A = (i,j)` and `B = (k,l)`:

.. math::
    :label: donnell-KG-entry

    K_{G\,[3(jm+i)+2,\;3(lm+k)+2]} = \int_{\mathcal{A}} \left( N_{xx}
    S^w_{A,x} S^w_{B,x} + N_{xy} \left( S^w_{A,x} S^w_{B,y} + S^w_{A,y}
    S^w_{B,x} \right) + N_{yy} S^w_{A,y} S^w_{B,y} \right) d\mathcal{A}

:eq:`donnell-KG-entry` is transcribed literally in ``fkG_num`` of
``cylshell_clpt_donnell_num.pyx``, where the four products of shape
functions appear in the same order. The plate writes the same expression in
an algebraically expanded form; the two are identical after collecting the
powers of `a` and `b`.

.. _tangent_clpt_donnell_grouping:

Which stress enters the geometric stiffness, and the degree of homogeneity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first integral of :eq:`donnell-KT` expands into the four classical
groups of :eq:`donnell-KT-classical`:

.. math::
    :label: donnell-K-groups

    \boldsymbol{K}_{C0} = \int_{\mathcal{A}} \boldsymbol{B}_L^\top \boldsymbol{C} \boldsymbol{B}_L \, d\mathcal{A} \qquad
    \boldsymbol{K}_{0L} = \int_{\mathcal{A}} \boldsymbol{B}_L^\top \boldsymbol{C} \boldsymbol{B}_{NL} \, d\mathcal{A} \qquad
    \boldsymbol{K}_{L0} = \boldsymbol{K}_{0L}^\top \qquad
    \boldsymbol{K}_{LL} = \int_{\mathcal{A}} \boldsymbol{B}_{NL}^\top \boldsymbol{C} \boldsymbol{B}_{NL} \, d\mathcal{A}

The membrane stress of :eq:`donnell-KG` is also split, according to the
strain that produces it:

.. math::
    :label: donnell-N-split

    \boldsymbol{N} = \bar{\boldsymbol{N}} + \boldsymbol{N}_L + \boldsymbol{N}_{NL}
    \qquad
    \boldsymbol{N}_L = \boldsymbol{A}\boldsymbol{\varepsilon}^{(0)}_L + \boldsymbol{B}\boldsymbol{\varepsilon}^{(1)}
    \qquad
    \boldsymbol{N}_{NL} = \boldsymbol{A} \left\{ \begin{matrix} \frac{1}{2}
    w_{,x}^2 \\[2pt] \frac{1}{2} w_{,y}^2 \\[2pt] w_{,x} w_{,y} \end{matrix}
    \right\}

where `\bar{\boldsymbol{N}}` is an optional prescribed pre-stress, used for
instance to superpose combined load cases, see ``c_cte`` and ``Nxx_cte``,
``Nyy_cte``, ``Nxy_cte`` in :meth:`.Shell.calc_kC`, and `\boldsymbol{B}` does
not contribute to `\boldsymbol{N}_{NL}` because `\boldsymbol{\varepsilon}^{(1)}`
is linear for the CLPT. In the code, `\bar{\boldsymbol{N}} + \boldsymbol{N}_L`
is computed in ``fkG_num``, from the arguments ``Nxx0, Nyy0, Nxy0`` and from
the linear strains of :eq:`donnell-BL`; and `\boldsymbol{N}_{NL}` is
computed separately, in ``fkC_num``, as ``NxxNL, NyyNL, NxyNL``, where the
``0.5`` factors of :eq:`donnell-N-split` are explicit. Since
:eq:`donnell-KG` is linear in `\boldsymbol{N}`:

.. math::
    :label: donnell-KGNL

    \boldsymbol{K}_G(\boldsymbol{N}) = \boldsymbol{K}_G(\bar{\boldsymbol{N}} +
    \boldsymbol{N}_L) + \boldsymbol{K}_{GNL}
    \qquad \text{with} \qquad
    \boldsymbol{K}_{GNL} = \boldsymbol{K}_G(\boldsymbol{N}_{NL})

With respect to `\boldsymbol{c}`: `\boldsymbol{K}_{C0}` is constant;
`\boldsymbol{K}_{0L}`, `\boldsymbol{K}_{L0}` and
`\boldsymbol{K}_G(\boldsymbol{N}_L)` are homogeneous of degree one; and
`\boldsymbol{K}_{LL}` and `\boldsymbol{K}_{GNL}` are homogeneous of degree
two. The implementation groups them as:

.. math::
    :label: donnell-grouping

    \begin{aligned}
        \texttt{fkC_num} \;\rightarrow\; \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{c}) + \boldsymbol{K}_{GNL}(\boldsymbol{c}) &= \boldsymbol{K}_{C0} + \boldsymbol{K}_{0L} + \boldsymbol{K}_{L0} + \boldsymbol{K}_{LL} + \boldsymbol{K}_{GNL} \\
        \texttt{fkG_num} \;\rightarrow\; \boldsymbol{K}_G(\boldsymbol{c}) &= \boldsymbol{K}_G(\bar{\boldsymbol{N}} + \boldsymbol{N}_L) \\
        \texttt{calc_kT} \;\rightarrow\; \boldsymbol{K}_T(\boldsymbol{c}) &= \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}(\boldsymbol{c}) + \boldsymbol{K}_{GNL}(\boldsymbol{c}) + \boldsymbol{K}_G(\boldsymbol{c})
    \end{aligned}

:eq:`donnell-grouping` is the statement written as a comment in ``fkC_num``
of both models, and the summation of the two matrices happens in
:meth:`.Shell.calc_kT`. The four constitutive
groups of :eq:`donnell-K-groups` are not assembled separately: the kernel
evaluates the whole product `(\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top
\boldsymbol{C} (\boldsymbol{B}_L + \boldsymbol{B}_{NL})` in expanded
symbolic form, in nine blocks, one per pair of displacements, and
`\boldsymbol{K}_{GNL}` is then added on top of the `w`--`w` block only,
under the comment ``# KGNL``.

**Why** `\boldsymbol{K}_{GNL}` **sits where it does.** Although
`\boldsymbol{K}_{GNL}` is geometric in nature, being a
`\boldsymbol{G}^\top \tilde{\boldsymbol{N}} \boldsymbol{G}` product, it is
deliberately collected together with the constitutive terms so that
`\boldsymbol{K}_G(\boldsymbol{c})` stays homogeneous of degree one:

.. math::

    \boldsymbol{K}_G(\lambda\boldsymbol{c}) = \lambda\,\boldsymbol{K}_G(\boldsymbol{c})

The linear buckling problem `(\boldsymbol{K}_{C0} + \lambda
\boldsymbol{K}_G(\boldsymbol{c}_0)) \boldsymbol{q} = \boldsymbol{0}`
needs this property, because there the load multiplier `\lambda` scales a
stress state obtained from a linear fundamental state
`\boldsymbol{c}_0`, and a `\boldsymbol{K}_G` containing a degree-two
term would make `\lambda` scale that state non-linearly, such that the
problem would no longer be a linear eigenvalue problem. With the grouping of
:eq:`donnell-grouping`, both requirements are met at once:
`\boldsymbol{K}_T` is the exact Jacobian of `\boldsymbol{F}_{int}`, so the
Newton-Raphson iterations converge quadratically, and `\boldsymbol{K}_G`
remains usable as the geometric stiffness of a linear buckling analysis. The
consequence is that ``fkG_num`` has no switch for `\boldsymbol{N}_{NL}` at
all: its signature takes no ``NLgeom`` argument, and its docstring states
why. :meth:`.Shell.calc_kG` still accepts ``NLgeom``, but only to force the
numerically integrated path of :ref:`tangent_clpt_donnell_paths`; it does
not change the value of `\boldsymbol{K}_G`.

Mapping onto the three-way split
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With :eq:`donnell-grouping` established, the three-way split of
:eq:`donnell-KT-split` is:

.. math::
    :label: donnell-three-way

    \begin{aligned}
        \boldsymbol{K}_{C0} &= \int_{\mathcal{A}} \boldsymbol{B}_L^\top \boldsymbol{C} \boldsymbol{B}_L \, d\mathcal{A} \\
        \boldsymbol{K}_{CNL}(\boldsymbol{c}) + \boldsymbol{K}_{GNL}(\boldsymbol{c}) &= \boldsymbol{K}_{0L} + \boldsymbol{K}_{L0} + \boldsymbol{K}_{LL} + \boldsymbol{K}_G(\boldsymbol{N}_{NL}) \\
        \boldsymbol{K}_G(\boldsymbol{c}) &= \boldsymbol{K}_G(\bar{\boldsymbol{N}} + \boldsymbol{N}_L)
    \end{aligned}

i.e. `\boldsymbol{K}_{C0}` is everything that ``fkC_num`` produces at
`\boldsymbol{c} = \boldsymbol{0}`, `\boldsymbol{K}_{CNL} +
\boldsymbol{K}_{GNL}` is everything else that ``fkC_num`` produces, and
`\boldsymbol{K}_G(\boldsymbol{c})` is what ``fkG_num`` produces. The
partition is clean, with the non-linear constitutive stiffness matrix
`\boldsymbol{K}_{CNL}` restricted to the terms of `\boldsymbol{B}_{NL}` of
:eq:`donnell-K-groups` and the geometric term `\boldsymbol{K}_{GNL}` written
next to it; :ref:`tangent_clpt_donnell_discrepancies` states the
consequence.

There is no separate code path for `\boldsymbol{K}_{C0}`,
`\boldsymbol{K}_{CNL}` and `\boldsymbol{K}_{GNL}`: ``fkC_num`` returns their
sum. `\boldsymbol{K}_{C0}`
alone is obtained either from the closed-form ``fk0`` or by calling
``fkC_num`` with `\boldsymbol{c} = \boldsymbol{0}` or with
``NLgeom=0``, both of which make ``wxi`` and ``weta`` vanish, since their
accumulation is guarded by ``if NLgeom == 1``. With `w_{,x} = w_{,y} = 0`
one has `\boldsymbol{B}_{NL} = \boldsymbol{0}` and `\boldsymbol{N}_{NL} =
\boldsymbol{0}`, so `\boldsymbol{K}_{CNL}` and `\boldsymbol{K}_{GNL}` vanish
identically, as a comment in ``fkC_num`` notes.

.. _tangent_clpt_donnell_closed_form:

The closed-form path
~~~~~~~~~~~~~~~~~~~~

The closed-form kernels evaluate the same `\boldsymbol{K}_{C0}` and
`\boldsymbol{K}_G`, over the full domain and for a constant laminate, by
separating the two directions. Every entry of :eq:`donnell-K-groups`
restricted to `\boldsymbol{B}_L` is a sum of products of the form

.. math::
    :label: donnell-separable

    \int_{\mathcal{A}} \frac{\partial^{p} S^{\alpha}_A}{\partial x^{p}}
    \frac{\partial^{q} S^{\beta}_B}{\partial y^{q}} \, d\mathcal{A}
    = \frac{ab}{4} \left( \frac{2}{a} \right)^{p} \left( \frac{2}{b}
    \right)^{q} \int_{-1}^{+1} P_i^{\alpha (p)} P_k^{\beta} \, d\xi
    \int_{-1}^{+1} P_j^{\alpha} P_l^{\beta (q)} \, d\eta

and the two one-dimensional integrals are evaluated analytically by the six
functions ``integral_ff``, ``integral_ffp``, ``integral_ffpp``,
``integral_fpfp``, ``integral_fpfpp`` and ``integral_fppfpp`` of
``panels/core/include/bardell.hpp``, declared at the top of
``plate_clpt_donnell.pyx`` and wrapped for Python as ``calc_integral_ff``
and so forth in ``panels/bardell.pyx``. They are hoisted out of the inner
loops, the `\xi` integrals inside the `i`, `k` loops and the `\eta`
integrals inside the `j`, `l` loops, which is what makes this path cheap: no
loop over integration points exists at all.

As an example, the first entry of `\boldsymbol{K}_{C0}` in ``fk0`` of
``plate_clpt_donnell.pyx`` reads

.. code-block:: cython

    k0v[c] += A11*b*fAuxifBuxi*gAugBu/a + A16*(fAufBuxi*gAuetagBu
            + fAuxifBu*gAugBueta) + A66*a*fAufBu*gAuetagBueta/b

which is the `u`--`u` entry of `\int \boldsymbol{B}_L^\top \boldsymbol{C}
\boldsymbol{B}_L \, d\mathcal{A}`, the factor `b/a` following from
:eq:`donnell-separable` with `p = q = 1`. The corresponding cylindrical
entry, in ``fk0`` of ``cylshell_clpt_donnell.pyx``, is identical, the two
implementations of `\boldsymbol{K}_{C0}` differing only by the five entries
carrying `1/r` and `1/r^2` that :eq:`donnell-BL` introduces through
`\boldsymbol{S}^w/r`.

Two properties of this path bound its use:

- ``fk0`` takes no `\boldsymbol{c}` and no constitutive matrix per
  point; its signature is ``fk0(shell, size, row0, col0)``. It can only
  ever produce `\boldsymbol{K}_{C0}`.
- ``fkG0`` takes the three membrane stress resultants as *scalars*,
  ``fkG0(Nxx, Nyy, Nxy, shell, size, row0, col0)``, so it evaluates
  `\boldsymbol{K}_G(\bar{\boldsymbol{N}})` for a membrane state uniform over
  the domain. The bodies of ``fkG0`` are identical for the plate and for the
  cylinder, consistent with `\boldsymbol{G}` of :eq:`donnell-G` carrying no
  `1/r`; the only entry assembled is character for character the same in
  both modules.

The geometric stiffness must therefore be integrated with ``fkG_num``, and
not ``fkG0``, whenever the pre-buckling resultants are not uniform, e.g.
near the edges of a cylinder whose fundamental state is not a pure membrane
state: ``fkG0`` cannot represent it.

.. _tangent_clpt_donnell_quadrature:

Numerical integration
~~~~~~~~~~~~~~~~~~~~~

All integrals of the numerically integrated path are computed in natural
coordinates using the Gauss-Legendre quadrature, with `n_x` and `n_y`
points obtained from ``scipy.special.roots_legendre``:

.. math::
    :label: donnell-quadrature

    \int_{\mathcal{A}} f \, d\mathcal{A} = \frac{ab}{4} \int_{-1}^{+1}
    \int_{-1}^{+1} f(\xi, \eta) \, d\xi \, d\eta
    \approx \frac{ab}{4} \sum_{p=1}^{n_x} \sum_{q=1}^{n_y} W_p W_q \,
    f(\xi_p, \eta_q)

For integration over a sub-domain `x_1 \le x \le x_2` and `y_1 \le y \le
y_2`, the factor `ab/4` becomes `(x_2-x_1)(y_2-y_1)/4`, computed as
``intx*inty``, and the Gauss points are mapped into the corresponding
interval of `\xi` and `\eta`, the interval itself being given by the
physical limits returned by :meth:`.Shell.integration_limits` and mapped
unconditionally, which gives exactly `\xi, \eta = \pm 1` for the full
domain; :meth:`.Shell.integration_points` returns the same points and
weights in physical coordinates. The factor `1/4` is not always visible in
the kernels: it is sometimes folded into the symbolic expansion, as in the
entries of ``fkC_num`` of ``plate_clpt_donnell_num.pyx``, and sometimes
written out, as ``weight*(intx*inty/4)`` in ``fkG_num`` of
``cylshell_clpt_donnell_num.pyx``. Both forms are the same number.

`\boldsymbol{F}_{int}` and `\boldsymbol{K}_T` must be integrated with the
same quadrature rule, which is why :meth:`.Shell.calc_fint` and
:meth:`.Shell.calc_kT` both default `n_x` and `n_y` to the attributes ``nx``
and ``ny`` of the shell, or to the shape of ``ABDnxny`` when it has one
matrix per point. The consistency between them is then exact at the
discrete level, whether or not the quadrature integrates the polynomials
exactly: :eq:`donnell-KT` is the derivative of :eq:`donnell-fint`
pointwise, hence also after applying the same weighted sum to both.

.. _tangent_clpt_donnell_multidomain:

Multi-domain assembly
~~~~~~~~~~~~~~~~~~~~~

For an assembly of panels, each :class:`.Shell` writes its own contribution
into a global matrix of size `\sum_p 3 m_p n_p`, offset by the attributes
``row_start`` and ``col_start`` and passed through as the arguments
``row0`` and ``col0`` of every kernel. Denoting by `\boldsymbol{L}_p` the
Boolean operator that selects the degrees of freedom of panel `p` from the
global vector, the assembly is:

.. math::
    :label: donnell-assembly

    \begin{aligned}
        \boldsymbol{F}_{int} &= \sum_{p} \boldsymbol{L}_p^\top \boldsymbol{F}_{int}^{(p)}(\boldsymbol{L}_p \boldsymbol{c}) + \boldsymbol{K}_{C}^{conn} \boldsymbol{c} \\
        \boldsymbol{K}_T &= \sum_{p} \boldsymbol{L}_p^\top \left( \boldsymbol{K}_{C}^{(p)} + \boldsymbol{K}_{G}^{(p)} \right) \boldsymbol{L}_p + \boldsymbol{K}_{C}^{conn}
    \end{aligned}

:eq:`donnell-assembly` is :meth:`.MultiDomain.calc_fint` and
:meth:`.MultiDomain.calc_kT`. Three points are relevant here:

- The panel-level blocks enter unchanged. :meth:`.MultiDomain.calc_kT`
  calls ``p.calc_kC(..., NLgeom=True)`` and ``p.calc_kG(..., NLgeom=True)``
  for each panel, i.e. exactly the two calls of :meth:`.Shell.calc_kT`, so
  :eq:`donnell-grouping` holds panel by panel.
- Both are assembled with ``finalize=False`` and the symmetrisation is
  applied once to the sum, because each panel only fills its own upper
  triangle. The load stiffness of the follower pressure loads, which is
  unsymmetric, is added after the symmetrisation, see
  :ref:`follower_pressure`.
- The connection matrix `\boldsymbol{K}_{C}^{conn}` is added after the
  loop, to `\boldsymbol{K}_T` and, as `\boldsymbol{K}_{C}^{conn}
  \boldsymbol{c}`, to `\boldsymbol{F}_{int}`. Its own derivation is
  out of the scope of this section, see :ref:`null_space`; what matters here
  is that it is added consistently to both, so that the assembled
  `\boldsymbol{K}_T` remains the derivative of the assembled
  `\boldsymbol{F}_{int}` whenever `\boldsymbol{K}_{C}^{conn}` is
  independent of `\boldsymbol{c}`.

.. _tangent_clpt_donnell_mapping:

Symbols and implementation variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following tables relate the symbols used herein to the variables of the
implementation, those of the plate kernels unless stated, the cylindrical
twins using the same names, and map the matrix terms.

.. list-table:: Symbols and implementation variables
    :header-rows: 1

    * - Symbol
      - Variable
      - Where
    * - `\boldsymbol{c}`
      - ``cs``, ``c``
      - argument of the kernels
    * - `m`, `n`
      - ``m``, ``n``
      - attributes of :class:`.Shell`, :meth:`.Shell.get_size`
    * - `a`, `b`, `r`
      - ``a``, ``b``, ``r``
      - attributes of :class:`.Shell`
    * - `P_i^u(\xi)`, `P_i^{u\prime}(\xi)`
      - ``fAu``, ``fAuxi``
      - ``ftab`` of the kernels
    * - `P_j^w(\eta)`, `P_j^{w\prime}`, `P_j^{w\prime\prime}`
      - ``gAw``, ``gAweta``, ``gAwetaeta``
      - ``gtab`` of the kernels
    * - `P_k^w(\xi)`, `P_l^w(\eta)` (term `B`)
      - ``fBw``, ``gBw``
      - ``ftab``, ``gtab`` of the kernels
    * - `\delta_{t1}`, `\delta_{r1}`, `\delta_{t2}`, `\delta_{r2}`
      - ``x1w``, ``x1wr``, ``x2w``, ``x2wr``
      - attributes of :class:`.Shell`
    * - `\partial w/\partial\xi`, `\partial w/\partial\eta`
      - ``wxi``, ``weta``
      - ``fkC_num``, ``calc_fint``
    * - `w_{,x}`, `w_{,y}`
      - ``wx``, ``wy``
      - ``fkC_num``
    * - `\boldsymbol{\varepsilon}^{(0)}`, `\boldsymbol{\varepsilon}^{(1)}`
      - ``exx, eyy, gxy``, ``kxx, kyy, kxy``
      - ``calc_fint``, ``fkG_num``
    * - `\boldsymbol{C}`, i.e. `\boldsymbol{A}`, `\boldsymbol{B}`,
        `\boldsymbol{D}`
      - ``Finput``, ``F``, ``A11`` ... ``D66``
      - argument of the kernels, :attr:`.Shell.ABD`, ``Shell.ABDnxny``
    * - `\boldsymbol{N}`, `\boldsymbol{M}`
      - ``Nxx, Nyy, Nxy``, ``Mxx, Myy, Mxy``
      - ``calc_fint``
    * - `\boldsymbol{N}_{NL}`
      - ``NxxNL``, ``NyyNL``, ``NxyNL``
      - ``fkC_num``
    * - `\bar{\boldsymbol{N}}`
      - ``Nxx0, Nyy0, Nxy0``; ``Shell.Nxx`` ...
      - arguments of ``fkG_num``
    * - `\bar{\boldsymbol{N}} + \boldsymbol{N}_L`
      - ``Nxx, Nyy, Nxy`` of ``fkG_num``
      - ``fkG_num``
    * - `x_2-x_1`, `y_2-y_1`
      - ``intx``, ``inty``
      - numerical kernels
    * - `W_p W_q`
      - ``weight``
      - numerical kernels
    * - `n_x`, `n_y`
      - ``nx``, ``ny``
      - attributes of :class:`.Shell`, arguments of the calc methods
    * - row term `A = (i,j)`, column term `B = (k,l)`
      - ``i, j``, ``k, l``
      - kernels
    * - `3(jm+i)`
      - ``DOF*(j*m + i)``
      - ``row``, ``col`` of the kernels
    * - offsets of `\boldsymbol{L}_p`
      - ``row0``, ``col0``
      - ``row_start``, ``col_start`` in :meth:`.MultiDomain.calc_kT`

The two last rows of the next table hold the terms whose placement differs
between the two namings.

.. list-table:: Matrix terms: notation of this section, name used by the
                code and producing function
    :header-rows: 1

    * - Herein
      - Code name
      - Function
    * - `\boldsymbol{K}_{C0}`
      - ``KC0``
      - ``fk0``
    * - `\boldsymbol{K}_{C0}`
      - ``KC0``
      - ``fkC_num`` with ``NLgeom=0``
    * - `\boldsymbol{K}_{CNL} = \boldsymbol{K}_{0L} + \boldsymbol{K}_{L0} +
        \boldsymbol{K}_{LL}`
      - ``KCNL``
      - ``fkC_num`` with ``NLgeom=1``
    * - `\boldsymbol{K}_G(\boldsymbol{c})`
      - ``KG(Nbar + N_L)``
      - ``fkG_num``
    * - `\boldsymbol{K}_G(\bar{\boldsymbol{N}})`
      - ``KG0``
      - ``fkG0``
    * - `\boldsymbol{F}_{int}`
      - ``fint``
      - ``calc_fint``
    * - `\boldsymbol{K}_T`
      - ``kT``
      - :meth:`.Shell.calc_kT`
    * - `\boldsymbol{K}_{GNL}`, next to `\boldsymbol{K}_{CNL}`
      - ``KGNL``
      - ``fkC_num``, under ``# KGNL``
    * - not in `\boldsymbol{K}_T`
      - ``KG0``, ``KG(c_cte)``
      - :meth:`.Shell.calc_kC`, added to ``kC``

.. _tangent_clpt_donnell_verification:

Verification
~~~~~~~~~~~~

Three test modules pin the claims of this section. The first two are
parametrised over ``plate_clpt_donnell`` and ``cylshell_clpt_donnell``, both
using an unsymmetric, unbalanced laminate so that `\boldsymbol{B} \neq
\boldsymbol{0}` and `A_{16}, A_{26} \neq 0`.

``tests/tests_shell/test_tangent_consistency.py`` verifies the Jacobian
claim of :eq:`donnell-KT` directly:

- ``test_tangent_is_derivative_of_fint`` is the directional Taylor test.
  Given an arbitrary state `\boldsymbol{c}` and direction
  `\boldsymbol{v}`, `\boldsymbol{F}_{int}(\boldsymbol{c} +
  h\boldsymbol{v}) = \boldsymbol{F}_{int}(\boldsymbol{c}) + h
  \boldsymbol{K}_T(\boldsymbol{c}) \boldsymbol{v} + O(h^2)`, so the
  relative error

  .. math::

      e(h) = \frac{\left\| \boldsymbol{F}_{int}(\boldsymbol{c} +
      h\boldsymbol{v}) - \boldsymbol{F}_{int}(\boldsymbol{c}) - h
      \boldsymbol{K}_T \boldsymbol{v} \right\|}{\left\| h \boldsymbol{K}_T
      \boldsymbol{v} \right\|}

  must fall proportionally to `h`. The test asserts a factor of at least
  five per decade over `h \in [10^{-6}, 10^{-2}]`, and that `e(h)/h` does
  not drift by more than a factor of two. An inconsistent tangent leaves a
  remainder of order `h` and `e(h)` plateaus instead.
- ``test_tangent_matches_finite_difference_jacobian`` compares every entry
  of `\boldsymbol{K}_T` against a central finite-difference Jacobian of
  `\boldsymbol{F}_{int}`.
- ``test_kG_is_homogeneous_of_degree_one`` asserts
  `\boldsymbol{K}_G(2\boldsymbol{c}) =
  2\boldsymbol{K}_G(\boldsymbol{c})`, which is the property that the
  placement of `\boldsymbol{K}_{GNL}` in :eq:`donnell-grouping` exists to
  preserve.
- ``test_fint_and_tangent_in_the_undeformed_state`` asserts
  `\boldsymbol{F}_{int}(\boldsymbol{0}) = \boldsymbol{0}` and
  `\boldsymbol{K}_T(\boldsymbol{0}) = \boldsymbol{K}_{C0}`, which is
  :eq:`donnell-three-way` at `\boldsymbol{c} = \boldsymbol{0}`.
- ``test_newton_raphson_converges_quadratically`` manufactures an
  equilibrium state and checks that the error is squared at each iteration.

``tests/tests_shell/test_nonlinear.py`` exercises the same tangent in a full
Newton-Raphson solution, deep in the postbuckling regime at about 37 times
the first linear buckling load, and asserts both the converged deflection
and a convergence order above `1.6`.

``tests/tests_shell/test_argument_validation.py`` pins the argument handling
around these matrices, and with it three claims of this section:

- ``test_KGNL_is_in_calc_kC_and_not_in_calc_kG`` asserts that
  `\boldsymbol{K}_G` is homogeneous of degree one, that ``NLgeom`` does not
  change its value, and that `\boldsymbol{K}_T` is exactly the sum of the
  two matrices of :eq:`donnell-grouping`.
- ``test_NLgeom_and_not_c_selects_the_large_displacement_matrix`` asserts
  that ``fkC_num`` returns `\boldsymbol{K}_{C0}` whenever ``NLgeom=0``,
  whatever `\boldsymbol{c}`, and that the result then matches the
  closed-form ``fk0``, as stated when mapping the code onto
  :eq:`donnell-three-way`.
- ``test_cylshell_approaches_plate_as_radius_grows`` asserts the monotonic
  convergence of :ref:`tangent_clpt_donnell_limit`, and
  ``test_cylshell_requires_positive_radius`` that `r = 0`, a negative `r`
  and an unset `r` are all rejected for the cylinder.

Together the first two modules support every claim made here about
`\boldsymbol{K}_T` being the exact Jacobian, about `\boldsymbol{K}_G` being
homogeneous of degree one, and about `\boldsymbol{K}_T(\boldsymbol{0}) =
\boldsymbol{K}_{C0}`. They do not separately test the decomposition of
:eq:`donnell-grouping` term by term, nor the closed-form path against the
numerically integrated one; those were verified independently while
preparing this section, by rebuilding `\boldsymbol{B}_L`,
`\boldsymbol{B}_{NL}` and `\boldsymbol{G}` from :eq:`donnell-BL`,
:eq:`donnell-BNL` and :eq:`donnell-G` at the Gauss points and comparing
against the compiled kernels, with the following results, for both models,
`m = n = 4`, `n_x = n_y = 10`, an unsymmetric laminate and a random
`\boldsymbol{c}` with deflections of twenty thicknesses.

.. list-table:: Relative difference between the compiled kernels and the
                equations of this section, rebuilt independently
    :header-rows: 1

    * - Kernel
      - Equation
      - Plate
      - Cylinder
    * - ``fkC_num``
      - `\int (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top \boldsymbol{C}
        (\boldsymbol{B}_L + \boldsymbol{B}_{NL}) + \boldsymbol{K}_{GNL}`
      - `4.0 \times 10^{-16}`
      - `2.3 \times 10^{-16}`
    * - ``fkG_num``
      - `\boldsymbol{K}_G(\bar{\boldsymbol{N}} + \boldsymbol{N}_L)`,
        :eq:`donnell-KG`
      - `5.7 \times 10^{-16}`
      - `4.9 \times 10^{-16}`
    * - ``calc_fint``
      - `\int (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top
        \hat{\boldsymbol{\sigma}}`, :eq:`donnell-fint`
      - `2.2 \times 10^{-16}`
      - `2.2 \times 10^{-16}`
    * - ``fk0``
      - ``fkC_num`` at `\boldsymbol{c} = \boldsymbol{0}`, `n_x = n_y
        = 30`
      - `4.7 \times 10^{-15}`
      - `4.6 \times 10^{-15}`
    * - ``fkG0``
      - ``fkG_num`` at `\boldsymbol{c} = \boldsymbol{0}`, `n_x = n_y
        = 30`
      - `8.6 \times 10^{-15}`
      - `8.8 \times 10^{-15}`

The last two rows also establish that the closed-form and the numerically
integrated paths agree to round-off once the quadrature is fine enough to
integrate the polynomials exactly, so the choice between them of
:ref:`tangent_clpt_donnell_paths` is a matter of cost and of capability,
not of accuracy.

.. _tangent_clpt_donnell_discrepancies:

Where the naming and the code do not line up
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two points deserve to be stated rather than smoothed over.

**The** `\boldsymbol{K}_C`--`\boldsymbol{K}_G` **boundary moves.** The
split of :eq:`donnell-KT-split` and the split of :eq:`donnell-KT-classical`
are not the same split, although both sum to the same `\boldsymbol{K}_T`.
Writing `\boldsymbol{K}_C` and `\boldsymbol{K}_G` for the classical
constitutive and geometric stiffness matrices, defined respectively as the
first and the second integral of :eq:`donnell-KT`:

.. math::
    :label: donnell-boundary

    \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL} =
    \boldsymbol{K}_C + \boldsymbol{K}_{GNL}
    \qquad
    \boldsymbol{K}_G(\boldsymbol{c}) = \boldsymbol{K}_G - \boldsymbol{K}_{GNL}

where the left-hand sides are the matrices returned by ``fkC_num`` and
``fkG_num``. The term `\boldsymbol{K}_{GNL}` is geometric by construction,
being `\int \boldsymbol{G}^\top \tilde{\boldsymbol{N}}_{NL} \boldsymbol{G} \,
d\mathcal{A}`, yet the implementation adds it to the constitutive terms
`\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL}`. What ``fkC_num`` *adds beyond*
`\boldsymbol{c} = \boldsymbol{0}` is therefore the sum `\boldsymbol{K}_{CNL}
+ \boldsymbol{K}_{GNL}`, and not `\boldsymbol{K}_{CNL}` alone, which keeps
its meaning of *the non-linear constitutive terms* of
:eq:`donnell-K-groups`. This section writes the sum explicitly wherever the
matrix of ``fkC_num`` is meant, as in :eq:`donnell-KT-split` and
:eq:`donnell-three-way`.

:meth:`.Shell.calc_kC` **does not always return** `\boldsymbol{K}_{C0} +
\boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}`. If ``c_cte`` is given or any of ``Nxx_cte``,
``Nyy_cte``, ``Nxy_cte`` is non-zero, a geometric block is added into the
matrix returned as ``kC``, either `\boldsymbol{K}_G(\boldsymbol{N}^{cte})`
through ``fkG0`` or `\boldsymbol{K}_G` evaluated at ``c_cte`` through
``fkG_num``. This is the superposition device of combined load cases, which
keeps a part of the stress state out of reach of the load multiplier
`\lambda`. With ``NLgeom=True`` and follower pressure loads, their load
stiffness :meth:`.Shell.calc_kCfollower` is added as well. The sum
`\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}` therefore describes the return
value of :meth:`.Shell.calc_kC` only when no constant pre-stress and no
follower load is declared. The two modules of
:ref:`tangent_clpt_donnell_verification` that verify the Jacobian set the
constant stresses to zero and do not exercise this branch; the third module
does, but only to check that it is reached safely, not the value of the
block.

Note on symbols
~~~~~~~~~~~~~~~

Three groups of symbols are introduced here because the implementation
distinguishes quantities that the classical treatment does not:

- `\boldsymbol{K}_{C0}`, `\boldsymbol{K}_{CNL}` and `\boldsymbol{K}_{GNL}`,
  for the constant constitutive part, the state-dependent constitutive part
  and the geometric part of the matrix returned by ``fkC_num``. The
  classical symbol for the first of them is `\boldsymbol{K}_0`;
  `\boldsymbol{K}_{C0}`, the linear constitutive stiffness matrix, is used
  throughout, in :eq:`donnell-K-groups` and in the three-way split of
  :eq:`donnell-KT-split`, which the implementation is organised around.
- `\bar{\boldsymbol{N}}`, `\boldsymbol{N}_L` and `\boldsymbol{N}_{NL}`, for the
  three contributions to the membrane stress entering :eq:`donnell-KG`. The
  decomposition is needed here because the implementation evaluates the
  third one separately, and places it on the other side of the
  `\boldsymbol{K}_C`--`\boldsymbol{K}_G` boundary.
- `\boldsymbol{L}_p`, for the Boolean selection operator of the multi-domain
  assembly of :eq:`donnell-assembly`, which has no counterpart in the
  single-domain treatment.


.. _tangent_kTv:

Product of the tangent stiffness matrix by a vector
---------------------------------------------------

:meth:`.Shell.calc_kTv` and :meth:`.MultiDomain.calc_kTv` return the
product `\boldsymbol{K}_T(\boldsymbol{c}) \boldsymbol{v}` of the tangent stiffness matrix of
:meth:`.Shell.calc_kT` by a vector, without assembling `\boldsymbol{K}_T`, for all the
geometrically non-linear models. Its cost is about that of one evaluation of
the internal force vector, instead of the assembly of `\boldsymbol{K}_T`, whose cost grows
with the square of the number of terms of the approximation. It is the
exact product, to round-off, and the exact directional derivative of
:meth:`.Shell.calc_fint`.

The product is what the prescribed displacements imposed exactly need at
each configuration, `-\partial \boldsymbol{R}_r/\partial \lambda = \boldsymbol{Z}^\top (\boldsymbol{F}_{ext} -
\boldsymbol{K}_T(\boldsymbol{c}) \boldsymbol{c}_{p1})`, see :ref:`null_space_pd`, and what matrix-free
iterative solvers and directional checks of the tangent need.

Derivation
~~~~~~~~~~

All the geometrically non-linear models, the plates and the Donnell and
Sanders cylinders of the CLPT, FSDT and TSDT, have at an integration point
the generalized strains

.. math::

    \boldsymbol{\varepsilon}(\boldsymbol{c}) = \left( \boldsymbol{B}_L + \tfrac{1}{2} \boldsymbol{B}_{NL} \right) \boldsymbol{c},
    \qquad \boldsymbol{\beta} = \boldsymbol{G} \boldsymbol{c}

where `\boldsymbol{\beta}` are the two rotations of the von Kármán terms, `w_{,x}`
and `w_{,y}` for the Donnell kinematics, `w_{,y} - v/r` replacing `w_{,y}`
for Sanders, `\boldsymbol{B}_L` is the linear strain operator and `\boldsymbol{B}_{NL}`, linear in
`\boldsymbol{\beta}`, enters only the membrane rows:

.. math::

    \frac{\partial \boldsymbol{B}_{NL}}{\partial \beta_x} = \begin{bmatrix} \boldsymbol{G}_x \\ 0
    \\ \boldsymbol{G}_y \\ 0 \end{bmatrix}, \qquad
    \frac{\partial \boldsymbol{B}_{NL}}{\partial \beta_y} = \begin{bmatrix} 0 \\ \boldsymbol{G}_y
    \\ \boldsymbol{G}_x \\ 0 \end{bmatrix}

The factor `1/2` of `\boldsymbol{B}_{NL}` gives the quadratic terms `\beta_x^2/2`,
`\beta_y^2/2` and `\beta_x \beta_y`. With the constitutive matrix `\boldsymbol{C}` of
``ABDnxny``, the internal force vector is `\boldsymbol{F}_{int} = \int (\boldsymbol{B}_L +
\boldsymbol{B}_{NL})^\top \boldsymbol{C} \boldsymbol{\varepsilon}(\boldsymbol{c}) dA`, and its directional derivative
along `\boldsymbol{v}` is

.. math::
    :label: kTv

    \boldsymbol{K}_T \boldsymbol{v} = \frac{d}{dt} \boldsymbol{F}_{int}(\boldsymbol{c} + t \boldsymbol{v}) \Big|_{t=0} =
    \int \left( (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top \boldsymbol{C} (\boldsymbol{B}_L + \boldsymbol{B}_{NL}) \boldsymbol{v} + \boldsymbol{G}^\top
    \tilde{\boldsymbol{N}} \boldsymbol{G} \boldsymbol{v} \right) dA

because the strains are a quadratic form whose bilinear form is `\boldsymbol{B}_L +
\boldsymbol{B}_{NL}`, such that `\frac{d}{dt} \boldsymbol{\varepsilon}(\boldsymbol{c} + t \boldsymbol{v}) = (\boldsymbol{B}_L +
\boldsymbol{B}_{NL}) \boldsymbol{v}`, and `\frac{d}{dt} \boldsymbol{B}_{NL}^\top \boldsymbol{\sigma} = \boldsymbol{G}^\top \tilde{\boldsymbol{N}} \boldsymbol{G}
\boldsymbol{v}` from the derivatives of `\boldsymbol{B}_{NL}` above, with `\tilde{\boldsymbol{N}}` the membrane
stresses. The two terms are those assembled by ``fkC_num`` and ``fkG_num``
of each model. The constant membrane stresses ``Nxx_cte``, ``Nyy_cte`` and
``Nxy_cte``, whose geometric stiffness :meth:`.Shell.calc_kC` adds to the
tangent stiffness matrix, are added to `\tilde{\boldsymbol{N}}`.

At each integration point, with the approximation functions tabulated once,
the kernels ``calc_kTv`` of ``panels/models/<model>_num.pyx`` compute

1. `\boldsymbol{\beta} = \boldsymbol{G} \boldsymbol{c}` and `\boldsymbol{G} \boldsymbol{v}`;
2. `\boldsymbol{\varepsilon} = (\boldsymbol{B}_L + \tfrac{1}{2} \boldsymbol{B}_{NL}) \boldsymbol{c}` and
   `\delta \boldsymbol{\varepsilon} = (\boldsymbol{B}_L + \boldsymbol{B}_{NL}) \boldsymbol{v}`;
3. the stresses `\boldsymbol{C} \boldsymbol{\varepsilon}`, of which `\tilde{\boldsymbol{N}}`, and `\boldsymbol{C}
   \delta \boldsymbol{\varepsilon}`;
4. the sum of :eq:`kTv` over the terms.

Each step is a loop over the `m n` terms, against a loop over the `m^2
n^2` pairs of terms for the assembly of `\boldsymbol{K}_T`.

The expressions are generated with SymPy, which also verifies, for every
model, that :eq:`kTv` is the derivative
of the internal force vector of one term, with a symbolic constitutive
matrix; `\boldsymbol{B}_{NL}` and `\boldsymbol{G}` being linear in the degrees of freedom, the
identity extends to any number of terms.

Follower pressure loads
~~~~~~~~~~~~~~~~~~~~~~~

With follower pressure loads, :meth:`.Shell.calc_kTv` adds the product of
their load stiffness :meth:`.Shell.calc_kCfollower` as the central
difference

.. math::

    -\frac{\partial \boldsymbol{F}_p}{\partial \boldsymbol{c}} \boldsymbol{v} = -\frac{\boldsymbol{F}_p(\boldsymbol{c} + t \boldsymbol{v})
    - \boldsymbol{F}_p(\boldsymbol{c} - t \boldsymbol{v})}{2 t}

which is exact for any `t`, the force vector `\boldsymbol{F}_p(\boldsymbol{c})` being affine in
`\boldsymbol{c}` for ``follower='linear'`` and quadratic for
``follower='quadratic'``, see :ref:`follower_pressure`. `t = \max(\|\boldsymbol{c}\|/\|\boldsymbol{v}\|,
1)` avoids the cancellation of large and small terms.

Verification
~~~~~~~~~~~~

The checks of the tangent stiffness matrix of
``tests/tests_shell/test_tangent_consistency.py`` are repeated for the
product, for all 9 models, in ``tests/tests_shell/test_tangent_product.py``:

- the product equals ``calc_kT(c) @ v`` with a relative difference below
  `10^{-12}`, also with a constant membrane stress state, one constitutive
  matrix per integration point, a partial integration domain, an offset in
  a larger vector, and dead and follower pressure loads at a load factor
  ``inc``;
- the directional Taylor test of :meth:`.Shell.calc_fint`, whose remainder
  falls as `O(h^2)`;
- the matrix built column by column from the product against the central
  difference of :meth:`.Shell.calc_fint`;
- in the undeformed state the product is `\boldsymbol{K}_{C0} \boldsymbol{v}`;
- Newton-Raphson with the tangent built from the product converges
  quadratically:

.. literalinclude:: ../../tests/tests_shell/test_tangent_product.py
    :pyobject: test_newton_raphson_converges_quadratically

``tests/multidomain/test_tangent_product_multidomain.py`` checks
:meth:`.MultiDomain.calc_kTv` with penalty connections and follower loads,
and `\partial \boldsymbol{R}_r/\partial \lambda` of the prescribed displacements against
the assembled tangent stiffness matrix.

Performance
~~~~~~~~~~~

Single shell, laminate ``[0, 45, -45, 90]``, the default integration points
of :func:`panels.shell.gauss_points`, `\max(2m - 1, 7)` by `\max(2n - 1,
7)`, one core; :meth:`.Shell.calc_kT` with the tabulated kernels, see
:ref:`tabulation`; the state `\boldsymbol{c}` and the vector `\boldsymbol{v}` of
``random_state`` of ``tests/tests_shell/test_tangent_consistency.py``. With
``panels`` 0.14.0, medians of 9 samples, 3 in each of 3 processes,
interleaved, after a warm-up; ratio: of ``calc_kT`` to ``calc_kTv``;
difference: relative difference between the product and ``calc_kT(c) @ v``,
the largest of the three sizes. With the former default, ``nx = 2m``,
``ny = 2n``, one point more in each direction, the times measured in the
same session, interleaved, are 3 to 16% longer:

.. list-table::
    :header-rows: 1

    * - model
      - `m = n`
      - size
      - ``calc_kT``
      - ``calc_fint``
      - ``calc_kTv``
      - ratio
      - difference
    * - ``plate_clpt_donnell``
      - 8 / 12 / 16
      - 192 / 432 / 768
      - 0.013 / 0.11 / 0.59 s
      - 0.80 / 3.5 / 11 ms
      - 1.1 / 5.0 / 16 ms
      - 12 / 23 / 37
      - `1.0 \times 10^{-15}`
    * - ``cylshell_fsdt_sanders``
      - 8 / 12 / 16
      - 320 / 720 / 1280
      - 0.10 / 1.1 / 6.0 s
      - 1.5 / 6.7 / 22 ms
      - 1.7 / 8.2 / 26 ms
      - 61 / 132 / 234
      - `9.1 \times 10^{-16}`
    * - ``plate_tsdt_donnell``
      - 8 / 12 / 16
      - 320 / 720 / 1280
      - 0.19 / 2.1 / 12 s
      - 1.6 / 7.6 / 24 ms
      - 1.7 / 8.7 / 27 ms
      - 110 / 246 / 441
      - `1.4 \times 10^{-15}`

The product costs 1.1 to 1.5 times :meth:`.Shell.calc_fint`, and the ratio
grows with the number of terms. With the previous kernels of ``panels``
0.12.0, whose approximation functions were evaluated in the innermost loops,
``calc_kT`` takes 2.8, 4.7 and 5.8 s for `m = n = 12`, measured in the same
way with the former default ``nx = 2m``, ``ny = 2n``, i.e. 500 to 610 times
the product with the same points. The times published with ``panels``
0.13.0 were longer, 7.0, 10.9 and 11.9 s with the previous kernels, 700
times the product of then.

Displacement-controlled post-buckling of the plate of
``tests/multidomain/test_null_space_pd.py``, the model of
``test_nonlinear_displacement_control`` with `m = n = 12` (407 reduced
degrees of freedom), prescribed displacements imposed exactly. Run time of
``Analysis.static``, on the same machine, medians of 3 runs in separate
processes, interleaved, after a warm-up, of ``panels`` before these
developments (0.12.0, with the previous kernels and the assembly of `\boldsymbol{K}_T`
for `\partial \boldsymbol{R}_r/\partial \lambda`), after them (0.13.0) and 0.14.0, with
``structsolve`` 0.7.0, 0.8.0 and 0.9.0. Before 0.9.0 the convergence was
checked with the relative criterion, ``relTOL = 1e-6`` by default;
``structsolve`` 0.9.0 checks with Crisfield's criterion, ``relTOL = 1e-9``
by default, and keeps the former with ``Analysis.convergence = 'relative'``.
The rows of 0.12.0 and 0.13.0 were measured with the integration points of
then, ``nx = 2m``, ``ny = 2n``, i.e. `24 \times 24`; those of 0.14.0 with
the default of :func:`panels.shell.gauss_points`, `23 \times 23`, in a later
session in which `24 \times 24` points, interleaved, gave times 1 to 6%
shorter than in the first one:

.. list-table::
    :header-rows: 1

    * - ``panels``
      - ``structsolve``
      - criterion
      - Newton-Raphson
      - modified Newton-Raphson
      - Riks
    * - 0.12.0
      - 0.7.0
      - relative, ``1e-6``
      - 116 s
      - 279 s
      - 149 s
    * - 0.12.0
      - 0.8.0
      - relative, ``1e-6``
      - 92.7 s
      - 73.8 s
      - 141 s
    * - 0.13.0
      - 0.7.0
      - relative, ``1e-6``
      - 4.84 s
      - 4.24 s
      - 7.71 s
    * - 0.13.0
      - 0.8.0
      - relative, ``1e-6``
      - 4.76 s
      - 3.40 s
      - 7.60 s
    * - 0.14.0
      - 0.9.0
      - relative, ``1e-6``
      - 4.36 s
      - 2.94 s
      - 6.86 s
    * - 0.14.0
      - 0.9.0
      - Crisfield, ``1e-9``
      - 5.33 s
      - 5.36 s
      - 8.41 s

The reactions at `\lambda = 1` agree within a relative `4 \times 10^{-6}`
with the relative criterion, and to ten significant digits with Crisfield's
criterion. Before, almost all the time was in the assembly of `\boldsymbol{K}_T`, about
3 s each with the previous kernels: with ``structsolve`` 0.7.0 inside
``calc_fext``, for `\partial \boldsymbol{R}_r/\partial \lambda`, at every iterate where
`\boldsymbol{K}_T` was not cached, e.g. at each iteration of the modified
Newton-Raphson method, whereas ``structsolve`` 0.8.0 evaluates `\partial
\boldsymbol{R}_r/\partial \lambda` only together with `\boldsymbol{K}_T`. After, with Crisfield's
criterion, 80 to 87% of the time is in the 37, 35 and 60 assemblies of
`\boldsymbol{K}_T` for the Newton iterations (31, 19 and 49 with the relative
criterion), and the products cost 0.19 to 0.32 s in total. The modified
Newton-Raphson method with Crisfield's criterion takes 16 increments and 35
assemblies with `23 \times 23` points, but 11 increments and 25 assemblies,
4.12 s, with `24 \times 24` points in the same session, whereas both
numbers of points integrate the matrices exactly: its iterations depend on
the round-off. The times
published with ``panels`` 0.13.0 for the first four rows were 2.1 to 2.5
times longer, e.g. 229 s and 11.8 s with ``structsolve`` 0.8.0 and the
Newton-Raphson method; the speed-ups from 0.12.0 to 0.13.0 are the same
within 7%, except with the modified Newton-Raphson method and
``structsolve`` 0.7.0, 77 then and 66 now.
