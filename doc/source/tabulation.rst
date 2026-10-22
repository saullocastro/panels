.. _tabulation:

Approximation functions tabulated at the integration points
===========================================================

The numerically integrated matrices and vectors of ``panels`` are sums over
the integration points of products of the approximation functions of the
Ritz method. This page describes how the kernels evaluate these functions:
before version 0.13.0 they were evaluated again in every loop over the terms
of the matrix, the innermost loop included; they are now evaluated once per
integration point and term, in tables. The results are the same, bit for
bit, and the assembly is several times faster.

Integrands
----------

Each field of a panel is approximated with the products of the hierarchical
functions of Bardell, see :ref:`theory_func_bardell`, along `\xi` and `\eta`:

.. math::

    u(\xi, \eta) = \sum_{j=0}^{n-1} \sum_{i=0}^{m-1} c_{u, ij}\, f_i(\xi)\,
    g_j(\eta)

where `f_i` and `g_j` depend on the boundary conditions of the field, i.e.
the flags ``x1u, x1ur, x2u, x2ur`` along `x` and ``y1u, y1ur, y2u, y2ur``
along `y`. The entry of a stiffness matrix between the terms `(i, j)` and
`(k, l)` is the sum over the `n_x \times n_y` Gauss-Legendre points

.. math::

    K_{(ij)(kl)} = \sum_{p=1}^{n_x} \sum_{q=1}^{n_y} w_p w_q\,
    \mathcal{F}\left(f_i(\xi_p), f_k(\xi_p), g_j(\eta_q), g_l(\eta_q),
    \ldots\right)

where `\mathcal{F}` combines the functions and their derivatives `f'`,
`f''`, `g'`, `g''` of each field with the constitutive matrix and, for the
non-linear matrices, with the displacement gradients at the point.

Previous implementation: functions in the innermost loop
--------------------------------------------------------

The kernels had the structure below, e.g. ``fkC_num`` of
``plate_clpt_donnell_num.pyx``, with seven functions per side::

    for ptx in range(nx):
        for pty in range(ny):
            xi, eta, weight = ...
            for i in range(m):
                fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                ...                                   # 7 functions of i
                for k in range(m):
                    fBu = f(k, xi, x1u, x1ur, x2u, x2ur)
                    ...                               # 7 functions of k
                    for j in range(n):
                        gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                        ...                           # 7 functions of j
                        for l in range(n):
                            gBu = f(l, eta, y1u, y1ur, y2u, y2ur)
                            ...                       # 7 functions of l
                            kCv[c] += weight*(...)    # the entries of (ij)(kl)

The value of `g_l(\eta_q)` does not depend on `i`, `k` or `j`, but it was
computed again for each of them: per integration point, `7 m^2 n^2`
evaluations in the innermost loop, `7 m^2 n` in the loop over `j`, and so
on, for only `7 (m + n)` distinct values. Each evaluation is a polynomial
of degree up to 29 in `\xi`, selected by the index of the term, see
``panels/core/src/bardell_functions.cpp``, which costs much more than the
arithmetic of the entry itself. For `m = n = 12` this is about 10^5
evaluations per integration point, against 168 distinct values.

The internal force vector, ``calc_fint``, has three loops over the `m n`
terms per integration point (the rotations, the strains, the forces), with
the functions evaluated again in each; the connections between panels and
the stiffeners have the same nested structure, along a line or over an
area.

Current implementation: tables per integration point
----------------------------------------------------

The functions are evaluated once per integration point and term, at the
start of the loop over the integration points, in a table along `x` and
another along `y`, and the loops over the terms read them::

    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)
    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi, eta, weight = ...
                # approximation functions evaluated once
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ...                               # one row per function
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    ...
                for i in range(m):
                    fAu = ftab[0, i]
                    ...
                    for k in range(m):
                        fBu = ftab[0, k]
                        ...
                        for j in range(n):
                            gAu = gtab[0, j]
                            ...
                            for l in range(n):
                                gBu = gtab[0, l]
                                ...
                                kCv[c] += weight*(...)

Per integration point the number of evaluations falls from `O(m^2 n^2)` to
`O(m + n)`, with tables of a few kilobytes; the loops over the pairs of
terms keep only the arithmetic of the entries, which is unchanged. The
functions read from the tables are the same numbers, combined in the same
order, such that the matrices and vectors are identical, bit for bit, to
those of the previous implementation.

The kernels that integrate along a line, e.g. the connections at a constant
`\xi` or the prescribed displacements, evaluate the functions across the
line at a single coordinate: their tables are filled once, before the
outermost loop over the terms.

Kernels
-------

All the kernels that evaluate the approximation functions in their loops
over the terms are tabulated:

- the 9 geometrically non-linear models, ``panels/models/<model>_num.pyx``:
  ``fkC_num``, ``fkG_num``, ``calc_fint``, ``fkM_num``, ``fkAx_num``,
  ``fkAy_num``, ``calc_fext_follower`` and ``fkCfollower_num``;
- the connections, ``panels/multidomain/connections``: ``kCSSxcte``,
  ``kCSSycte``, ``kCBFxcte``, ``kCBFycte`` (including the shear
  deformation theories), ``kCpd`` (prescribed displacements) and
  ``kCSB_dmg`` (skin-base connection with damage, the cohesive zone of the
  double cantilever beam, see :ref:`cohesive_zone`);
- the blade stiffeners, ``panels/stiffener/models``.

The analytical kernels, which use closed-form integrals of the products of
the functions, e.g. ``fk0`` or ``kCSB``, are not affected. The kernel
``calc_kTv`` of the product of the tangent stiffness matrix by a vector,
see :ref:`tangent_product`, is written with the tables from the start.

Implementation
--------------

The rewriting is done by :mod:`panels.dev.tabulate`, applied by the SymPy
generators of the modules of the shear deformation theories, of the
cylinders with the Sanders kinematics and of the follower pressure loads,
and with ``python -m panels.dev.tabulate <file.pyx> ...`` to the modules
written by hand. For each kernel it

1. finds the calls ``NAME = f(IDX, COORD, x1F, x1Fr, x2F, x2Fr)``, with
   ``f``, ``fp`` or ``fpp``, the flags of the field ``F`` along `x` (or
   `y`) and an optional suffix of the panel, e.g. ``x1u1`` for the panel 1
   of a connection, and fails on a call in another form;
2. groups them in one table per direction, panel and coordinate, with one
   row per function and as many columns as the loop over ``IDX``;
3. fills the tables right before the loops over the terms of the loop over
   the integration points, after checking that the coordinates are not
   modified afterwards, and replaces each call by the read of the table;
4. declares and allocates the tables outside the ``nogil`` block.

A kernel already tabulated is left unchanged, such that applying it twice
does nothing. The helper ``_terms_sdt`` of the connections of the shear
deformation theories, whose flags come from an array, was tabulated by
hand in the same way.

Verification
------------

- The matrices and vectors of every tabulated kernel were compared, bit for
  bit, with those of the previous implementation, recording the output of
  each call while running 20 test files of the models, of the follower
  pressures, of the partial integration domains, of the aerodynamic
  matrices, of the connections, of the cohesive zone and of the stiffened
  panels: the 73946 calls of 92 kernels gave identical outputs.
- ``tests/test_tabulate.py`` checks the rewriting and that no kernel of the
  package evaluates the approximation functions in its loops any more.
- The tests of the package, among which the consistency of the tangent
  stiffness matrix with the internal force vector,
  ``tests/tests_shell/test_tangent_consistency.py``, are unchanged and
  pass.

Performance
-----------

Single shell, laminate ``[0, 45, -45, 90]``, ``nx = 2m``, ``ny = 2n``
integration points, a follower pressure, one core, times in seconds before
and after the tabulation, with the speed-up:

.. list-table::
    :header-rows: 1

    * - model, `m = n`
      - ``fkC_num`` (``NLgeom``)
      - ``fkG_num`` (``NLgeom``)
      - ``calc_fint``
      - ``fkCfollower_num``
      - ``fkM_num``
    * - ``plate_clpt_donnell``, 8
      - 0.26 → 0.028 (9.5)
      - 0.11 → 0.0039 (29)
      - 0.020 → 0.0028 (7.2)
      - 0.35 → 0.023 (15)
      - 0.18 → 0.0085 (21)
    * - ``plate_clpt_donnell``, 12
      - 4.95 → 0.30 (17)
      - 2.03 → 0.033 (62)
      - 0.17 → 0.015 (11)
      - 6.21 → 0.24 (26)
      - 3.11 → 0.087 (36)
    * - ``plate_clpt_donnell``, 16
      - 39.6 → 1.65 (24)
      - 16.1 → 0.17 (97)
      - 0.76 → 0.051 (15)
      - 49.0 → 1.33 (37)
      - 24.0 → 0.48 (50)
    * - ``cylshell_fsdt_sanders``, 12
      - 8.48 → 1.54 (5.5)
      - 2.34 → 0.074 (32)
      - 0.23 → 0.021 (11)
      - 6.26 → 0.38 (16)
      - 4.05 → 0.12 (33)
    * - ``plate_tsdt_donnell``, 12
      - 10.3 → 2.92 (3.5)
      - 1.53 → 0.040 (39)
      - 0.23 → 0.022 (10)
      - 6.20 → 0.24 (26)
      - 5.01 → 0.54 (9.2)

The speed-up grows with the number of terms. It is smaller for ``fkC_num``
of the shear deformation theories, whose integrands have more arithmetic
per pair of terms, which the tabulation does not change.

Connections between two plates with `m = n = 8` and `12`, the skin-base
connection with damage integrated with `50 \times 25` points:

.. list-table::
    :header-rows: 1

    * - kernel
      - `m = n = 8`
      - `m = n = 12`
    * - ``kCSSxcte`` (the three blocks)
      - 1.5 → 0.32 ms (4.9)
      - 9.8 → 0.66 ms (15)
    * - ``kCSB_dmg`` (the three blocks)
      - 3.60 → 0.106 s (34)
      - 27.5 → 0.49 s (57)
    * - ``kCpd``
      - 0.19 → 0.057 ms (3.3)
      - 1.1 → 0.16 ms (7.1)

The double cantilever beam with cohesive zone of
``tests/multidomain/test_dcb_damage.py`` runs in 2 min 45 s instead of
7 min 28 s, its non-linear matrices of the panels and its connections
being tabulated; its cohesive zone is assembled with cached operators by
default, or with ``kCSB_dmg`` with ``use_kernels=True``.
