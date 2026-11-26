.. _tabulation:

Approximation functions tabulated at the integration points
===========================================================

The numerically integrated matrices and vectors of ``panels`` are sums over
the integration points of products of the approximation functions of the
Ritz method. This page describes how the kernels evaluate these functions:
once per integration point and term, in tables, instead of again in every
loop over the terms of the matrix, the innermost loop included, and keep
these tables on the shell from one call to the next. The results are the
same, bit for bit, as with the functions evaluated in the loops, and the
assembly is several times faster.

Integrands
----------

Each field of a panel is approximated with the products of the hierarchical
functions of Bardell, see :ref:`theory_func_bardell`, along `\xi` and `\eta`:

.. math::

    u(\xi, \eta) = \sum_{j=0}^{n-1} \sum_{i=0}^{m-1} c_{u, ij}\, P_i(\xi)\,
    P_j(\eta)

where `P_i` and `P_j` depend on the boundary conditions of the field, i.e.
the flags ``x1u, x1ur, x2u, x2ur`` along `x` and ``y1u, y1ur, y2u, y2ur``
along `y`. The entry of a stiffness matrix between the terms `(i, j)` and
`(k, l)` is the sum over the `n_x \times n_y` Gauss-Legendre points

.. math::

    K_{(ij)(kl)} = \sum_{p=1}^{n_x} \sum_{q=1}^{n_y} w_p w_q\,
    \mathcal{F}\left(P_i(\xi_p), P_k(\xi_p), P_j(\eta_q), P_l(\eta_q),
    \ldots\right)

where `\mathcal{F}` combines the functions and their derivatives `f'`,
`f''`, `g'`, `g''` of each field with the constitutive matrix and, for the
non-linear matrices, with the displacement gradients at the point.

Functions evaluated in the innermost loop
-----------------------------------------

A direct implementation has the structure below, e.g. ``fkC_num`` of
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

The value of `P_l(\eta_q)` does not depend on `i`, `k` or `j`, but it was
computed again for each of them: per integration point, `7 m^2 n^2`
evaluations in the innermost loop, `7 m^2 n` in the loop over `j`, and so
on, for only `7 (m + n)` distinct values. Each evaluation is a polynomial
of degree up to 29 in `\xi`, selected by the index of the term and
evaluated by Horner's rule, see :ref:`theory_func_bardell`, which costs
much more than the arithmetic of the entry itself. For `m = n = 12` this is about 10^5
evaluations per integration point, against 168 distinct values.

The internal force vector, ``calc_fint``, has three loops over the `m n`
terms per integration point (the rotations, the strains, the forces), with
the functions evaluated again in each; the connections between panels and
the stiffeners have the same nested structure, along a line or over an
area.

Tables per integration point
----------------------------

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
those of the direct implementation.

The kernels that integrate along a line, e.g. the connections at a constant
`\xi` or the prescribed displacements, evaluate the functions across the
line at a single coordinate: their tables are filled once, before the
outermost loop over the terms.

Tables kept per domain across calls
-----------------------------------

The integration points of a shell, its numbers of terms and its boundary
flags do not change from one call of a kernel to the next, e.g. between the
iterations of a non-linear analysis, and neither do the tables. The kernels
of the models, ``panels/models/<model>_num.pyx``, therefore keep them on
the shell, with one row per integration point along `x` (or `y`)::

    ftab, ftab_new = lookup_table(shell, __name__, 'fkC_num', 'ftab',
                                  ftab_key, (nx, 7, m))
    ...
    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi, eta, weight = ...
                if ftab_new:
                    for i in range(m):
                        ftab[ptx, 0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                        ...
                for i in range(m):
                    fAu = ftab[ptx, 0, i]
                    ...
    if ftab_new:
        store_table(shell, __name__, 'fkC_num', 'ftab', ftab_key, ftab)

The first call fills the tables, at all the points, and stores them on the
shell; the next calls only read them. The values are computed by the same
code, at the same coordinates, as when they were filled at every call, such
that the matrices and vectors are again identical, bit for bit.

Each table is stored with a key of all the values it depends on, see
:mod:`panels.basis_tables`:

- the numbers of terms, ``m`` or ``n``, and of integration points, ``nx``
  or ``ny``;
- the Gauss-Legendre abscissae and the limits of the integration domain in
  natural coordinates, ``xi1, xi2`` or ``eta1, eta2``, which follow from
  :meth:`.Shell.integration_limits`;
- the boundary flags of the fields of its rows, e.g. ``x1u, x1ur, x2u,
  x2ur``;
- the dimensions entering its rows, e.g. ``a`` of the derivatives
  ``(2/a)*fp(...)``.

A call whose key differs from the stored one, after a change of any of
these values, fills a new table and replaces the stored one: a shell keeps
at most one table per kernel and table name, a few tens of kilobytes, e.g.
`240 \times 7 \times 24` values along `x` for a domain of 240 points and
`m = 24`. The tables are looked up and stored while the kernel holds the
GIL; a stored table is never modified, only replaced, such that kernels
running at the same time in several threads read consistent tables.
:func:`.basis_tables.clear_tables` removes the tables of a shell.

The tables of the functions at a single point given as argument, e.g. for
the field outputs, and those of the connections and of the stiffeners,
which depend on several domains, are still filled at every call.

Kernels
-------

All the kernels that evaluate the approximation functions in their loops
over the terms are tabulated:

- the 9 geometrically non-linear models, ``panels/models/<model>_num.pyx``:
  ``fkC_num``, ``fkG_num``, ``calc_fint``, ``fkM_num``, ``fkAx_num``,
  ``fkAy_num``, ``calc_fext_follower`` and ``fkCfollower_num``; the kernels of
  ``plate_clpt_gl_num.pyx`` and ``plate_clpt_tl_num.pyx``, see
  :ref:`plate_clpt_gl` and :ref:`plate_clpt_tl`, are generated with the
  tables of the functions and of their derivatives in `x` and `y`;
- the connections, ``panels/multidomain/connections``: ``kCSSxcte``,
  ``kCSSycte``, ``kCBFxcte``, ``kCBFycte`` (including the shear
  deformation theories), ``kCpd`` (prescribed displacements) and
  ``kCSB_dmg`` (skin-base connection with damage, the cohesive zone of the
  double cantilever beam, see :ref:`cohesive_zone`);
- the blade stiffeners, ``panels/stiffener/models``.

The analytical kernels, which use closed-form integrals of the products of
the functions, e.g. ``fk0`` or :func:`.fkCSB` of the skin-base connection,
are not affected. The kernel
``calc_kTv`` of the product of the tangent stiffness matrix by a vector,
see :ref:`tangent_product`, is written with the tables.

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
4. declares and allocates the tables outside the ``nogil`` block;
5. for a kernel with a ``shell`` argument, keeps the tables on the shell,
   see :func:`panels.dev.tabulate.keep_tables`: the key gathers the names
   that the rows of each table depend on, following the assignments of the
   coordinates in the loops over the points back to the values set before
   the ``nogil`` block, and it fails when this dependence cannot be
   established, e.g. for a coordinate assigned conditionally.

A kernel already tabulated is left unchanged, such that applying it twice
does nothing. The generators of all the modules
``panels/models/<model>_num.pyx`` apply :func:`.keep_tables` to the text
they write. The helper ``_terms_sdt`` of the connections of the shear
deformation theories, whose flags come from an array, was tabulated by
hand in the same way. The matrices and vectors of the tabulated kernels
are identical, bit for bit, to those of the direct implementation, the
functions being the same numbers combined in the same order.

Performance
-----------

Per integration point the tabulation removes `O(m^2 n^2)` evaluations of
polynomials of degree up to 29 and keeps only the arithmetic of the entries.
The speed-up therefore grows with the number of terms, and it is smaller for
the integrands with more arithmetic per pair of terms, e.g. the constitutive
stiffness matrix ``fkC_num`` of the shear deformation theories: for
`m = n = 12` it is about 30 to 60 for the geometric stiffness matrices and
3.5 to 17 for the non-linear constitutive stiffness matrices, measured
when the functions were evaluated with ``pow()``.

Keeping the tables across calls removes the remaining `O(m + n)`
evaluations per point. Their share is largest for many points and few
terms per point, as in the long domains of the coupons: for a domain of
`240 \times 6` points and `m = 24`, `n = 4`, ``fkC_num`` with ``NLgeom=1``,
``fkG_num`` and ``calc_fint`` together take 16 % less time for
``'plate_clpt_donnell'`` and 28 % less for ``'plate_fsdt_donnell'`` from the
second call on than with the tables filled at every call (Windows, MSVC).
The evaluation by Horner's rule, see :ref:`theory_func_bardell`, makes
filling the tables at the first call cheap as well: the first call takes
14 % and 27 % less time than with the tables filled from the polynomials
evaluated with ``pow()``.

Operators per domain
--------------------

For the flat plates ``'plate_clpt_donnell'`` and ``'plate_fsdt_donnell'``
the tables can be taken one step further: the products of the tabulated
functions at all the points, which do not change from one evaluation to
the next, are kept once per domain as the operators of the generalized
derivatives, and the loops over the pairs of terms become products of dense
matrices. This is the option ``plate_kernels='blas'`` of
:class:`.MultiDomain`, see :ref:`plate_blas`.
