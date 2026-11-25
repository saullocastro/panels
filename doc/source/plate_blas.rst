.. _plate_blas:

Plate kernels as matrix products
================================

The tables of :ref:`tabulation` remove the evaluations of the approximation
functions from the loops over the pairs of terms, but the loops remain:
per integration point, the Cython kernels of the tangent stiffness matrix
add one product of scalars for each pair of terms and each pair of degrees
of freedom, `O(m^2 n^2)` operations of a few flops each, in a scalar loop.
For the flat plates of the von Kármán kinematics, the models
``'plate_clpt_donnell'`` and ``'plate_fsdt_donnell'``, the same sums can be
written as products of dense matrices, which the BLAS library of NumPy
computes at a much higher rate. The option ``plate_kernels='blas'`` of
:class:`.MultiDomain` selects this formulation, implemented in
:mod:`panels.multidomain.plate_blas`; the default ``'cython'`` keeps the
kernels of :ref:`tabulation`.

From tables per point to operators per domain
---------------------------------------------

Every integrand of the plate kernels is a function of the generalized
derivatives of the fields at the integration point, `u_{,x}, u_{,y},
\ldots, w_{,xx}` (CLPT) or `\phi_{x,x}, \ldots` (FSDT), which are linear in
the Ritz constants of the domain. The values of the approximation functions
and of their derivatives at the points, the tables of :ref:`tabulation`,
depend only on the number of terms, the dimensions of the domain, the
boundary flags and the points, not on the configuration. They are therefore
combined once per domain into the operator `\boldsymbol{D}` that maps the
Ritz constants to the generalized derivatives at all the points, see
:class:`.plate_blas.PlateOperators`, and kept with the assembly while the
domain does not change. Each evaluation then has three steps:

1. the generalized derivatives at all the points, `\boldsymbol{D}
   \boldsymbol{c}`, a matrix-vector product;
2. per point, the small matrix of the strains, the stresses and the matrix
   `\boldsymbol{H}_p` (6 to 12 rows), see the Notes of
   :mod:`panels.multidomain.plate_blas`;
3. the matrix `\boldsymbol{D}^\top \boldsymbol{H} \boldsymbol{D}`, one GEMM
   per pair of fields, or the vector `\boldsymbol{D}^\top \boldsymbol{g}`
   of the internal forces or of the product of the tangent by a vector.

The arithmetic is that of the Cython kernels, regrouped: the matrices and
vectors are equal to the round-off (about `10^{-15}` of the largest entry
in ``tests/multidomain/test_plate_blas.py``, which checks `10^{-12}`), and
the matrices keep the pattern of the Cython kernels, the upper triangle by
pairs of terms with ``finalize=False``. The memory of the operators grows
with the number of points times the number of terms of the domain, e.g.
about 5 MB for an FSDT domain with `m = 15`, `n = 6` and `60 \times 10`
points.

Where it applies
----------------

The operators replace the Cython kernels in :meth:`.MultiDomain.calc_kC` and
:meth:`.MultiDomain.calc_kG` with ``NLgeom=True`` and a configuration ``c``,
:meth:`.MultiDomain.calc_fint` and :meth:`.MultiDomain.calc_kTv`, for the
domains of :func:`.plate_blas.is_eligible`: a constant constitutive matrix,
the whole domain integrated, no constant membrane stresses and no follower
pressure load. Every other domain and every other call, e.g. the linear
matrices of a buckling analysis, use the Cython kernels, such that the
option can be switched on for any assembly. The cohesive coupons of
:mod:`.delamination`, whose tangent is assembled at each iteration, are
the intended use, see :ref:`cohesive_zone_mixed_mode`.

API
---

.. automodule:: panels.multidomain.plate_blas
    :members:
