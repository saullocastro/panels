.. _theory_func_bardell:

Bardell's hierarchical functions
================================

Every domain of ``panels``, a :class:`.Shell`, a panel of a
:class:`.MultiDomain` assembly or the flange of a :class:`.BladeStiff2D`,
approximates its displacement fields with the Ritz method, as series of
products of one-dimensional functions along `x` and `y`. The functions are
the hierarchical Legendre-based polynomials in Rodrigues form applied by
Bardell [bardell1991]_ in the hierarchical finite element method, and used
as approximation functions of semi-analytical models of plates, shells and
stiffened panels by Castro and Donadon [castro2017Multidomain]_. They are
coded in C++ up to 30 terms and accessed from Python with
:mod:`panels.bardell`, where their expressions are given.


The functions
-------------

The functions `P_i(\xi)`, `i = 0, 1, \dots, 29`, are defined over the
natural coordinate `-1 \le \xi \le 1`, which maps the length of the domain,
`\xi = 2x/a - 1` along `x` and `\eta = 2y/b - 1` along `y`. The first four
terms are cubic Hermite polynomials and the terms `i \ge 4` are polynomials
of degree `i` that vanish together with their first derivative at both ends.
Only the first four terms therefore carry a value or a slope at the ends,
each one at one end only:

========== ============== ================ ============== ================ ==========
term `i`    `P_i(-1)`      `P_i'(-1)`       `P_i(+1)`      `P_i'(+1)`       flag
========== ============== ================ ============== ================ ==========
0           1              0                0              0                ``x1t``
1           0              `1/2`            0              0                ``x1r``
2           0              0                1              0                ``x2t``
3           0              0                0              `1/2`            ``x2r``
`\ge 4`     0              0                0              0
========== ============== ================ ============== ================ ==========

where the prime is the derivative with respect to `\xi`, and
`\partial/\partial x = (2/a) \, \partial/\partial \xi`. The set is
hierarchical: increasing the number of terms, ``m`` along `x` and ``n``
along `y`, adds functions without changing the previous ones.


Evaluation in floating point
----------------------------

The C++ functions ``f``, ``fp`` and ``fpp`` (one term) and ``vec_f``,
``vec_fp`` and ``vec_fpp`` (all the terms), in
``panels/core/src/bardell_functions.cpp``, evaluate each polynomial by
Horner's rule, without ``pow()``: the first four terms in `\xi`, the terms
`i \ge 4` and their derivatives, which are even or odd in `\xi`, in
`\xi^2`; see the Notes of :mod:`panels.bardell`. The coefficients are the
exact rationals of the definition rounded to double precision. The file is
generated with SymPy by ``theory/func/bardell/bardell_functions_C.py`` and
is not edited by hand.

The terms of high degree have large coefficients of alternating sign, e.g.
up to about `1.3 \times 10^6` for `P_{29}`, whose largest absolute value
over `-1 \le \xi \le 1` is about `2 \times 10^{-4}`: any evaluation of
these monomial forms in double precision loses digits by cancellation.
Against the exact value, computed in rational arithmetic, the error
relative to the largest absolute value of each term grows by a factor of
about 2.5 per term, from one unit round-off for the first terms to about
`7 \times 10^{-7}` for `P_{29}`, `4 \times 10^{-7}` for `P_{29}'` and
`7 \times 10^{-8}` for `P_{29}''`. Up to panels 0.17.0 the monomials were
summed with ``pow()`` and coefficients printed with 15 significant digits,
with errors up to `1.6 \times 10^{-5}`. The matrices and vectors therefore
differ from those of 0.17.0 at round-off level, more with more terms, e.g.
by about `10^{-15}` of the largest entry for `m = 3`, `n = 4` and up to
`10^{-9}` for `m = 24`, `n = 4`.


Boundary flags
--------------

The four flags multiply the first four terms: ``x1t`` and ``x1r`` the terms
of the value (translation) and of the slope (rotation) at `\xi = -1`,
``x2t`` and ``x2r`` those at `\xi = +1`. A flag ``0.`` removes its term,
which makes the value or the slope of every function of the series zero at
that end, an essential boundary condition imposed exactly on the whole
edge; a flag ``1.`` keeps the term, and the corresponding natural boundary
condition follows from the energy. Any other non-zero flag only scales the
term.

Each field of a domain has its own flags along `x` and along `y`,
attributes of :class:`.Shell` named after the edge, the field and the kind
of term:

- ``x1u``, ``x1ur``, ``x2u``, ``x2ur``: the flags ``x1t``, ``x1r``,
  ``x2t``, ``x2r`` of the functions of `u` along `x`, i.e. at the edges `x =
  0` and `x = a`;
- ``y1u``, ``y1ur``, ``y2u``, ``y2ur``: the same along `y`, at the edges `y =
  0` and `y = b`;
- likewise for `v` and `w`, and for the rotations `\phi_x` and `\phi_y` of
  the models based on shear deformation theories, e.g. ``x1phix``,
  ``y2phiyr``.

For example, ``x1w = 0`` imposes `w = 0` and ``x1wr = 0`` imposes `w_{,x} =
0` on the edge `x = 0`. The combinations of the flags that give the usual
supports are listed in :ref:`boundary_conditions`. A removed term keeps its
place in the vector of the Ritz constants, see :meth:`.Shell.get_size`, with
null rows and columns in the matrices, which the solvers of ``structsolve``
remove before solving.


Integrals and structural matrices
---------------------------------

An entry of a structural matrix is the integral over the domain of the
product of two terms of the series, `(i, j)` and `(k, l)`, or of their
derivatives, multiplied by the constitutive coefficients. When these
coefficients are constant over the domain and the integration domain is the
whole shell, the double integral separates in the two directions:

.. math::
    :label: bardell-separation

    \int_0^b \int_0^a \frac{\partial^p P_i}{\partial x^p}
    \frac{\partial^q P_k}{\partial x^q} \,
    \frac{\partial^r P_j}{\partial y^r} \frac{\partial^s P_l}{\partial y^s}
    \, dx \, dy =
    \left(\frac{2}{a}\right)^{p + q} \frac{a}{2}
    \int_{-1}^{1} P_i^{(p)} P_k^{(q)} \, d\xi
    \left(\frac{2}{b}\right)^{r + s} \frac{b}{2}
    \int_{-1}^{1} P_j^{(r)} P_l^{(s)} \, d\eta

with `p, q, r, s \le 2`, where `f` and `g` carry the flags of the fields of
each term along `x` and `y`. The six one-dimensional integrals,
:func:`.calc_integral_ff`, :func:`.calc_integral_ffp`,
:func:`.calc_integral_ffpp`, :func:`.calc_integral_fpfp`,
:func:`.calc_integral_fpfpp` and :func:`.calc_integral_fppfpp`, are
tabulated in closed form, generated with SymPy in rational arithmetic and
exact. They give the closed-form (analytical) matrices of the models, e.g.
the linear constitutive, geometric, mass and aerodynamic matrices of a
:class:`.Shell`, :meth:`.Shell.calc_kC`, :meth:`.Shell.calc_kG`,
:meth:`.Shell.calc_kM` and :meth:`.Shell.calc_kA`, and the penalty
connections along the edges, see :mod:`panels.multidomain.connections`.

When the integrand does not separate, i.e. for the non-linear matrices and
internal forces, for a stiffness given per integration point by
``ABDnxny``, for the follower pressure loads, or for a partial integration
domain, see :meth:`.Shell.is_partial_domain`, the matrices are integrated
numerically with ``nx`` by ``ny`` Gauss-Legendre points,
:meth:`.Shell.integration_points`, evaluating the functions and their first
and second derivatives at each point, see :func:`.calc_vec_f`,
:func:`.calc_vec_fp` and :func:`.calc_vec_fpp`. The same functions give the
field outputs, :meth:`.Shell.uvw`, :meth:`.Shell.strain` and
:meth:`.Shell.stress`, through the field modules
:mod:`panels.models.clpt_field` and :mod:`panels.models.fsdt_tsdt_field`.
The values of the functions at the integration points can be tabulated once
per kernel, see :ref:`tabulation`.

The product of two terms is a polynomial of degree up to 58, integrated
exactly by a Gauss-Legendre quadrature of 30 points: the closed-form
integrals agree with it up to the round-off of the functions, and
Eq. :eq:`bardell-separation` gives the physical integrals, e.g.
`\int_0^a P_{i,x} P_{k,x} \, dx = (2/a)^2 (a/2)` ``calc_integral_fpfp(i, k,
...)``.
