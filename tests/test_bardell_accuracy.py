r"""Accuracy of the evaluation of Bardell's functions by Horner's rule

The C++ functions of ``panels/core/src/bardell_functions.cpp`` evaluate each
polynomial by Horner's rule, in `\xi` or in `\xi^2`, with correctly rounded
coefficients. They are compared here against the exact value: the exact
rational coefficients of the definition evaluated in rational arithmetic
(:class:`fractions.Fraction`) at the same double-precision `\xi`, then
rounded. The vector functions are compared with the scalar ones.

Notes
-----
For a polynomial `p(\xi) = \sum_k a_k \xi^k` of degree `d`, the value
computed by Horner's rule has an error not larger than `\gamma_{2d} \sum_k
|a_k| |\xi|^k`, with `\gamma_{2d} = 2du/(1 - 2du)` and `u` the unit
round-off, Higham (2002), Chapter 5 [1]_. The test checks the stricter
`2u \sum_k |a_k|` over `-1 \le \xi \le 1`.

References
----------
.. [1] Higham, N. J., "Accuracy and Stability of Numerical Algorithms",
   2nd ed., SIAM, Philadelphia, 2002, Chapter 5, Polynomials,
   pp. 93-104. https://doi.org/10.1137/1.9780898718027.ch5

"""
from fractions import Fraction
from math import factorial

import numpy as np

from panels.bardell import (calc_f, calc_fp, calc_fpp, calc_vec_f,
                            calc_vec_fp, calc_vec_fpp)

NUM = 30
EPS = np.finfo(np.float64).eps


def _factorial2(n):
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def exact_coefficients(i, nder=0):
    r"""Exact coefficients `a_0, a_1, \dots` of the derivative ``nder`` of
    the term ``i``, with unit boundary flags, from the definition of
    :mod:`panels.bardell`"""
    if i < 4:
        a = [[Fraction(1, 2), Fraction(-3, 4), 0, Fraction(1, 4)],
             [Fraction(1, 8), Fraction(-1, 8), Fraction(-1, 8), Fraction(1, 8)],
             [Fraction(1, 2), Fraction(3, 4), 0, Fraction(-1, 4)],
             [Fraction(-1, 8), Fraction(-1, 8), Fraction(1, 8), Fraction(1, 8)]][i]
        a = [Fraction(c) for c in a]
    else:
        r = i + 1
        a = [Fraction(0)]*r
        for n in range(r//2 + 1):
            p = r - 2*n - 1
            if p < 0:
                continue
            den = 2**n*factorial(n)*factorial(p)
            a[p] += Fraction((-1)**n*_factorial2(2*r - 2*n - 7), den)
    for _ in range(nder):
        a = [k*c for k, c in enumerate(a)][1:]
    return a


def eval_exact(a, xi):
    """Polynomial with Fraction coefficients at the double ``xi``, exactly,
    rounded to double at the end"""
    x = Fraction(xi)
    s = Fraction(0)
    for c in reversed(a):
        s = s*x + c
    return float(s)


def test_horner_against_exact_rational_arithmetic():
    r"""Errors against the exact value at 401 points of `-1 \le \xi \le 1`,
    for every term ``i`` and derivative: the compiled Horner evaluation has
    a maximum absolute error not larger than twice the unit round-off times
    `\sum_k |a_k|`, the scale of the round-off of any evaluation of the
    monomial form, well within the bound `\gamma_{2n} \sum_k |a_k|` of
    Horner's rule"""
    rng = np.random.default_rng(1)
    xis = np.concatenate([np.linspace(-1, 1, 201), rng.uniform(-1, 1, 200)])
    for nder, fn in enumerate((calc_f, calc_fp, calc_fpp)):
        for i in range(NUM):
            a = exact_coefficients(i, nder)
            sum_abs = float(sum(abs(c) for c in a))
            exact = np.array([eval_exact(a, x) for x in xis])
            horner = np.array([fn(i, x, 1., 1., 1., 1.) for x in xis])
            e_horner = np.abs(horner - exact).max()
            assert e_horner <= 2*EPS*sum_abs, (nder, i)


def test_vec_equals_scalar():
    r"""The vector functions ``calc_vec_f``, ``calc_vec_fp`` and
    ``calc_vec_fpp`` give, bit for bit, the scalar functions of each term,
    with arbitrary boundary flags"""
    flags = (0.3, 1.7, 0., 2.)
    for xi in np.linspace(-1, 1, 41):
        for vec, fn in ((calc_vec_f, calc_f), (calc_vec_fp, calc_fp),
                        (calc_vec_fpp, calc_fpp)):
            v = vec(xi, *flags)
            assert np.array_equal(v, [fn(i, xi, *flags) for i in range(NUM)])
