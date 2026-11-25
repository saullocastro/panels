r"""Accuracy of the evaluation of Bardell's functions by Horner's rule

The C++ functions of ``panels/core/src/bardell_functions.cpp`` evaluate each
polynomial by Horner's rule, in `\xi` or in `\xi^2`, with correctly rounded
coefficients. Up to panels 0.17.0 they summed the monomials with ``pow()``
and coefficients printed with 15 significant digits. Both are compared here
against the exact value: the exact rational coefficients of the definition
evaluated in rational arithmetic (:class:`fractions.Fraction`) at the same
double-precision `\xi`, then rounded.
"""
import os
import re
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


def pow_form_expressions():
    """C expressions of ``f``, ``fp`` and ``fpp`` of panels 0.17.0, stored in
    ``bardell_functions_pow_0.17.0.txt``, as ``{(nder, i): expression}`` with
    ``pow(xi, k)`` written ``xi**k``"""
    path = os.path.join(os.path.dirname(__file__),
                        'bardell_functions_pow_0.17.0.txt')
    nders = {'f': 0, 'fp': 1, 'fpp': 2}
    out = {}
    with open(path) as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            name, i, expr = line.split(' ', 2)
            out[(nders[name], int(i))] = compile(
                re.sub(r'pow\(xi, (\d+)\)', r'xi**\1', expr.strip()),
                '<pow form>', 'eval')
    return out


def eval_pow_form(code, xi):
    """Evaluation of panels 0.17.0: Python applies the same floating-point
    operations in the same order as the compiled C++, ``xi**k`` calling the
    C ``pow()``"""
    return eval(code, {'xi': xi, 'xi1t': 1., 'xi1r': 1., 'xi2t': 1.,
                       'xi2r': 1.})


def test_horner_at_least_as_accurate_as_pow_form():
    r"""Errors against the exact value at 401 points of `-1 \le \xi \le 1`,
    for every term ``i`` and derivative: the compiled Horner evaluation has
    a maximum absolute error not larger than that of the evaluation of
    panels 0.17.0 with ``pow()``, or not larger than one unit round-off
    times `\sum_k |a_k|`, the scale of the round-off of any evaluation of
    the monomial form; and not larger than twice that scale, well within
    the bound `\gamma_{2n} \sum_k |a_k|` of Horner's rule. Over all the
    terms of each family the largest error decreases, absolute and relative
    to the largest absolute value of the term"""
    codes = pow_form_expressions()
    rng = np.random.default_rng(1)
    xis = np.concatenate([np.linspace(-1, 1, 201), rng.uniform(-1, 1, 200)])
    for nder, fn in enumerate((calc_f, calc_fp, calc_fpp)):
        worst_pow = worst_horner = 0.
        worst_pow_rel = worst_horner_rel = 0.
        for i in range(NUM):
            a = exact_coefficients(i, nder)
            sum_abs = float(sum(abs(c) for c in a))
            exact = np.array([eval_exact(a, x) for x in xis])
            powf = np.array([eval_pow_form(codes[(nder, i)], x) for x in xis])
            horner = np.array([fn(i, x, 1., 1., 1., 1.) for x in xis])
            scale = np.abs(exact).max()
            e_pow = np.abs(powf - exact).max()
            e_horner = np.abs(horner - exact).max()
            assert e_horner <= max(e_pow, EPS*sum_abs), (nder, i)
            assert e_horner <= 2*EPS*sum_abs, (nder, i)
            worst_pow = max(worst_pow, e_pow)
            worst_horner = max(worst_horner, e_horner)
            worst_pow_rel = max(worst_pow_rel, e_pow/scale)
            worst_horner_rel = max(worst_horner_rel, e_horner/scale)
        assert worst_horner < worst_pow
        assert worst_horner_rel < worst_pow_rel


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

