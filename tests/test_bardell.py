r"""Bardell's hierarchical functions of :mod:`panels.bardell`

Values and slopes at the ends `\xi = \pm 1`, role of the boundary flags and
closed-form integrals against a Gauss-Legendre quadrature.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss

from panels.bardell import (calc_f, calc_fp, calc_vec_f, calc_vec_fp,
                            calc_vec_fpp, calc_integral_ff,
                            calc_integral_ffp, calc_integral_fpfp,
                            calc_integral_fppfpp)

NUM = 30


def test_end_values_and_flags():
    flags = (1., 1., 1., 1.) # x1t, x1r, x2t, x2r
    xis = np.linspace(-1, 1, 2001)
    fmax = np.abs([calc_vec_f(xi, *flags) for xi in xis]).max(axis=0)
    fpmax = np.abs([calc_vec_fp(xi, *flags) for xi in xis]).max(axis=0)
    # only the first four terms have a value or a slope at the ends
    expected = {
        # term: (f(-1), f'(-1), f(+1), f'(+1))
        0: (1., 0., 0., 0.),
        1: (0., 0.5, 0., 0.),
        2: (0., 0., 1., 0.),
        3: (0., 0., 0., 0.5),
        }
    for i in range(NUM):
        ends = (calc_f(i, -1., *flags), calc_fp(i, -1., *flags),
                calc_f(i, +1., *flags), calc_fp(i, +1., *flags))
        # round-off of the evaluation in floating point, which grows with
        # the degree of the polynomial
        tol = 1e-4*np.array([fmax[i], fpmax[i], fmax[i], fpmax[i]])
        assert np.all(np.abs(np.subtract(ends, expected.get(i, (0., 0., 0.,
            0.)))) <= tol), i
    # a flag 0 removes its term: here a clamped end at xi = -1
    x1t, x1r = 0., 0.
    for xi in np.linspace(-1, 1, 7):
        f = calc_vec_f(xi, x1t, x1r, 1., 1.)
        assert f[0] == 0. and f[1] == 0.
        assert f[2] == calc_f(2, xi, 1., 1., 1., 1.)
    assert np.allclose(calc_vec_f(-1., x1t, x1r, 1., 1.), 0., atol=1e-4*fmax)
    assert np.allclose(calc_vec_fp(-1., x1t, x1r, 1., 1.), 0.,
                       atol=1e-4*fpmax)
    # any other non-zero flag only scales its term
    assert np.isclose(calc_fp(1, -1., 1., 2., 1., 1.), 1.)


def test_integrals_against_quadrature():
    # products of two terms are polynomials of degree up to 58, integrated
    # exactly by 30 Gauss-Legendre points
    xis, ws = leggauss(30)
    fx = (1., 0., 1., 1.) # x1t, x1r, x2t, x2r of the term i
    fy = (0., 1., 1., 0.) # x1t, x1r, x2t, x2r of the term j
    F = np.array([calc_vec_f(xi, *fx) for xi in xis])
    Fp = np.array([calc_vec_fp(xi, *fx) for xi in xis])
    Fpp = np.array([calc_vec_fpp(xi, *fx) for xi in xis])
    G = np.array([calc_vec_f(xi, *fy) for xi in xis])
    Gp = np.array([calc_vec_fp(xi, *fy) for xi in xis])
    Gpp = np.array([calc_vec_fpp(xi, *fy) for xi in xis])
    for func, A, B in ((calc_integral_ff, F, G),
                       (calc_integral_ffp, F, Gp),
                       (calc_integral_fpfp, Fp, Gp),
                       (calc_integral_fppfpp, Fpp, Gpp)):
        quad = (A*ws[:, None]).T @ B
        closed = np.array([[func(i, j, *fx, *fy) for j in range(NUM)]
                           for i in range(NUM)])
        assert np.allclose(closed, quad, rtol=1e-6,
                           atol=1e-6*np.abs(quad).max()), func.__name__
    # physical integral over 0 <= x <= a of the product of the derivatives
    # with respect to x of two terms
    a = 0.7
    i, j = 5, 7
    dfdx = Fp*2/a
    ref = np.sum(ws*a/2*dfdx[:, i]*dfdx[:, j])
    assert np.isclose((2/a)**2*a/2*calc_integral_fpfp(i, j, *fx, *fx), ref)
