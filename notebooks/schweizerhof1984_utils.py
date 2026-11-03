r"""Utilities of the notebook schweizerhof1984_cantilever_postbuckling.ipynb

Cantilever cylinder of Schweizerhof and Ramm (1984), Section 6.4, under a
follower (hydrostatic) pressure: linear buckling factors of Table 4 and the
Riks analysis of the imperfect shell 1 of Fig. 13, with the half-wave sector
of ``cylshell_clpt_sanders`` and the imperfection added exactly for the
Sanders kinematics. The model is that of the test
``tests/tests_shell/test_schweizerhof1984_cantilever_postbuckling.py``,
imported here, such that the notebook and the test use the same definitions.

"""
import os
import sys
import time

import numpy as np
from structsolve import Analysis
from structsolve.stopping import value_reached

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.abspath(os.path.join(HERE, '..', 'tests', 'tests_shell'))
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)

from test_schweizerhof1984_cantilever_postbuckling import (  # noqa: E402
        build_shell, buckling_K, imperfection, ImperfectShell, w_max, R, t,
        E, NU)

#: Table 4 of [1], p. 1111, read on the page: (shell, L/R, R/t, n), then
#: (k_c, k_f) of B1 and of B2 (k_f with the symmetrization of Eq. (63));
#: k_f of B1 of shell 3 is not given
TABLE4 = [
    ((1, 6., 100., 3), (10.89, 9.70), (9.11, 8.10)),
    ((2, 3., 100., 4), (20.85, 19.58), (16.43, 15.42)),
    ((3, 3., 500., 6), (45.83, None), (36.48, 35.48)),
]
#: k_f of B2 with the symmetrizations of Eqs. (64) and (65)
TABLE4_B2_KF_64_65 = {1: (8.12, 8.09), 2: (15.46, 15.38), 3: (35.50, 35.46)}

# Fig. 13 of [1], p. 1112, read on the page rendered at 300 dpi: the axes
# located from the frame and the ticks (K: 87.4 px per unit, w/t: 267.9 px
# per unit), the points at the centre of the thick line, about 7 px wide.
# Reading precision about +-0.05 in K and +-0.01 in w/t (+-0.02 below
# w/t = 0.1, where the curve is nearly vertical). (w/t, K)
#: solid line, imperfect shell, 1 % of t
FIG13_IMPERFECT = [
    (0.017, 5.10), (0.022, 5.56), (0.039, 6.02), (0.049, 6.48),
    (0.073, 6.93), (0.138, 7.39), (0.174, 7.52), (0.211, 7.62),
    (0.286, 7.72), (0.398, 7.78), (0.510, 7.82), (0.696, 7.88),
    (0.883, 7.91), (1.069, 7.92), (1.443, 7.93), (2.003, 7.96),
    (2.563, 7.99), (2.936, 7.99)]
#: thin horizontal line, perfect shell
FIG13_PERFECT_K = 8.04


def riks(bc='B1', xi=0.01, wstop=3., m=14, nt=8, inc=1., max_inc=None,
         num_eigvals=6, max_steps=None):
    r"""Riks analysis of the cantilever under the follower pressure, the
    load factor being ``K`` of Eq. (60), up to ``w_max/t = wstop``

    ``xi > 0``: imperfection of ``xi*t`` in the shape of the lowest linear
    buckling mode of the follower load; ``xi = 0``: the perfect shell, on
    its fundamental path, up to ``K = wstop`` instead.

    With ``inertia=True`` and ``inertia_num_eigvals=num_eigvals``: the number
    of negative real eigenvalues of the unsymmetric tangent stiffness matrix
    among the ``num_eigvals`` of smallest magnitude, by ARPACK on its LU
    factorization, and the sign of its determinant, at every converged
    increment; structsolve counts the eigenvalues of ``S K_T S``, ``S =
    |diag(K0)|^(-1/2)`` of the linear stiffness matrix.

    Returns a dict with the load factors ``K``, the maximum displacement
    ``w`` (``|w|``, without the imperfection) over ``t``, the counts
    ``nneg``, the signs ``det``, the buckling factor ``K_f`` of the model,
    the run time and the maximum relative residual of the converged
    increments.

    """
    s, kP = build_shell(bc, m=m, nt=nt)
    K_f, mode = buckling_K(s, kP)
    c0 = imperfection(s, mode, xi) if xi > 0 else np.zeros_like(mode)
    sm = ImperfectShell(s, c0, kP)
    an = Analysis(sm.calc_fext, sm.calc_fint, sm.calc_kC, sm.calc_kG)
    an.NL_method = 'arc_length_riks'
    an.final_load_factor = None
    an.maxArcLength = 1e4
    an.initialInc = inc
    an.maxInc = inc if max_inc is None else max_inc
    an.inertia = True
    an.inertia_num_eigvals = num_eigvals
    if xi > 0:
        an.stop_criterion = value_reached(lambda c: w_max(s, c)/t,
                                          wstop)
    else:
        an.stop_criterion = lambda an: an.increments[-1] >= wstop
    if max_steps is not None:
        crit = an.stop_criterion
        an.stop_criterion = lambda an: crit(an) or len(an.increments) >= max_steps
    t0 = time.perf_counter()
    an.static(NLgeom=True, silent=True)
    elapsed = time.perf_counter() - t0
    res = max(np.linalg.norm(sm.calc_fint(c, inc=lbd)
                             - lbd*sm.calc_fext(inc=1.))
              / np.linalg.norm(sm.calc_fext(inc=1.))
              for lbd, c in zip(an.increments, an.cs))
    w = [w_max(s, c)/t for c in an.cs]
    return dict(K=list(map(float, an.increments)), w=list(map(float, w)),
                nneg=list(an.num_negative_eigvals), det=list(an.det_signs),
                K_f=float(K_f), time=elapsed, max_residual=float(res))


def bifurcation_load(bc='B1', m=14, nt=8, K1=None, K2=None, tol=1e-3):
    r"""Load factor ``K`` of the first bifurcation of the fundamental path of
    the perfect shell under the follower pressure, where the sign of the
    determinant of the tangent stiffness matrix changes, bracketed between
    ``K1`` and ``K2`` (by default 0.9 and 1.1 times the linear buckling
    factor) and bisected to ``tol``, with Newton-Raphson at fixed ``K``"""
    s, kP = build_shell(bc, m=m, nt=nt)
    K_f = buckling_K(s, kP)[0]
    sm = ImperfectShell(s, np.zeros(s.get_size()), kP)
    nz = np.flatnonzero(np.abs(sm.kT0).sum(axis=1).A.ravel() > 0)
    fext = sm.calc_fext()
    state = dict(c=np.zeros(s.get_size()))

    def det_sign(K):
        c = state['c'].copy()
        for _ in range(30):
            r = sm.calc_fint(c, inc=K) - K*fext
            if np.linalg.norm(r) <= 1e-10*K*np.linalg.norm(fext):
                break
            kT = sm.calc_kT(c, inc=K).toarray()[np.ix_(nz, nz)]
            c[nz] -= np.linalg.solve(kT, r[nz])
        else:
            raise RuntimeError('Newton-Raphson did not converge')
        state['c'] = c
        kT = sm.calc_kT(c, inc=K).toarray()[np.ix_(nz, nz)]
        return np.linalg.slogdet(kT)[0]

    K1 = 0.9*K_f if K1 is None else K1
    K2 = 1.1*K_f if K2 is None else K2
    assert det_sign(K1) > 0 and det_sign(K2) < 0
    while K2 - K1 > tol:
        K = (K1 + K2)/2
        if det_sign(K) > 0:
            K1 = K
        else:
            K2 = K
    return (K1 + K2)/2, K_f
