r"""Linear buckling of annular plates under a uniform compression on both
edges, ``'annular_clpt_donnell'``

The least buckling loads of full annuli, clamped, simply supported, free
or movable on each edge, against Yamaki (1958), Table 1, and against the
roots of his stability conditions, Eqs. (5)-(9), for the radius ratios
`\beta = 0.3` and `0.5`; the roots against the table; and the least load of
the clamped annulus against the same table for the three annular models.
The same comparisons with few terms (names ending in ``_small``): one case
for each kind of edge on each side, and the clamped annulus of the three
models, with the bounds given in the Notes.

Notes
-----
An annulus of outer radius `a` and inner radius `b = \beta a` compressed
by the force `p` per unit length uniformly distributed along both edges
has the membrane state `N_{\rho\rho} = N_{\theta\theta} = -p`, constant
over the plate, the solution of Lamé with equal radial stresses on both
edges. Yamaki [1]_, Eqs. (1)-(2) and (15), writes the critical load as

.. math::

    p = k^2 \frac{D}{a^2}, \qquad D = \frac{E h^3}{12 (1 - \nu^2)}

and his Table 1 lists the least `k`, not `k^2`, with the number `n` of
nodal diameters of the buckling mode, for `\nu = 0.3` and twelve cases of
boundary conditions: clamped (`w = w_{,\rho} = 0`), simply supported (`w =
M_\rho = 0`), free (`M_\rho = 0` and the Kirchhoff shear force equal to the
normal component of the applied load) and movable (`w_{,\rho} = 0`, and the
same condition on the shear force), his Eqs. (6)-(9). The conditions on the
forces are the natural boundary conditions of the energy of the Ritz method
with the geometric stiffness of the constant membrane state. Yamaki's Case
1, `\beta = 0.5`: `k = 12.15`, `p a^2/D = 147.6`, below the clamped strip of
width `a - b`, `k = 2 \pi/(1 - \beta)`, his Eq. (27).

Several modes of the simply supported and free cases are axisymmetric, `n
= 0`: the full annulus is made of two half annuli with free radial edges,
connected with ``'SSycte'``, so that every `n` is represented. In-plane
boundary conditions do not change the geometric stiffness of a constant
membrane state; `u = v = 0` on the outer edge removes the in-plane rigid
motions.

With `w = W(\rho) \cos n\theta` and `\rho` in units of `a`, Yamaki's Eq.
(5) gives `W = A J_n(k\rho) + B Y_n(k\rho) + C \rho^n + D \rho^{-n}` (`C +
D \ln \rho` for `n = 0`), and his Eqs. (6)-(9) the conditions on each edge
among

.. math::

    W = 0, \qquad W' = 0, \qquad
    W'' + \nu \left(\frac{W'}{\rho} - \frac{n^2 W}{\rho^2}\right) = 0

    \frac{d}{d\rho} \left(W'' + \frac{W'}{\rho} - \frac{n^2 W}{\rho^2}
    \right) - (1 - \nu) \frac{n^2}{\rho^2} \left(W' - \frac{W}{\rho}
    \right) + k^2 W' = 0

(clamped: the first two; simply supported: the first and the third; free:
the last two; movable: the second and the last). The least root `k` of the
`4 \times 4` determinant over `n`, computed by :func:`exact_k`, differs
from Yamaki's Table 1 by at most one unit of its last digit, except Case 9
(outer edge free, inner edge clamped) at `\beta = 0.3`, where the table
gives 2.18 and the root is `k = 2.1674` with `n = 2` (0.6 %); panels
agrees with the roots within 0.1 %.

The annuli of few terms, `m = 6` terms in `\rho` and `n = 4` in
`\theta`, are Rayleigh-Ritz solutions: the approximation functions
satisfy the essential conditions and the penalty of ``'SSycte'`` only
adds stiffness, so that their least load is not below the least root,
itself within 0.6 % of the table. The excess of the cases of
:data:`CASES_SMALL` over the table is 0.60 % at most (Case 9, `\beta =
0.3`, 1.2 % above the root `k = 2.1674`; the others within 0.11 % of the
table), bounded by 1 % as a regression bound. The clamped annulus, `\beta
= 0.5`, converges more slowly: `k = 12.56581` for the three models, 3.4 %
above the table and the root, `k = 12.1553`; it is compared with this
value of the code, to `10^{-5}`. The clamped edges need more than four
terms in `\rho`, the four functions of the edge degrees of freedom of `w`
being removed by `w = w_{,\rho} = 0` on both edges.

References
----------
.. [1] Yamaki, N., "Buckling of a thin annular plate under uniform
   compression", Journal of Applied Mechanics, Vol. 25, 1958, pp. 267-273,
   Table 1. https://doi.org/10.1115/1.4011755

"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import jv, jvp, yv, yvp
from structsolve import lb

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels import modelDB


E, NU, H = 70e9, 0.3, 1e-3
A = 1.
D = E*H**3/(12*(1 - NU**2))
#: flags (w, w_rho) of the clamped (C), simply supported (S), free (F) and
#: movable (M) edges, 0 for a zero value
EDGE = {'C': (0, 0), 'S': (0, 1), 'F': (1, 1), 'M': (1, 0)}
#: Yamaki (1958), Table 1, least `k` for `\beta = 0.3` and `0.5`, by case
#: number and (outer, inner) edges, `\nu = 0.3`
YAMAKI = {
    (1, 'C', 'C'): (8.63, 12.15),
    (2, 'C', 'S'): (7.06, 9.42),
    (3, 'C', 'F'): (3.19, 3.65),
    (4, 'C', 'M'): (4.71, 6.39),
    (5, 'S', 'C'): (6.16, 8.73),
    (6, 'S', 'S'): (4.75, 6.40),
    (7, 'S', 'F'): (1.61, 1.32),
    (8, 'S', 'M'): (2.40, 3.18),
    (9, 'F', 'C'): (2.18, 3.14),
    (10, 'F', 'S'): (1.34, 1.32),
    (11, 'M', 'C'): (3.84, 5.69),
    (12, 'M', 'S'): (2.87, 3.64),
    }
BETAS = (0.3, 0.5)
CASES = [(case, beta) for case in YAMAKI for beta in BETAS]
#: one case for each kind of outer and of inner edge, for the small models
CASES_SMALL = [((3, 'C', 'F'), 0.3), ((8, 'S', 'M'), 0.5),
               ((9, 'F', 'C'), 0.3), ((12, 'M', 'S'), 0.5)]
MODELS = [k for k in modelDB.db if k.startswith('annular_clpt_')]


def _rows(n, k, rho, edge):
    funcs = [(f(n, k*rho), k*fp(n, k*rho, 1), k**2*fp(n, k*rho, 2),
              k**3*fp(n, k*rho, 3)) for f, fp in ((jv, jvp), (yv, yvp))]
    if n == 0:
        funcs += [(1., 0., 0., 0.),
                  (np.log(rho), 1/rho, -1/rho**2, 2/rho**3)]
    else:
        funcs += [(rho**n, n*rho**(n - 1), n*(n - 1)*rho**(n - 2),
                   n*(n - 1)*(n - 2)*rho**(n - 3)),
                  (rho**-n, -n*rho**(-n - 1), n*(n + 1)*rho**(-n - 2),
                   -n*(n + 1)*(n + 2)*rho**(-n - 3))]
    r = rho
    cond = {'W': [W for W, W1, W2, W3 in funcs],
            'W1': [W1 for W, W1, W2, W3 in funcs],
            'M': [W2 + NU*(W1/r - n**2*W/r**2) for W, W1, W2, W3 in funcs],
            'V': [W3 + W2/r - W1/r**2 - n**2*W1/r**2 + 2*n**2*W/r**3
                  - (1 - NU)*n**2/r**2*(W1 - W/r) + k**2*W1
                  for W, W1, W2, W3 in funcs]}
    return [cond[c] for c in {'C': ('W', 'W1'), 'S': ('W', 'M'),
                              'F': ('M', 'V'), 'M': ('W1', 'V')}[edge]]


def _det(k, n, beta, outer, inner):
    M = np.array(_rows(n, k, 1., outer) + _rows(n, k, beta, inner))
    return np.linalg.det(M/np.abs(M).max(axis=1)[:, None])


def exact_k(beta, outer, inner, nmax=8, kmax=14.):
    """Least root `k < kmax` of Yamaki's stability conditions over the
    numbers of nodal diameters `n < nmax`, edges ``'C'``, ``'S'``, ``'F'``
    or ``'M'``"""
    best = np.inf
    ks = np.linspace(0.3, kmax, 1400)
    for n in range(nmax):
        d = np.array([_det(k, n, beta, outer, inner) for k in ks])
        for i in np.nonzero(np.sign(d[:-1]) != np.sign(d[1:]))[0]:
            k = brentq(_det, ks[i], ks[i + 1], args=(n, beta, outer, inner))
            if abs(_det(k, n, beta, outer, inner)) < 1e-8:
                best = min(best, k)
                break
    return best


def annulus_k(beta, outer, inner, model='annular_clpt_donnell', m=10, n=12):
    r"""Least `k = a \sqrt{p/D}` of the full annulus, two half annuli
    connected with ``'SSycte'``, under `N_{xx} = N_{yy} = -1`"""
    halves = []
    for _ in range(2):
        s = Shell(model=model, a=A*(1 - beta), b=np.pi*A, r=A, m=m, n=n,
                  plyt=H, stack=[0], laminaprop=(E, NU))
        for e in ('y1', 'y2'):
            for f in ('u', 'v', 'w', 'ur', 'vr', 'wr'):
                setattr(s, e + f, 1)
        s.x1u = s.x1v = 0
        s.x2u = s.x2v = 1
        s.x1w, s.x1wr = EDGE[outer]
        s.x2w, s.x2wr = EDGE[inner]
        s.Nxx = -1.
        s.Nyy = -1.
        halves.append(s)
    p1, p2 = halves
    conn = [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0),
            dict(p1=p1, p2=p2, func='SSycte', ycte1=0, ycte2=p2.b)]
    md = MultiDomain([p1, p2], conn)
    eigvals, _ = lb(md.reduce(md.calc_kC(silent=True)),
                    md.reduce(md.calc_kG(silent=True)), silent=True,
                    num_eigvalues=4)
    return np.sqrt(eigvals[0]*A**2/D)


@pytest.mark.parametrize('case, beta', CASES)
def test_exact_roots_against_yamaki_table(case, beta):
    """The least roots of Yamaki's (1958) stability conditions against his
    Table 1, within 0.6 %"""
    _, outer, inner = case
    ref = YAMAKI[case][BETAS.index(beta)]
    assert np.isclose(exact_k(beta, outer, inner), ref, rtol=6e-3)


@pytest.mark.slow
@pytest.mark.parametrize('case, beta', CASES)
def test_yamaki1958_table1(case, beta):
    """Least `k` against the roots of Yamaki's (1958) stability conditions,
    within 0.1 %, and against his Table 1, within 0.6 %"""
    _, outer, inner = case
    ref = YAMAKI[case][BETAS.index(beta)]
    k = annulus_k(beta, outer, inner)
    assert np.isclose(k, exact_k(beta, outer, inner), rtol=1e-3)
    assert np.isclose(k, ref, rtol=6e-3)


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS)
def test_clamped_models(model):
    r"""Clamped annulus, `\beta = 0.5`, for the three annular models,
    against Yamaki (1958), Table 1, Case 1, `k = 12.15`, within 0.1 %"""
    assert np.isclose(annulus_k(0.5, 'C', 'C', model=model), 12.15,
                      rtol=1e-3)


@pytest.mark.parametrize('case, beta', CASES_SMALL)
def test_yamaki1958_table1_small(case, beta):
    """Least `k`, `m = 6`, `n = 4`, against Yamaki's (1958) Table 1: not
    below it by more than 0.6 %, above it by less than 1 %"""
    _, outer, inner = case
    ref = YAMAKI[case][BETAS.index(beta)]
    k = annulus_k(beta, outer, inner, m=6, n=4)
    assert (1 - 6e-3)*ref < k < (1 + 1e-2)*ref


@pytest.mark.parametrize('model', MODELS)
def test_clamped_models_small(model):
    r"""Clamped annulus, `\beta = 0.5`, `m = 6`, `n = 4`, for the three
    annular models: the value of the code, `k = 12.56581`, to `10^{-5}`,
    above Yamaki (1958), Table 1, Case 1, `k = 12.15`"""
    k = annulus_k(0.5, 'C', 'C', model=model, m=6, n=4)
    assert np.isclose(k, 12.56581, rtol=1e-5)
    assert k > 12.15
