r"""Annular-sector plates of the first-order shear deformation theory,
``'annular_fsdt_donnell'``, ``'annular_fsdt_gl'`` and ``'annular_fsdt_tl'``

Thin annuli of ``'annular_fsdt_donnell'`` against the classical plate
results used for ``'annular_clpt_donnell'``: the free vibration of the
annulus clamped outside and free inside against the exact roots of the
frequency equation of Leissa (1969), Eqs. (1.11)-(1.13) and (1.17); the
buckling under a uniform compression against Yamaki (1958), Table 1; the
buckling under a torque against Irie, Yamada and Tsujino (1983), Table 1.
Moderately thick annuli under a torque against the Mindlin plate results of
Bagheri, Kiani and Eslami (2019), Table 2, where the transverse shear
lowers the buckling torque by up to 47 % in the cases used. The natural
frequencies of a thick annulus against the Ritz FSDT values of Dozio
(2015), Table 5, for four combinations of free, simply supported and
clamped edges. The three
models: the same buckling torque of a thick annulus; internal force vectors
that agree to the second order in the displacements; the strains of an
in-plane rigid rotation and of a rigid rotation about an axis inclined to
the plane; the tangent stiffness against the derivative of the internal
force vector; the laminate offset and the limit `r \to \infty` against the
flat plates.

Notes
-----
The full annulus of outer radius `a` and inner radius `b = \beta a` is made
of two half annuli with free radial edges connected with ``'SSycte'``, so
that every number `n` of nodal diameters is represented. Edge conditions,
named inner-outer for the torque as in Bagheri, Kiani and Eslami [4]_ and
outer-inner for the compression as in Yamaki [2]_: clamped (C, `w =
\phi_x = \phi_y = 0`), simply supported (S, `w = \phi_y = 0`, the
tangential rotation fixed, Bagheri et al. Eq. (11)) and free (F). For `h/a
= 0.001` the transverse shear changes the buckling torques of Bagheri et
al. by less than 0.6 % from the classical values of Irie, Yamada and
Tsujino [3]_ (their Table 1), and the models of the shear deformation
theory reproduce the classical tables.

The torque `T` on the outer edge, reacted on the inner edge (`u = v = 0`),
gives the membrane state `N_{\rho\theta} = T/(2 \pi \rho^2)`, `N_{\rho\rho}
= N_{\theta\theta} = 0` [4]_, Eqs. (10) and (12); it is applied as the
force per unit length `q = T/(2 \pi a^2)` along `\boldsymbol{e}_\theta` on
the outer edge, and the linear static case and the geometric stiffness of
its membrane state are computed by :meth:`.MultiDomain.linear_buckling`. The
critical torque is given as `\tau = T/(2 \pi D)`, `D = E h^3/(12 (1 -
\nu^2))`, [3]_ and [4]_, Eq. (26), `\nu = 0.3`. Bagheri et al. [4]_ use the
transverse shear stiffness `A_{44} = A_{55} = E h/(2 (1 + \nu))` without a
shear correction factor, their Eq. (7): the thick annuli are computed with
``Shell.fsdt_shear_correction = 1``, which reproduces their Table 2. The
default of :class:`.Shell`, ``fsdt_shear_correction = 'rohwer'``, gives
the shear correction factor `5/6` of a homogeneous plate and lower
torques, not tested; with `m = 12`, `n = 16`, edges inner-outer, `\tau`
of [4]_ and of panels with the factor 1 and with ``'rohwer'``, and their
differences from [4]_:

=====  =====  =====  =======  =======  ========  =======  ========
edges  `h/a`  ratio  [4]_     1        diff (%)  rohwer   diff (%)
=====  =====  =====  =======  =======  ========  =======  ========
C-C    0.05   0.3    53.758   53.761   +0.005    52.421   -2.49
C-C    0.05   0.5    150.414  150.415  +0.001    145.088  -3.54
C-C    0.10   0.3    39.261   39.262   +0.002    36.699   -6.53
C-C    0.10   0.5    98.436   98.436   0.000     90.368   -8.20
S-S    0.05   0.3    34.976   34.976   -0.001    34.584   -1.12
S-S    0.05   0.5    100.749  100.749  0.000     98.990   -1.75
S-S    0.10   0.3    29.912   29.912   +0.001    28.804   -3.70
S-S    0.10   0.5    79.632   79.632   +0.001    75.433   -5.27
C-F    0.05   0.3    12.215   12.216   +0.007    12.122   -0.76
C-F    0.05   0.5    33.524   33.524   0.000     33.165   -1.07
C-F    0.10   0.3    11.030   11.030   +0.002    10.773   -2.33
C-F    0.10   0.5    29.174   29.174   0.000     28.273   -3.09
F-C    0.05   0.3    9.904    9.908    +0.038    9.820    -0.84
F-C    0.05   0.5    29.607   29.608   +0.002    29.273   -1.13
F-C    0.10   0.3    8.877    8.877    +0.003    8.660    -2.45
F-C    0.10   0.5    25.626   25.625   -0.002    24.801   -3.22
=====  =====  =====  =======  =======  ========  =======  ========

The ratio is `\beta`. The transverse shear lowers the torques the most for
the thicker plates and the stiffer edges, and the factor `5/6` adds up to
a further 8 % to it.

The uniform compression `p` on both edges gives the constant membrane state
`N_{\rho\rho} = N_{\theta\theta} = -p`, applied as ``Nxx = Nyy = -1``;
Yamaki [2]_ tabulates `k = a \sqrt{p/D}`. The free vibration of the annulus
clamped at the outer radius and free at the inner radius has the frequency
parameters `\lambda^2 = \omega a^2 \sqrt{\rho_m h/D}` at the roots of the
determinant of Leissa [1]_, Eqs. (1.11)-(1.13) and (1.17), computed by
:func:`exact_lambda2`.

A rigid rotation `t` of the plane about its axis has `u = (r - x)(1 - \cos
t)`, `v = (r - x) \sin t`, `w = \phi_x = \phi_y = 0`, linear in `x`; the
membrane strains of ``'annular_fsdt_gl'`` and ``'annular_fsdt_tl'`` vanish
and those of ``'annular_fsdt_donnell'`` are `\varepsilon_{xx} =
\varepsilon_{yy} = -(1 - \cos t)`. A rigid rotation `t` about an axis inclined to the plane, with the
rotations those of the rotated normal, is fitted by least squares over a
quarter annulus: the transverse shear strains of ``'annular_fsdt_gl'`` are of
`O(t^2)`, while all the strains of ``'annular_fsdt_tl'`` are of `O(t^5)`,
see :mod:`panels.models.annular_fsdt_tl_num`.

For `r \to \infty` the strains of the annular models are those of the flat
plates ``'plate_fsdt_donnell'``, ``'plate_fsdt_gl'`` and
``'plate_fsdt_tl'``.

Dozio [5]_, Table 5, gives the eight lowest `\lambda = \omega a^2
\sqrt{\rho h/D}` of an isotropic annulus, `\beta = 0.3`, `h/a = 0.2`, `\nu
= 0.3`, with a first-order theory obtained from his model HOT1 with a zero
transverse normal stress and the shear correction factor `5/6`, his Eq.
(30): ``fsdt_shear_correction = 5/6``. His edges, inner-outer ([5]_,
Table 1): clamped, all displacements zero, `u = v = w = \phi_x = \phi_y =
0`; simply supported, `v = w = \phi_y = 0`; free. His list has the modes of
both families, the transverse (`w, \phi_x, \phi_y`) and the in-plane (`u,
v`) ones, uncoupled for one isotropic layer; the modes of `p \ge 1` nodal
diameters, pairs of equal frequencies in the full annulus, are merged. With
`m = 14`, `n = 16` (the same to five digits with `m = 16`, `n = 20`)
panels reproduces his 32 values within 0.005 %, with one exception: for
the annulus free inside and clamped outside, panels has a transverse pair
`\lambda = 40.294`, one nodal diameter and the second radial order, between
his sixth and seventh values (39.424, in-plane, and 40.534, the in-plane
torsional mode), that his Table 5 does not list; the TSDT model gives the
same mode at 40.720. The test skips it, the reason of its absence from [5]_
being unknown.

The tests ``test_*_small`` repeat the cases with few terms (`m = 6`, `n =
4` for the torque, `m = n = 6` for the compression and the frequencies, `m =
n = 3` and `4` for the tangent stiffness and the offset), with plates of `h/a
\ge 0.01`, as the transverse shear locks the thin plates with few terms (with
`h/a = 0.01` and `m = n = 6` the torque of the clamped annulus is already 40
% above the classical value of [3]_). Their bounds, measured on hpc12: the
buckling torque of the clamped annulus, `h/a = 0.10`, is 53.27954 for the
three models, 36 % above the 39.261 of [4]_ (9.4 % with `m = n = 6`), which
the Ritz values of the full-size tests approach from above; the
compression `k` of `h/a = 0.01` differs from [2]_ by -0.06 % (C-S) and
+0.09 % (F-C), the transverse shear lowering it slightly, within the
tolerance 0.5 %; the three lowest `\lambda` of [5]_ are reproduced within
-0.003 % to +0.133 %, Ritz values from above with the rounding of the table,
within the bounds `-10^{-4}` and `2 \times 10^{-3}`.

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, 1969, Eqs.
   (1.11)-(1.13) and (1.17), p. 2.
.. [2] Yamaki, N., "Buckling of a thin annular plate under uniform
   compression", Journal of Applied Mechanics, Vol. 25, 1958, pp. 267-273,
   Table 1. https://doi.org/10.1115/1.4011755
.. [3] Irie, T., Yamada, G., and Tsujino, M., "Buckling loads of annular
   plates subjected to a torque", Journal of Sound and Vibration, Vol. 86,
   No. 1, 1983, pp. 145-146, Table 1.
   https://doi.org/10.1016/0022-460X(83)90951-3
.. [4] Bagheri, H., Kiani, Y., and Eslami, M. R., "Buckling of moderately
   thick annular plates subjected to torque", Archive of Mechanical
   Engineering, Vol. 66, No. 2, 2019, pp. 209-227, Eqs. (7), (10), (11),
   (12) and (26), Tables 1 and 2.
   https://doi.org/10.24425/ame.2019.128445
.. [5] Dozio, L., "Refined 2-D theories for free vibration analysis of
   annular plates: Unified Ritz formulation and numerical assessment",
   Computers and Structures, Vol. 147, 2015, pp. 250-258, Eq. (30), Tables
   1 and 5. https://doi.org/10.1016/j.compstruc.2014.10.015

"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import iv, ivp, jv, jvp, kv, kvp, yv, yvp
from structsolve import freq, lb

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels import modelDB


E, NU, RHO = 70e9, 0.3, 2700.
A = 1.
#: the annular models and the flat plates they tend to for r -> infinity
MODELS = [k for k in modelDB.db if k.startswith('annular_fsdt_')]
PLATE = {k: k.replace('annular_', 'plate_') for k in MODELS}
DONNELL = 'annular_fsdt_donnell'
#: flags (w, phix, phiy) of the clamped (C), simply supported (S) and free
#: (F) edges, 0 for a zero value
EDGE = {'C': (0, 0, 0), 'S': (0, 1, 0), 'F': (1, 1, 1)}
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')


def bending_D(h):
    return E*h**3/(12*(1 - NU**2))


def half_annulus(beta, h, model, m, n, outer, inner, shear=None):
    r"""Half annulus, `a(1 - \beta)` wide, of outer radius `a`, free radial
    edges, the edge conditions ``outer`` and ``inner`` for `w`, `\phi_x`,
    `\phi_y` and free in-plane edges"""
    s = Shell(model=model, a=A*(1 - beta), b=np.pi*A, r=A, m=m, n=n,
              plyt=h, stack=[0], laminaprop=(E, NU), rho=RHO)
    if shear is not None:
        s.fsdt_shear_correction = shear
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in FIELDS:
            setattr(s, e + f, 1)
            setattr(s, e + f + 'r', 1)
    s.x1w, s.x1phix, s.x1phiy = EDGE[outer]
    s.x2w, s.x2phix, s.x2phiy = EDGE[inner]
    return s


def full_annulus(halves):
    p1, p2 = halves
    conn = [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0),
            dict(p1=p1, p2=p2, func='SSycte', ycte1=0, ycte2=p2.b)]
    return MultiDomain([p1, p2], conn)


def tau(beta, inner, outer, h, model=DONNELL, m=12, n=16, shear=None):
    r"""Least `\tau = T/(2 \pi D)` of the annulus under a torque on the outer
    edge, the inner edge with `u = v = 0`"""
    T = 2*np.pi*bending_D(h)
    halves = []
    for _ in range(2):
        s = half_annulus(beta, h, model, m, n, outer, inner, shear)
        s.x2u = s.x2v = 0
        s.add_distr_load_fixed_x(0., funcy=lambda y: T/(2*np.pi*A**2) + 0*y)
        halves.append(s)
    md = full_annulus(halves)
    eigvals, _ = md.linear_buckling(num_eigvalues=6, silent=True)
    return np.abs(eigvals).min()


def yamaki_k(beta, outer, inner, h=1e-3, model=DONNELL, m=10, n=12):
    r"""Least `k = a \sqrt{p/D}` of the annulus under `N_{xx} = N_{yy} =
    -1`, the outer edge with `u = v = 0`"""
    halves = []
    for _ in range(2):
        s = half_annulus(beta, h, model, m, n, outer, inner)
        s.x1u = s.x1v = 0
        s.Nxx = -1.
        s.Nyy = -1.
        halves.append(s)
    md = full_annulus(halves)
    eigvals, _ = lb(md.reduce(md.calc_kC(silent=True)),
                    md.reduce(md.calc_kG(silent=True)), silent=True,
                    num_eigvalues=4)
    return np.sqrt(eigvals[0]*A**2/bending_D(h))


def _det(lam2, n, beta):
    k = np.sqrt(lam2)
    cols = []
    for f, fp in ((jv, jvp), (yv, yvp), (iv, ivp), (kv, kvp)):
        W0, W1 = f(n, k), k*fp(n, k, 1)
        W, W1i, W2, W3 = (f(n, k*beta), k*fp(n, k*beta, 1),
                          k**2*fp(n, k*beta, 2), k**3*fp(n, k*beta, 3))
        b = beta
        M = W2 + NU*(W1i/b - n**2*W/b**2)
        V = (W3 + W2/b - W1i/b**2 - n**2*W1i/b**2 + 2*n**2*W/b**3
             - (1 - NU)*n**2/b**2*(W1i - W/b))
        col = np.array([W0, W1, M, V])
        cols.append(col/np.abs(col).max())
    return np.linalg.det(np.array(cols).T)


def exact_lambda2(n, beta, lo, hi):
    """Root in ``[lo, hi]`` of the frequency equation of the annulus clamped
    outside and free inside (Leissa 1969, Eqs. (1.11)-(1.13) and (1.17))"""
    return brentq(_det, lo, hi, args=(n, beta))


#: Irie, Yamada and Tsujino (1983), Table 1, `\tau` for `\beta = 0.3` and
#: `0.5`, edges (inner, outer), `\nu = 0.3`
IRIE = {
    ('C', 'C'): (61.97, 186.1),
    ('S', 'S'): (37.10, 110.6),
    ('C', 'F'): (12.80, 35.80),
    ('F', 'C'): (10.60, 32.05),
    }
#: Bagheri, Kiani and Eslami (2019), Table 2, `\tau` for `\beta = 0.3` and
#: `0.5`, edges (inner, outer), `h/a = 0.05` and `0.10`, `\nu = 0.3`
BAGHERI = {
    (('C', 'C'), 0.05): (53.758, 150.414),
    (('C', 'C'), 0.10): (39.261, 98.436),
    (('S', 'S'), 0.05): (34.976, 100.749),
    (('S', 'S'), 0.10): (29.912, 79.632),
    (('C', 'F'), 0.05): (12.215, 33.524),
    (('C', 'F'), 0.10): (11.030, 29.174),
    (('F', 'C'), 0.05): (9.904, 29.607),
    (('F', 'C'), 0.10): (8.877, 25.626),
    }
#: Yamaki (1958), Table 1, `k` for `\beta = 0.3` and `0.5`, edges (outer,
#: inner), `\nu = 0.3`
YAMAKI = {
    ('C', 'C'): (8.63, 12.15),
    ('C', 'S'): (7.06, 9.42),
    ('S', 'S'): (4.75, 6.40),
    ('C', 'F'): (3.19, 3.65),
    ('F', 'C'): (2.18, 3.14),
    }
BETAS = (0.3, 0.5)


@pytest.mark.slow
@pytest.mark.parametrize('beta', BETAS)
@pytest.mark.parametrize('edges', list(IRIE))
def test_irie1983_thin(edges, beta):
    r"""Torque, `h/a = 0.001`: least `\tau` against Irie, Yamada and
    Tsujino (1983), Table 1, within 1 %"""
    ref = IRIE[edges][BETAS.index(beta)]
    assert np.isclose(tau(beta, *edges, h=1e-3), ref, rtol=1e-2)


@pytest.mark.slow
@pytest.mark.parametrize('beta', BETAS)
@pytest.mark.parametrize('edges, delta', list(BAGHERI))
def test_bagheri2019_thick(edges, delta, beta):
    r"""Torque, `h/a = 0.05` and `0.10`, no shear correction: least `\tau`
    against Bagheri, Kiani and Eslami (2019), Table 2, within 0.1 %"""
    ref = BAGHERI[(edges, delta)][BETAS.index(beta)]
    got = tau(beta, *edges, h=delta*A, shear=1.)
    assert np.isclose(got, ref, rtol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS)
def test_thick_models(model):
    r"""Clamped annulus, `\beta = 0.3`, `h/a = 0.10`, for the three models,
    against Bagheri, Kiani and Eslami (2019), Table 2, `\tau = 39.261`, within
    0.1 %"""
    got = tau(0.3, 'C', 'C', h=0.1*A, model=model, shear=1.)
    assert np.isclose(got, 39.261, rtol=1e-3)


def test_shear_correction_one():
    r"""``fsdt_shear_correction = 1``: the transverse shear stiffness of
    ``Shell.ABD`` is `G h`"""
    h = 0.1
    s = half_annulus(0.3, h, DONNELL, 4, 4, 'C', 'C', shear=1.)
    G = E/(2*(1 + NU))
    assert np.allclose(np.diag(s.ABD)[6:], G*h, rtol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize('beta', BETAS)
@pytest.mark.parametrize('edges', list(YAMAKI))
def test_yamaki1958_thin(edges, beta):
    r"""Uniform compression, `h/a = 0.001`: least `k` against Yamaki (1958),
    Table 1, within 1 %"""
    ref = YAMAKI[edges][BETAS.index(beta)]
    assert np.isclose(yamaki_k(beta, *edges), ref, rtol=1e-2)


@pytest.mark.slow
@pytest.mark.parametrize('beta', [0.4, 0.6])
def test_leissa_thin(beta):
    r"""Free vibration, clamped outside and free inside, `h/a = 0.001`:
    `\lambda^2` of `n = 0, 1, 1, 2, 2` against the exact roots (Leissa 1969,
    Eqs. (1.11)-(1.13) and (1.17)), within 0.1 %"""
    h = 1e-3
    halves = [half_annulus(beta, h, DONNELL, 10, 12, 'C', 'F')
              for _ in range(2)]
    for s in halves:
        s.x1u = s.x1v = 0
    md = full_annulus(halves)
    eigvals, _ = freq(md.reduce(md.calc_kC(silent=True)),
                      md.reduce(md.calc_kM(silent=True)),
                      sparse_solver=False, silent=True, num_eigvalues=8)
    l2 = np.sort(np.sqrt(-eigvals.real))*A**2*np.sqrt(RHO*h/bending_D(h))
    #NOTE roots of n = 0, 1, 2 within 10 % of Leissa (1969), Table 2.21
    guess = {0.4: (13.54, 19.80, 31.34), 0.6: (25.60, 28.52, 36.60)}[beta]
    ref = [exact_lambda2(n, beta, 0.9*guess[n], 1.1*guess[n])
           for n in (0, 1, 1, 2, 2)]
    assert np.allclose(l2[:5], ref, rtol=1e-3)


#: fields fixed at the edges of the thick annulus of the frequencies:
#: clamped, simply supported, free
DOZIO_EDGES = {'C': FIELDS, 'S': ('v', 'w', 'phiy'), 'F': ()}
#: reference eight lowest lambda, by edges (inner, outer), and the index of
#: the frequencies of panels without a reference value
DOZIO = {
    ('F', 'S'): ((4.5328, 11.205, 12.711, 15.875, 20.740, 27.931, 30.530,
                  31.301), ()),
    ('F', 'F'): ((4.6160, 7.8878, 11.126, 15.146, 15.659, 18.778, 26.674,
                  27.279), ()),
    ('F', 'C'): ((10.365, 15.895, 25.384, 35.763, 36.860, 39.424, 40.534,
                  43.901), (6,)),
    ('C', 'C'): ((30.146, 30.878, 33.775, 39.638, 47.999, 48.220, 52.997,
                  57.793), ()),
    }


def dozio_lambdas(edges, m, n):
    r"""Distinct `\lambda = \omega a^2 \sqrt{\rho h/D}` of the full annulus
    of Dozio (2015), Table 5, `\beta = 0.3`, `h/a = 0.2`, shear correction
    factor `5/6`, edges (inner, outer) of :data:`DOZIO_EDGES`, in ascending
    order"""
    inner, outer = edges
    h = 0.2*A
    halves = []
    for _ in range(2):
        s = Shell(model=DONNELL, a=0.7*A, b=np.pi*A, r=A, m=m, n=n,
                  plyt=h, stack=[0], laminaprop=(E, NU), rho=RHO)
        s.fsdt_shear_correction = 5/6
        for e in ('x1', 'x2', 'y1', 'y2'):
            for f in FIELDS:
                setattr(s, e + f, 1)
                setattr(s, e + f + 'r', 1)
        for e, kind in (('x1', outer), ('x2', inner)):
            for f in DOZIO_EDGES[kind]:
                setattr(s, e + f, 0)
        halves.append(s)
    md = full_annulus(halves)
    eigvals, _ = freq(md.reduce(md.calc_kC(silent=True)),
                      md.reduce(md.calc_kM(silent=True)),
                      sparse_solver=False, silent=True, num_eigvalues=40,
                      symmetric=True)
    lam = np.sort(np.sqrt(np.abs(eigvals)))*A**2*np.sqrt(RHO*h/bending_D(h))
    #NOTE rigid-body modes excluded, pairs of equal frequencies merged
    values = []
    for v in lam[np.isfinite(lam) & (lam > 0.1)]:
        if not values or v/values[-1] - 1 > 1e-4:
            values.append(v)
    return np.array(values)


@pytest.mark.slow
@pytest.mark.parametrize('edges', list(DOZIO))
def test_dozio2015_thick(edges):
    r"""Free vibration, `\beta = 0.3`, `h/a = 0.2`, shear correction factor
    `5/6`: the eight lowest `\lambda = \omega a^2 \sqrt{\rho h/D}` of both
    families of modes against Dozio (2015), Table 5, within 0.01 %"""
    values = dozio_lambdas(edges, 14, 16)
    ref, skip = DOZIO[edges]
    got = np.delete(values, skip)[:8]
    assert np.allclose(got, ref, rtol=1e-4)


def _sector(model, m=6, n=6):
    s = Shell(model=model, a=0.5*A, b=np.pi/2*A, r=A, m=m, n=n, plyt=0.02,
              stack=[0, 45, -45], laminaprop=(E, NU), rho=RHO)
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in FIELDS:
            setattr(s, e + f, 1)
            setattr(s, e + f + 'r', 1)
    s.x1u = s.x1v = s.x1w = 0
    return s


def test_internal_force_linear_limit():
    r"""``calc_fint`` of the three models at `\epsilon \boldsymbol{c}`: the
    differences between the models and from `\boldsymbol{K}_0 \epsilon
    \boldsymbol{c}` are of `O(\epsilon^2)` (ratio between 3.8 and 4.2 when
    `\epsilon` halves) and below `10^{-3}` of the linear force at the
    smaller `\epsilon`"""
    rng = np.random.default_rng(0)
    c0 = None
    out = {}
    for model in MODELS:
        s = _sector(model)
        size = s.get_size()
        if c0 is None:
            c0 = rng.standard_normal(size)*1e-3
        K0 = s.calc_kC(silent=True)
        fw = modelDB.db[model]['matrices_num']
        out[model] = [(fw.calc_fint(eps*c0, s.ABD, s, size, 0, s.nx, s.ny),
                       K0 @ (eps*c0)) for eps in (1e-2, 5e-3)]
    for model in MODELS:
        dl = [np.abs(f - kc).max() for f, kc in out[model]]
        assert 3.8 < dl[0]/dl[1] < 4.2
        assert dl[1] < 1e-3*np.abs(out[model][1][1]).max()
        if model != DONNELL:
            dm = [np.abs(f - g[0]).max()
                  for (f, _), g in zip(out[model], out[DONNELL])]
            assert 3.8 < dm[0]/dm[1] < 4.2


@pytest.mark.parametrize('model', MODELS)
def test_inplane_rigid_rotation(model):
    r"""Rigid rotation `t = 0.3` of the plane about its axis, exact in the
    approximation: zero membrane strains for ``'annular_fsdt_gl'`` and
    ``'annular_fsdt_tl'``, `\varepsilon_{xx} = \varepsilon_{yy} = -(1 -
    \cos t)` for ``'annular_fsdt_donnell'``"""
    t = 0.3
    s = _sector(model, m=4, n=4)
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in FIELDS:
            setattr(s, e + f, 1)
    size = s.get_size()
    xs, ys = np.meshgrid(np.linspace(0.05, 0.95, 7)*s.a,
                         np.linspace(0.05, 0.95, 7)*s.b)
    xs, ys = xs.ravel(), ys.ravel()
    target = {'u': (A - xs)*(1 - np.cos(t)), 'v': (A - xs)*np.sin(t)}
    c = _fit(s, size, xs, ys, target)
    strains = s.strain(c, xs=xs, ys=ys, NLgeom=True)[1]
    for name in ('gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz'):
        assert np.abs(strains[name]).max() < 1e-10
    for name in ('exx', 'eyy'):
        if model == DONNELL:
            assert np.allclose(strains[name], -(1 - np.cos(t)), rtol=1e-8)
        else:
            assert np.abs(strains[name]).max() < 1e-10


def _fit(s, size, xs, ys, target):
    """Ritz constants whose fields fit ``target`` (dict of field values at
    ``xs``, ``ys``, zero for the missing fields) by least squares"""
    fuvw = modelDB.db[s.model]['field'].fuvw
    cols = []
    for k in range(size):
        e = np.zeros(size)
        e[k] = 1.
        cols.append(np.concatenate(fuvw(e, s, xs, ys)))
    Amat = np.array(cols).T
    rhs = np.concatenate([target.get(f, np.zeros_like(xs)) for f in FIELDS])
    c, *_ = np.linalg.lstsq(Amat, rhs, rcond=None)
    assert np.abs(Amat @ c - rhs).max() < 1e-9
    return np.ascontiguousarray(c)


def _rotated_fields(t, axis, xs, ys):
    r"""Fields of the rigid rotation `t` about the unit ``axis`` of the
    plane, with the rotations of the rotated normal"""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    Q = np.eye(3) + np.sin(t)*K + (1 - np.cos(t))*K @ K
    th = ys/A
    erho = np.array([np.cos(th), np.sin(th), 0*th])
    ex, ey = -erho, np.array([-np.sin(th), np.cos(th), 0*th])
    ez = np.array([0*th, 0*th, -1 + 0*th])
    X0 = (A - xs)*erho
    Ug = Q @ X0 - X0
    n = Q @ ez
    dot = lambda p, q: (p*q).sum(axis=0)
    return {'u': dot(Ug, ex), 'v': dot(Ug, ey), 'w': dot(Ug, ez),
            'phix': dot(n, ex), 'phiy': dot(n, ey)}


def test_rigid_rotation_tl():
    r"""Rigid rotations `t = 0.2` and `0.1` about an axis at 45 degrees to
    the plane, fitted over a quarter annulus with the rotations of the
    rotated normal: the membrane and bending strains of the three models
    below `10^{-12}`, except the membrane strains of
    ``'annular_fsdt_donnell'``; the transverse shear strains of
    ``'annular_fsdt_tl'`` of `O(t^5)` (fitted order between 4.5 and 5.5),
    those of ``'annular_fsdt_gl'`` of `O(t^2)` (between 1.8 and 2.2) and
    more than 1000 times larger at `t = 0.1`"""
    axis = np.array([np.cos(0.4), np.sin(0.4), 1.])/np.sqrt(2)
    xg, yg = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 30))
    shear = {model: [] for model in MODELS}
    for t in (0.2, 0.1):
        for model in MODELS:
            s = _sector(model, m=4, n=14)
            for e in ('x1', 'x2', 'y1', 'y2'):
                for f in FIELDS:
                    setattr(s, e + f, 1)
            size = s.get_size()
            xs, ys = xg.ravel()*s.a, yg.ravel()*s.b
            c = _fit(s, size, xs, ys, _rotated_fields(t, axis, xs, ys))
            xe, ye = np.meshgrid(np.linspace(0.1, 0.9, 5)*s.a,
                                 np.linspace(0.1, 0.9, 5)*s.b)
            st = s.strain(c, xs=xe.ravel(), ys=ye.ravel(), NLgeom=True)[1]
            names = ('kxx', 'kyy', 'kxy')
            if model != DONNELL:
                names = ('exx', 'eyy', 'gxy') + names
            for name in names:
                assert np.abs(st[name]).max() < 1e-12
            shear[model].append(max(np.abs(st['gyz']).max(),
                                    np.abs(st['gxz']).max()))
    order = lambda v: np.log2(v[0]/v[1])
    tl, gl = shear['annular_fsdt_tl'], shear['annular_fsdt_gl']
    assert 4.5 < order(tl) < 5.5
    assert 1.8 < order(gl) < 2.2
    assert tl[1] < 1e-3*gl[1]


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS)
def test_tangent_is_derivative_of_fint(model):
    """``fkC_num(NLgeom=1) + fkG_num`` against central differences of
    ``calc_fint``"""
    fw = modelDB.db[model]['matrices_num']
    s = _sector(model, m=5, n=5)
    size = s.get_size()
    c = np.random.default_rng(1).standard_normal(size)*1e-3
    fint = lambda cc: fw.calc_fint(cc, s.ABD, s, size, 0, s.nx, s.ny)
    kT = (s.calc_kC(c=c, NLgeom=True, silent=True)
          + s.calc_kG(c=c, NLgeom=True, silent=True)).toarray()
    dc = 1e-7
    J = np.column_stack([(fint(c + dc*e) - fint(c - dc*e))/(2*dc)
                         for e in np.eye(size)])
    assert np.abs(kT - J).max() < 1e-7*np.abs(kT).max()


@pytest.mark.parametrize('model', MODELS)
def test_r_not_larger_than_a_raises(model):
    """An inner radius `r - a \\le 0` is rejected"""
    s = Shell(model=model, a=1., b=1., r=1., m=4, n=4,
              plyt=1e-3, stack=[0], laminaprop=(E, NU))
    with pytest.raises(ValueError):
        s.calc_kC(silent=True)


def _square(model, r, offset, m=8):
    r"""Square panel, `a = b = 1`, `h = 0.01`, `m = n` terms, with `u = v = w =
    0` on the four edges, of radius ``r`` for the annular models"""
    kw = dict(r=r) if model.startswith('annular') else {}
    s = Shell(model=model, a=1., b=1., m=m, n=m, plyt=0.01, stack=[0],
              laminaprop=(E, NU), rho=RHO, offset=offset, **kw)
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in FIELDS:
            setattr(s, e + f, 1)
            setattr(s, e + f + 'r', 1)
        for f in 'uvw':
            setattr(s, e + f, 0)
    return s


@pytest.mark.slow
@pytest.mark.parametrize('model', MODELS)
def test_offset_plate_limit(model):
    r"""Laminate offset `d = h/2`, stiffness and mass: for `r = 10^4 a` the
    frequencies of the annular model are those of the flat plate with the
    same offset, within 3e-4; the offset raises them by more than 30 %"""
    def omegas(s):
        eigvals, _ = freq(s.calc_kC(silent=True), s.calc_kM(silent=True),
                          sparse_solver=False, silent=True, num_eigvalues=4)
        return np.sort(np.sqrt(-eigvals.real))[:4]
    got = omegas(_square(model, 1e4, 0.005))
    ref = omegas(_square(PLATE[model], None, 0.005))
    assert np.allclose(got, ref, rtol=3e-4)
    assert got[0] > 1.3*omegas(_square(model, 1e4, 0.))[0]


def _omegas(s):
    eigvals, _ = freq(s.calc_kC(silent=True), s.calc_kM(silent=True),
                      sparse_solver=False, silent=True, num_eigvalues=4)
    return np.sort(np.sqrt(-eigvals.real))[:4]


@pytest.mark.parametrize('model', MODELS)
def test_torque_small(model):
    r"""Torque, clamped annulus, `\beta = 0.3`, `h/a = 0.10`, no shear
    correction, `m = 6`, `n = 4`: least `\tau` above Bagheri, Kiani and
    Eslami (2019), Table 2, 39.261, and equal to the value of the code,
    53.27954, for the three models"""
    got = tau(0.3, 'C', 'C', h=0.1*A, model=model, m=6, n=4, shear=1.)
    assert got > 39.261
    assert np.isclose(got, 53.27954, rtol=1e-5)


#: k of the code for m = n = 6, h/a = 0.01, beta = 0.5, by edges (outer,
#: inner)
YAMAKI_SMALL = {('C', 'S'): 9.414388, ('F', 'C'): 3.142901}


@pytest.mark.parametrize('edges', list(YAMAKI_SMALL))
def test_yamaki1958_small(edges):
    r"""Uniform compression, `h/a = 0.01`, `\beta = 0.5`, `m = n = 6`: least
    `k` against Yamaki (1958), Table 1, within 0.5 %, and equal to the value
    of the code"""
    got = yamaki_k(0.5, *edges, h=0.01, m=6, n=6)
    assert np.isclose(got, YAMAKI[edges][1], rtol=5e-3)
    assert np.isclose(got, YAMAKI_SMALL[edges], rtol=1e-5)


@pytest.mark.parametrize('edges', [('F', 'S'), ('F', 'C')])
def test_dozio2015_small(edges):
    r"""Free vibration, the annulus of :func:`test_dozio2015_thick`, `m = n =
    6`: the three lowest `\lambda` not below Dozio (2015), Table 5, by more
    than his rounding, `10^{-4}`, and at most 0.2 % above it"""
    got = dozio_lambdas(edges, 6, 6)[:3]
    ref = np.array(DOZIO[edges][0][:3])
    assert np.all(got > (1 - 1e-4)*ref)
    assert np.all(got < (1 + 2e-3)*ref)


@pytest.mark.parametrize('model', MODELS)
def test_tangent_small(model):
    """``fkC_num(NLgeom=1) + fkG_num`` against central differences of
    ``calc_fint``, `m = n = 3`"""
    fw = modelDB.db[model]['matrices_num']
    s = _sector(model, m=3, n=3)
    size = s.get_size()
    c = np.random.default_rng(1).standard_normal(size)*1e-3
    fint = lambda cc: fw.calc_fint(cc, s.ABD, s, size, 0, s.nx, s.ny)
    kT = (s.calc_kC(c=c, NLgeom=True, silent=True)
          + s.calc_kG(c=c, NLgeom=True, silent=True)).toarray()
    dc = 1e-7
    J = np.column_stack([(fint(c + dc*e) - fint(c - dc*e))/(2*dc)
                         for e in np.eye(size)])
    assert np.abs(kT - J).max() < 1e-7*np.abs(kT).max()


@pytest.mark.parametrize('model', MODELS)
def test_offset_small(model):
    r"""Laminate offset `d = h/2`, `m = n = 4`: for `r = 10^4 a` the
    frequencies of the annular model are those of the flat plate with the
    same offset, within 3e-4; the offset raises them by more than 30 %"""
    got = _omegas(_square(model, 1e4, 0.005, m=4))
    ref = _omegas(_square(PLATE[model], None, 0.005, m=4))
    assert np.allclose(got, ref, rtol=3e-4)
    assert got[0] > 1.3*_omegas(_square(model, 1e4, 0., m=4))[0]
