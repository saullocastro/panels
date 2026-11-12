r"""Annular-sector plates with the total-Lagrangian third-order shear
deformation theory, ``'annular_tsdt_tl'``

The linear kernels against those of ``'annular_tsdt_donnell'`` and
``'annular_tsdt_gl'``; thin annuli against the classical-plate results of
the literature (free vibration of an annulus clamped outside and free
inside against the exact roots of Leissa (1969), buckling under a uniform
compression against Yamaki (1958), Table 1, buckling under a torque against
Irie, Yamada and Tsujino (1983), Table 1); a thick antisymmetric cross-ply
sector of large radius against the TSDT frequencies of Reddy (1984); finite
rigid rotations; a thin cantilever strip of large radius in large bending
against the elastica; the tangent stiffness against the derivative of the
internal force vector.

Notes
-----
The thin and thick cases, the annuli and their references are those of the
tests of ``'annular_tsdt_donnell'`` and ``'annular_tsdt_gl'``, whose
helpers are imported: at zero displacements the kernels of the three models
coincide. Yamaki [2]_ gives `k = a \sqrt{p/D}` for the compression `p` on
both edges, Irie et al. [3]_ `\tau_B = T_B/(2 \pi D)` for the torque `T_B`
on the outer edge; the frequency equation of the clamped-free annulus is
that of Leissa [1]_, Eqs. (1.11)-(1.13) and (1.17). Reddy's TSDT [4]_
gives, for the simply supported square `[0/90]` plate, `a/h = 5`, `E_1/E_2
= 10`, `G_{12} = G_{13} = 0.6 E_2`, `G_{23} = 0.5 E_2`, `\nu_{12} = 0.25`,
`\bar\omega = \omega a^2/h \sqrt{\rho/E_2} = 6.9887`, as reproduced by Adim
et al. [5]_, Table 3.

The strains of ``'annular_tsdt_tl'`` are the coefficients of the
Green-Lagrange strains of the field `\boldsymbol{X} + z \boldsymbol{d} +
p(z) \boldsymbol{q}`, with the second-order director `\boldsymbol{d}`:
under a finite rigid rotation `t` only the error of the director remains,
the transverse shear strains being of `O(t^5)`, while the third-order and
transverse shear strains of ``'annular_tsdt_gl'``, linear, are of
`O(t^2)`.

The reference of the cantilever strip is the inextensible elastica under
a tip force `P` of fixed direction, normal to the undeformed strip,
`\theta'' = -\alpha \cos\theta` in `s/L`, `\alpha = P L^2/(EI)`,
`\theta(0) = 0`, `\theta'(1) = 0`, solved by shooting in :func:`elastica`:
for `\alpha = 1` the tip rotation is 0.461 rad, the tip deflection `w =
0.3017208 L` and the axial shortening `-u = 0.0564332 L`. The strip of
`m = 14` radial terms gives `w` 0.21 % and `-u` 0.47 % below these values;
with `m = 10` terms and the whole load in one increment the same, to the
digits shown (hpc12), which the tolerances of 0.3 % and 0.6 % bound.

The thin annuli, the torque, the compression and the cross-ply plate of
``'annular_tsdt_tl'`` with few terms are checked by the tests
``test_*_small`` of the module of ``'annular_tsdt_donnell'`` and
``'annular_tsdt_gl'``, whose model list includes ``'annular_tsdt_tl'``. The
constitutive stiffness matrices of ``'annular_tsdt_tl'`` and
``'annular_tsdt_gl'`` of the quarter annulus differ by `2.7 \times
10^{-10}` of their largest term with `m = n = 3` and `4` and by `9.5
\times 10^{-13}` with `m = n = 5` (hpc12): the tolerance of the small test
is `10^{-9}`, that of the full-size one `10^{-12}`.

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, 1969, Eqs.
   (1.11)-(1.13) and (1.17), p. 2.
.. [2] Yamaki, N., "Buckling of a thin annular plate under uniform
   compression", J. Appl. Mech., 25, 267-273, 1958, Table 1.
   https://doi.org/10.1115/1.4011755
.. [3] Irie, T., Yamada, G. and Tsujino, M., "Buckling loads of annular
   plates subjected to a torque", J. Sound Vib., 86(1), 145-146, 1983,
   Table 1. https://doi.org/10.1016/0022-460X(83)90951-3
.. [4] Reddy, J. N., "A simple higher-order theory for laminated composite
   plates", J. Appl. Mech., 51, 745-752, 1984.
   https://doi.org/10.1115/1.3167719
.. [5] Adim, B., Daouadji, T. H. and Rabahi, A., Int. J. Adv. Struct.
   Eng., 8, 103-117, 2016, Table 3.

"""
import os
import sys

import numpy as np
import pytest

from panels import modelDB
from panels.shell import Shell

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_annular_tsdt import (A, D, E, H, NU, RHO, REDDY_TSDT, YAMAKI,
                               annulus, exact_lambda2, set_edge)
from test_annular_fsdt import _fit, _rotated_fields
from test_fsdt_tsdt_gl import set_flags

MODEL = 'annular_tsdt_tl'
GL = 'annular_tsdt_gl'
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')


def _sector(model, m=6, n=6, free=False):
    """Quarter annulus `r = 1`, `a = 0.5`, unsymmetric laminate, `h =
    0.06`, with `u = v = w = 0` at the outer edge unless ``free``"""
    s = Shell(model=model, a=0.5*A, b=np.pi/2*A, r=A, m=m, n=n, plyt=0.02,
              stack=[0, 45, -45], laminaprop=(E, NU), rho=RHO, offset=0.005)
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in FIELDS:
            setattr(s, e + f, 1)
            setattr(s, e + f + 'r', 1)
    if not free:
        s.x1u = s.x1v = s.x1w = 0
    return s


@pytest.mark.slow
def test_linear_kernels():
    r"""At zero displacements: ``calc_kC`` (``NLgeom`` 0 and 1) equal to
    that of ``'annular_tsdt_donnell'`` and ``'annular_tsdt_gl'``,
    ``calc_kG`` of a membrane state and ``calc_kM`` equal to those of
    ``'annular_tsdt_gl'``, to rounding"""
    s, gl, dn = (_sector(mo) for mo in (MODEL, GL, 'annular_tsdt_donnell'))
    c0 = np.zeros(s.get_size())
    K0 = s.calc_kC(silent=True)
    tol = 1e-12*abs(K0).max()
    assert abs(K0 - gl.calc_kC(silent=True)).max() < tol
    assert abs(K0 - dn.calc_kC(silent=True)).max() < tol
    assert abs(s.calc_kC(c=c0, NLgeom=True, silent=True) - K0).max() < tol
    for p in (s, gl):
        p.Nxx, p.Nyy, p.Nxy = -1., 0.5, 0.3
    KG = s.calc_kG(silent=True)
    assert abs(KG - gl.calc_kG(silent=True)).max() < 1e-12*abs(KG).max()
    KM = s.calc_kM(silent=True)
    assert abs(KM - gl.calc_kM(silent=True)).max() < 1e-12*abs(KM).max()


def test_internal_force_linear_limit():
    r"""``calc_fint`` at `\epsilon \boldsymbol{c}` against `\boldsymbol{K}_0
    \epsilon \boldsymbol{c}`: the difference is of `O(\epsilon^2)` (ratio
    between 3.8 and 4.2 when `\epsilon` halves) and below `10^{-3}` of the
    linear force at the smaller `\epsilon`"""
    s = _sector(MODEL)
    size = s.get_size()
    c0 = np.random.default_rng(0).standard_normal(size)*1e-3
    K0 = s.calc_kC(silent=True)
    fw = modelDB.db[MODEL]['matrices_num']
    dl, lin = [], []
    for eps in (1e-2, 5e-3):
        f = fw.calc_fint(eps*c0, s.ABD, s, size, 0, s.nx, s.ny)
        lin.append(K0 @ (eps*c0))
        dl.append(np.abs(f - lin[-1]).max())
    assert 3.8 < dl[0]/dl[1] < 4.2, dl
    assert dl[1] < 1e-3*np.abs(lin[1]).max()


@pytest.mark.slow
def test_leissa_clamped_free_annulus():
    r"""Thin annulus clamped outside and free inside, `\beta = 0.4`, `\nu =
    1/3`: `\lambda^2 = \omega a^2 \sqrt{\rho h/D}` for `n = 0, 1, 1, 2, 2`
    against the exact roots of Leissa (1969), within 0.1 %"""
    from structsolve import freq
    beta, nu = 0.4, 1/3
    md = annulus(MODEL, beta, 'C', 'F', nu=nu)
    eigvals, _ = freq(md.reduce(md.calc_kC(silent=True)),
                      md.reduce(md.calc_kM(silent=True)),
                      sparse_solver=False, silent=True, num_eigvalues=8)
    Dn = E*H**3/(12*(1 - nu**2))
    l2 = np.sort(np.sqrt(-eigvals.real))*A**2*np.sqrt(RHO*H/Dn)
    guess = (13.50, 19.39, 31.34)
    ref = np.array([exact_lambda2(n, beta, guess[n]) for n in (0, 1, 1, 2, 2)])
    assert np.allclose(l2[:5], ref, rtol=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize('case', list(YAMAKI))
def test_yamaki1958_compression(case):
    r"""Thin annulus under `N_{xx} = N_{yy} = -1`: `k = a \sqrt{p/D}`
    against Yamaki (1958), Table 1, within 0.6 %, the rounding of the
    table"""
    from structsolve import lb
    outer, inner, beta = case
    md = annulus(MODEL, beta, outer, inner)
    for s in md.panels:
        s.Nxx = -1.
        s.Nyy = -1.
    eigvals, _ = lb(md.reduce(md.calc_kC(silent=True)),
                    md.reduce(md.calc_kG(silent=True)), silent=True,
                    num_eigvalues=4)
    assert np.isclose(np.sqrt(eigvals[0]*A**2/D), YAMAKI[case], rtol=6e-3)


@pytest.mark.slow
def test_irie1983_torsion():
    r"""Thin annulus clamped on both edges, `\beta = 0.3`, under a torque on
    the outer edge, the pre-buckling state from a linear static case:
    `\tau_B = T_B/(2 \pi D)` against Irie et al. (1983), Table 1, 61.97,
    within 1 %"""
    md = annulus(MODEL, 0.3, 'C', 'C', n=16)
    T = 2*np.pi*D
    for s in md.panels:
        s.x1u = s.x1v = 1
        s.add_distr_load_fixed_x(0., funcy=lambda y: T/(2*np.pi*A**2) + 0*y)
    eigvals, _ = md.linear_buckling(num_eigvalues=6, silent=True)
    assert np.isclose(np.abs(eigvals).min(), 61.97, rtol=1e-2)


@pytest.mark.slow
def test_thick_cross_ply_plate_limit():
    r"""Simply supported square `[0/90]` sector, `a/h = 5`, `E_1/E_2 = 10`,
    `r = 10^4 a`: `\bar\omega = \omega a^2/h \sqrt{\rho/E_2}` against
    Reddy's TSDT (1984), 6.9887, within 0.1 %"""
    from structsolve import freq
    E2, rho, a = 1e9, 1500., 1.
    h = a/5
    s = Shell(model=MODEL, a=a, b=a, r=1e4*a, m=12, n=12, plyt=h/2,
              stack=[0, 90], rho=rho,
              laminaprop=(10*E2, E2, 0.25, 0.6*E2, 0.6*E2, 0.5*E2))
    for e in ('x1', 'x2', 'y1', 'y2'):
        set_edge(s, e, 'S')
    eigvals, _ = freq(s.calc_kC(silent=True), s.calc_kM(silent=True),
                      sparse_solver=False, silent=True, num_eigvalues=4)
    omega = np.sort(np.sqrt(-eigvals.real))[0]
    assert np.isclose(omega*a**2/h*np.sqrt(rho/E2), REDDY_TSDT[10], rtol=1e-3)


def rotation_strains(model, t, axis):
    r"""Largest absolute strains of the groups membrane, `\kappa`,
    `\kappa^{(3)}`, `\gamma` and `\gamma^{(2)}` at 25 inner points of a
    free quarter annulus whose Ritz constants fit, by least squares, the
    rigid rotation `t` about the unit ``axis``, with the rotations of the
    rotated normal"""
    s = _sector(model, m=4, n=14, free=True)
    size = s.get_size()
    xg, yg = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 30))
    xs, ys = xg.ravel()*s.a, yg.ravel()*s.b
    c = _fit(s, size, xs, ys, _rotated_fields(t, axis, xs, ys))
    xe, ye = np.meshgrid(np.linspace(0.1, 0.9, 5)*s.a,
                         np.linspace(0.1, 0.9, 5)*s.b)
    st = s.strain(c, xs=xe.ravel(), ys=ye.ravel(), NLgeom=True)[1]
    groups = (('exx', 'eyy', 'gxy'), ('kxx', 'kyy', 'kxy'),
              ('kxx3', 'kyy3', 'kxy3'), ('gyz', 'gxz'), ('gyz2', 'gxz2'))
    return np.array([max(np.abs(st[nm]).max() for nm in g) for g in groups])


#: rotation axes: in the plane of the annulus, and at 45 degrees to it
AXES = {'in-plane': np.array([np.cos(0.4), np.sin(0.4), 0.]),
        'inclined': np.array([np.cos(0.4), np.sin(0.4), 1.])/np.sqrt(2)}


@pytest.mark.parametrize('axis', list(AXES))
def test_rigid_rotation(axis):
    r"""Rigid rotations `t = 0.2` and `0.1` of a free quarter annulus,
    `h = 0.06`: ``'annular_tsdt_tl'`` has membrane, bending and third-order
    strains below `10^{-9}`, the fitting error, and transverse shear strains
    of `O(t^5)` (fitted order between 4.5 and 5.5; hpc12, in-plane axis:
    `\gamma` 3.94e-5, 1.24e-6, inclined axis: 6.92e-6, 2.19e-7). For the axis
    in the plane the linear third-order and transverse shear strains of
    ``'annular_tsdt_gl'`` vanish (below `10^{-10}`); for the inclined axis
    they are of `O(t^2)` (order between 1.8 and 2.2), more than 1000 times
    those of ``'annular_tsdt_tl'`` at `t = 0.1`"""
    tl = [rotation_strains(MODEL, t, AXES[axis]) for t in (0.2, 0.1)]
    gl = [rotation_strains(GL, t, AXES[axis]) for t in (0.2, 0.1)]
    order = lambda v0, v1: np.log2(v0/v1)
    assert max(tl[0][:3].max(), tl[1][:3].max()) < 1e-9
    for k in (3, 4):
        assert 4.5 < order(tl[0][k], tl[1][k]) < 5.5, tl
    big_gl = [g[2:].max() for g in gl]
    if axis == 'in-plane':
        assert max(big_gl) < 1e-10, gl
    else:
        assert 1.8 < order(*big_gl) < 2.2, gl
        assert tl[1].max() < 1e-3*big_gl[1]


def strip_tip(model, alpha=1., inc=0.1, m=14):
    r"""Tip ``(w, -u)/L`` of a strip of an annulus of outer radius `r = 10^4
    L`, ``L = 1`` radially, ``b = 0.04``, ``h = 2e-3``, ``nu = 0``, clamped
    at the outer edge (``w,x = 0`` too), under a tip force along `z` of
    ``alpha = P L^2/EI``, mean over the tip, ``m`` terms radially, load
    increment ``inc``"""
    from structsolve import Analysis
    L, b, h, Ev = 1., 0.04, 2e-3, 70e9
    P = alpha*Ev*h**3*b/12/L**2
    s = Shell(model=model, a=L, b=b, r=1e4*L, stack=[0], plyt=h,
              laminaprop=(Ev, 0.), m=m, n=4)
    set_flags(s, ('x2', 'y1', 'y2'), FIELDS, **{'': 1., 'r': 1.})
    set_flags(s, ('x1', ), FIELDS, **{'': 0., 'r': 1.})
    s.x1wr = 0.
    s._rebuild()
    for yv in np.linspace(0, b, 5):
        s.add_point_load(L, yv, 0., 0., P/5, cte=False)
    an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    an.initialInc = an.maxInc = inc
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    f = s.uvw(an.cs[-1], xs=np.full(5, L), ys=np.linspace(0, b, 5))[1]
    return f['w'].mean()/L, -f['u'].mean()/L


def elastica(alpha):
    r"""Tip ``(w, -u)/L`` and rotation of the inextensible elastica of a
    cantilever, `\theta'' = -\alpha \cos\theta` in `s/L`, `\theta(0) = 0`,
    `\theta'(1) = 0`, by shooting on the tip rotation"""
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq

    def shoot(th_tip):
        return solve_ivp(lambda s, y: [y[1], -alpha*np.cos(y[0])], [1., 0.],
                         [th_tip, 0.], rtol=1e-11, atol=1e-13,
                         dense_output=True)
    th = brentq(lambda t: shoot(t).y[0, -1], 1e-9, 1.5)
    s = np.linspace(0, 1, 4001)
    t = shoot(th).sol(s)[0]
    return np.trapezoid(np.sin(t), s), np.trapezoid(1 - np.cos(t), s), th


def test_elastica():
    """The shooting solution for ``alpha = 1``: ``w = 0.3017208 L``, ``-u =
    0.0564332 L``, tip rotation 0.461 rad"""
    w, u, th = elastica(1.)
    assert np.allclose((w, u), (0.3017208, 0.0564332), rtol=1e-6)
    assert np.isclose(th, 0.461, atol=1e-3)


@pytest.mark.slow
def test_thin_cantilever_strip_against_elastica():
    r"""Strip of a large annulus, `L/h = 500`, `\alpha = 1`, tip rotation
    0.46 rad: ``'annular_tsdt_tl'`` within 0.3 % and 0.6 % of the elastica
    (hpc12: `w` -0.21 %, `-u` -0.47 %), ``'annular_tsdt_gl'`` more than 8 %
    and 20 % above it (+10.5 %, +24.5 %); both against the stored hpc12
    values, rtol `10^{-5}`"""
    ELASTICA = elastica(1.)[:2]
    w, u = strip_tip(MODEL)
    wg, ug = strip_tip(GL)
    assert np.isclose(w, ELASTICA[0], rtol=3e-3)
    assert np.isclose(u, ELASTICA[1], rtol=6e-3)
    assert wg > 1.08*ELASTICA[0] and ug > 1.2*ELASTICA[1]
    assert np.allclose((w, u), (0.3010774, 0.0561698), rtol=1e-5)
    assert np.allclose((wg, ug), (0.3333425, 0.0702331), rtol=1e-5)



def test_thin_cantilever_strip_small():
    r"""The strip of :func:`test_thin_cantilever_strip_against_elastica`
    with `m = 10` radial terms and the whole load in one increment:
    ``'annular_tsdt_tl'`` within 0.3 % and 0.6 % of the elastica"""
    ELASTICA = elastica(1.)[:2]
    w, u = strip_tip(MODEL, inc=1., m=10)
    assert np.isclose(w, ELASTICA[0], rtol=3e-3)
    assert np.isclose(u, ELASTICA[1], rtol=6e-3)


def test_tangent_is_derivative_of_fint():
    r"""``calc_kC(NLgeom=True) + calc_kG`` against central differences of
    ``calc_fint`` at a state of rotations of some tenths of a radian, and
    ``calc_kTv`` against the product of that matrix"""
    fw = modelDB.db[MODEL]['matrices_num']
    s = _sector(MODEL, m=5, n=5)
    size = s.get_size()
    rng = np.random.default_rng(1)
    c = 0.02*rng.standard_normal(size)
    fint = lambda cc: fw.calc_fint(cc, s.ABD, s, size, 0, s.nx, s.ny)
    kT = (s.calc_kC(c=c, NLgeom=True, silent=True)
          + s.calc_kG(c=c, NLgeom=True, silent=True)).toarray()
    dc = 1e-7
    for k in rng.choice(size, 20, replace=False):
        e = np.zeros(size)
        e[k] = dc
        fd = (fint(c + e) - fint(c - e))/(2*dc)
        assert np.allclose(fd, kT[:, k], rtol=1e-5, atol=1e-7*np.abs(kT).max())
    v = rng.standard_normal(size)
    kTv = fw.calc_kTv(c, v, s.ABD, s, size, 0, s.nx, s.ny)
    assert np.allclose(kTv, kT @ v, rtol=1e-10, atol=1e-10*np.abs(kT @ v).max())


def test_r_not_larger_than_a_raises():
    """An inner radius `r - a \\le 0` is rejected"""
    s = Shell(model=MODEL, a=1., b=1., r=1., m=4, n=4,
              plyt=H, stack=[0], laminaprop=(E, NU))
    with pytest.raises(ValueError):
        s.calc_kC(silent=True)


def test_linear_kernels_small():
    r""":func:`test_linear_kernels` with `m = n = 4`, the matrices equal
    within `10^{-9}` of their largest term"""
    s, gl, dn = (_sector(mo, m=4, n=4)
                 for mo in (MODEL, GL, 'annular_tsdt_donnell'))
    c0 = np.zeros(s.get_size())
    K0 = s.calc_kC(silent=True)
    tol = 1e-9*abs(K0).max()
    assert abs(K0 - gl.calc_kC(silent=True)).max() < tol
    assert abs(K0 - dn.calc_kC(silent=True)).max() < tol
    assert abs(s.calc_kC(c=c0, NLgeom=True, silent=True) - K0).max() < tol
    for p in (s, gl):
        p.Nxx, p.Nyy, p.Nxy = -1., 0.5, 0.3
    KG = s.calc_kG(silent=True)
    assert abs(KG - gl.calc_kG(silent=True)).max() < 1e-9*abs(KG).max()
    KM = s.calc_kM(silent=True)
    assert abs(KM - gl.calc_kM(silent=True)).max() < 1e-9*abs(KM).max()
