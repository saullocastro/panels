r"""Annular-sector plates with the third-order shear deformation theory,
``'annular_tsdt_donnell'`` and ``'annular_tsdt_gl'``

Thin annuli against the classical-plate results of the literature: free
vibration of an annulus clamped outside and free inside against the exact
roots of Leissa (1969); buckling under a uniform compression on both edges
against Yamaki (1958), Table 1; buckling under a torque against Irie,
Yamada and Tsujino (1983), Table 1. A thick antisymmetric cross-ply sector
of large radius against the TSDT frequencies of Reddy (1984) for the
simply supported square plate. The laminate offset against the flat
plate, the Green-Lagrange model against the von Kármán one in the linear
limit, the tangent stiffness against the derivative of the internal force
vector, and the check of the radii.

Notes
-----
The thin annuli have `h/a \approx 10^{-3}`: the TSDT tends to the
classical plate theory, whose solutions the references give, as `h \to
0`. A clamped edge has `u = v = w = \phi_x = \phi_y = 0` and a zero normal
slope of `w`, such that the whole section is fixed, as for the flat
plates of panels; a simply supported edge has `w = 0` and a zero rotation
`\phi` about the normal of the edge.

Yamaki [2]_ gives `k = a \sqrt{p/D}` for the compression `p` on both edges,
`a` the outer radius; Irie et al. [3]_ give `\tau_B = T_B/(2 \pi D)` for
the torque `T_B` on the outer edge, the inner edge fixed in its plane,
with the pre-buckling state `N_{xy} = -T/(2 \pi \rho^2)` of a linear
static case. The frequency equation of the annulus clamped outside and free
inside is that of Leissa [1]_, Eqs. (1.11)-(1.13) and (1.17), see
:func:`exact_lambda2`.

For `r \to \infty` the strains are those of ``'plate_tsdt_donnell'`` and
``'plate_tsdt_gl'``, the difference being of the order of `a/r`. Reddy's
TSDT [4]_ gives, for the simply supported (SS-1) square `[0/90]` plate,
`a/h = 5`, `E_1/E_2 = 10` and `40`, `G_{12} = G_{13} = 0.6 E_2`, `G_{23}
= 0.5 E_2`, `\nu_{12} = 0.25`, the frequencies `\bar\omega = \omega a^2/h
\sqrt{\rho/E_2} = 6.9887` and `9.0871`, as reproduced by Adim et al. [5]_,
Table 3, against the three-dimensional values 6.9845 and 8.5625 of Noor
[6]_.

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, 1969, Eqs.
   (1.11)-(1.13) and (1.17), p. 2.
.. [2] Yamaki, N., "Buckling of a thin annular plate under uniform
   compression", J. Appl. Mech., 25, 267-273, 1958, Table 1.
   https://doi.org/10.1115/1.4011755
.. [3] Irie, T., Yamada, G. and Tsujino, Y., "Buckling loads of annular
   plates subjected to torque", J. Sound Vib., 86(1), 145-146, 1983,
   Table 1. https://doi.org/10.1016/0022-460X(83)90951-3
.. [4] Reddy, J. N., "A simple higher-order theory for laminated composite
   plates", J. Appl. Mech., 51, 745-752, 1984.
   https://doi.org/10.1115/1.3167719
.. [5] Adim, B., Daouadji, T. H. and Rabahi, A., Int. J. Adv. Struct.
   Eng., 8, 103-117, 2016, Table 3.
.. [6] Noor, A. K., "Free vibrations of multilayered composite plates",
   AIAA J., 11(7), 1038-1039, 1973.

"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import iv, ivp, jv, jvp, kv, kvp, yv, yvp
from structsolve import freq, lb

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels import modelDB


E, NU, H, RHO = 70e9, 0.3, 1e-3, 2700.
A = 1.
D = E*H**3/(12*(1 - NU**2))
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
#: the annular TSDT models
MODELS = [k for k in modelDB.db if k.startswith('annular_tsdt_')]
PLATE = {k: k.replace('annular_', 'plate_') for k in MODELS}


def set_edge(s, edge, kind):
    """Edge ``'C'`` (clamped), ``'S'`` (simply supported, `w` and the
    rotation along the edge zero, the tangential displacement zero) or
    ``'F'`` (free)"""
    for f in FIELDS:
        setattr(s, edge + f, 1)
        setattr(s, edge + f + 'r', 1)
    if kind == 'C':
        for f in FIELDS:
            setattr(s, edge + f, 0)
        setattr(s, edge + 'wr', 0)
    elif kind == 'S':
        tangential, rotation = (('v', 'phiy') if edge[0] == 'x'
                                else ('u', 'phix'))
        for f in ('w', tangential, rotation):
            setattr(s, edge + f, 0)
    elif kind != 'F':
        raise ValueError(kind)


def annulus(model, beta, outer, inner, nu=NU, m=10, n=12, **kwargs):
    """Full annulus of outer radius ``A``, two half annuli connected with
    ``'SSycte'``"""
    halves = []
    for _ in range(2):
        s = Shell(model=model, a=A*(1 - beta), b=np.pi*A, r=A, m=m, n=n,
                  plyt=H, stack=[0], laminaprop=(E, nu), rho=RHO, **kwargs)
        for e in ('y1', 'y2'):
            set_edge(s, e, 'F')
        set_edge(s, 'x1', outer)
        set_edge(s, 'x2', inner)
        halves.append(s)
    p1, p2 = halves
    conn = [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0),
            dict(p1=p1, p2=p2, func='SSycte', ycte1=0, ycte2=p2.b)]
    return MultiDomain([p1, p2], conn)


# Leissa (1969): annulus clamped outside, free inside, nu = 1/3


def _det(lam2, n, beta, nu=1/3):
    k = np.sqrt(lam2)
    cols = []
    for f, fp in ((jv, jvp), (yv, yvp), (iv, ivp), (kv, kvp)):
        W0, W1 = f(n, k), k*fp(n, k, 1)
        W, W1i, W2, W3 = (f(n, k*beta), k*fp(n, k*beta, 1),
                          k**2*fp(n, k*beta, 2), k**3*fp(n, k*beta, 3))
        b = beta
        M = W2 + nu*(W1i/b - n**2*W/b**2)
        V = (W3 + W2/b - W1i/b**2 - n**2*W1i/b**2 + 2*n**2*W/b**3
             - (1 - nu)*n**2/b**2*(W1i - W/b))
        col = np.array([W0, W1, M, V])
        cols.append(col/np.abs(col).max())
    return np.linalg.det(np.array(cols).T)


def exact_lambda2(n, beta, guess):
    """Root of the frequency equation of the clamped-free annulus near
    ``guess``"""
    return brentq(_det, 0.9*guess, 1.1*guess, args=(n, beta))


@pytest.mark.parametrize('model', MODELS)
def test_leissa_clamped_free_annulus(model):
    r"""Thin annulus clamped outside and free inside, `\beta = 0.4`, `\nu =
    1/3`: `\lambda^2 = \omega a^2 \sqrt{\rho h/D}` for `n = 0, 1, 1, 2, 2`
    against the exact roots of Leissa (1969), within 0.1 %"""
    beta, nu = 0.4, 1/3
    md = annulus(model, beta, 'C', 'F', nu=nu)
    eigvals, _ = freq(md.reduce(md.calc_kC(silent=True)),
                      md.reduce(md.calc_kM(silent=True)),
                      sparse_solver=False, silent=True, num_eigvalues=8)
    Dn = E*H**3/(12*(1 - nu**2))
    l2 = np.sort(np.sqrt(-eigvals.real))*A**2*np.sqrt(RHO*H/Dn)
    guess = (13.50, 19.39, 31.34)
    ref = np.array([exact_lambda2(n, beta, guess[n]) for n in (0, 1, 1, 2, 2)])
    assert np.allclose(l2[:5], ref, rtol=1e-3)


#: Yamaki (1958), Table 1, least k, (outer, inner, beta): k, nu = 0.3
YAMAKI = {('C', 'C', 0.5): 12.15, ('S', 'S', 0.3): 4.75}


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('case', list(YAMAKI))
def test_yamaki1958_compression(case, model):
    r"""Thin annulus under `N_{xx} = N_{yy} = -1`: `k = a \sqrt{p/D}`
    against Yamaki (1958), Table 1, within 0.6 %, the rounding of the
    table"""
    outer, inner, beta = case
    md = annulus(model, beta, outer, inner)
    for s in md.panels:
        s.Nxx = -1.
        s.Nyy = -1.
    eigvals, _ = lb(md.reduce(md.calc_kC(silent=True)),
                    md.reduce(md.calc_kG(silent=True)), silent=True,
                    num_eigvalues=4)
    assert np.isclose(np.sqrt(eigvals[0]*A**2/D), YAMAKI[case], rtol=6e-3)


@pytest.mark.parametrize('model', MODELS)
def test_irie1983_torsion(model):
    r"""Thin annulus clamped on both edges, `\beta = 0.3`, under a torque on
    the outer edge, the pre-buckling state from a linear static case:
    `\tau_B = T_B/(2 \pi D)` against Irie et al. (1983), Table 1, 61.97,
    within 1 %"""
    md = annulus(model, 0.3, 'C', 'C', n=16)
    T = 2*np.pi*D
    for s in md.panels:
        # the outer edge free to rotate in its plane, loaded tangentially
        s.x1u = s.x1v = 1
        s.add_distr_load_fixed_x(0., funcy=lambda y: T/(2*np.pi*A**2) + 0*y)
    eigvals, _ = md.linear_buckling(num_eigvalues=6, silent=True)
    assert np.isclose(np.abs(eigvals).min(), 61.97, rtol=1e-2)


#: Reddy (1984) TSDT, [0/90], a/h = 5, by E1/E2, as reproduced by Adim et al.
REDDY_TSDT = {10: 6.9887, 40: 9.0871}


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('E1E2', list(REDDY_TSDT))
def test_thick_cross_ply_plate_limit(E1E2, model):
    r"""Simply supported square `[0/90]` sector, `a/h = 5`, `r = 10^4 a`:
    `\bar\omega = \omega a^2/h \sqrt{\rho/E_2}` against Reddy's TSDT
    (1984), within 0.1 %"""
    E2, rho, a = 1e9, 1500., 1.
    h = a/5
    s = Shell(model=model, a=a, b=a, r=1e4*a, m=12, n=12, plyt=h/2,
              stack=[0, 90], rho=rho,
              laminaprop=(E1E2*E2, E2, 0.25, 0.6*E2, 0.6*E2, 0.5*E2))
    for e in ('x1', 'x2', 'y1', 'y2'):
        set_edge(s, e, 'S')
    eigvals, _ = freq(s.calc_kC(silent=True), s.calc_kM(silent=True),
                      sparse_solver=False, silent=True, num_eigvalues=4)
    omega = np.sort(np.sqrt(-eigvals.real))[0]
    assert np.isclose(omega*a**2/h*np.sqrt(rho/E2), REDDY_TSDT[E1E2],
                      rtol=1e-3)


def _square(model, r, offset):
    r"""Square panel, `a = b = 1`, `h = 0.05`, with `u = v = w = 0` on the
    four edges, of radius ``r`` for the annular models"""
    kw = dict(r=r) if model.startswith('annular') else {}
    s = Shell(model=model, a=1., b=1., m=8, n=8, plyt=0.05, stack=[0],
              laminaprop=(E, NU), rho=RHO, offset=offset, **kw)
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, e + f, 0)
    return s


def _omegas(s, num=4):
    eigvals, _ = freq(s.calc_kC(silent=True), s.calc_kM(silent=True),
                      sparse_solver=False, silent=True, num_eigvalues=num)
    return np.sort(np.sqrt(-eigvals.real))[:num]


@pytest.mark.parametrize('model', MODELS)
def test_offset_plate_limit(model):
    r"""Laminate offset `d = h/2`, stiffness and mass: for `r = 10^4 a` the
    frequencies of the annular model are those of the plate model with the
    same offset; the offset raises them by more than 30 %"""
    got = _omegas(_square(model, 1e4, 0.025))
    ref = _omegas(_square(PLATE[model], None, 0.025))
    assert np.allclose(got, ref, rtol=3e-4)
    assert got[0] > 1.3*_omegas(_square(model, 1e4, 0.))[0]


def test_gl_equals_donnell_in_the_linear_limit():
    """Same linear frequencies; same internal force vector for a
    configuration without in-plane displacements, for which the
    Green-Lagrange and the von Kármán membrane strains coincide"""
    s1 = annulus('annular_tsdt_donnell', 0.4, 'C', 'F').panels[0]
    s2 = annulus('annular_tsdt_gl', 0.4, 'C', 'F').panels[0]
    assert np.allclose(_omegas(s1, 6), _omegas(s2, 6), rtol=1e-9)
    size = s1.get_size()
    c = np.random.default_rng(0).standard_normal(size)*1e-2
    c[0::5] = 0
    c[1::5] = 0
    f = [modelDB.db[s.model]['matrices_num'].calc_fint(
        c, s.ABD, s, size, 0, s.nx, s.ny) for s in (s1, s2)]
    assert np.allclose(f[0], f[1], rtol=1e-12, atol=1e-12*np.abs(f[0]).max())
    c = np.random.default_rng(1).standard_normal(size)*1e-2
    f = [modelDB.db[s.model]['matrices_num'].calc_fint(
        c, s.ABD, s, size, 0, s.nx, s.ny) for s in (s1, s2)]
    assert not np.allclose(f[0], f[1], rtol=1e-6)


@pytest.mark.parametrize('model', MODELS)
def test_tangent_is_derivative_of_fint(model):
    """``fkC_num(NLgeom=1) + fkG_num`` against central differences of
    ``calc_fint``, and ``calc_kTv`` against the product of that matrix"""
    fw = modelDB.db[model]['matrices_num']
    s = Shell(model=model, a=0.5, b=np.pi/2, r=A, m=5, n=5, plyt=0.02,
              stack=[0, 45], laminaprop=(E, NU), rho=RHO, offset=0.005)
    size = s.get_size()
    c = np.random.default_rng(1).standard_normal(size)*1e-3
    fint = lambda cc: fw.calc_fint(cc, s.ABD, s, size, 0, s.nx, s.ny)
    kT = (s.calc_kC(c=c, NLgeom=True, silent=True)
          + s.calc_kG(c=c, NLgeom=True, silent=True)).toarray()
    dc = 1e-7
    J = np.column_stack([(fint(c + dc*e) - fint(c - dc*e))/(2*dc)
                         for e in np.eye(size)])
    assert np.abs(kT - J).max() < 1e-7*np.abs(kT).max()
    v = np.random.default_rng(2).standard_normal(size)
    kTv = fw.calc_kTv(c, v, s.ABD, s, size, 0, s.nx, s.ny)
    assert np.allclose(kTv, kT @ v, rtol=1e-10, atol=1e-10*np.abs(kT @ v).max())


@pytest.mark.parametrize('model', MODELS)
def test_r_not_larger_than_a_raises(model):
    """An inner radius `r - a \\le 0` is rejected"""
    s = Shell(model=model, a=1., b=1., r=1., m=4, n=4,
              plyt=H, stack=[0], laminaprop=(E, NU))
    with pytest.raises(ValueError):
        s.calc_kC(silent=True)
