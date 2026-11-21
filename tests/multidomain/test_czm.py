r"""Mixed-mode multilinear cohesive interface ``'CZM'``

Verification of :mod:`panels.multidomain.connections.czm`:

- the point law (:func:`.evaluate_laws`) dissipates, along a path of
  constant mode ratio up to complete failure, the sum of the
  Benzeggagh-Kenane toughnesses of the superposed laws, to round-off;
- along paths that change the mode ratio, including unloading and closure,
  no damage variable decreases, and the dissipated energy at failure lies
  between the sums of the pure-mode toughnesses of the laws;
- the consistent tangent equals central finite differences of the tractions;
- the separation operators of ``'CZM'`` are those of ``'SB_TSL'`` for the
  classical laminated plate theory, and a pristine ``'CZM'`` interface
  between two FSDT plates tends to the perfect bond ``'SB'`` as the penalty
  stiffness grows.

The law values are synthetic, chosen for verification; set A has the
toughnesses and the penalty stiffness of AS4/PEEK of Turon et al. (2018),
Table 1 and p. 509.

"""
import numpy as np
import pytest

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections.czm import (BilinearLaw, law_with_ratio,
        evaluate_laws, CohesiveInterface)


def laws_A():
    return [BilinearLaw(GIc=0.969, GIIc=1.719, tauI=80., tauII=100.,
                        KI=1e5, eta=2.)]


def laws_B():
    kappa = (0.5/1.5)*(80./60.)**2
    with pytest.warns(UserWarning, match='exponent'):
        bridging = law_with_ratio(0.4, 0., 3., 1e3, 0.5, kappa)
    return [law_with_ratio(0.5, 1.5, 60., 1e5, 2., kappa),
            law_with_ratio(0.2, 0.4, 20., 1e4, 1.5, kappa),
            bridging]


def run(laws, path, mode_ratio='per-law'):
    """Work, damage history and final dissipation along a path"""
    r = np.zeros((len(laws), 1))
    W = 0.
    t_old = np.zeros(3)
    d_old = np.zeros(3)
    Ds = []
    for d in path:
        tau, _, r, D = evaluate_laws(laws, *[np.array([x]) for x in d], r,
                                     mode_ratio, tangent=False)
        t = tau[:, 0]
        W += (0.5*(t + t_old)*(np.asarray(d) - d_old)).sum()
        t_old, d_old = t, np.asarray(d, float)
        Ds.append(D[:, 0].copy())
    return W, np.array(Ds)


@pytest.mark.parametrize('laws', [laws_A, laws_B])
@pytest.mark.parametrize('B', [0., 0.2, 0.5, 0.8, 1.])
def test_energy_constant_mode_ratio(laws, B):
    laws = laws()
    kappa = laws[0].KII/laws[0].KI
    if B == 1.:
        v = np.array([1., 0., 0.])
    elif B == 0.:
        v = np.array([0., 0., 1.])
    else:
        v = np.array([np.sqrt(B/(kappa*(1 - B))), 0., 1.])
    v /= np.linalg.norm(v)
    # breakpoints of each law along the path, where the traction is not
    # smooth, so that the trapezoidal rule is exact
    lam1 = (kappa*v[0]**2 + v[2]**2)/np.sqrt(kappa**2*v[0]**2 + v[2]**2)
    tb = [0.]
    for law in laws:
        Go, Gc = law.Go(B), law.Gc(B)
        if Go > 0:
            KB = law.KI*(1 - B) + law.KII*B
            lo = np.sqrt(2*Go/KB)
            tb += [lo/lam1, lo*Gc/Go/lam1]
    t = np.unique(np.concatenate([np.linspace(0, 1.2*max(tb), 3001), tb]))
    W, D = run(laws, [ti*v for ti in t])
    expected = sum(law.Gc(B) for law in laws)
    assert np.all(D[-1] == 1.)
    assert abs(W - expected) < 1e-10*expected


def _paths(dmax, rng):
    s = np.linspace(0., 1., 1500)
    yield [np.array([0, 0, x*dmax/8]) for x in s] + \
          [np.array([x*3*dmax, 0, dmax/8]) for x in s]
    yield [np.array([x*dmax/8, 0, 0]) for x in s] + \
          [np.array([dmax/8, 0, x*3*dmax]) for x in s]
    ang = np.linspace(0, 6*np.pi, 8000)
    mag = np.linspace(0, 3*dmax, 8000)
    p = [m*np.array([np.cos(a), 0.3*np.sin(2*a), np.sin(a)])
         for a, m in zip(ang, mag)]
    yield [np.array([q[0], q[1], q[2] if q[2] > 0 else 1e-3*q[2]]) for q in p]


@pytest.mark.parametrize('laws', [laws_A, laws_B])
def test_no_healing_and_bounds(laws):
    laws = laws()
    rng = np.random.default_rng(1)
    dmax = max(2*max(l.GIc, l.GIIc)/l.tauI for l in laws)
    lo = sum(min(l.GIc, l.GIIc) for l in laws)
    hi = sum(max(l.GIc, l.GIIc) for l in laws)
    for path in _paths(dmax, rng):
        W, D = run(laws, path)
        assert np.all(np.diff(D, axis=0) >= -1e-14)
        assert np.all(D[-1] == 1.)
        # W is the dissipated energy, nothing stored at complete failure
        # apart from the contact, kept small by the path
        assert lo*(1 - 1e-3) <= W <= hi*(1 + 1e-3)


@pytest.mark.parametrize('laws', [laws_A, laws_B])
def test_tangent_finite_differences(laws):
    laws = laws()
    rng = np.random.default_rng(2)
    scale = max(l.tauI/l.KI for l in laws)*5
    n = 400
    d = rng.normal(size=(3, n))*scale
    d[2] = np.where(rng.uniform(size=n) < 0.8, abs(d[2]), d[2])
    r_old = rng.uniform(0, 0.6, (len(laws), n))
    _, T, r_new, _ = evaluate_laws(laws, *d, r_old)
    h = 1e-5*np.linalg.norm(d, axis=0)
    Tfd = np.zeros_like(T)
    ok = np.ones(n, bool)
    for j in range(3):
        e = np.zeros((3, n))
        e[j] = h
        tp, _, rp, _ = evaluate_laws(laws, *(d + e), r_old, tangent=False)
        tm, _, rm, _ = evaluate_laws(laws, *(d - e), r_old, tangent=False)
        ok &= np.all((rp > r_old) == (rm > r_old), axis=0)
        ok &= np.all((rp >= 1) == (rm >= 1), axis=0)
        ok &= np.sign(d[2] + h) == np.sign(d[2] - h)
        Tfd[:, j] = (tp - tm)/(2*h)
    scale_T = np.abs(Tfd).max(axis=(0, 1))
    ok &= scale_T > 0
    err = np.abs(T - Tfd).max(axis=(0, 1))[ok]/scale_T[ok]
    assert ok.sum() > n/2
    assert err.max() < 1e-7


def _plates(model, m=8, n=6, a=40., b=20.):
    kw = dict(a=a, b=b, m=m, n=n, plyt=0.14, stack=[0]*8,
              laminaprop=(133150., 10950., 0.316, 5190., 5190., 3500.),
              model=model)
    top = Shell(group='top', x0=0, y0=0, **kw)
    bot = Shell(group='bot', x0=0, y0=0, **kw)
    return top, bot


def test_operators_equal_sb_tsl_clpt():
    top, bot = _plates('plate_clpt_donnell')
    conn = [dict(p1=top, p2=bot, func='CZM', laws=laws_A(), nr_x_gauss=12,
                 nr_y_gauss=5)]
    md = MultiDomain([bot, top], conn, conn_method='penalty',
                     pd_method='penalty')
    iface = md.cohesive_interfaces()[0]
    Bu, Bv, Bw, w, dofs = md._tsl_operators(top, bot, 12, 5)
    for B_czm, B_tsl in ((iface.Bu, Bu), (iface.Bv, Bv), (iface.Bw, Bw)):
        np.testing.assert_allclose(B_czm[:, dofs].toarray(),
                                   np.asarray(B_tsl.todense() if hasattr(
                                       B_tsl, 'todense') else B_tsl),
                                   atol=1e-12)
    # Gauss-Legendre rules of numpy and scipy, equal to round-off
    np.testing.assert_allclose(iface.weights, w, rtol=1e-12)


def _cantilever(model, conn_func, K=None):
    top, bot = _plates(model)
    for p in (top, bot):
        p.x1u = p.x1v = p.x1w = 0
        p.x1ur = p.x1vr = p.x1wr = 0
        if 'fsdt' in model:
            p.x1phix = p.x1phiy = 0
    if conn_func == 'SB':
        conn = [dict(p1=top, p2=bot, func='SB')]
        md = MultiDomain([bot, top], conn)
    else:
        # the linear analysis uses the pristine secant stiffness, the
        # strength and toughness do not matter
        law = BilinearLaw(GIc=1e9, GIIc=1e9, tauI=1e3, tauII=1e3, KI=K,
                          eta=2.)
        conn = [dict(p1=top, p2=bot, func='CZM', laws=[law], nr_x_gauss=20,
                     nr_y_gauss=10)]
        md = MultiDomain([bot, top], conn)
    top.add_distr_load_fixed_x(top.a, funcx=None, funcy=None,
                               funcz=lambda y: 1.)
    kC = md.reduce(md.calc_kC())
    fext = md.reduce(md.calc_fext())
    from structsolve import solve
    c = md.expand(solve(kC, fext, silent=True))
    res = md.uvw(c, group='top', gridx=5, gridy=5)
    return np.max(np.abs(res['w']))


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'plate_fsdt_donnell'])
def test_pristine_interface_tends_to_sb(model):
    w_sb = _cantilever(model, 'SB')
    errs = []
    for K in (1e5, 1e6, 1e7):
        errs.append(abs(_cantilever(model, 'CZM', K) - w_sb)/w_sb)
    # converges to the exact bond, the error decreasing with the stiffness
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-3


def test_law_without_shear_strength_fails_on_closed_sliding():
    """A law with G_IIc = 0 fails on round-off sliding of a closed
    interface (B = 1, zero onset); a small positive G_IIc does not"""
    kappa = 0.2
    with pytest.warns(UserWarning, match='shear strength'):
        zero = law_with_ratio(0.25, 0., 5.5, 9.17e4, 2.9, kappa)
    small = law_with_ratio(0.25, 1e-4, 5.5, 9.17e4, 2.9, kappa)
    du, dv, dw = np.array([1e-10]), np.zeros(1), np.array([-1e-5])
    for law, r_exp in ((zero, 1.), (small, 0.)):
        _, _, r, d = evaluate_laws([law], du, dv, dw, np.zeros((1, 1)),
                                   tangent=False)
        assert r[0, 0] == r_exp