r"""Mixed-mode multilinear cohesive interface ``'CZM'``

Verification of :mod:`panels.multidomain.connections.czm`:

- the point law (:func:`.evaluate_laws`) dissipates, along a path of
  constant mode ratio up to complete failure, the sum of the
  Benzeggagh-Kenane toughnesses of the superposed laws, to round-off;
- along paths that change the mode ratio, including unloading and closure,
  no damage variable decreases, and the dissipated energy at failure lies
  between the sums of the pure-mode toughnesses of the laws;
- the consistent tangent equals central finite differences of the tractions,
  also with ``mode_ratio='summed'`` and `\eta = 1`, and that of a
  contact-only interface those of its internal force vector;
- the free energy and the resultants of a uniform separation are those of
  the closed form, for a law and for the contact;
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

def test_open_only_law_does_not_grow_damage_when_closed():
    """open_only: closed sliding leaves the threshold unchanged, also for
    G_IIc = 0; an opening grows it as for the default law"""
    kappa = 0.2
    law = law_with_ratio(0.25, 0., 5.5, 9.17e4, 2.9, kappa, open_only=True)
    ref = law_with_ratio(0.25, 1e-4, 5.5, 9.17e4, 2.9, kappa)
    r0 = np.full((1, 2), 0.1)
    du = np.array([1e-3, 1e-3])
    dw = np.array([-1e-4, 0.])
    _, T, r, _ = evaluate_laws([law], du, np.zeros(2), dw, r0)
    assert np.all(r == 0.1)
    assert np.all(np.isfinite(T))
    dw = np.array([5e-3, 5e-3])
    du = np.zeros(2)
    _, _, r1, _ = evaluate_laws([law], du, du, dw, np.zeros((1, 2)),
                                tangent=False)
    _, _, r2, _ = evaluate_laws([ref], du, du, dw, np.zeros((1, 2)),
                                tangent=False)
    assert np.allclose(r1, r2) and np.all(r1 > 0)

def _bridging(slip_offset):
    kappa = 0.2112
    return law_with_ratio(0.25, 0., 5.5, 9.17e4, 2.9, kappa, open_only=True,
                          slip_offset=slip_offset)


def test_slip_offset_requires_open_only():
    with pytest.raises(ValueError, match='open_only'):
        law_with_ratio(0.25, 1e-4, 5.5, 9.17e4, 2.9, 0.2, slip_offset=True)


def test_slip_offset_leaves_other_laws_unchanged():
    """Offsets change nothing for laws without slip_offset, and a
    slip_offset law with zero offsets is the open_only law"""
    rng = np.random.default_rng(3)
    n = 200
    d = rng.normal(size=(3, n))*1e-3
    s0 = rng.normal(size=(2, n))*1e-3
    r_old = rng.uniform(0, 0.5, (3, n))
    laws = laws_B()
    out0 = evaluate_laws(laws, *d, r_old)
    out1 = evaluate_laws(laws, *d, r_old, s0=s0)
    for a, b in zip(out0, out1):
        assert np.array_equal(a, b)
    oo = [_bridging(False)]
    so = [_bridging(True)]
    for a, b in zip(evaluate_laws(oo, *d, r_old[:1]),
                    evaluate_laws(so, *d, r_old[:1], s0=np.zeros((2, n)))):
        assert np.array_equal(a, b)
    # undamaged tractions do not depend on the offsets
    t0, _, _, _ = evaluate_laws(so, *d, np.zeros((1, n)), tangent=False)
    t1, _, _, _ = evaluate_laws(so, *d, np.zeros((1, n)), s0=s0,
                                tangent=False)
    closed = d[2] <= 0
    assert np.array_equal(t0[:, closed], t1[:, closed])


def _point_history(laws, path):
    """Converged history of one point along ``path``, with the commit
    rule of CohesiveInterface: thresholds from the previous offset, then
    the offset follows the slip where the point is closed"""
    r = np.zeros((len(laws), 1))
    s0 = np.zeros((2, 1))
    rs, diss = [], 0.
    for du, dv, dw in path:
        sep = [np.array([x]) for x in (du, dv, dw)]
        _, _, r_new, _ = evaluate_laws(laws, *sep, r, tangent=False, s0=s0)
        S = (du - s0[0, 0])**2 + (dv - s0[1, 0])**2 if laws[0].slip_offset \
            else du**2 + dv**2
        N = max(dw, 0.)**2
        law = laws[0]
        P = law.KII*S + law.KI*N
        B = law.KII*S/P if P > 0 else 0.
        diss += law.Gc(B)*(r_new[0, 0] - r[0, 0])
        r = r_new
        if dw <= 0:
            s0[:, 0] = du, dv
        rs.append(r[0, 0])
    return np.array(rs), diss


@pytest.mark.parametrize('nsteps', [50, 400])
def test_slip_offset_closed_slide_reopen(nsteps):
    """A point opened to r = r_pre, closed, slid while closed and reopened:
    open_only alone fails it at once without dissipation; with slip_offset
    the threshold stays at r_pre and grows continuously to failure, the
    dissipation is G_Ic, the same for any step size"""
    law = _bridging(True)
    lo = law.tauI/law.KI
    lf = 2*law.GIc/law.tauI
    w1 = lo + 0.3*(lf - lo)
    up = [(0., 0., w) for w in np.linspace(0, w1, nsteps)]
    close = [(0., 0., w) for w in np.linspace(w1, -1e-5, nsteps)]
    slide = [(s, 0., -1e-5) for s in np.linspace(0, 1e-3, nsteps)]
    # first reopened state with an opening of 1e-3 times the slip, the
    # limit of small increments
    reopen = [(1e-3, 0., 1e-6)] +         [(1e-3, 0., w) for w in np.linspace(1e-6, 1.2*lf, 4*nsteps)[1:]]
    path = up + close + slide + reopen
    i_re = len(up) + len(close) + len(slide)
    r, diss = _point_history([law], path)
    r_pre = r[len(up) - 1]
    assert 0.25 < r_pre < 0.35
    assert np.all(r[len(up):i_re] == r_pre)
    # first reopened states: no jump
    first = i_re
    assert abs(r[first] - r_pre) < 1e-12
    assert np.all(np.diff(r) >= 0)
    assert np.max(np.diff(r)) < 2./nsteps
    assert r[-1] == 1.
    assert abs(diss - law.GIc) < 1e-12*law.GIc
    # the open_only law without the offset fails at once on reopening
    r_oo, diss_oo = _point_history([_bridging(False)], path)
    assert r_oo[first] == 1.
    assert diss_oo < 0.4*law.GIc


def test_slip_offset_tangent_finite_differences():
    laws = laws_B()[:2] + [_bridging(True)]
    rng = np.random.default_rng(4)
    scale = max(l.tauI/l.KI for l in laws)*5
    n = 400
    d = rng.normal(size=(3, n))*scale
    d[2] = np.where(rng.uniform(size=n) < 0.8, abs(d[2]), d[2])
    s0 = d[:2] + rng.normal(size=(2, n))*scale*0.3
    r_old = rng.uniform(0, 0.6, (len(laws), n))
    _, T, r_new, _ = evaluate_laws(laws, *d, r_old, s0=s0)
    h = 1e-5*np.linalg.norm(d, axis=0)
    Tfd = np.zeros_like(T)
    ok = np.ones(n, bool)
    for j in range(3):
        e = np.zeros((3, n))
        e[j] = h
        tp, _, rp, _ = evaluate_laws(laws, *(d + e), r_old, tangent=False,
                                     s0=s0)
        tm, _, rm, _ = evaluate_laws(laws, *(d - e), r_old, tangent=False,
                                     s0=s0)
        ok &= np.all((rp > r_old) == (rm > r_old), axis=0)
        ok &= np.all((rp >= 1) == (rm >= 1), axis=0)
        ok &= np.sign(d[2] + h) == np.sign(d[2] - h)
        Tfd[:, j] = (tp - tm)/(2*h)
    # the bridging law grows damage at some of the compared points
    grows = (r_new[2] > r_old[2]) & (r_new[2] < 1) & ok
    assert grows.sum() > 20
    scale_T = np.abs(Tfd).max(axis=(0, 1))
    ok &= scale_T > 0
    err = np.abs(T - Tfd).max(axis=(0, 1))[ok]/scale_T[ok]
    assert ok.sum() > n/2
    assert err.max() < 1e-7


def test_slip_offset_commit_interface():
    """CohesiveInterface keeps the offsets as converged history: updated
    only by commit, where the point is closed"""
    top, bot = _plates('plate_clpt_donnell')
    law = _bridging(True)
    conn = [dict(p1=top, p2=bot, func='CZM', laws=[law], nr_x_gauss=4,
                 nr_y_gauss=2)]
    md = MultiDomain([bot, top], conn, conn_method='penalty',
                     pd_method='penalty')
    iface = md.cohesive_interfaces()[0]
    npts = iface.npts
    sep = {}
    iface.separations = lambda c: sep[c]
    lo = law.tauI/law.KI
    lf = 2*law.GIc/law.tauI
    one = np.ones(npts)
    sep[0] = (0*one, 0*one, (lo + 0.3*(lf - lo))*one)
    sep[1] = (1e-3*one, 0*one, -1e-5*one)
    sep[2] = (1e-3*one, 0*one, 1.01*(lo + 0.3*(lf - lo))*one)
    iface.commit(0)
    r_pre = iface.r.copy()
    assert np.all(iface.s0 == 0)
    iface.state(1)
    assert np.all(iface.s0 == 0)
    iface.commit(1)
    assert np.all(iface.s0[0] == 1e-3) and np.all(iface.s0[1] == 0)
    assert np.array_equal(iface.r, r_pre)
    diss = iface.dissipated
    iface.commit(2)
    assert np.all(iface.s0[0] == 1e-3)
    assert np.all(iface.r > r_pre) and np.all(iface.r < r_pre + 0.02)
    # pure mode I growth, G_c = G_Ic
    np.testing.assert_allclose(iface.dissipated - diss,
        law.GIc*np.sum(iface.weights*(iface.r - r_pre)[0]), rtol=1e-12)
    iface.reset()
    assert np.all(iface.s0 == 0)


def _opening_driven(shear_free=False):
    return law_with_ratio(0.25, 0., 5.5, 9.17e4, 2.9, 0.2112,
                          opening_driven=True, shear_free=shear_free)


def test_opening_driven_options():
    with pytest.raises(ValueError, match='exclude'):
        BilinearLaw(0.25, 0., 5.5, 0., 9.17e4, 2.9, KII=1e4, open_only=True,
                    slip_offset=True, opening_driven=True)
    with pytest.raises(ValueError, match='shear_free'):
        law_with_ratio(0.25, 1e-4, 5.5, 9.17e4, 2.9, 0.2, shear_free=True)
    assert _opening_driven(True).KII == 0.
    assert 'opening_driven=True' in repr(_opening_driven())


@pytest.mark.parametrize('shear_free', [False, True])
def test_opening_driven_pristine_and_unflagged_unchanged(shear_free):
    """Below onset the flagged law has the undamaged tractions (K_II du,
    K_II dv, K_I dw), and the other laws are unchanged"""
    rng = np.random.default_rng(5)
    n = 300
    law = _opening_driven(shear_free)
    lo = law.tauI/law.KI
    d = rng.uniform(-1, 1, (3, n))*0.9*lo
    d[:2] *= 100.
    t, _, r, _ = evaluate_laws([law], *d, np.zeros((1, n)))
    assert np.all(r == 0)
    K = np.array([law.KII, law.KII, law.KI])[:, None]
    np.testing.assert_allclose(t, K*d, rtol=1e-14, atol=0)
    others = laws_B()[:2]
    r_old = rng.uniform(0, 0.5, (3, n))
    a = evaluate_laws(others + [law], *(d*20), r_old)
    b = evaluate_laws(others, *(d*20), r_old[:2])
    a1 = evaluate_laws([law], *(d*20), r_old[2:])
    np.testing.assert_allclose(a[0], b[0] + a1[0], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(a[1], b[1] + a1[1], rtol=1e-13, atol=1e-9)
    np.testing.assert_array_equal(a[2][:2], b[2])
    np.testing.assert_array_equal(a[3][:2], b[3])


@pytest.mark.parametrize('shear_free', [False, True])
def test_opening_driven_mode_I_dissipation(shear_free):
    """DCB-type monotonic opening: the work is G_Ic (trapezoidal rule exact
    with the breakpoints of the law)"""
    law = _opening_driven(shear_free)
    lo = law.tauI/law.KI
    lf = 2*law.GIc/law.tauI
    w = np.unique(np.concatenate([np.linspace(0, 1.2*lf, 2001), [lo, lf]]))
    W, D = run([law], [np.array([0., 0., x]) for x in w])
    assert D[-1, 0] == 1.
    assert abs(W - law.GIc) < 1e-10*law.GIc


def test_opening_driven_shear_free_any_path():
    """K_II = 0: the work to failure is G_Ic for a mixed path: opening
    with slip, sliding while open, closing, sliding while closed and
    reopening with slip"""
    law = _opening_driven(True)
    lo = law.tauI/law.KI
    lf = 2*law.GIc/law.tauI
    s = np.linspace(0., 1., 4001)
    w1 = lo + 0.4*(lf - lo)
    path = ([np.array([x*1e-2, x*3e-3, x*w1]) for x in s]
            + [np.array([1e-2 + x*5e-2, 3e-3, w1]) for x in s]
            + [np.array([6e-2, 3e-3, w1 - x*(w1 + 1e-5)]) for x in s]
            + [np.array([6e-2 - x*0.1, 3e-3, -1e-5]) for x in s]
            + [np.array([-4e-2 + x*1e-2, 3e-3, -1e-5 + x*(1.5*lf + 1e-5)])
               for x in s])
    W, D = run([law], path)
    assert D[-1, 0] == 1.
    # trapezoidal rule across the kinks at onset and failure
    assert abs(W - law.GIc) < 1e-4*law.GIc


@pytest.mark.parametrize('shear_free', [False, True])
def test_opening_driven_sliding_does_not_damage(shear_free):
    """Closed sliding, and open sliding below the opening at onset, leave
    the threshold unchanged, however large the slip"""
    law = _opening_driven(shear_free)
    lo = law.tauI/law.KI
    r0 = np.full((1, 4), 0.1)
    du = np.array([1e-3, 1., 1e-3, 1.])
    dw = np.array([-1e-4, 0., 0.5*lo, 0.5*lo])
    # threshold 0.1 corresponds to an opening above lo
    _, T, r, d = evaluate_laws([law], du, np.zeros(4), dw, r0)
    assert np.all(r == 0.1)
    assert np.all(np.isfinite(T))
    _, _, r, _ = evaluate_laws([law], du, np.zeros(4), dw, np.zeros((1, 4)),
                               tangent=False)
    assert np.all(r == 0.)


@pytest.mark.parametrize('nsteps', [50, 400])
def test_opening_driven_closed_slide_reopen(nsteps):
    """Opened to r_pre, closed, slid while closed, reopened with the slip
    kept: the threshold stays at r_pre, then grows continuously; the
    bookkeeping is G_Ic, for any step size"""
    law = _opening_driven()
    lo = law.tauI/law.KI
    lf = 2*law.GIc/law.tauI
    w1 = lo + 0.3*(lf - lo)
    up = [(0., 0., w) for w in np.linspace(0, w1, nsteps)]
    close = [(0., 0., w) for w in np.linspace(w1, -1e-5, nsteps)]
    slide = [(s, 0., -1e-5) for s in np.linspace(0, 1e-3, nsteps)]
    reopen = [(1e-3, 0., w) for w in np.linspace(1e-6, 1.2*lf, 4*nsteps)]
    path = up + close + slide + reopen
    r = np.zeros((1, 1))
    rs, diss = [], 0.
    for du, dv, dw in path:
        sep = [np.array([x]) for x in (du, dv, dw)]
        _, _, r_new, _ = evaluate_laws([law], *sep, r, tangent=False)
        diss += law.GIc*(r_new[0, 0] - r[0, 0])
        r = r_new
        rs.append(r[0, 0])
    rs = np.array(rs)
    i_re = len(up) + len(close) + len(slide)
    r_pre = rs[len(up) - 1]
    assert 0.25 < r_pre < 0.35
    assert np.all(rs[len(up):i_re + 1] == r_pre)
    assert np.all(np.diff(rs) >= 0)
    assert np.max(np.diff(rs)) < 2./nsteps
    assert rs[-1] == 1.
    assert abs(diss - law.GIc) < 1e-12*law.GIc


@pytest.mark.parametrize('shear_free', [False, True])
def test_opening_driven_tangent_finite_differences(shear_free):
    laws = laws_B()[:2] + [_opening_driven(shear_free)]
    rng = np.random.default_rng(6)
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
    grows = (r_new[2] > r_old[2]) & (r_new[2] < 1) & ok
    assert grows.sum() > 20
    # with shear stiffness, the shear traction depends on the opening
    if not shear_free:
        sel = grows & (np.abs(d[0]) > 0.1*scale)
        assert np.all(np.abs(T[0, 2, sel]) > 0)
    scale_T = np.abs(Tfd).max(axis=(0, 1))
    ok &= scale_T > 0
    err = np.abs(T - Tfd).max(axis=(0, 1))[ok]/scale_T[ok]
    assert ok.sum() > n/2
    assert err.max() < 1e-7


def test_opening_driven_commit_interface():
    """The bookkeeping of CohesiveInterface uses G_Ic for the flagged law,
    whatever the slip"""
    top, bot = _plates('plate_clpt_donnell')
    law = _opening_driven()
    conn = [dict(p1=top, p2=bot, func='CZM', laws=[law], nr_x_gauss=4,
                 nr_y_gauss=2)]
    md = MultiDomain([bot, top], conn, conn_method='penalty',
                     pd_method='penalty')
    iface = md.cohesive_interfaces()[0]
    npts = iface.npts
    lo = law.tauI/law.KI
    lf = 2*law.GIc/law.tauI
    one = np.ones(npts)
    sep = {0: (5e-3*one, 0*one, (lo + 0.3*(lf - lo))*one)}
    iface.separations = lambda c: sep[c]
    iface.commit(0)
    assert np.all(iface.r > 0.29) and np.all(iface.r < 0.31)
    np.testing.assert_allclose(iface.dissipated,
        law.GIc*np.sum(iface.weights*iface.r[0]), rtol=1e-12)


def _wake(kappa=0.2112):
    return law_with_ratio(0.25, 0., 5.5, 9.17e4, 2.9, kappa, open_only=True,
                          wake_offset=True)


def test_wake_offset_options():
    with pytest.raises(ValueError, match='open_only'):
        law_with_ratio(0.25, 0., 5.5, 9.17e4, 2.9, 0.2, wake_offset=True)
    with pytest.raises(ValueError, match='exclude'):
        law_with_ratio(0.25, 0., 5.5, 9.17e4, 2.9, 0.2, open_only=True,
                       wake_offset=True, slip_offset=True)
    assert 'wake_offset=True' in repr(_wake())


def test_wake_offset_zero_offsets_is_open_only_and_others_unchanged():
    rng = np.random.default_rng(7)
    n = 200
    d = rng.normal(size=(3, n))*1e-3
    s0w = rng.normal(size=(2, n))*1e-3
    r_old = rng.uniform(0, 0.5, (1, n))
    for a, b in zip(evaluate_laws([_wake()], *d, r_old,
                                  s0_wake=np.zeros((2, n))),
                    evaluate_laws([_bridging(False)], *d, r_old)):
        assert np.array_equal(a, b)
    laws = laws_B()
    r3 = rng.uniform(0, 0.5, (3, n))
    for a, b in zip(evaluate_laws(laws, *d, r3),
                    evaluate_laws(laws, *d, r3, s0_wake=s0w)):
        assert np.array_equal(a, b)


def test_wake_offset_tangent_finite_differences():
    laws = laws_B()[:2] + [_wake()]
    rng = np.random.default_rng(8)
    scale = max(l.tauI/l.KI for l in laws)*5
    n = 400
    d = rng.normal(size=(3, n))*scale
    d[2] = np.where(rng.uniform(size=n) < 0.8, abs(d[2]), d[2])
    s0w = d[:2] + rng.normal(size=(2, n))*scale*0.3
    r_old = rng.uniform(0, 0.6, (len(laws), n))
    _, T, r_new, _ = evaluate_laws(laws, *d, r_old, s0_wake=s0w)
    h = 1e-5*np.linalg.norm(d, axis=0)
    Tfd = np.zeros_like(T)
    ok = np.ones(n, bool)
    for j in range(3):
        e = np.zeros((3, n))
        e[j] = h
        tp, _, rp, _ = evaluate_laws(laws, *(d + e), r_old, tangent=False,
                                     s0_wake=s0w)
        tm, _, rm, _ = evaluate_laws(laws, *(d - e), r_old, tangent=False,
                                     s0_wake=s0w)
        ok &= np.all((rp > r_old) == (rm > r_old), axis=0)
        ok &= np.all((rp >= 1) == (rm >= 1), axis=0)
        ok &= np.sign(d[2] + h) == np.sign(d[2] - h)
        Tfd[:, j] = (tp - tm)/(2*h)
    grows = (r_new[2] > r_old[2]) & (r_new[2] < 1) & ok
    assert grows.sum() > 20
    scale_T = np.abs(Tfd).max(axis=(0, 1))
    ok &= scale_T > 0
    err = np.abs(T - Tfd).max(axis=(0, 1))[ok]/scale_T[ok]
    assert ok.sum() > n/2
    assert err.max() < 1e-7


def test_wake_offset_commit_interface():
    """The wake offset follows the slip, open or closed, until the
    crack-tip law has failed at the point, and is fixed afterwards"""
    top, bot = _plates('plate_clpt_donnell')
    kappa = 0.2112
    tip = law_with_ratio(0.28, 1.24, 62., 1e6, 2.9, kappa)
    br = _wake(kappa)
    conn = [dict(p1=top, p2=bot, func='CZM', laws=[tip, br], nr_x_gauss=4,
                 nr_y_gauss=2)]
    md = MultiDomain([bot, top], conn, conn_method='penalty',
                     pd_method='penalty')
    iface = md.cohesive_interfaces()[0]
    npts = iface.npts
    one = np.ones(npts)
    lo = br.tauI/br.KI
    lf_tip = 2*tip.GIc/tip.tauI
    sep = {0: (1e-3*one, 0*one, 0.5*lf_tip*one),     # open, tip damaged
           1: (2e-3*one, 0*one, -1e-5*one),          # closed, slides
           2: (3e-3*one, 0*one, 2*lf_tip*one),       # tip fails
           3: (8e-3*one, 0*one, -1e-5*one),          # closed, slides
           4: (8e-3*one, 0*one, (2*lf_tip + 2*lo)*one)}  # reopens
    iface.separations = lambda c: sep[c]
    iface.commit(0)
    assert np.all(iface.d[0] < 1) and np.all(iface.s0w[0] == 1e-3)
    iface.commit(1)
    assert np.all(iface.s0w[0] == 2e-3)
    iface.commit(2)
    assert np.all(iface.d[0] == 1.)
    # offset of the state where the tip failed is not taken: fixed at the
    # last state with the tip intact
    assert np.all(iface.s0w[0] == 2e-3)
    r2 = iface.r[1].copy()
    iface.commit(3)
    assert np.all(iface.s0w[0] == 2e-3)
    assert np.array_equal(iface.r[1], r2)
    # on reopening, B of the bridging law follows the slip since the wake
    # formed (6e-3), not zero as with slip_offset
    diss = iface.dissipated
    iface.commit(4)
    assert np.all(iface.r[1] >= r2)
    du = 6e-3
    P = br.KII*du**2 + br.KI*(2*lf_tip + 2*lo)**2
    B = br.KII*du**2/P
    np.testing.assert_allclose(iface.dissipated - diss,
        br.Gc(B)*np.sum(iface.weights*(iface.r[1] - r2)), rtol=1e-12)
    iface.reset()
    assert np.all(iface.s0w == 0)


def test_law_input_errors():
    """Each invalid law raises its error: no shear toughness without
    ``KII``, `G_{Ic}` not above the elastic energy at onset, `G_{IIc}` not
    above it; and an unknown ``mode_ratio`` of :func:`.evaluate_laws`"""
    with pytest.raises(ValueError, match='needs KII'):
        BilinearLaw(0.25, 0., 5.5, 0., 9e4, 2.)
    with pytest.raises(ValueError, match='G_Ic = 0.0001 must exceed'):
        BilinearLaw(1e-4, 1., 80., 100., 1e5, 2.)
    with pytest.raises(ValueError, match='G_IIc = 0.01 must exceed'):
        BilinearLaw(1., 0.01, 30., 100., 1e5, 2., KII=1e5)
    z = np.zeros(1)
    with pytest.raises(ValueError, match="mode_ratio must be 'per-law' or "
                                         "'summed'"):
        evaluate_laws(laws_A(), z, z, z, np.zeros((1, 1)), mode_ratio='max')


def laws_eta1():
    kappa = 0.5
    return [law_with_ratio(0.5, 1.5, 60., 1e5, 1., kappa),
            law_with_ratio(0.2, 0.4, 20., 1e4, 1., kappa)]


@pytest.mark.parametrize('laws', [laws_B, laws_eta1])
def test_tangent_finite_differences_summed(laws):
    """``mode_ratio='summed'``, also with the linear Benzeggagh-Kenane
    exponent `\\eta = 1`, whose derivative of `B^\\eta` is 1 at `B = 0`
    (closed form): the consistent tangent equals central finite differences
    of the tractions, as in test_tangent_finite_differences"""
    from panels.multidomain.connections.czm import _dbeta
    np.testing.assert_array_equal(_dbeta(np.array([0., 0.3]), 1.), [1., 1.])
    laws = laws()
    rng = np.random.default_rng(3)
    scale = max(l.tauI/l.KI for l in laws)*5
    n = 400
    d = rng.normal(size=(3, n))*scale
    d[2] = np.where(rng.uniform(size=n) < 0.8, abs(d[2]), d[2])
    r_old = rng.uniform(0, 0.6, (len(laws), n))
    _, T, r_new, _ = evaluate_laws(laws, *d, r_old, 'summed')
    h = 1e-5*np.linalg.norm(d, axis=0)
    Tfd = np.zeros_like(T)
    ok = np.ones(n, bool)
    for j in range(3):
        e = np.zeros((3, n))
        e[j] = h
        tp, _, rp, _ = evaluate_laws(laws, *(d + e), r_old, 'summed',
                                     tangent=False)
        tm, _, rm, _ = evaluate_laws(laws, *(d - e), r_old, 'summed',
                                     tangent=False)
        ok &= np.all((rp > r_old) == (rm > r_old), axis=0)
        ok &= np.all((rp >= 1) == (rm >= 1), axis=0)
        ok &= np.sign(d[2] + h) == np.sign(d[2] - h)
        Tfd[:, j] = (tp - tm)/(2*h)
    assert np.any(r_new[:, ok] > r_old[:, ok])
    scale_T = np.abs(Tfd).max(axis=(0, 1))
    ok &= scale_T > 0
    err = np.abs(T - Tfd).max(axis=(0, 1))[ok]/scale_T[ok]
    assert ok.sum() > n/2
    assert err.max() < 1e-7


def test_tangent_at_zero_separation_is_pristine():
    """At zero separation the tangent of a pristine law is
    `\\mathrm{diag}(K_{II}, K_{II}, K_I)`, the limit of the tangent for an
    opening and for a closure (closed form), which the linear analyses use"""
    law = laws_A()[0]
    z = np.zeros(1)
    _, T, _, _ = evaluate_laws([law], z, z, z, np.zeros((1, 1)))
    np.testing.assert_array_equal(T[:, :, 0],
                                  np.diag([law.KII, law.KII, law.KI]))


def test_interface_input_errors_and_warnings():
    """:class:`.CohesiveInterface` raises its errors, for domains of
    different positions, without ``col_start`` and for a contact-only
    interface without ``k_contact``, and warns for a law that violates Eq.
    (1) of Turon et al. (2018) and for ``mode_ratio='summed'`` with
    different ratios `K_{II}/K_I`; an assembly without connections has no
    cohesive interface"""
    top, bot = _plates('plate_clpt_donnell')
    with pytest.raises(ValueError, match='col_start of the domains'):
        CohesiveInterface(top, bot, laws_A(), 4, 2, 10)
    other = Shell(group='o', x0=1., y0=0, a=top.a, b=top.b, m=4, n=4,
                  plyt=0.14, stack=[0]*8, laminaprop=top.laminaprop,
                  model='plate_clpt_donnell')
    with pytest.raises(ValueError, match="must have the same x0"):
        CohesiveInterface(top, other, laws_A(), 4, 2, 10)
    md = MultiDomain([bot, top])
    assert md.cohesive_interfaces() == []
    size = md.get_size()
    with pytest.raises(ValueError, match='contact-only interface needs '
                                         'k_contact'):
        CohesiveInterface(top, bot, [], 4, 2, size)
    bad = BilinearLaw(0.969, 1.719, 80., 100., 1e5, 2., KII=1e5)
    assert bad.consistency_error() > 1e-6
    with pytest.warns(UserWarning, match='violates Eq. \\(1\\)'):
        CohesiveInterface(top, bot, [bad], 4, 2, size)
    with pytest.warns(UserWarning, match="mode_ratio='summed' with different"):
        CohesiveInterface(top, bot, laws_A() + [laws_B()[0]], 4, 2, size,
                          mode_ratio='summed')


def test_contact_only_tangent_finite_differences():
    """Contact-only interface, ``laws=[]``: the tangent of the closed
    points equals central finite differences of the internal force vector,
    and its product by a vector equals that of :meth:`.kTv` (other
    formulation); without contact the tangent is null"""
    top, bot = _plates('plate_clpt_donnell', m=6, n=4)
    md = MultiDomain([bot, top])
    md.get_size()
    iface = CohesiveInterface(top, bot, [], 6, 3, md.get_size(),
                              k_contact=1e5, contact_eps=1e-5)
    rng = np.random.default_rng(4)
    c = rng.normal(size=iface.size)*1e-3
    _, _, dw = iface.separations(c)
    assert np.any(dw < -1e-5) and np.any(dw > 0)
    K = iface.kT(c)
    v = rng.normal(size=iface.size)*1e-3
    h = 1e-6
    fd = (iface.fint(c + h*v) - iface.fint(c - h*v))/(2*h)
    np.testing.assert_allclose(K @ v, fd, rtol=1e-6, atol=1e-6*abs(fd).max())
    np.testing.assert_allclose(K @ v, iface.kTv(c, v), rtol=1e-10,
                               atol=1e-10*abs(fd).max())
    assert iface.kT(np.zeros(iface.size)).nnz == 0


@pytest.mark.parametrize('contact', [False, True])
def test_energy_and_resultants_uniform_separation(contact):
    """Uniform separation of a pristine interface, closed form over the
    area `a b`: free energy `\\frac{1}{2} (K_{II} \\Delta_u^2 + K_I
    \\Delta_w^2) a b` and resultants `-(K_{II} \\Delta_u, 0, K_I \\Delta_w)
    a b` of a law below its onset; for a contact-only interface with the
    penetration `x > \\epsilon`, energy `k (x^2/2 - \\epsilon x/2 +
    \\epsilon^2/6) a b` and normal resultant `k (x - \\epsilon/2) a b`,
    :func:`.contact_traction`"""
    top, bot = _plates('plate_clpt_donnell', m=4, n=4)
    md = MultiDomain([bot, top])
    size = md.get_size()
    area = top.a*top.b
    one = np.ones(4*3)
    if contact:
        k, eps, x = 1e5, 1e-3, 5e-3
        iface = CohesiveInterface(top, bot, [], 4, 3, size, k_contact=k,
                                  contact_eps=eps)
        iface.separations = lambda c: (0*one, 0*one, -x*one)
        energy = k*(x*x/2 - eps*x/2 + eps*eps/6)*area
        res = np.array([0., 0., k*(x - eps/2)*area])
    else:
        law = laws_A()[0]
        iface = CohesiveInterface(top, bot, [law], 4, 3, size)
        du, dw = 1e-4, 2e-4
        iface.separations = lambda c: (du*one, 0*one, dw*one)
        energy = 0.5*(law.KII*du**2 + law.KI*dw**2)*area
        res = -np.array([law.KII*du, 0., law.KI*dw])*area
    assert iface.energy(None) == pytest.approx(energy, rel=1e-12)
    np.testing.assert_allclose(iface.resultants(None), res, rtol=1e-12,
                               atol=1e-12*abs(res).max())


def test_wake_offset_without_crack_tip_law_is_fixed():
    """Interface with ``wake_offset`` laws only: no law marks the crack
    tip, so the wake offsets do not follow the slip and stay zero (the
    offsets follow where a law without ``wake_offset`` has not failed)"""
    top, bot = _plates('plate_clpt_donnell')
    md = MultiDomain([bot, top])
    iface = CohesiveInterface(top, bot, [_wake()], 4, 2, md.get_size())
    one = np.ones(iface.npts)
    iface.separations = lambda c: (2e-3*one, 0*one, -1e-5*one)
    iface.commit(None)
    assert iface._tip == []
    assert np.all(iface.s0w == 0.)
