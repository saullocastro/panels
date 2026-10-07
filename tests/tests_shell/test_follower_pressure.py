r"""Verification of the follower pressure of :meth:`.Shell.add_pressure_load`

A follower pressure stays normal to the deformed mid-surface and acts on its
deformed area, see ``theory/shells/follower_pressure/follower_pressure.py``.
Its force vector `F_p(c)` enters the residual through
:meth:`.Shell.calc_fint`, and its load stiffness `-\partial F_p/\partial c`,
:meth:`.Shell.calc_kCfollower`, enters :meth:`.Shell.calc_kT`, both at the
load factor ``inc``.

The checks, for all 9 models and for the first-order (``'linear'``) and
complete (``'quadratic'``) area vectors:

1. in the undeformed state the follower force equals the dead pressure force;
2. directional Taylor test of `F_p`: the relative remainder of `F_p(c + h d) -
   F_p(c) + h K_p d` is at round-off for the first-order area vector, for
   which `F_p` is affine, and falls as `O(h)` for the quadratic one;
3. finite-difference Jacobian of the total residual against `K_T`;
4. quadratic convergence of Newton-Raphson and of the Riks method of
   ``structsolve``, with the load factor passed to the callbacks;
5. symmetry of `K_p` when the theory predicts it, and unsymmetry otherwise;
6. the rigid rotation of a ring under a hoop prestress, a zero-energy mode
   only with the follower load stiffness (Sanders kinematics).
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.special import roots_legendre
from structsolve import Analysis, solve

from panels import modelDB
from panels.shell import Shell

from test_tangent_consistency import MODELS, make_shell, random_state

TRUNCATIONS = ['linear', 'quadratic']
PRESSURES = ['constant', 'function', 'patch']


def add_pressure(s, kind, follower, cte=False, scale=1.):
    if kind == 'constant':
        s.add_pressure_load(-2.e4*scale, follower=follower, cte=cte)
    elif kind == 'function':
        s.add_pressure_load(lambda x, y: scale*1.e4*(1 + x/s.a - 2*y**2/s.b**2),
                            follower=follower, cte=cte)
    elif kind == 'patch':
        s.add_pressure_load(lambda x, y: -scale*3.e4*(1 + x*y/(s.a*s.b)),
                            x1=0.05, x2=0.2, y1=0.03, y2=0.15,
                            follower=follower, cte=cte)
    else:
        raise ValueError(kind)


def fext_follower_by_quadrature(s, func, follower, c, x1=None, x2=None,
                                y1=None, y2=None):
    r"""Independent integration of `\int p N.a(c) dA` with the shape functions
    of ``field.fg`` and finite differences of the displacements"""
    x1 = 0. if x1 is None else x1
    x2 = s.a if x2 is None else x2
    y1 = 0. if y1 is None else y1
    y2 = s.b if y2 is None else y2
    fg = modelDB.db[s.model]['field'].fg
    size = s.get_size()
    g = np.zeros((5, size))
    dofs = modelDB.db[s.model]['dofs']
    kw = 1/s.r if s.is_curved() else 0.
    kv = kw if 'sanders' in s.model else 0.

    def uvw(x, y):
        fg(g, x, y, s)
        return g[:3] @ c, g[:3].copy()

    out = np.zeros(size)
    pts, wts = roots_legendre(s.nx)
    ptsy, wtsy = roots_legendre(s.ny)
    e = 1.e-6
    for xi, wx in zip(pts, wts):
        x = x1 + (xi + 1)*(x2 - x1)/2
        for eta, wy in zip(ptsy, wtsy):
            y = y1 + (eta + 1)*(y2 - y1)/2
            (u, v, w), N = uvw(x, y)
            ux_, uy_ = [(uvw(x + e*s.a, y)[0] - uvw(x - e*s.a, y)[0])/(2*e*s.a),
                        (uvw(x, y + e*s.b)[0] - uvw(x, y - e*s.b)[0])/(2*e*s.b)]
            ux, vx, wx_ = ux_
            uy, vy, wy_ = uy_
            A = np.array([1 + ux, vx, wx_])
            B = np.array([uy, 1 + vy + kw*w, wy_ - kv*v])
            if follower == 'quadratic':
                a = np.cross(A, B)
            else:
                a = np.array([-wx_, -wy_ + kv*v, 1 + ux + vy + kw*w])
            out += wx*wy*(x2 - x1)/2*(y2 - y1)/2*func(x, y)*(a @ N)
    return out


@pytest.mark.parametrize('kind', PRESSURES)
@pytest.mark.parametrize('model', MODELS)
def test_undeformed_state_equals_dead_load(model, kind):
    s = make_shell(model)
    add_pressure(s, kind, follower=None)
    dead = s.calc_fext()
    for follower in TRUNCATIONS:
        s.clear_loads()
        add_pressure(s, kind, follower=follower)
        c0 = np.zeros(s.get_size())
        assert np.allclose(s.calc_fext(), dead, rtol=1e-14, atol=0)
        assert np.allclose(s.calc_fext(c=c0), dead, rtol=1e-14, atol=0)
        assert np.allclose(s.calc_fext_follower(c0), dead, rtol=1e-13,
                           atol=1e-12*np.abs(dead).max())
        assert np.all(s.calc_fint(c0) == 0)


@pytest.mark.parametrize('follower', TRUNCATIONS)
@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_clpt_sanders',
                                   'cylshell_fsdt_donnell'])
def test_force_against_independent_quadrature(model, follower):
    r"""The kernels against `\int p N.a dA` with the area vector computed
    from finite differences of the displacement field"""
    s = make_shell(model)
    func = lambda x, y: -1.e4*(1 + x/s.a)
    s.add_pressure_load(func, x1=0.05, follower=follower)
    rng = np.random.default_rng(11)
    c = random_state(s, rng)
    ref = fext_follower_by_quadrature(s, func, follower, c, x1=0.05)
    assert np.allclose(s.calc_fext_follower(c), ref, rtol=1e-6,
                       atol=1e-6*np.abs(ref).max())


@pytest.mark.parametrize('kind', PRESSURES)
@pytest.mark.parametrize('follower', TRUNCATIONS)
@pytest.mark.parametrize('model', MODELS)
def test_taylor_follower_force(model, follower, kind):
    s = make_shell(model)
    add_pressure(s, kind, follower=follower)
    rng = np.random.default_rng(0)
    c = random_state(s, rng)
    d = random_state(s, rng)
    F = lambda c: s.calc_fext_follower(c)
    Kd = -(s.calc_kCfollower(c=c) @ d)
    assert np.linalg.norm(Kd) > 0
    F0 = F(c)
    steps = [1.e-2, 1.e-3, 1.e-4, 1.e-5]
    errors = [np.linalg.norm(F(c + h*d) - F0 - h*Kd)/np.linalg.norm(h*Kd)
              for h in steps]
    if follower == 'linear':
        # F_p is affine in c
        assert max(errors) < 1.e-8, errors
    else:
        for h, prev, err in zip(steps[1:], errors[:-1], errors[1:]):
            assert err < 0.2*prev, errors
        coefficients = [err/h for h, err in zip(steps, errors)]
        assert max(coefficients)/min(coefficients) < 1.5, errors


@pytest.mark.parametrize('follower', TRUNCATIONS)
@pytest.mark.parametrize('model', MODELS)
def test_finite_difference_jacobian_of_residual(model, follower):
    r"""`-dR/dc`, with `R = F_{ext}(inc) - F_{int}(c, inc)`, against
    `K_T(c, inc)`, every entry"""
    s = make_shell(model)
    add_pressure(s, 'constant', follower=follower, scale=100.)
    add_pressure(s, 'patch', follower=follower, cte=True, scale=100.)
    rng = np.random.default_rng(7)
    c = random_state(s, rng)
    inc = 0.7
    N = c.shape[0]
    step = 1.e-8
    J = np.empty((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = step
        J[:, j] = (s.calc_fint(c + e, inc=inc) - s.calc_fint(c - e, inc=inc))/(2*step)
    K = s.calc_kT(c=c, inc=inc).toarray()
    err = np.linalg.norm(K - J)/np.linalg.norm(J)
    assert err < 1.e-5, 'KT differs from d(fint)/dc by %.3e' % err
    # the load stiffness is small compared to KT for these thin panels, so
    # the error must also be small compared to it
    kCf = s.calc_kCfollower(c=c, inc=inc).toarray()
    err_f = np.linalg.norm(K - J)/np.linalg.norm(kCf)
    assert err_f < 1.e-2, 'KT - d(fint)/dc is %.3e of kCfollower' % err_f


@pytest.mark.parametrize('model', MODELS)
def test_load_factor(model):
    r"""The incremented loads scale with ``inc``, the constant ones not"""
    s = make_shell(model)
    add_pressure(s, 'constant', follower='quadratic', cte=False)
    add_pressure(s, 'function', follower='linear', cte=True)
    rng = np.random.default_rng(1)
    c = random_state(s, rng)
    s2 = make_shell(model)
    add_pressure(s2, 'constant', follower='quadratic', cte=False, scale=2.5)
    add_pressure(s2, 'function', follower='linear', cte=True)
    assert np.allclose(s.calc_fint(c, inc=2.5), s2.calc_fint(c, inc=1.),
                       rtol=1e-12)
    assert np.allclose(s.calc_kCfollower(c=c, inc=2.5).toarray(),
                       s2.calc_kCfollower(c=c).toarray(), rtol=1e-12,
                       atol=1e-12*abs(s2.calc_kCfollower(c=c)).max())
    # fext(c) is the load vector of the configuration c
    assert np.allclose(s.calc_fext(inc=2.5, c=c),
                       s2.calc_fext(c=c), rtol=1e-12)
    R = s.calc_fext(inc=2.5) - s.calc_fint(c, inc=2.5)
    Rdead = s.calc_fext(inc=2.5) - s.calc_fint(c, inc=2.5)*0
    assert not np.allclose(R, Rdead)


def follower_shell(model):
    """Panel with the follower load as the only load, for Newton-Raphson"""
    s = make_shell(model)
    s.add_pressure_load(-8.e3 if model.startswith('cyl') else 1.5e3,
                        follower='quadratic', cte=False)
    s.add_pressure_load(lambda x, y: 2.e3*np.sin(np.pi*x/s.a),
                        y1=0.05, y2=0.15, follower='linear', cte=False)
    return s


class Recorder(object):
    r"""Wraps ``Shell.calc_fint`` recording the relative residual of each call

    The residual is `R = inc F_{ext} - F_{int}(c, inc)`, as in
    ``structsolve``, with the load factor ``inc`` passed by the solver.
    """
    def __init__(self, s):
        self.s = s
        self.fext = s.calc_fext()
        self.an = None
        self.calls = []

    def calc_fint(self, c, inc=1., silent=True):
        fint = self.s.calc_fint(c, inc=inc)
        R = inc*self.fext - fint
        ref = max(np.linalg.norm(inc*self.fext), np.linalg.norm(fint))
        step = len(self.an.increments)
        self.calls.append((inc, np.linalg.norm(R)/ref, step))
        return fint


def check_quadratic(errors, floor=1.e-12):
    r"""Every iteration from ``e_k < 1e-2`` above the round-off ``floor``
    gives ``e_k+1 <= 10 e_k**1.8``, i.e. a superlinear rate of order about 2
    (a linear rate fails it from ``e_k ~ 1e-4`` on). Returns the number of
    iterations checked and the orders ``log(e2/e1)/log(e1/e0)``"""
    checked = 0
    for e0, e1 in zip(errors[:-1], errors[1:]):
        if floor < e0 < 1.e-2:
            assert e1 <= max(10*e0**1.8, floor), errors
            checked += 1
    orders = [np.log(e2/e1)/np.log(e1/e0)
              for e0, e1, e2 in zip(errors[:-2], errors[1:-1], errors[2:])
              if e2 > floor and e1 < 1e-2]
    return checked, orders


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_clpt_sanders',
                                   'cylshell_fsdt_sanders'])
def test_newton_raphson_converges_quadratically(model):
    s = follower_shell(model)
    rec = Recorder(s)
    an = Analysis(s.calc_fext, rec.calc_fint, s.calc_kC, s.calc_kG)
    rec.an = an
    an.initialInc = 0.25
    an.relTOL = 1.e-12
    an.maxNumIter = 20
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    c = an.cs[-1]
    # deflections well above the thickness, the non-linear terms matter
    h = len(s.stack)*s.plyt
    wmax = np.abs(s.uvw(c, gridx=9, gridy=9)[1]['w']).max()
    assert wmax > 2*h, wmax
    R = s.calc_fext() - s.calc_fint(c)
    assert np.linalg.norm(R) <= 1.e-12*np.linalg.norm(s.calc_fext())
    for step in range(len(an.increments)):
        errors = [e for inc, e, st in rec.calls if st == step]
        checked, orders = check_quadratic(errors)
        assert checked >= 1, (step, errors)
    # the follower terms matter: the dead load gives another solution
    s_dead = make_shell(model)
    for load in s.pressure_loads_inc:
        s_dead.pressure_loads_inc.append(load[:5])
    an_dead = Analysis(s_dead.calc_fext, s_dead.calc_fint, s_dead.calc_kC,
                       s_dead.calc_kG)
    an_dead.initialInc = 0.25
    an_dead.static(NLgeom=True, silent=True)
    assert np.isclose(an_dead.increments[-1], 1.)
    assert (np.linalg.norm(an_dead.cs[-1] - c)
            > 1.e-4*np.linalg.norm(c))


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_clpt_sanders'])
def test_riks_with_follower_pressure(model):
    s = follower_shell(model)
    rec = Recorder(s)
    an = Analysis(s.calc_fext, rec.calc_fint, s.calc_kC, s.calc_kG)
    rec.an = an
    an.NL_method = 'arc_length_riks'
    an.initialInc = 0.3
    an.relTOL = 1.e-12
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    fext = s.calc_fext()
    for lbd, c in zip(an.increments, an.cs):
        R = lbd*fext - s.calc_fint(c, inc=lbd)
        assert np.linalg.norm(R) <= 1.e-10*np.linalg.norm(lbd*fext)
    # quadratic convergence of the bordered system in the steps of the arc
    # length method, the last increment is corrected by Newton-Raphson
    checked = 0
    for step in range(len(an.increments) - 1):
        errors = [e for inc, e, st in rec.calls if st == step]
        checked += check_quadratic(errors)[0]
    assert checked >= 1
    # the same path as Newton-Raphson at the load factor 1
    s2 = follower_shell(model)
    an2 = Analysis(s2.calc_fext, s2.calc_fint, s2.calc_kC, s2.calc_kG)
    an2.initialInc = 0.25
    an2.relTOL = 1.e-12
    an2.static(NLgeom=True, silent=True)
    assert np.allclose(an.cs[-1], an2.cs[-1], rtol=1e-6,
                       atol=1e-8*np.abs(an2.cs[-1]).max())


def bc_shell(model, edges):
    r"""Square panel with ``edges`` a dict of the flags of each edge"""
    s = make_shell(model)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for field in 'uvw':
            setattr(s, edge + field, edges.get(edge + field, 1.))
    s._rebuild()
    return s


def asym(K):
    K = K.toarray()
    return np.linalg.norm(K - K.T)/np.linalg.norm(K)


@pytest.mark.parametrize('model', MODELS)
def test_symmetry(model):
    r"""Symmetric when p is uniform over the whole domain and, on each edge,
    `w = 0` or the normal displacement is zero"""
    ss = {e + 'w': 0. for e in ('x1', 'x2', 'y1', 'y2')}
    normal = {'x1u': 0., 'x2u': 0., 'y1v': 0., 'y2v': 0.}
    mixed = {'x1w': 0., 'x2u': 0., 'y1v': 0., 'y2w': 0.}
    # symmetric: w = 0 on the whole boundary, the normal displacement zero
    for edges in (ss, normal, mixed):
        s = bc_shell(model, edges)
        s.add_pressure_load(-1.e4, follower=True)
        assert asym(s.calc_kCfollower()) < 1.e-12, edges
    # unsymmetric: a free edge, a patch, a varying pressure
    free = dict(ss)
    free['x2w'] = 1.
    s = bc_shell(model, free)
    s.add_pressure_load(-1.e4, follower=True)
    assert asym(s.calc_kCfollower()) > 1.e-3
    s = bc_shell(model, ss)
    s.add_pressure_load(-1.e4, x1=0.1, follower=True)
    assert asym(s.calc_kCfollower()) > 1.e-3
    s = bc_shell(model, ss)
    s.add_pressure_load(lambda x, y: -1.e4*(1 + x/s.a), follower=True)
    assert asym(s.calc_kCfollower()) > 1.e-3
    # the matrix in calc_kC is not symmetrized
    s = bc_shell(model, free)
    s.add_pressure_load(-1.e4, follower=True, cte=False)
    c = np.zeros(s.get_size())
    kC = s.calc_kC(c=c, NLgeom=True, inc=1.)
    kC0 = s.calc_kC(c=c)
    assert np.allclose((kC - kC0).toarray(), s.calc_kCfollower().toarray(),
                       atol=1e-12*abs(kC).max())


def ring_shell(model, v_free=False):
    r"""Ring in generalized plane strain: a quarter of the circumference with
    symmetry planes at the straight edges and at the ends"""
    E, nu, h, R = 70.e9, 0.3, 0.01, 1.
    s = Shell(model=model)
    s.r = R
    s.a = 0.5*R
    s.b = np.pi/2*R
    s.stack = [0.]
    s.plyt = h
    G = E/(2*(1 + nu))
    s.laminaprop = (E, E, nu, G, G, G)
    s.m = 4
    s.n = 15
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, e + f, 1.)
            setattr(s, e + f + 'r', 1.)
    s.x1u = s.x2u = 0.
    s.x1vr = s.x2vr = 0.
    s.x1wr = s.x2wr = 0.
    s.x1phiy = s.x2phiy = 1.
    s.x1phix = s.x2phix = 0.
    s.y1phix = s.y2phix = 1.
    if not v_free:
        s.y1v = s.y2v = 0.
        s.y1wr = s.y2wr = 0.
        s.y1phiy = s.y2phiy = 0.
    s._rebuild()
    return s


@pytest.mark.parametrize('model', ['cylshell_clpt_sanders',
                                   'cylshell_fsdt_sanders',
                                   'cylshell_tsdt_sanders'])
def test_rigid_rotation_of_ring_is_neutral_only_for_follower(model):
    r"""With Sanders' kinematics the hoop prestress `N_{yy} \beta_y^2`, with
    `\beta_y = w_{,y} - v/r`, gives the rigid rotation `v = const` a negative
    energy under a dead external pressure, a spurious buckling mode at a
    vanishing load. The follower load stiffness cancels it exactly."""
    s = ring_shell(model, v_free=True)
    s.add_pressure_load(-1., cte=False, follower=True)
    size = s.get_size()
    # least-squares fit of v = 1, u = w = 0 (the minimum-norm solution has
    # phix = phiy = 0 for the FSDT and TSDT, the rotation of the transverse
    # normal phiy + v/r follows the rigid rotation)
    fg = modelDB.db[s.model]['field'].fg
    g = np.zeros((5, size))
    rows, rhs = [], []
    for x in np.linspace(0, s.a, 7):
        for y in np.linspace(0, s.b, 31):
            fg(g, x, y, s)
            rows += [gi.copy() for gi in g[:3]]
            rhs += [0., 1., 0.]
    c_rot = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    assert np.linalg.norm(np.array(rows) @ c_rot - rhs) < 1e-8
    k0 = s.calc_kC()
    assert abs(c_rot @ k0 @ c_rot) < 1e-10*abs(k0).max()*(c_rot @ c_rot)
    # the membrane prestress of the external pressure p = -1, Nyy = p r,
    # imposed directly since the rigid rotation makes k0 singular
    s.Nyy = -1.*s.r
    kG = s.calc_kG()
    kF = s.calc_kCfollower()
    eG = c_rot @ (kG @ c_rot)
    eF = c_rot @ (kF @ c_rot)
    # compressive hoop force: negative energy of the rotation, -p int dA
    assert np.isclose(eG, -s.a*s.b, rtol=1e-10)
    # cancelled by the load stiffness, int p v**2/r dA
    assert abs(eG + eF) < 1e-10*abs(eG)


def test_validation_of_arguments():
    s = make_shell('plate_clpt_donnell')
    with pytest.raises(ValueError, match='follower must be'):
        s.add_pressure_load(1., follower='exact')
    s.add_pressure_load(1., follower=True)
    assert s.pressure_loads[-1][5] == 'linear'
    s.add_pressure_load(1., follower=False)
    assert s.pressure_loads[-1][5] is None
    assert not make_shell('plate_clpt_donnell').has_follower_loads()
    assert s.has_follower_loads()


def test_load_surface_factor():
    r"""``zp`` scales the pressure by ``1 + zp/r`` for cylinders only"""
    for model, factor in (('cylshell_fsdt_sanders', 1 + 0.01/0.4),
                          ('plate_fsdt_donnell', 1.)):
        s = make_shell(model)
        s.add_pressure_load(-1.e3)
        f1 = s.calc_fext()
        rng = np.random.default_rng(2)
        c = random_state(s, rng)
        for follower in (None, 'linear'):
            s.clear_loads()
            s.add_pressure_load(-1.e3, zp=0.01, follower=follower)
            assert np.allclose(s.calc_fext(), factor*f1, rtol=1e-13)
        s.clear_loads()
        s.add_pressure_load(-1.e3, follower=True)
        fp = s.calc_fext_follower(c)
        s.clear_loads()
        s.add_pressure_load(-1.e3, zp=0.01, follower=True)
        assert np.allclose(s.calc_fext_follower(c), factor*fp, rtol=1e-13)


@pytest.mark.parametrize('model', ['plate_clpt_donnell', 'cylshell_fsdt_sanders'])
def test_linear_static_with_follower_load(model):
    r"""``structsolve.Analysis.static(NLgeom=False)`` evaluates the follower
    load in the current configuration of the geometrically linear structure,
    which for the first-order area vector is ``(k0 + kCfollower) c = fext``,
    a non-symmetric linear system"""
    from structsolve import static
    s = make_shell(model)
    s.add_pressure_load(-2.e5, follower=True, cte=False)
    s.add_pressure_load(lambda x, y: 1.e5*(1 + x/s.a), x1=0.1, follower=True,
                        cte=False)
    an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    an.relTOL = 1.e-12
    increments, cs = an.static(NLgeom=False, silent=True)
    k0 = s.calc_kC()
    kCf = s.calc_kCfollower()
    fext = s.calc_fext()
    ref = static(k0 + kCf, fext, silent=True)[1][0]
    assert np.allclose(cs[0], ref, rtol=1e-8, atol=1e-10*np.abs(ref).max())
    # the follower terms matter
    c_dead = solve(k0, fext, silent=True)
    assert np.linalg.norm(c_dead - ref) > 1e-4*np.linalg.norm(ref)
