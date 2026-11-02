r"""Connection ``'BFxycte'`` between the edge `x_1 =` constant of ``p1`` and
the edge `y_2 =` constant of ``p2``

- the penalty matrix is the line integral of the penalty energy of
  :mod:`panels.multidomain.connections.kCBFxycte`, for the models with 3
  and 5 DOFs per term, any angle and both orders of the panels;
- patch tests: the displacements of a rigid-body motion of the assembly,
  and of a homogeneous strain that keeps the angle between the domains,
  computed in 3D and written in the frame of each domain, store no energy
  in the connection and satisfy the constraints of the null-space method,
  whereas a strain that changes the angle does not;
- a constant strain is carried across the joint exactly, with the
  null-space and the penalty methods;
- a T-joint gives the natural frequencies of the same structure built with
  ``'BFxcte'``, the second domain rotated by 90 degrees in its plane: with
  the null-space method the two Ritz spaces are the same and so are the
  frequencies, to round-off, for every number of terms;
- the wing-box of Elalfy, De Breuker and Castro with the ribs joined to the
  spars (slow).

The null space of the penalty matrix is compared with the basis of the
null-space method in ``test_null_space.py``.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.linalg import eigh
from structsolve.sparseutils import remove_null_cols

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import nullspace

from test_bf_sdt import _flags, _field, LAMINAPROP


MODELS = ['plate_clpt_donnell', 'plate_fsdt_donnell', 'plate_tsdt_donnell']
ISO = (70.e9, 0.3)


def _cs(angle):
    t = np.radians(angle)
    c, s = np.cos(t), np.sin(t)
    return (0. if abs(c) < 1e-14 else c), (0. if abs(s) < 1e-14 else s)


def _pair(model1, model2=None, seed=None, r2=None, a1=0.3, b1=0.6, b2=0.2):
    model2 = model1 if model2 is None else model2
    kw = dict(x0=0, y0=0, plyt=0.2e-3, laminaprop=LAMINAPROP)
    p1 = Shell(a=a1, b=b1, m=7, n=6, model=model1, stack=[0, 45, -45, 90],
               **kw)
    p2 = Shell(a=b1, b=b2, m=5, n=6, model=model2, stack=[0, 90, 90, 0],
               r=r2, **kw)
    _flags(p1, seed)
    _flags(p2, None if seed is None else seed + 1)
    return p1, p2


@pytest.mark.parametrize('order', ['p1 first', 'p2 first'])
@pytest.mark.parametrize('angle', [90., 30., -120., 180.])
@pytest.mark.parametrize('models', [('plate_clpt_donnell',) * 2,
    ('plate_clpt_donnell', 'cylshell_clpt_sanders'),
    ('plate_fsdt_donnell',) * 2, ('plate_tsdt_donnell',) * 2,
    ('plate_fsdt_donnell', 'cylshell_fsdt_sanders')])
def test_energy(models, angle, order):
    r"""`\boldsymbol{c}^\top \boldsymbol{K} \boldsymbol{c}` against the line
    integral of the penalty energy evaluated from the displacement fields,
    with the rotations `\phi_x` of ``p1`` and `\phi_y` of ``p2``, which are
    `-w_{,x}` and `-w_{,y} + v/r` for the CLPT"""
    r2 = 0.8 if 'cylshell' in models[1] else None
    p1, p2 = _pair(*models, seed=3, r2=r2)
    kt, kr = 1.3e7, 2.1e3
    xcte1, ycte2 = 0.4*p1.a, p2.b
    panels = [p1, p2] if order == 'p1 first' else [p2, p1]
    md = MultiDomain(panels, [dict(p1=p1, p2=p2, func='BFxycte', xcte1=xcte1,
                     ycte2=ycte2, angle=angle, kt=kt, kr=kr)],
                     conn_method='penalty')
    K = md.get_kC_conn().toarray()
    xi, wg = np.polynomial.legendre.leggauss(30)
    L = p1.b
    s = (xi + 1)*L/2
    c_, s_ = _cs(angle)
    rng = np.random.default_rng(0)
    for _ in range(3):
        c = rng.standard_normal(md.get_size())
        f1 = _field(p1, c, np.full_like(s, xcte1), s)
        f2 = _field(p2, c, s, np.full_like(s, ycte2))
        e = (kt*((f1['u'] + c_*f2['v'] - s_*f2['w'])**2
                 + (f1['v'] - f2['u'])**2
                 + (f1['w'] - s_*f2['v'] - c_*f2['w'])**2)
             + kr*(f1['phix'] + f2['phiy'])**2)
        ref = np.sum(wg*e)*L/2
        assert np.isclose(c @ K @ c, ref, rtol=1e-11)


def _frames(angle):
    r"""Columns `e_x, e_y, e_z` of the frames of ``p1`` and ``p2``"""
    c, s = _cs(angle)
    E1 = np.eye(3)
    E2 = np.column_stack([E1[:, 1], -c*E1[:, 0] + s*E1[:, 2],
                          s*E1[:, 0] + c*E1[:, 2]])
    return E1, E2


def _targets(o, E, t, A):
    r"""Fields of a panel with origin ``o`` and frame ``E`` for the
    displacement `\boldsymbol{t} + \boldsymbol{A} \boldsymbol{r}`, with no
    transverse shear"""
    ex, ey, ez = E.T
    def comp(e):
        return lambda x, y: (e @ (t + A @ o) + (e @ A @ ex)*x + (e @ A @ ey)*y)
    return dict(u=comp(ex), v=comp(ey), w=comp(ez),
                phix=lambda x, y: -(ez @ A @ ex) + 0.*x,
                phiy=lambda x, y: -(ez @ A @ ey) + 0.*x)


def _basis(p, grid=6):
    r"""Values of `u, v, w, \phi_x, \phi_y` of the approximation functions
    of ``p`` on a grid, one column per Ritz constant, and the points of the
    grid, `\phi_x = -w_{,x}` and `\phi_y = -w_{,y}` for the CLPT"""
    xis = etas = np.linspace(-1, 1, grid)
    xs, ys = [v.ravel() for v in np.meshgrid((xis + 1)*p.a/2,
                                             (etas + 1)*p.b/2)]
    dofs = p.get_size()//(p.m*p.n)
    #NOTE the field and the derivatives of each row block, see
    #     nullspace.constraint_matrix() for the ordering
    rows = [[(0, 0, 0, 1.)], [(1, 0, 0, 1.)], [(2, 0, 0, 1.)]]
    if dofs == 3:
        rows += [[(2, 1, 0, -1.)], [(2, 0, 1, -1.)]]
    else:
        rows += [[(3, 0, 0, 1.)], [(4, 0, 0, 1.)]]
    N = np.zeros((5*xs.size, p.get_size()))
    for k, terms in enumerate(rows):
        for field, dx, dy, coeff in terms:
            F = nullspace._eval(p, field, dx, 'x', xis)
            G = nullspace._eval(p, field, dy, 'y', etas)
            block = coeff*np.kron(G, F)
            N[k*xs.size:(k + 1)*xs.size, field::dofs] += block
    return N, xs, ys, ('u', 'v', 'w', 'phix', 'phiy')


class _Patch:
    r"""Assembly of the patch tests, called with `\boldsymbol{t}` and
    `\boldsymbol{A}` to return the penalty energy, its scale, the constraints
    of the null-space method and their scale for the fields of the
    displacement `\boldsymbol{t} + \boldsymbol{A} \boldsymbol{r}`"""
    def __init__(self, model, angle, conn_angle=None):
        self.p1, self.p2 = p1, p2 = _pair(model)
        xcte1, ycte2 = p1.a, 0.
        self.E1, self.E2 = _frames(angle)
        self.o1 = np.zeros(3)
        #NOTE the point (0, ycte2) of p2 is the point (xcte1, 0) of p1
        self.o2 = self.o1 + xcte1*self.E1[:, 0] - ycte2*self.E2[:, 1]
        conn = dict(p1=p1, p2=p2, func='BFxycte', xcte1=xcte1, ycte2=ycte2,
                    angle=angle if conn_angle is None else conn_angle,
                    kt=1., kr=1.)
        md = MultiDomain([p1, p2], [conn], conn_method='penalty')
        self.size = md.get_size()
        self.K = md.get_kC_conn()
        self.B = nullspace.constraint_matrix(conn, self.size)
        self.bases = [_basis(p1), _basis(p2)]

    def __call__(self, t, A):
        c = np.zeros(self.size)
        for p, o, E, (N, xs, ys, fields) in zip((self.p1, self.p2),
                (self.o1, self.o2), (self.E1, self.E2), self.bases):
            targets = _targets(o, E, t, A)
            f = np.concatenate([targets[k](xs, ys) for k in fields])
            cp, *_ = np.linalg.lstsq(N, f, rcond=None)
            assert np.allclose(N @ cp, f, atol=1e-10*np.abs(f).max())
            c[p.col_start:p.col_end] = cp
        K, B = self.K, self.B
        return (c @ K @ c, np.abs(c) @ abs(K) @ np.abs(c), B @ c,
                abs(B) @ np.abs(c))


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


@pytest.mark.parametrize('angle', [90., 30., 135., 180.])
@pytest.mark.parametrize('model', MODELS)
def test_patch(model, angle):
    r"""A rigid-body motion and a homogeneous strain that keeps the angle
    between the domains store no energy in the connection and satisfy its
    constraints; a strain that changes the angle, or the connection with the
    angle of the opposite sign, does not

    With the common axis `\boldsymbol{s}` and the normals `\boldsymbol{n}_1`,
    `\boldsymbol{n}_2`, a homogeneous strain `\boldsymbol{\varepsilon}`
    rotates the normal of each domain about `\boldsymbol{s}` by `-\boldsymbol{n}_i
    \cdot \boldsymbol{\varepsilon} \boldsymbol{t}_i`, with `\boldsymbol{t}_i =
    \boldsymbol{s} \times \boldsymbol{n}_i`, and the two rotations are equal
    when `\boldsymbol{\varepsilon}` has no shear in the plane normal to
    `\boldsymbol{s}`, here `\varepsilon_{xz} = 0` in the frame of ``p1``, with
    `\varepsilon_{xx} = \varepsilon_{zz}`; for coplanar domains, `\theta =
    180°`, the rotations are always equal.
    """
    rng = np.random.default_rng(5)
    t = 1e-3*rng.standard_normal(3)
    W = _skew(1e-3*rng.standard_normal(3))
    S = 1e-3*rng.standard_normal((3, 3))
    S = S + S.T
    #NOTE an isotropic strain in the plane x1-z1, normal to the common axis,
    #     and any strain involving the common axis y1
    S[0, 2] = S[2, 0] = 0.
    S[2, 2] = S[0, 0]
    patch = _Patch(model, angle)
    for A in (W, S, W + S):
        U, scale, Bc, Bscale = patch(t, A)
        assert abs(U) < 1e-12*scale
        assert np.all(np.abs(Bc) < 1e-12*Bscale.max())
    if angle == 180.:
        #NOTE coplanar domains, whose rotations about the common axis are
        #     equal for any homogeneous strain
        return
    #NOTE a shear in the plane normal to the common axis changes the angle
    shear = np.zeros((3, 3))
    shear[0, 2] = shear[2, 0] = 1e-3
    U, scale, Bc, Bscale = patch(t, W + S + shear)
    assert U > 1e-3*scale
    assert np.abs(Bc).max() > 1e-3*Bscale.max()
    #NOTE the frame of p2 of the opposite angle
    U, scale, Bc, Bscale = _Patch(model, angle, conn_angle=-angle)(t, W)
    assert U > 1e-3*scale


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
@pytest.mark.parametrize('model', MODELS)
def test_constant_strain_across_the_joint(model, conn_method):
    r"""L-joint clamped at the root of the common edge and pulled along it
    with the same stress `\sigma` in both domains, of different thicknesses,
    with `\nu = 0`: the strain `\sigma/E` is constant in both domains and the
    displacement along the edge at the free end is `\sigma L/E`"""
    E, sigma = 70.e9, 50.e6
    kw = dict(x0=0, y0=0, stack=[0.], laminaprop=(E, 0.), model=model)
    p1 = Shell(a=0.2, b=0.5, m=6, n=6, plyt=1.e-3, **kw)
    p2 = Shell(a=0.5, b=0.15, m=6, n=6, plyt=2.5e-3, **kw)
    _flags(p1, None)
    _flags(p2, None)
    #NOTE clamped at y1 = 0 and x2 = 0, the root of the common edge
    #NOTE the flags of the slopes of u and v are not set, which would
    #     prevent the strain at the root
    for p, edge in ((p1, 'y1'), (p2, 'x1')):
        for d in ['u', 'v', 'w', 'wr'] + (['phix', 'phiy'] if 'dt' in model else []):
            setattr(p, edge + d, 0.)
    h1, h2 = 1.e-3, 2.5e-3
    p1.add_distr_load_fixed_y(p1.b, funcy=lambda x: sigma*h1 + 0.*x)
    p2.add_distr_load_fixed_x(p2.a, funcx=lambda y: sigma*h2 + 0.*y)
    conn = [dict(p1=p1, p2=p2, func='BFxycte', xcte1=p1.a, ycte2=0.)]
    md = MultiDomain([p1, p2], conn, conn_method=conn_method)
    c = md.linear_static()
    ref = sigma*p1.b/E
    xs = np.linspace(0, p1.a, 5)
    ys = np.linspace(0, p1.b, 7)
    X, Y = [v.ravel() for v in np.meshgrid(xs, ys)]
    f1 = _field(p1, c, X, Y)
    f2 = _field(p2, c, Y, X*p2.b/p1.a)
    rtol = 1e-8
    assert np.allclose(f1['v'], sigma/E*Y, rtol=0, atol=rtol*ref)
    assert np.allclose(f2['u'], sigma/E*Y, rtol=0, atol=rtol*ref)
    for f in (f1, f2):
        for k in ('w', 'phix', 'phiy'):
            assert np.allclose(f[k], 0., atol=rtol*ref/p1.b)
    assert np.allclose(f1['u'], 0., atol=rtol*ref)
    assert np.allclose(f2['v'], 0., atol=rtol*ref)


def _tjoint(func, conn_method, m, kr=None):
    r"""Natural frequencies of a T-joint cantilever: ``p1``, a square plate
    clamped at `y_1 = 0`, and the web ``p2`` normal to it along its line
    `x_1 = a_1/2`, clamped at the root of the common edge, connected by
    ``'BFxycte'`` at the edge `y_2 = 0` of ``p2``, or the same web rotated
    by 90 degrees in its plane, ``q2``, connected by ``'BFxcte'`` at its
    edge `x_2 = a_2`; the plies of ``q2`` are rotated by 90 degrees"""
    kw = dict(x0=0, y0=0, plyt=0.25e-3, laminaprop=LAMINAPROP, rho=1600.)
    a1, hw = 0.3, 0.08
    p1 = Shell(a=a1, b=a1, m=m, n=m, stack=[0, 45, -45, 90, 90, -45, 45, 0],
               **kw)
    if func == 'BFxycte':
        p2 = Shell(a=a1, b=hw, m=m, n=m, stack=[0, 90, 0, 90, 90, 0], **kw)
        root = 'x1'
        conn = dict(p1=p1, p2=p2, func=func, xcte1=a1/2, ycte2=0.)
    else:
        p2 = Shell(a=hw, b=a1, m=m, n=m, stack=[90, 0, 90, 0, 0, 90], **kw)
        root = 'y1'
        conn = dict(p1=p1, p2=p2, func=func, xcte1=a1/2, xcte2=hw)
    _flags(p1, None)
    _flags(p2, None)
    for p, edge in ((p1, 'y1'), (p2, root)):
        for d in ('u', 'v', 'w', 'wr'):
            setattr(p, edge + d, 0.)
    if kr is not None:
        conn['kr'] = kr
    md = MultiDomain([p1, p2], [conn], conn_method=conn_method)
    K = md.calc_kC(silent=True)
    M = md.calc_kM(silent=True)
    if conn_method == 'null-space':
        K, M = md.reduce(K), md.reduce(M)
    #NOTE the Ritz constants removed by the flags of the clamped edges
    K, M, _ = remove_null_cols(K, M, silent=True)
    ev = eigh(K.toarray(), M.toarray(), eigvals_only=True,
              subset_by_index=[0, 5])
    return np.sqrt(ev)/(2*np.pi)


#NOTE natural frequencies in Hz of the T-joint of _tjoint() with the
#     null-space method and 12 terms per direction, the same for 'BFxycte'
#     and 'BFxcte' to 2e-8
TJOINT_12 = [44.77052834, 121.73221846, 153.75926469, 199.22316713,
             203.62399878, 267.40525323]


@pytest.mark.parametrize('m', [6, 8, 10])
def test_tjoint_same_as_BFxcte(m):
    r"""With the null-space method the T-joint of ``'BFxycte'`` and that of
    ``'BFxcte'``, with the web rotated in its plane, have the same Ritz
    space, the functions of Bardell along `y_2` of the web spanning those
    along `-x_2` of the rotated web, and the same constraints, and therefore
    the same natural frequencies to round-off; with the penalty method the
    two are also the same, the penalty constants of :func:`.calc_kt_kr`
    using `A_{22}, D_{22}` of the web and `A_{11}, D_{11}` of the rotated
    web, and they are within 2.2% of the null-space method"""
    f = _tjoint('BFxycte', 'null-space', m)
    assert np.allclose(f, _tjoint('BFxcte', 'null-space', m), rtol=1e-7)
    f_pen = _tjoint('BFxycte', 'penalty', m)
    assert np.allclose(f_pen, _tjoint('BFxcte', 'penalty', m), rtol=1e-6)
    assert np.allclose(f_pen, f, rtol=0.022)


def test_tjoint_convergence():
    r"""The constraints of the null-space method are exact, so the Ritz
    spaces of the T-joint are nested as the number of terms grows, and the
    six natural frequencies decrease monotonically, from 0.25% to 1.8%
    above those of 12 terms with 6 terms, and within 0.6% with 10 terms"""
    fs = np.array([_tjoint('BFxycte', 'null-space', m) for m in (6, 8, 10)])
    ref = np.array(TJOINT_12)
    assert np.all(np.diff(np.vstack([fs, ref]), axis=0) < 0)
    assert np.all(fs[-1]/ref - 1 < 0.006)


def _wingbox(rib_spar, conn_method, P=500.):
    r"""High-AR-t5 wing-box of Elalfy, De Breuker and Castro (preprint,
    Tables 1 and 2): span 1010 mm, chord 50.5 mm, height 12.12 mm, skins,
    ribs 1 mm, spars 5 mm, 9 bays, 8 ribs and a tip rib, aluminium, clamped
    at the root, with a tip force ``P`` given by a pressure on the upper skin
    over the last 10 mm of the span, all the domains with
    ``'plate_clpt_donnell'``

    Skins `(X, Y, Z)`, spars `(X, -Z, Y)`, connected to the skins with
    ``'BFycte'``, ribs `(-Z, Y, X)`, connected to the skins with
    ``'BFxcte'``, and, with ``rib_spar=True``, to the spars with
    ``'BFxycte'``, the edge `x = a` of the spar of each bay being the edge
    `y = 0` or `y = W` of the rib at the end of the bay.

    """
    E, NU, RHO = 73.e9, 0.33, 2795.
    L, W, H = 1.010, 50.5e-3, 12.12e-3
    nbays, dx_load = 9, 10.e-3
    a = L/nbays

    def plate(a, b, t, m, n):
        p = Shell(a=a, b=b, r=None, stack=[0.], plyt=t, laminaprop=(E, NU),
                  rho=RHO, model='plate_clpt_donnell', m=m, n=n)
        _flags(p, None)
        return p

    doms, conn, prev = [], [], None
    for i in range(nbays):
        top, bot = plate(a, W, 1.e-3, 8, 8), plate(a, W, 1.e-3, 8, 8)
        spars = [plate(a, H, 5.e-3, 8, 5) for _ in range(2)]
        rib = plate(H, W, 1.e-3, 4, 6)
        conn += [dict(p1=top, p2=spars[0], func='BFycte', ycte1=0., ycte2=0.),
                 dict(p1=bot, p2=spars[0], func='BFycte', ycte1=0., ycte2=H),
                 dict(p1=top, p2=spars[1], func='BFycte', ycte1=W, ycte2=0.),
                 dict(p1=bot, p2=spars[1], func='BFycte', ycte1=W, ycte2=H),
                 dict(p1=top, p2=rib, func='BFxcte', xcte1=a, xcte2=0.),
                 dict(p1=bot, p2=rib, func='BFxcte', xcte1=a, xcte2=H)]
        if rib_spar:
            conn += [dict(p1=spars[0], p2=rib, func='BFxycte', xcte1=a, ycte2=0.),
                     dict(p1=spars[1], p2=rib, func='BFxycte', xcte1=a, ycte2=W)]
        bay = [top, bot] + spars
        if prev is None:
            for p in bay:
                for d in ('u', 'v', 'w', 'wr'):
                    setattr(p, 'x1' + d, 0.)
        else:
            conn += [dict(p1=p1, p2=p2, func='SSxcte', xcte1=a, xcte2=0.)
                     for p1, p2 in zip(prev, bay)]
        prev = bay
        doms += bay + [rib]
    top.add_pressure_load(P/(W*dx_load), x1=a - dx_load, cte=False)
    md = MultiDomain(doms, conn, conn_method=conn_method)
    return md, top


#NOTE tip deflection in mm and first buckling load in N for P = 500 N,
#     without and with the ribs joined to the spars
WINGBOX = {'penalty': ((453.5219, 172.77), (453.4117, 172.78)),
           'null-space': ((447.5362, 190.46), (447.3922, 190.41))}


@pytest.mark.slow
@pytest.mark.parametrize('conn_method', ['penalty', 'null-space'])
def test_wingbox_ribs_joined_to_spars(conn_method):
    r"""Wing-box of Elalfy, De Breuker and Castro with and without the ribs
    joined to the spars with ``'BFxycte'``

    The first linear buckling load of the High-AR-t5 case is 173.11 N with
    Abaqus, Table 3, p. 8 of the preprint. With the terms of the model, 8 x
    8 for the skins, the penalty method gives 172.77 N and the null-space
    method 190.46 N: the default penalty constants of :func:`.calc_kt_kr`
    make the box 10% more flexible in buckling. Joining the ribs to the
    spars changes the tip deflection by -0.03% and the buckling load by
    less than 0.03%: the local buckling of the skins is little affected by
    the thin ribs. The buckling load is not bound to increase with the
    additional constraints, which also change the pre-buckling state.
    """
    for rib_spar, (w_ref, P_ref) in zip((False, True), WINGBOX[conn_method]):
        md, tip = _wingbox(rib_spar, conn_method)
        c = md.linear_static()
        w = md.uvw(c, None, gridx=3, gridy=3, eval_panel=tip)['w'][0][:, -1].mean()
        ev, _ = md.linear_buckling(num_eigvalues=2)
        assert np.isclose(w*1e3, w_ref, rtol=1e-5)
        assert np.isclose(500.*ev[0], P_ref, rtol=1e-4)
