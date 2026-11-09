r"""Web of a ring frame connected to a cylinder with ``'BFxcte'``

An annular-sector web, model ``'annular_clpt_donnell'``, connected at the end
`x = 0` of a cylinder clamped there: the connection clamps the edge of the
web, whose frequencies must be those of the annulus clamped on that edge
and free on the other, from the frequency equation of Leissa (1969), with
the penalty and with the null-space connections. Verified:

- the connection at the reference surface of the cylinder, the outer edge
  of the web at the radius of the cylinder, against the roots of the
  annulus clamped outside and free inside;
- the connection at a face of the cylinder, key ``'face'``: an internal
  frame, outer edge of the web on the inner surface, against the same
  roots, and an external frame, inner edge of the web on the outer surface,
  against the roots of the annulus clamped inside and free outside, this
  frequency equation against the values of Southwell given by Leissa
  (1969), Table 2.31;
- the face of a cylinder whose reference surface is that face, offset
  `h/2`, gives the matrices of the connection at the reference surface;
  the same internal frame modelled with the reference surface of the
  cylinder at its inner face or at its mid-surface and ``face='lower'``;
- the two rigid-body motions of the assembly that the approximation
  functions represent exactly, the translation along the axis and the
  rotation about it, give no strain energy and satisfy the constraints;
- the penalty stiffness matrix of the face connection gives the penalty
  energy evaluated from the fields;
- the geometry of the connection is checked, a mismatch raising a
  ``ValueError``.

Notes
-----
With ``dict(p1=cylinder, p2=web, func='BFxcte', xcte1=x_f, xcte2=0)`` the
connection imposes `u_{cyl} = w_{web}`, `v_{cyl} = v_{web}`, `w_{cyl} =
-u_{web}` and the continuity of the rotation about the circumferential
axis, `w_{cyl,x} = w_{web,x}`, at the station `x_f`, see
:mod:`panels.models.annular_clpt_donnell_num`. With ``face='lower'`` or
``face='upper'`` the displacements of the cylinder are those of its inner or
outer surface, `z_f = d \mp h/2`, with the Kirchhoff-Love kinematics, `u(z_f)
= u - z_f w_{,x}` and `v(z_f) = (1 + z_f/r) v - z_f w_{,y}`, the term `z_f
v/r` with the Sanders-Koiter kinematics, see
:mod:`panels.multidomain.connections.kCBFxcte_face`. At a clamped end of
the cylinder all of them vanish, which clamps the connected edge of the
web through the connection alone (the web itself has free edges).

The reference values are the roots of the frequency equation of the
annulus, from the Bessel-function solution of Leissa [1]_, Eq. (1.17),
`W = A J_n(k\rho) + B Y_n(k\rho) + C I_n(k\rho) + D K_n(k\rho)`, with `W =
W' = 0` at the clamped edge and the bending moment and the
Kelvin-Kirchhoff edge reaction of Leissa's Eqs. (1.11)-(1.13) zero at the
free edge, see :func:`exact_lambda2`, `\lambda^2 = \omega r_o^2
\sqrt{\rho_m h/D}` with the outer radius `r_o` of the web, `\nu = 1/3`,
`\beta = 0.4` the ratio of the inner to the outer radius. Clamped outside:
13.5004, 19.3892 and 31.3384 for `n = 0, 1, 2`, Leissa's Table 2.21 giving
13.54, 19.80 and 31.34. Clamped inside: 9.07188, 9.12934 and 10.3655.
Leissa's Table 2.31 gives the ratios `\beta` of Southwell (his ref.
2.37) at which `\lambda^2` takes given values, `\nu = 0.3`; the roots of
the frequency equation are at these ratios to within 0.006 in the range
`\beta = 0.27` to `0.65` used here. Leissa's Table 2.29 (`\nu = 1/3`,
from his ref. 2.42) gives 7.036, 9.096 and 10.37 for `\beta = 0.4`: its
`n = 0` value differs from the root, 9.07188, and from the trend of
Table 2.31, and is not used. The cylinder, two half cylinders of length `0.5 r`
clamped at both ends, has its lowest frequencies well above those of the
web, and the lowest modes of the assembly are those of the web for `n =
0, 1, 1, 2, 2` nodal diameters. The null-space connection reproduces the
roots to 1e-5; the penalty connection, with the default penalty
constants, gives values up to 0.3 % lower.

With the reference surface of the cylinder at its inner face, offset `d =
h/2`, the inner face is `z_f = 0` and the face connection is the
connection at the reference surface. The same cylinder described with its
mid-surface, radius `r + h/2` and `d = 0`, connected at its lower face,
differs only by the thin-shell approximations of the laminate offset: for
a web of height `0.2 r` at mid-length of a simply supported cylinder of
length `0.5 r`, `h/r = 0.002`, the eight lowest frequencies agree to 1.3e-5
with the null-space method (1.3 % for `h/r = 0.02`), whereas the web
attached to the mid-surface, its outer radius larger by `h/2`, gives
frequencies 0.9 % lower. The penalty method differs more, 0.7 %, because
the default penalty constants depend on the stiffness of the laminate about
its reference surface, which the offset changes.

A rigid motion `\boldsymbol{t} + \boldsymbol{\omega} \times
\boldsymbol{X}` moves the points of the normal of the cylinder with the
Kirchhoff-Love kinematics and the Sanders-Koiter rotation `v/r`: the
translation along the axis, `u = 1` in the cylinder and `w = 1` in the web,
and the rotation about the axis, `v = r \Omega` in the cylinder and `v =
\rho \Omega` in the web, with `\rho = r_o - x` the radius of a point of the
web, give zero strains in both models and zero penalty energy when the
edge of the web is at the radius of the face, `\rho = r + z_f`. The
approximation functions represent these fields exactly, constants and
linear functions; the transverse translations and the tilts, `\cos\theta`
and `\sin\theta` along the circumference, are represented only
approximately and are not tested.

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, 1969, Eqs.
   (1.11)-(1.13) and (1.17), p. 2, Table 2.21, p. 23, and Tables 2.29 and
   2.31, pp. 28-29.

"""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
from scipy.special import iv, ivp, jv, jvp, kv, kvp, yv, yvp
from scipy.sparse import coo_matrix
from structsolve import freq
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.bardell import calc_vec_f, calc_vec_fp
from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import kCBFxcte_face, nullspace


E, NU, H, RHO = 70e9, 1/3, 1e-3, 2700.
R = 1.
D = E*H**3/(12*(1 - NU**2))
BETA = 0.4
#: roots of the frequency equation of Leissa (1969), Eqs. (1.11)-(1.13) and
#: (1.17), beta = 0.4, nu = 1/3, n = 0, 1, 2, clamped outside, free inside
EXACT = (13.5004, 19.3892, 31.3384)
#: roots of the same frequency equation, beta = 0.4, nu = 1/3, n = 0, 1, 2,
#: clamped inside, free outside
EXACT_INNER = (9.07188, 9.12934, 10.3655)
#: Leissa (1969), Table 2.31, after Southwell, clamped inside, free
#: outside, nu = 0.3: n, lambda^2 and the ratio b/a at which it occurs
SOUTHWELL = ((0, 6.25, 0.276), (0, 25.0, 0.642), (1, 9.00, 0.397),
             (1, 21.2, 0.603), (1, 25.0, 0.634), (2, 9.00, 0.349),
             (2, 16.0, 0.522), (3, 25.0, 0.59))
RTOL = {'null-space': 1e-5, 'penalty': 5e-3}
WEB = 'annular_clpt_donnell'
CYL = 'cylshell_clpt_sanders'


def _det(lam2, n, beta, clamped, nu=NU):
    r"""Determinant of the frequency equation of an annulus of outer radius
    1 and inner radius ``beta``, clamped on the edge ``clamped``,
    ``'outer'`` or ``'inner'``, and free on the other, Poisson's ratio
    ``nu``"""
    k = np.sqrt(lam2)
    rc, rf = (1., beta) if clamped == 'outer' else (beta, 1.)
    cols = []
    for f, fp in ((jv, jvp), (yv, yvp), (iv, ivp), (kv, kvp)):
        W0, W1c = f(n, k*rc), k*fp(n, k*rc, 1)
        W, W1, W2, W3 = (f(n, k*rf), k*fp(n, k*rf, 1),
                         k**2*fp(n, k*rf, 2), k**3*fp(n, k*rf, 3))
        b = rf
        M = W2 + nu*(W1/b - n**2*W/b**2)
        V = (W3 + W2/b - W1/b**2 - n**2*W1/b**2 + 2*n**2*W/b**3
             - (1 - nu)*n**2/b**2*(W1 - W/b))
        col = np.array([W0, W1c, M, V])
        cols.append(col/np.abs(col).max())
    return np.linalg.det(np.array(cols).T)


def exact_lambda2(n, beta, guess, clamped):
    """Root of the frequency equation of the annulus near ``guess``"""
    return brentq(_det, 0.9*guess, 1.1*guess, args=(n, beta, clamped))


def _free(p):
    for f in 'uvw':
        for e in ('x1', 'x2', 'y1', 'y2'):
            setattr(p, e + f, 1)
            setattr(p, e + f + 'r', 1)


def _cylinder(r, length, offset=0., clamped=True, h=H, m=8, n=12):
    r"""Half cylinder of reference radius ``r``, its ends `x = 0, L`
    clamped or free"""
    c = Shell(model=CYL, a=length, b=np.pi*r, r=r, m=m, n=n, plyt=h,
              stack=[0], laminaprop=(E, NU), rho=RHO, offset=offset)
    _free(c)
    if clamped:
        for f in 'uvw':
            for e in ('x1', 'x2'):
                setattr(c, e + f, 0)
                setattr(c, e + f + 'r', 0)
    return c


def _web(r_outer, r_inner, h=H, m=10, n=12):
    r"""Half annulus, free edges"""
    w = Shell(model=WEB, a=r_outer - r_inner, b=np.pi*r_outer, r=r_outer,
              m=m, n=n, plyt=h, stack=[0], laminaprop=(E, NU), rho=RHO)
    _free(w)
    return w


def _assembly(cyls, webs, conn_web, conn_method):
    r"""Two half cylinders and two half webs closed with ``'SSycte'``, the
    webs connected to the cylinders by ``conn_web(cylinder, web)``"""
    conn = []
    for p1, p2 in (cyls, webs):
        conn += [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0),
                 dict(p1=p1, p2=p2, func='SSycte', ycte1=0, ycte2=p2.b)]
    for c, w in zip(cyls, webs):
        conn.append(conn_web(c, w))
    return MultiDomain(cyls + webs, conn, conn_method=conn_method)


def _lambda2(md, r_outer, num=8):
    K = md.reduce(md.calc_kC(silent=True))
    M = md.reduce(md.calc_kM(silent=True))
    eigvals, _ = freq(K, M, sparse_solver=False, silent=True,
                      num_eigvalues=num)
    return np.sort(np.sqrt(-eigvals.real))*r_outer**2*np.sqrt(RHO*H/D)


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_web_clamped_by_cylinder_end(conn_method):
    """Web at the clamped end of a cylinder against the roots of the
    frequency equation of Leissa (1969), `\\beta = 0.4`"""
    beta = 0.4
    cyls, webs = [], []
    for k in range(2):
        c = Shell(model='cylshell_clpt_sanders', a=0.5*R, b=np.pi*R, r=R,
                  m=8, n=12, plyt=H, stack=[0], laminaprop=(E, NU), rho=RHO)
        w = Shell(model='annular_clpt_donnell', a=R*(1 - beta), b=np.pi*R,
                  r=R, m=10, n=12, plyt=H, stack=[0], laminaprop=(E, NU),
                  rho=RHO)
        for f in 'uvw':
            for e in ('x1', 'x2'):
                setattr(c, e + f, 0)
                setattr(c, e + f + 'r', 0)
            for e in ('x1', 'x2', 'y1', 'y2'):
                setattr(w, e + f, 1)
                setattr(w, e + f + 'r', 1)
            for e in ('y1', 'y2'):
                setattr(c, e + f, 1)
                setattr(c, e + f + 'r', 1)
        cyls.append(c)
        webs.append(w)
    conn = []
    for p1, p2 in (cyls, webs):
        conn += [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0),
                 dict(p1=p1, p2=p2, func='SSycte', ycte1=0, ycte2=p2.b)]
    for c, w in zip(cyls, webs):
        conn.append(dict(p1=c, p2=w, func='BFxcte', xcte1=0, xcte2=0))
    md = MultiDomain(cyls + webs, conn, conn_method=conn_method)
    K = md.reduce(md.calc_kC(silent=True))
    M = md.reduce(md.calc_kM(silent=True))
    eigvals, _ = freq(K, M, sparse_solver=False, silent=True, num_eigvalues=8)
    l2 = np.sort(np.sqrt(-eigvals.real))*R**2*np.sqrt(RHO*H/D)
    ref = np.array(EXACT)[[0, 1, 1, 2, 2]]
    assert np.allclose(l2[:5], ref, rtol=RTOL[conn_method])


@pytest.mark.parametrize('clamped', ['outer', 'inner'])
def test_exact_roots(clamped):
    """The roots of the frequency equation against :data:`EXACT` and
    :data:`EXACT_INNER`"""
    ref = EXACT if clamped == 'outer' else EXACT_INNER
    for n, value in enumerate(ref):
        assert np.isclose(exact_lambda2(n, BETA, value, clamped), value,
                          rtol=1e-5)


def test_clamped_inside_against_southwell():
    r"""The frequency equation of the annulus clamped inside and free
    outside, `\nu = 0.3`, against the ratios `\beta` of Southwell given
    by Leissa (1969), Table 2.31, within 0.006"""
    for n, lam2, beta in SOUTHWELL:
        root = brentq(lambda b: _det(lam2, n, b, 'inner', nu=0.3),
                      beta - 0.03, beta + 0.03)
        assert abs(root - beta) < 0.006


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_internal_frame_on_inner_surface(conn_method):
    """Outer edge of the web on the inner surface of a cylinder clamped at
    `x = 0`, ``face='lower'``, against the roots of the annulus clamped
    outside and free inside (Leissa 1969, Eqs. (1.11)-(1.13) and (1.17))"""
    r_o = R - H/2
    cyls = [_cylinder(R, 0.5*R) for _ in range(2)]
    webs = [_web(r_o, BETA*r_o) for _ in range(2)]
    md = _assembly(cyls, webs, lambda c, w: dict(p1=c, p2=w, func='BFxcte',
                   xcte1=0, xcte2=0, face='lower'), conn_method)
    l2 = _lambda2(md, r_o)
    ref = np.array(EXACT)[[0, 1, 1, 2, 2]]
    assert np.allclose(l2[:5], ref, rtol=RTOL[conn_method])


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_external_frame_on_outer_surface(conn_method):
    """Inner edge of the web on the outer surface of a cylinder clamped at
    `x = 0`, ``xcte2=web.a`` and ``face='upper'``, against the roots of the
    annulus clamped inside and free outside (Leissa 1969, Eqs.
    (1.11)-(1.13) and (1.17)), :data:`EXACT_INNER`"""
    r_i = R + H/2
    r_o = r_i/BETA
    cyls = [_cylinder(R, 0.5*R) for _ in range(2)]
    webs = [_web(r_o, r_i) for _ in range(2)]
    md = _assembly(cyls, webs, lambda c, w: dict(p1=c, p2=w, func='BFxcte',
                   xcte1=0, xcte2=w.a, face='upper'), conn_method)
    l2 = _lambda2(md, r_o)
    ref = np.array(EXACT_INNER)[[0, 1, 1, 2, 2]]
    assert np.allclose(l2[:5], ref, rtol=RTOL[conn_method])


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_face_at_reference_surface(conn_method):
    """With the offset `d = h/2` of the cylinder its inner face is its
    reference surface: ``face='lower'`` gives the stiffness matrix and the
    five lowest frequencies, those of the web, of the connection without
    ``'face'``"""
    out = []
    for face in (None, 'lower'):
        cyls = [_cylinder(R, 0.5*R, offset=H/2) for _ in range(2)]
        webs = [_web(R, BETA*R) for _ in range(2)]
        md = _assembly(cyls, webs, lambda c, w: dict(p1=c, p2=w,
                       func='BFxcte', xcte1=0, xcte2=0, face=face),
                       conn_method)
        out.append((md.calc_kC(silent=True).toarray(), _lambda2(md, R)))
    assert np.allclose(out[1][0], out[0][0], rtol=1e-12,
                       atol=1e-12*np.abs(out[0][0]).max())
    assert np.allclose(out[1][1][:5], out[0][1][:5], rtol=1e-6)


def test_inner_face_offset_or_mid_surface():
    """Web at mid-length of a simply supported cylinder, its outer edge on
    the inner surface, null-space method: the cylinder with its reference
    surface at that surface, offset `h/2` and no ``'face'``, against the
    cylinder with its reference surface at mid-thickness and
    ``face='lower'``, within 1e-4; a web attached to the mid-surface of the
    cylinder, its outer radius larger by `h/2`, differs by more than 0.5 %"""
    h, length, beta = 2e-3, 0.5*R, 0.8
    out = []
    for r, offset, face, r_web in ((R, h/2, None, R),
                                   (R + h/2, 0., 'lower', R),
                                   (R + h/2, 0., None, R + h/2)):
        cyls = [_cylinder(r, length, offset=offset, clamped=False, h=h,
                          m=10) for _ in range(2)]
        for c in cyls:
            for e in ('x1', 'x2'):
                setattr(c, e + 'v', 0)
                setattr(c, e + 'w', 0)
            c.x1u = 0
        webs = [_web(r_web, beta*R, h=h, m=8) for _ in range(2)]
        md = _assembly(cyls, webs, lambda c, w: dict(p1=c, p2=w,
                       func='BFxcte', xcte1=length/2, xcte2=0, face=face),
                       'null-space')
        out.append(_lambda2(md, R, num=10)[:8])
    assert np.allclose(out[1], out[0], rtol=1e-4)
    assert np.abs(out[2]/out[0] - 1).max() > 5e-3


def _flags(p, field, direction):
    return tuple(float(getattr(p, direction + e + 'uvw'[field] + s))
                 for e, s in (('1', ''), ('1', 'r'), ('2', ''), ('2', 'r')))


def _fit(p, field, func, c):
    r"""Ritz constants of ``field`` of ``p`` fitted to ``func(x, y)`` by
    least squares on a grid of Gauss-Legendre points, written into ``c``"""
    xis = leggauss(2*p.m)[0]
    etas = leggauss(2*p.n)[0]
    F = np.array([calc_vec_f(xi, *_flags(p, field, 'x'))[:p.m]
                  for xi in xis])
    G = np.array([calc_vec_f(eta, *_flags(p, field, 'y'))[:p.n]
                  for eta in etas])
    #NOTE Ritz constant of term (i, j) at 3*(j*m + i), point (k, l) at
    #     row l*len(xis) + k
    A = np.kron(G, F)
    X, Y = np.meshgrid((xis + 1)*p.a/2, (etas + 1)*p.b/2)
    values = func(X.ravel(), Y.ravel())
    coef = np.linalg.lstsq(A, values, rcond=None)[0]
    assert np.allclose(A @ coef, values, rtol=0, atol=1e-12)
    c[p.col_start + 3*np.arange(p.m*p.n) + field] = coef


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_rigid_body_motions(conn_method):
    """Free cylinder with an internal frame on its inner surface at `x = 0`
    and an external frame on its outer surface at `x = L`: the translation
    along the axis and the rotation about it give no strain energy and,
    with the null-space method, satisfy the constraints"""
    length = 0.5*R
    cyls = [_cylinder(R, length, clamped=False) for _ in range(2)]
    r_int, r_ext = R - H/2, R + H/2
    webs_int = [_web(r_int, BETA*r_int) for _ in range(2)]
    webs_ext = [_web(r_ext/BETA, r_ext) for _ in range(2)]
    conn = []
    for ps in (cyls, webs_int, webs_ext):
        conn += [dict(p1=ps[0], p2=ps[1], func='SSycte', ycte1=ps[0].b,
                      ycte2=0),
                 dict(p1=ps[0], p2=ps[1], func='SSycte', ycte1=0,
                      ycte2=ps[1].b)]
    for c, wi, we in zip(cyls, webs_int, webs_ext):
        conn += [dict(p1=c, p2=wi, func='BFxcte', xcte1=0, xcte2=0,
                      face='lower'),
                 dict(p1=c, p2=we, func='BFxcte', xcte1=length, xcte2=we.a,
                      face='upper')]
    md = MultiDomain(cyls + webs_int + webs_ext, conn,
                     conn_method=conn_method)
    K = md.calc_kC(silent=True)
    size = md.get_size()
    webs = webs_int + webs_ext
    # translation along the axis: u of the cylinder, w of the webs
    c_t = np.zeros(size)
    for p in cyls:
        _fit(p, 0, lambda x, y: np.ones_like(x), c_t)
    for p in webs:
        _fit(p, 2, lambda x, y: np.ones_like(x), c_t)
    # rotation about the axis, Omega = 1: v = r in the cylinder, v = rho =
    # r_o - x in the webs
    c_r = np.zeros(size)
    for p in cyls:
        _fit(p, 1, lambda x, y: R*np.ones_like(x), c_r)
    for p in webs:
        _fit(p, 1, lambda x, y, p=p: p.r - x, c_r)
    for c in (c_t, c_r):
        energy = c @ (K @ c)
        assert abs(energy) < 1e-10*abs(K.diagonal()).max()*(c @ c)
        if conn_method == 'null-space':
            for ci in conn:
                B = nullspace.constraint_matrix(ci, size)
                assert np.abs(B @ c).max() < 1e-10*np.abs(c).max()


def _field(p, field, x, der_x, der_y, etas, c):
    r"""``field`` of ``p``, or its first derivatives, at `x` and at the
    natural coordinates ``etas`` along `y`"""
    xi = 2*x/p.a - 1
    if der_x:
        fx = calc_vec_fp(xi, *_flags(p, field, 'x'))[:p.m]*2/p.a
    else:
        fx = calc_vec_f(xi, *_flags(p, field, 'x'))[:p.m]
    out = []
    for eta in etas:
        if der_y:
            gy = calc_vec_fp(eta, *_flags(p, field, 'y'))[:p.n]*2/p.b
        else:
            gy = calc_vec_f(eta, *_flags(p, field, 'y'))[:p.n]
        out.append(np.kron(gy, fx)
                   @ c[p.col_start + 3*np.arange(p.m*p.n) + field])
    return np.array(out)


@pytest.mark.parametrize('face', ['lower', 'upper'])
def test_penalty_energy_from_fields(face):
    r"""The penalty matrix of :func:`.kCBFxcte_face.fkCBFxcte_face` against
    the penalty energy computed from the fields of a random vector of Ritz
    constants, with the Gauss-Legendre rule along the connection"""
    kt, kr = 1.3e6, 2.1e2
    c1 = _cylinder(R, 0.5*R, clamped=False, m=6, n=7)
    if face == 'lower':
        w1 = _web(R - H/2, BETA*(R - H/2), m=5, n=6)
        xcte2 = 0.
    else:
        w1 = _web((R + H/2)/BETA, R + H/2, m=5, n=6)
        xcte2 = w1.a
    md = MultiDomain([c1, w1], [])
    size = md.get_size()
    xf = 0.3*c1.a
    z = kCBFxcte_face.face_coordinate(c1, face)
    kC = kCBFxcte_face.fkCBFxcte_face(kt, kr, c1, w1, xf, xcte2, z, size)
    K = finalize_symmetric_matrix(coo_matrix(kC).tocsr())
    c = np.random.default_rng(1).standard_normal(size)
    etas, wts = leggauss(20)
    u1 = (_field(c1, 0, xf, 0, 0, etas, c)
          - z*_field(c1, 2, xf, 1, 0, etas, c))
    v1 = ((1 + z/R)*_field(c1, 1, xf, 0, 0, etas, c)
          - z*_field(c1, 2, xf, 0, 1, etas, c))
    w1_ = _field(c1, 2, xf, 0, 0, etas, c)
    wx1 = _field(c1, 2, xf, 1, 0, etas, c)
    u2 = _field(w1, 0, xcte2, 0, 0, etas, c)
    v2 = _field(w1, 1, xcte2, 0, 0, etas, c)
    w2 = _field(w1, 2, xcte2, 0, 0, etas, c)
    wx2 = _field(w1, 2, xcte2, 1, 0, etas, c)
    integrand = (kt*((u1 - w2)**2 + (v1 - v2)**2 + (w1_ + u2)**2)
                 + kr*(wx1 - wx2)**2)
    energy = 0.5*c1.b/2*np.sum(wts*integrand)
    assert np.isclose(0.5*c @ (K @ c), energy, rtol=1e-10)


def test_geometry_errors():
    """A web whose edge is not at the radius of the connected surface, a
    face on the wrong side of the web, an unknown face and a web spanning
    another angle raise a ``ValueError``, also from :class:`.MultiDomain`"""
    c = _cylinder(R, 0.5*R)
    w_ref = _web(R, BETA*R)
    w_in = _web(R - H/2, BETA*R)
    w_out = _web((R + H/2)/BETA, R + H/2)
    w_angle = Shell(model=WEB, a=R*(1 - BETA), b=0.5*np.pi*R, r=R, m=10,
                    n=12, plyt=H, stack=[0], laminaprop=(E, NU), rho=RHO)
    bad = [dict(p1=c, p2=w_ref, func='BFxcte', xcte1=0, xcte2=0,
                face='lower'),
           dict(p1=c, p2=w_in, func='BFxcte', xcte1=0, xcte2=0),
           dict(p1=c, p2=w_out, func='BFxcte', xcte1=0, xcte2=0,
                face='upper'),
           dict(p1=c, p2=w_in, func='BFxcte', xcte1=0, xcte2=0,
                face='inner'),
           dict(p1=c, p2=w_angle, func='BFxcte', xcte1=0, xcte2=0)]
    for ci in bad:
        with pytest.raises(ValueError):
            kCBFxcte_face.check_bfxcte(ci)
    good = [dict(p1=c, p2=w_ref, func='BFxcte', xcte1=0, xcte2=0),
            dict(p1=c, p2=w_in, func='BFxcte', xcte1=0, xcte2=0,
                 face='lower'),
            dict(p1=c, p2=w_out, func='BFxcte', xcte1=0, xcte2=w_out.a,
                 face='upper')]
    assert np.allclose([kCBFxcte_face.check_bfxcte(ci) for ci in good],
                       [0., -H/2, H/2], rtol=0, atol=1e-15)
    for conn_method in ('null-space', 'penalty'):
        md = MultiDomain([c, w_ref], [bad[0]], conn_method=conn_method)
        with pytest.raises(ValueError):
            md.reduce(md.calc_kC(silent=True))
