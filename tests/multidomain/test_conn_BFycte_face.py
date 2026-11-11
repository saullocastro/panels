r"""Blades connected to a face of a cylinder with ``'BFycte'``

Verified, for the key ``'face'`` of the connection ``'BFycte'``, see
:mod:`panels.multidomain.connections.kCBFycte_face`:

- the rigid-body motions of a closed cylinder with an internal blade on its
  inner surface and an external blade on its outer surface that the
  approximation functions represent exactly, the translation along the
  axis and the rotation about it, give no strain energy and no penalty
  energy, and satisfy the constraints of the null-space method, for the
  classical (CLPT), first-order (FSDT) and third-order (TSDT) theories and
  for a cylinder with the Sanders-Koiter and with the Donnell kinematics,
  the face following the Sanders-Koiter kinematics by default;
- with ``kinematics='model'`` a Donnell cylinder issues a ``UserWarning``
  and the rotation about the axis leaves the residuals `-z_f \Omega` of
  `v_1(z_f) = w_2` and `-\Omega` of the rotation about `x`;
- the equivalent offset model: a cylinder whose reference surface is its
  inner face, offset `h/2`, connected at its lower face, `z_f = 0`, gives
  the frequencies of the connection at the reference surface, the kernels
  of :mod:`panels.multidomain.connections.kCBFycte`, to 1e-9, with both
  methods;
- the penalty energy of :func:`.kCBFycte_face.fkCBFycte_face` equals that
  of the constraints of :func:`.nullspace.constraint_matrix`;
- the key ``'kinematics'`` without the key ``'face'`` raises a
  ``ValueError`` for ``'BFycte'`` and ``'BFxcte'``, with both methods, and
  so does a geometry that does not match.

Notes
-----
The local `y` axis of a blade connected with ``'BFycte'`` points towards the
axis of the cylinder, so that its edge `y_2 = 0` is the root of an internal
blade, on the inner surface, ``face='lower'``, and its edge `y_2 = b_2` the
root of an external blade, on the outer surface, ``face='upper'``. A rigid
rotation `\Omega` about the axis gives `v = r \Omega` in the cylinder and,
in a blade, whose `z` axis is the circumferential direction, `w_2 = \rho
\Omega`, `\rho` being the radius of a point of the blade, `\rho = r + z_f -
y_2` for the internal blade and `\rho = r + z_f + b_2 - y_2` for the
external one: the root of the blade is at the radius of the face, `r +
z_f`. With the Sanders-Koiter kinematics at the face, `v_1(z_f) = (1 +
z_f/r) v_1 - z_f w_{1,y} = (r + z_f) \Omega` and the rotation of the normal
about `x`, `-w_{1,y} + v_1/r`, is `\Omega`, that of the blade, `-w_{2,y} =
\Omega` (`\phi_y = \Omega` for the FSDT and TSDT plates, `\phi_y = 0` in
the cylinder, whose rotation is `\phi_y + v/r`). The transverse
translations and the tilts, `\cos\theta` and `\sin\theta` along the
circumference, are represented only approximately and are not tested.

With the reference surface of the skin at its inner face, offset `d = h/2`,
the lower face is `z_f = d - h/2 = 0`, and the connection at that face must
be the connection at the reference surface: the face connection of
:func:`.kCBFycte_face.fkCBFycte_face` and of the null-space constraints then
reproduces the kernels of :mod:`panels.multidomain.connections.kCBFycte`
for a skin with the Sanders-Koiter kinematics.

"""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix
from structsolve import freq
from structsolve.sparseutils import finalize_symmetric_matrix

import panels.modelDB as modelDB
from panels.bardell import calc_vec_f
from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import kCBFycte_face, nullspace


E, NU, H, RHO = 70e9, 0.3, 2e-3, 2700.
R = 1.
L = 0.5
HB = 0.05
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
#: models of the cylinder and of the blades
MODELS = [('cylshell_clpt_sanders', 'plate_clpt_donnell'),
          ('cylshell_clpt_donnell', 'plate_clpt_gl'),
          ('cylshell_fsdt_sanders', 'plate_fsdt_donnell'),
          ('cylshell_fsdt_donnell', 'plate_fsdt_donnell'),
          ('cylshell_tsdt_sanders', 'plate_tsdt_donnell'),
          ('cylshell_tsdt_donnell', 'plate_tsdt_donnell')]


def _fields(p):
    return FIELDS[:modelDB.db[p.model]['dofs']]


def _set(p, ends, value):
    for e in ends:
        for f in _fields(p):
            setattr(p, e + f, value)
            setattr(p, e + f + 'r', 1)


def _skin(model, r=R, offset=0., m=6, n=8, supported=False):
    r"""Half cylinder of reference radius ``r``, free edges or, with
    ``supported``, `u = v = w = 0` at the ends"""
    p = Shell(model=model, a=L, b=np.pi*r, r=r, m=m, n=n, plyt=H, stack=[0],
              laminaprop=(E, NU), rho=RHO, offset=offset)
    _set(p, ('x1', 'x2', 'y1', 'y2'), 1)
    if supported:
        for e in ('x1', 'x2'):
            for f in 'uvw':
                setattr(p, e + f, 0)
    return p


def _blade(model, hb=HB, m=6, n=6, supported=False):
    p = Shell(model=model, a=L, b=hb, m=m, n=n, plyt=H, stack=[0],
              laminaprop=(E, NU), rho=RHO)
    _set(p, ('x1', 'x2', 'y1', 'y2'), 1)
    if supported:
        for e in ('x1', 'x2'):
            for f in 'uvw':
                setattr(p, e + f, 0)
    return p


def _closed(skins):
    p1, p2 = skins
    return [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0),
            dict(p1=p1, p2=p2, func='SSycte', ycte1=0, ycte2=p2.b)]


def _cylinder(models, conn_method=None, kinematics=None):
    r"""Closed free cylinder of two halves, an internal blade on the inner
    face of the first and an external blade on the outer face of the
    second, both at `y = 0`"""
    skins = [_skin(models[0]) for _ in range(2)]
    blades = [_blade(models[1]) for _ in range(2)]
    extra = {} if kinematics is None else dict(kinematics=kinematics)
    faces = [dict(p1=skins[0], p2=blades[0], func='BFycte', ycte1=0,
                  ycte2=0, face='lower', **extra),
             dict(p1=skins[1], p2=blades[1], func='BFycte', ycte1=0,
                  ycte2=blades[1].b, face='upper', **extra)]
    kwargs = {} if conn_method is None else dict(conn_method=conn_method)
    md = MultiDomain(skins + blades, _closed(skins) + faces, **kwargs)
    return md, skins, blades, faces


def _flags(p, field, direction):
    return tuple(float(getattr(p, direction + edge + field + r))
                 for edge, r in (('1', ''), ('1', 'r'), ('2', ''), ('2', 'r')))


def _fit(p, field, func, c):
    r"""Ritz constants of the field number ``field`` of ``p`` fitted to
    ``func(x, y)`` on a grid of Gauss-Legendre points, written into ``c``"""
    name = FIELDS[field]
    xis = leggauss(2*p.m)[0]
    etas = leggauss(2*p.n)[0]
    F = np.array([calc_vec_f(xi, *_flags(p, name, 'x'))[:p.m] for xi in xis])
    G = np.array([calc_vec_f(eta, *_flags(p, name, 'y'))[:p.n]
                  for eta in etas])
    A = np.kron(G, F)
    X, Y = np.meshgrid((xis + 1)*p.a/2, (etas + 1)*p.b/2)
    values = func(X.ravel(), Y.ravel())
    coef = np.linalg.lstsq(A, values, rcond=None)[0]
    assert np.allclose(A @ coef, values, rtol=0, atol=1e-12)
    dofs = modelDB.db[p.model]['dofs']
    c[p.col_start + dofs*np.arange(p.m*p.n) + field] = coef


def _rigid(md, skins, blades, faces):
    r"""Ritz constants of the translation along the axis and of the
    rotation `\Omega = 1` about it, see the Notes of the module"""
    size = md.get_size()
    c_t = np.zeros(size)
    for p in skins + blades:
        _fit(p, 0, lambda x, y: np.ones_like(x), c_t)
    c_r = np.zeros(size)
    for p in skins:
        _fit(p, 1, lambda x, y: R*np.ones_like(x), c_r)
    for p, ci in zip(blades, faces):
        z = kCBFycte_face.face_coordinate(ci['p1'], ci['face'])
        root = R + z + ci['ycte2']
        _fit(p, 2, lambda x, y, root=root: root - y, c_r)
        if modelDB.db[p.model]['dofs'] == 5:
            _fit(p, 4, lambda x, y: np.ones_like(x), c_r)
    return c_t, c_r


@pytest.mark.parametrize('models', MODELS)
@pytest.mark.parametrize('conn_method', [None, 'penalty'])
def test_rigid_body_motions(conn_method, models):
    r"""Translation along the axis and rotation about it: no strain energy,
    no penalty energy and, with the default ``conn_method`` of
    :class:`.MultiDomain`, ``'null-space'``, no residual of the
    constraints"""
    md, skins, blades, faces = _cylinder(models, conn_method)
    if conn_method is None:
        assert md.conn_method == 'null-space'
        assert all(md._null_space(ci) for ci in faces)
    K = md.calc_kC(silent=True)
    size = md.get_size()
    for c in _rigid(md, skins, blades, faces):
        energy = c @ (K @ c)
        assert abs(energy) < 1e-10*abs(K.diagonal()).max()*(c @ c)
        if conn_method is None:
            for ci in md.conn:
                B = nullspace.constraint_matrix(ci, size)
                assert np.abs(B @ c).max() < 1e-10*np.abs(c).max()


def test_donnell_kinematics_of_the_model():
    r"""``kinematics='model'`` at the faces of a Donnell cylinder: a
    ``UserWarning``, and the rotation about the axis leaves the residual
    `-z_f \Omega` of `v_1(z_f) = w_2` and `-\Omega` of the rotation, the
    other quantities satisfied"""
    models = ('cylshell_clpt_donnell', 'plate_clpt_donnell')
    md, skins, blades, faces = _cylinder(models, 'penalty',
                                         kinematics='model')
    size = md.get_size()
    _, c_r = _rigid(md, skins, blades, faces)
    for ci in faces:
        with pytest.warns(UserWarning, match='Donnell cylinder'):
            B = nullspace.constraint_matrix(ci, size)
        q = (B @ c_r).reshape(4, -1)
        z = kCBFycte_face.face_coordinate(ci['p1'], ci['face'])
        assert np.allclose(q[0], 0, atol=1e-10)
        assert np.allclose(q[1], -z, rtol=1e-8, atol=0)
        assert np.allclose(q[2], 0, atol=1e-10)
        assert np.allclose(q[3], -1., rtol=1e-8, atol=0)


def _freqs(md, num=8):
    r"""Eigenvalues of :func:`structsolve.freq`, dense solver"""
    K = md.reduce(md.calc_kC(silent=True))
    M = md.reduce(md.calc_kM(silent=True))
    eigvals, _ = freq(K, M, sparse_solver=False, silent=True,
                      num_eigvalues=num)
    return np.sort(eigvals.real)


def _supported(model_skin, conn_method, skin_kw, conn_kw, hb=HB):
    r"""Closed cylinder of two halves supported at the ends, `u = v = w =
    0`, with an internal blade at `y = 0` of each half"""
    skins = [_skin(model_skin, supported=True, **skin_kw) for _ in range(2)]
    blades = [_blade('plate_clpt_donnell', hb=hb, supported=True)
              for _ in range(2)]
    conn = _closed(skins) + [dict(p1=s, p2=b, func='BFycte', ycte1=0,
                                  ycte2=0, **conn_kw)
                             for s, b in zip(skins, blades)]
    return MultiDomain(skins + blades, conn, conn_method=conn_method)


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_face_at_reference_surface(conn_method):
    r"""Reference surface of the cylinder at its inner face, offset `h/2`:
    ``face='lower'`` is `z_f = 0`, and the frequencies are those of the
    connection at the reference surface, kernels of
    :mod:`panels.multidomain.connections.kCBFycte`, to 1e-9"""
    skin_kw = dict(r=R - H/2, offset=H/2)
    ref = _freqs(_supported('cylshell_clpt_sanders', conn_method, skin_kw,
                            {}))
    face = _freqs(_supported('cylshell_clpt_sanders', conn_method, skin_kw,
                             dict(face='lower')))
    assert np.allclose(face, ref, rtol=1e-9, atol=0)


@pytest.mark.parametrize('face', ['lower', 'upper'])
@pytest.mark.parametrize('models,kinematics',
        [(MODELS[0], 'sanders'), (MODELS[1], 'sanders'),
         (MODELS[1], 'model'), (MODELS[4], 'sanders')])
def test_penalty_and_null_space_agree(models, kinematics, face):
    r"""The penalty energy of :func:`.kCBFycte_face.fkCBFycte_face`, `k_t =
    k_r = 1`, equals that of the constraints of
    :func:`.nullspace.constraint_matrix`, `\frac{1}{2} \frac{a_1}{2} \sum_k
    \sum_l w_l q_k(\xi_l)^2`, with the Gauss-Legendre weights `w_l` of their
    points, for a random vector of Ritz constants"""
    s = _skin(models[0], m=7, n=6)
    b = _blade(models[1], m=5, n=4)
    md = MultiDomain([s, b], [])
    size = md.get_size()
    ycte1 = 0.3*s.b
    ycte2 = 0. if face == 'lower' else b.b
    ci = dict(p1=s, p2=b, func='BFycte', ycte1=ycte1, ycte2=ycte2,
              face=face, kinematics=kinematics)
    z = kCBFycte_face.face_coordinate(s, face)
    kC = kCBFycte_face.fkCBFycte_face(1., 1., s, b, ycte1, ycte2, z, size,
                                      kinematics=kinematics)
    K = finalize_symmetric_matrix(coo_matrix(kC).tocsr())
    if kinematics == 'model':
        with pytest.warns(UserWarning, match='Donnell cylinder'):
            B = nullspace.constraint_matrix(ci, size)
    else:
        B = nullspace.constraint_matrix(ci, size)
    c = np.random.default_rng(3).standard_normal(size)
    q = (B @ c).reshape(4, -1)
    wts = leggauss(q.shape[1])[1]
    assert np.isclose(0.5*c @ (K @ c), 0.5*s.a/2*np.sum(wts*q**2),
                      rtol=1e-10)


@pytest.mark.parametrize('func', ['BFycte', 'BFxcte'])
@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_kinematics_without_face(conn_method, func):
    r"""The key ``'kinematics'`` without the key ``'face'`` raises a
    ``ValueError``; ``kinematics=None`` is accepted"""
    s = _skin('cylshell_clpt_sanders')
    b = _blade('plate_clpt_donnell')
    if func == 'BFycte':
        ci = dict(p1=s, p2=b, func=func, ycte1=0, ycte2=0)
    else:
        b.b = s.b
        ci = dict(p1=s, p2=b, func=func, xcte1=0, xcte2=0)
    md = MultiDomain([s, b], [dict(ci, kinematics='sanders')],
                     conn_method=conn_method)
    with pytest.raises(ValueError, match="requires the key 'face'"):
        if conn_method == 'penalty':
            md.calc_kC(silent=True)
        else:
            md.get_Z()
    md = MultiDomain([s, b], [dict(ci, kinematics=None)],
                     conn_method=conn_method)
    md.calc_kC(silent=True)


def test_geometry_errors():
    r"""A root of the blade on the side opposite to the face, connected
    lines of different lengths or an unknown face raise a ``ValueError``"""
    s = _skin('cylshell_clpt_sanders')
    b = _blade('plate_clpt_donnell')
    ci = dict(p1=s, p2=b, func='BFycte', ycte1=0)
    with pytest.raises(ValueError, match='ycte2'):
        kCBFycte_face.check_bfycte(dict(ci, ycte2=b.b, face='lower'))
    with pytest.raises(ValueError, match='ycte2'):
        kCBFycte_face.check_bfycte(dict(ci, ycte2=0, face='upper'))
    with pytest.raises(ValueError, match="'face' of a 'BFycte'"):
        kCBFycte_face.check_bfycte(dict(ci, ycte2=0, face='middle'))
    b2 = _blade('plate_clpt_donnell')
    b2.a = 0.9*L
    with pytest.raises(ValueError, match='same length'):
        kCBFycte_face.check_bfycte(dict(ci, p2=b2, ycte2=0, face='lower'))
    assert kCBFycte_face.check_bfycte(dict(ci, ycte2=0)) == 0.
    assert np.isclose(kCBFycte_face.check_bfycte(
                      dict(ci, ycte2=0, face='lower')), -H/2)
