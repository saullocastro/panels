r"""Verification of the total-Lagrangian Kirchhoff-Love plate, ``plate_clpt_tl``

The model has the Green-Lagrange membrane strains of ``plate_clpt_gl``
and the changes of curvature of the deformed mid-surface. The checks:

1. a rigid rotation of a free plate, in plane, out of plane or about an
   oblique axis, by up to 1 rad, gives zero strains, curvatures and internal
   forces;
2. the tangent stiffness matrix ``calc_kC + calc_kG`` is the central
   difference of ``calc_fint`` at a state with a rotation of 1 rad, and so
   is ``calc_kTv``;
3. a bent plate rotated rigidly by `\theta` keeps its strains, hence its
   bending energy, whereas the bending energy of ``plate_clpt_gl`` is
   multiplied by `\cos^2\theta`: the purpose of the model;
4. the linear kernels are those of ``plate_clpt_donnell``, the geometric
   stiffness of a constant membrane stress that of ``plate_clpt_gl``, and for
   small deflections the non-linear response is that of
   ``plate_clpt_gl`` and ``plate_clpt_donnell``;
5. a cantilever strip under a tip force of constant direction follows the
   inextensible elastica, bent out of plane as well as in plane;
6. the default integration, `3p` points along a direction, is exact.

The tangent stiffness matrix, ``calc_kTv`` and the follower pressures are
also checked with the other models in ``test_tangent_consistency.py``,
``test_tangent_product.py`` and ``test_follower_pressure.py``.

Notes
-----
The changes of curvature are measured with the area vector
`\boldsymbol{a} = \boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y}` of the
deformed mid-surface:

.. math::

    \kappa_{xx} = -\boldsymbol{a} \cdot \boldsymbol{u}_{,xx}, \quad
    \kappa_{yy} = -\boldsymbol{a} \cdot \boldsymbol{u}_{,yy}, \quad
    \kappa_{xy} = -2 \boldsymbol{a} \cdot \boldsymbol{u}_{,xy}

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from structsolve import Analysis, lb, solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels import modelDB
from panels.models import clpt_field
from panels.shell import Shell, gauss_points

from test_plate_clpt_gl import (plate, rigid_rotation, AXES, elastica,
                                cantilever_strip)

TL = 'plate_clpt_tl'
GL = 'plate_clpt_gl'
VK = 'plate_clpt_donnell'


def grid(s, nx=9, ny=7):
    xs = np.linspace(0, s.a, nx)
    ys = np.linspace(0, s.b, ny)
    return [np.ascontiguousarray(v.ravel()) for v in np.meshgrid(xs, ys)]


def fit(s, U, X, Y):
    r"""Ritz constants of the displacements ``U`` (3 x points) by least
    squares, exact for polynomials of degree up to three"""
    size = s.get_size()
    A = np.zeros((3*X.size, size))
    for k in range(size):
        e = np.zeros(size)
        e[k] = 1.
        A[0::3, k], A[1::3, k], A[2::3, k] = clpt_field.fuvw(e, s, X, Y, 1)[:3]
    U = np.asarray(U).T.ravel()
    c = np.linalg.lstsq(A, U, rcond=None)[0]
    assert np.abs(A@c - U).max() < 1e-12*max(1., np.abs(U).max())
    return c


def bent_and_rotated(s, axis, angle):
    r"""A bent state ``w = k (x^2 + x y)/2``, with ``w`` at most 0.2 times the
    thickness, and the same state rotated rigidly, ``X -> R X``"""
    X, Y = grid(s)
    h = sum(s.plyts)
    k = 0.2*h/s.a**2
    X0 = np.vstack([X, Y, 0*X])
    Ub = np.vstack([0*X, 0*X, k*(X**2 + X*Y)/2])
    R = Rotation.from_rotvec(np.asarray(axis)/np.linalg.norm(axis)*angle).as_matrix()
    Ur = R@(X0 + Ub) - X0
    return fit(s, Ub, X, Y), fit(s, Ur, X, Y)


def energies(s, c, n=12):
    r"""Membrane and bending strain energies by Gauss-Legendre quadrature of
    the strains of :func:`.clpt_field.fstrain`, the laminate being symmetric
    (``B = 0``)"""
    s._rebuild()
    s.r = 0.
    xi, wi = np.polynomial.legendre.leggauss(n)
    X, Y = [np.ascontiguousarray(v.ravel()) for v in
            np.meshgrid((xi + 1)/2*s.a, (xi + 1)/2*s.b)]
    W = np.outer(wi, wi).ravel()*s.a*s.b/4
    e = np.array(clpt_field.fstrain(c, s, X, Y, 1, 1))
    A, D = s.ABD[:3, :3], s.ABD[3:, 3:]
    assert np.abs(s.ABD[:3, 3:]).max() < 1e-12*np.abs(A).max()
    Em = 0.5*np.einsum('ip,ij,jp,p->', e[:3], A, e[:3], W)
    Eb = 0.5*np.einsum('ip,ij,jp,p->', e[3:], D, e[3:], W)
    return Em, Eb, e


@pytest.mark.parametrize('axis', list(AXES))
@pytest.mark.parametrize('angle', [0.1, 0.5, 1.0])
def test_rigid_rotation_zero_strains(axis, angle):
    s = plate(TL, stack=[0, 45], free=True, m=6, n=5)
    c, X, Y = rigid_rotation(s, AXES[axis], angle)
    fint = s.calc_fint(c)
    # reference: the internal forces of the uniform strain exx = 1e-3
    ce = np.zeros_like(c)
    ce[0::3] = np.linalg.lstsq(
        np.column_stack([clpt_field.fuvw(np.eye(s.get_size())[k], s, X, Y, 1)[0]
                         for k in range(0, s.get_size(), 3)]),
        1e-3*X, rcond=None)[0]
    fe = s.calc_fint(ce)
    strains = np.array(clpt_field.fstrain(c, s, X, Y, 1, 1))
    assert np.abs(fint).max() < 1e-9*np.abs(fe).max()
    assert np.abs(strains[:3]).max() < 1e-12
    # round-off of the second derivatives of the fitted state, 1e-9 of the
    # curvature h/a^2 of a deflection of the order of the thickness
    assert np.abs(strains[3:]).max() < 1e-11


def test_tangent_is_central_difference_at_large_rotation():
    r"""Every column of ``calc_kC(c, NLgeom=True) + calc_kG(c, NLgeom=True)``,
    and of ``calc_kTv``, against the central difference of ``calc_fint``, at
    a bent and stretched state rotated by 1 rad about an oblique axis plus a
    random perturbation, with an unsymmetric laminate (``B != 0``)"""
    s = plate(TL, stack=[0, 45, 30], free=True, m=5, n=4)
    X, Y = grid(s)
    h = sum(s.plyts)
    X0 = np.vstack([X, Y, 0*X])
    U = np.vstack([2e-3*X, -1e-3*Y, 3*h*(X**2/s.a**2 - X*Y/(s.a*s.b))])
    R = Rotation.from_rotvec(np.array([1., 2., 3.])/np.sqrt(14)).as_matrix()
    c = fit(s, R@(X0 + U) - X0, X, Y)
    rng = np.random.default_rng(3)
    c += 1e-2*h*rng.standard_normal(c.size)
    KT = (s.calc_kC(c=c, NLgeom=True) + s.calc_kG(c=c, NLgeom=True)).toarray()
    N = c.size
    step = 1e-7*np.abs(c).max()
    FD = np.zeros((N, N))
    for j in range(N):
        dc = np.zeros(N)
        dc[j] = step
        FD[:, j] = (s.calc_fint(c + dc) - s.calc_fint(c - dc))/(2*step)
    assert np.abs(KT - FD).max() < 1e-6*np.abs(KT).max()
    v = rng.standard_normal(N)
    kTv = s.calc_kTv(c, v)
    assert np.linalg.norm(kTv - KT@v) < 1e-12*np.linalg.norm(KT@v)


@pytest.mark.parametrize('axis', list(AXES))
@pytest.mark.parametrize('angle', [0.5, 1.0])
def test_bent_plate_rotated_keeps_its_bending_energy(axis, angle):
    r"""The strains of a bent plate are invariant under a rigid rotation
    `\boldsymbol{X} \to \boldsymbol{R} \boldsymbol{X}`, pointwise, so its
    membrane and bending energies too. With ``plate_clpt_gl`` the membrane
    energy is invariant as well, but the curvatures `-w_{,\alpha\beta}` of
    the fixed frame are those of the bent state times `\cos\theta` for a
    rotation about an in-plane axis, and the bending energy that times
    `\cos^2\theta`, 0.77 at 0.5 rad and 0.29 at 1 rad; about the normal it is
    invariant, and about the oblique axis `(1, 2, 3)/\sqrt{14}` the factor
    is that of the normal of the rotated plane, `(R_{33})^2`"""
    out = {}
    for model in (TL, GL):
        s = plate(model, stack=[0, 90, 0], free=True, m=6, n=5)
        cb, cr = bent_and_rotated(s, AXES[axis], angle)
        Emb, Ebb, eb = energies(s, cb)
        Emr, Ebr, er = energies(s, cr)
        out[model] = (Emb, Ebb, Emr, Ebr, eb, er)
    Emb, Ebb, Emr, Ebr, eb, er = out[TL]
    # bending dominates, the membrane energy is that of w,x^2/2
    assert Emb < 0.1*Ebb
    # the membrane strains, at most 3e-9, are the difference of gradients
    # of order one in the rotated state, hence the round-off of 1e-12
    assert np.abs(er[:3] - eb[:3]).max() < 1e-11
    # and the curvatures that of the second derivatives of the fitted
    # displacements, of order a
    assert np.abs(er[3:] - eb[3:]).max() < 1e-8*np.abs(eb[3:]).max()
    assert np.isclose(Ebr, Ebb, rtol=1e-8)
    assert np.isclose(Emr, Emb, rtol=1e-5)
    Emb, Ebb, Emr, Ebr, eb, er = out[GL]
    n = np.asarray(AXES[axis], float)/np.linalg.norm(AXES[axis])
    R = Rotation.from_rotvec(n*angle).as_matrix()
    assert np.isclose(Emr, Emb, rtol=1e-5)
    assert np.isclose(Ebr, R[2, 2]**2*Ebb, rtol=1e-8)
    if axis != 'in-plane z':
        assert Ebr < 0.95*Ebb


def test_linear_kernels():
    r"""The linear stiffness and mass are those of ``plate_clpt_donnell``,
    the geometric stiffness of a constant membrane stress state that of
    ``plate_clpt_gl``, analytical (``fkG0``) and numerical, and the
    geometric stiffness of a linear state with moments has the ``u-w`` and
    ``v-w`` blocks of the moments besides those of ``plate_clpt_gl``"""
    tl = plate(TL, stack=[30, -45, 0, 90])
    gl = plate(GL, stack=[30, -45, 0, 90])
    vk = plate(VK, stack=[30, -45, 0, 90])
    size = tl.get_size()
    rel = lambda A, B: abs(A - B).max()/abs(B).max()
    assert rel(tl.calc_kC(), vk.calc_kC()) == 0
    assert rel(tl.calc_kC(c=np.zeros(size), NLgeom=True), vk.calc_kC()) < 1e-12
    assert rel(tl.calc_kM(), vk.calc_kM()) == 0
    for s in (tl, gl):
        s.Nxx, s.Nyy, s.Nxy = -100., 30., 20.
    assert rel(tl.calc_kG(), gl.calc_kG()) == 0
    num = modelDB.db[TL]['matrices_num']
    kG_num = finalize_symmetric_matrix(num.fkG_num(np.zeros(size), tl.ABD, tl, size,
                                       0, 0, tl.nx, tl.ny, -100., 30., 20.))
    assert rel(kG_num, gl.calc_kG()) < 1e-12
    for s in (tl, gl):
        s.Nxx, s.Nyy, s.Nxy = 0., 0., 0.
    # a bent state: the moments couple u, v with w, through the quadratic
    # terms w,x u,xx + w,y v,xx - (u,x + v,y) w,xx of kxx and their
    # analogues, the other blocks being those of plate_clpt_gl; homogeneous
    # of degree one in c
    rng = np.random.default_rng(1)
    c = 1e-4*rng.standard_normal(size)
    kG_tl = tl.calc_kG(c=c).toarray()
    kG_gl = gl.calc_kG(c=c).toarray()
    w = np.arange(2, size, 3)
    uv = np.setdiff1d(np.arange(size), w)
    assert np.abs(kG_gl[np.ix_(uv, w)]).max() == 0
    assert np.abs(kG_tl[np.ix_(uv, w)]).max() > 1e-6*np.abs(kG_tl).max()
    for idx in (w, uv):
        assert rel(kG_tl[np.ix_(idx, idx)], kG_gl[np.ix_(idx, idx)]) < 1e-12
    assert rel(tl.calc_kG(c=2*c), 2*tl.calc_kG(c=c)) < 1e-12
    # without moments, a membrane state of a symmetric laminate, the
    # blocks of plate_clpt_gl
    tl = plate(TL)
    gl = plate(GL)
    c_m = 1e-4*rng.standard_normal(tl.get_size())
    c_m[2::3] = 0.
    assert rel(tl.calc_kG(c=c_m), gl.calc_kG(c=c_m)) < 1e-12


@pytest.mark.parametrize('stack', [(0, 90, -45, 45, 45, -45, 90, 0),
                                   (30, -45, 0, 90)])
def test_linear_buckling(stack):
    r"""With the constant stress of ``fkG0`` the eigenvalue is that of
    ``plate_clpt_gl``; from the stresses of a linear static solution it is
    the same for the symmetric laminate, whose prebuckling state has no
    moments, and differs by the order of the membrane strain for the
    unsymmetric one, through the moments"""
    eig = {}
    for model in (TL, GL):
        s = plate(model, stack=stack, m=10, n=10)
        s.Nxx = -1.
        eig[model, 'fkG0'] = lb(s.calc_kC(), s.calc_kG(), silent=True)[0][0]
        s.Nxx = 0.
        s.x2u = 1.
        s.add_distr_load_fixed_x(s.a, funcx=lambda y: -1., funcy=None,
                                 funcz=None, cte=True)
        k0 = s.calc_kC()
        c0 = solve(k0, s.calc_fext(), silent=True)
        eig[model, 'fkG_num'] = lb(k0, s.calc_kG(c=c0), silent=True)[0][0]
    print(eig)
    assert np.isclose(eig[TL, 'fkG0'], eig[GL, 'fkG0'], rtol=1e-12)
    tol = 1e-10 if stack[0] == 0 else 1e-4
    assert np.isclose(eig[TL, 'fkG_num'], eig[GL, 'fkG_num'], rtol=tol)


@pytest.mark.parametrize('p, wtol', [(1.e2, 3e-5), (1.e3, 2e-4), (2.e4, 2e-3)])
def test_small_deflection_agrees_with_gl_and_von_karman(p, wtol):
    r"""Simply supported plate under a uniform pressure, centre deflections of
    about 1.1, 3.4 and 11 times the thickness: the three models agree. The
    non-linear terms of the curvatures of ``plate_clpt_tl``, ``w,x u,xx +
    w,y v,xx - (u,x + v,y) w,xx`` for ``kxx``, are of the order of the
    membrane strain times the curvature, and make the deflection smaller than
    that of ``plate_clpt_gl`` by about 2e-5, 1.1e-4 and 1.3e-3 (the membrane
    strains, about ``(w/a)^2``, are 3e-5, 3e-4 and 3e-3)"""
    res = {}
    for model in (TL, GL, VK):
        s = plate(model)
        s.add_pressure_load(p, cte=False)
        an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
        an.initialInc = 0.25
        an.maxInc = 0.25
        an.static(NLgeom=True, silent=True)
        assert np.isclose(an.increments[-1], 1.)
        c = an.cs[-1]
        res[model] = s.uvw(c, xs=np.array([s.a/2]), ys=np.array([s.b/2]))[1]['w'].max()
    print(p, res)
    assert np.isclose(res[TL], res[GL], rtol=wtol)
    assert np.isclose(res[TL], res[VK], rtol=wtol)


@pytest.mark.parametrize('alpha, inc', [(1., 0.1), (3., 0.05)])
def test_cantilever_strip_out_of_plane_against_elastica(alpha, inc):
    r"""Out-of-plane bending of the strip of ``test_plate_clpt_gl.py`` (``L =
    1``, ``b = 0.04``, ``t = 2e-3``, ``nu = 0``, ``m = 14``), ``alpha = P
    L^2/EI = 1`` and ``3``, tip rotations of 0.46 and 1.0 rad: the elastica
    gives ``w = 0.30172 L``, ``u = -0.05643 L`` and ``w = 0.6033 L``, ``u =
    -0.2554 L``, against ``alpha/3 L`` of the linear beam followed by
    ``plate_clpt_gl``. The tolerance, 1e-4, is set by the Ritz
    approximation of the slope with 14 terms (the observed difference is
    about 1e-6 at ``alpha = 1``); the axial strain, of the order of
    ``alpha t^2/(12 L^2) = 3e-7 alpha``, and with it the stretch of the area
    vector ``|a| - 1`` and the extensibility of the strip, are smaller"""
    w_ref, u_ref, th_ref = elastica(alpha)
    tl = cantilever_strip(TL, 'out-of-plane', alpha, inc)
    print('elastica', (w_ref, u_ref, th_ref), TL, tl)
    assert np.isclose(tl[0], w_ref, rtol=1e-4)
    assert np.isclose(tl[1], u_ref, rtol=1e-4)


def test_cantilever_strip_in_plane_against_elastica():
    r"""In-plane bending, ``alpha = 2``, carried by the membrane strains: the
    same as ``plate_clpt_gl``, ``v = 0.4935 L``, ``u = -0.1606 L``"""
    w_ref, u_ref, th_ref = elastica(2.)
    tl = cantilever_strip(TL, 'in-plane', 2., 0.1)
    gl = cantilever_strip(GL, 'in-plane', 2., 0.1)
    print('elastica', (w_ref, u_ref), TL, tl, GL, gl)
    assert np.isclose(tl[0], w_ref, rtol=0.005)
    assert np.isclose(tl[1], u_ref, rtol=0.01)
    assert np.isclose(tl[0], gl[0], rtol=1e-4)


def test_default_integration_is_exact():
    r"""The default ``nx = 3 p`` of ``plate_clpt_tl`` (21 points for ``m =
    8``, against 15 for ``plate_clpt_gl``) integrates ``calc_fint`` and the
    tangent stiffness matrix exactly: more points change nothing. At this
    state, with gradients of order 0.5, the 15 x 9 points of
    ``plate_clpt_gl`` are not exact but within 1e-12, the terms of degree
    above 29 being products of five or six of the highest functions, whereas
    10 x 6 points are off by 1e-5"""
    assert [gauss_points(m, (6, -2)) for m in (1, 4, 5, 8, 15)] == [9, 9, 12, 21, 42]
    s = plate(TL, stack=[0, 45, 30], free=True, m=8, n=5)
    assert (s.nx, s.ny) == (21, 12)
    s2 = plate(GL, m=8, n=5)
    assert (s2.nx, s2.ny) == (15, 9)
    # every term of the approximation with a large amplitude, gradients of
    # order 0.5, such that the products of six functions carry weight
    rng = np.random.default_rng(0)
    c = 0.1*rng.standard_normal(s.get_size())

    def run(nx, ny):
        f = s.calc_fint(c, nx=nx, ny=ny)
        k = (s.calc_kC(c=c, NLgeom=True, nx=nx, ny=ny)
             + s.calc_kG(c=c, NLgeom=True, nx=nx, ny=ny)).toarray()
        return f, k

    f0, k0 = run(None, None)
    f1, k1 = run(s.nx + 3, s.ny + 3)
    f2, k2 = run(15, 9)
    f3, k3 = run(10, 6)
    rel = lambda a, b: np.abs(a - b).max()/np.abs(b).max()
    assert rel(f0, f1) < 1e-13
    assert rel(k0, k1) < 1e-13
    assert rel(f2, f1) < 1e-11
    assert rel(k2, k1) < 1e-11
    assert rel(f3, f1) > 1e-6
    assert rel(k3, k1) > 1e-6
