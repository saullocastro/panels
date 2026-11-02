r"""Verification of the plate with Green-Lagrange membrane strains, ``plate_clpt_gl``

The model, ``theory/shells/plate_clpt_gl``, has the membrane strains

.. math::

    \varepsilon_{xx} = u_{,x} + \tfrac{1}{2}(u_{,x}^2 + v_{,x}^2 + w_{,x}^2), \quad
    \varepsilon_{yy} = v_{,y} + \tfrac{1}{2}(u_{,y}^2 + v_{,y}^2 + w_{,y}^2), \quad
    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} + w_{,x} w_{,y}

and the changes of curvature of ``plate_clpt_donnell``. The checks:

1. a rigid rotation of a free plate, in plane, out of plane or about an
   oblique axis, by up to 1 rad, gives zero strains, internal forces and
   strain energy, which the von Kármán strains of ``plate_clpt_donnell`` do
   not;
2. the linear kernels and the linear analyses (static, buckling, natural
   frequencies) are those of ``plate_clpt_donnell``, the geometric stiffness
   only adding the ``u-u`` and ``v-v`` blocks;
3. for small deflections the non-linear static response is that of
   ``plate_clpt_donnell``;
4. a cantilever strip under a tip force of constant direction bent in its
   plane follows the inextensible elastica, solved here by shooting, while
   the von Kármán plate gives the linear beam; bent out of plane, the strip
   follows the linear beam, the changes of curvature being those of the von
   Kármán plate, measured in the fixed frame (the limitation of the model).

The consistency of the tangent stiffness matrix with the internal force
vector, of ``calc_kTv`` and of the follower pressure are checked with the
other models in ``test_tangent_consistency.py``, ``test_tangent_product.py``
and ``test_follower_pressure.py``.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from structsolve import Analysis, freq, lb, solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels import modelDB
from panels.models import clpt_field
from panels.shell import Shell

GL = 'plate_clpt_gl'
VK = 'plate_clpt_donnell'
LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)


def plate(model, stack=(0, 90, -45, 45, 45, -45, 90, 0), free=False, m=8, n=8):
    s = Shell(model=model, a=0.4, b=0.3, stack=list(stack), plyt=0.125e-3,
              laminaprop=LAMINAPROP, rho=1.6e3, m=m, n=n)
    # simply supported, or free
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, edge + f, 1. if (free or f != 'w') else 0.)
            setattr(s, edge + f + 'r', 1.)
    if not free:
        s.x1u = 0.
        s.y1v = 0.
    return s


def rigid_rotation(s, axis, angle):
    r"""Ritz constants of ``u = (R - I) X`` over the plate, least squares on
    a grid (exact, the approximation functions containing the linear ones)"""
    xs = np.linspace(0, s.a, 9)
    ys = np.linspace(0, s.b, 7)
    X, Y = [np.ascontiguousarray(v.ravel()) for v in np.meshgrid(xs, ys)]
    size = s.get_size()
    A = np.zeros((3*X.size, size))
    for k in range(size):
        e = np.zeros(size)
        e[k] = 1.
        A[0::3, k], A[1::3, k], A[2::3, k] = clpt_field.fuvw(e, s, X, Y, 1)[:3]
    R = Rotation.from_rotvec(np.asarray(axis)/np.linalg.norm(axis)*angle).as_matrix()
    U = ((R - np.eye(3))@np.vstack([X, Y, 0*X])).T.ravel()
    c = np.linalg.lstsq(A, U, rcond=None)[0]
    assert np.abs(A@c - U).max() < 1e-12
    return c, X, Y


AXES = {'in-plane z': (0, 0, 1), 'out-of-plane x': (1, 0, 0),
        'out-of-plane y': (0, 1, 0), 'oblique': (1, 2, 3)}


@pytest.mark.parametrize('axis', list(AXES))
@pytest.mark.parametrize('angle', [0.1, 0.5, 1.0])
def test_rigid_rotation_zero_membrane_energy(axis, angle):
    out = {}
    for model in (GL, VK):
        s = plate(model, stack=[0, 45], free=True, m=6, n=5)
        c, X, Y = rigid_rotation(s, AXES[axis], angle)
        fint = s.calc_fint(c)
        # reference: the internal forces and the strain energy of the
        # uniform strain exx = 1e-3, u = 1e-3 x
        ce = rigid_rotation(s, (0, 0, 1), 0.)[0]
        ce[0::3] = np.linalg.lstsq(
            np.column_stack([clpt_field.fuvw(np.eye(s.get_size())[k], s, X, Y, 1)[0]
                             for k in range(0, s.get_size(), 3)]),
            1e-3*X, rcond=None)[0]
        fe = s.calc_fint(ce)
        strains = np.array(clpt_field.fstrain(c, s, X, Y, 1, 1))
        # strain energy density, averaged over the grid, against that of
        # the uniform strain
        F = s.ABD
        energy = np.mean(np.einsum('ip,ij,jp->p', strains, F, strains))/(F[0, 0]*1e-6)
        out[model] = (np.abs(fint).max()/np.abs(fe).max(), energy,
                      np.abs(strains).max())
    rel, energy, strain = out[GL]
    assert rel < 1e-9
    assert energy < 1e-18
    assert strain < 1e-12
    # the von Karman strains of the same rotation, e.g. exx = cos t - 1 in
    # plane, are of the order of angle**2, or angle**4 out of plane
    assert out[VK][2] > 0.1*min(angle**2, angle**4/4)


def test_linear_kernels_equal_von_karman():
    vk = plate(VK, stack=[30, -45, 0, 90])
    gl = plate(GL, stack=[30, -45, 0, 90])
    size = gl.get_size()
    rng = np.random.default_rng(1)
    c = 1e-4*rng.standard_normal(size)
    rel = lambda A, B: abs(A - B).max()/abs(B).max()
    assert rel(gl.calc_kC(), vk.calc_kC()) == 0
    assert rel(gl.calc_kC(c=c), vk.calc_kC(c=c)) < 1e-13
    assert rel(gl.calc_kM(), vk.calc_kM()) == 0
    # geometric stiffness: the w-w block of the von Karman plate, plus u-u
    # and v-v blocks, and no coupling between u, v and w
    w = np.arange(2, size, 3)
    uv = np.setdiff1d(np.arange(size), w)
    kG_gl = gl.calc_kG(c=c).toarray()
    kG_vk = vk.calc_kG(c=c).toarray()
    assert np.abs(kG_gl[np.ix_(w, w)] - kG_vk[np.ix_(w, w)]).max() < 1e-12*np.abs(kG_vk).max()
    assert np.abs(kG_gl[np.ix_(uv, w)]).max() == 0
    assert np.abs(kG_gl[np.ix_(uv, uv)]).max() > 0.1*np.abs(kG_vk).max()
    # the analytical fkG0 equals the numerical fkG_num at a constant stress
    for s in (gl, vk):
        s.Nxx, s.Nyy, s.Nxy = -100., 30., 20.
    num = modelDB.db[GL]['matrices_num']
    kG_num = finalize_symmetric_matrix(num.fkG_num(np.zeros(size), gl.ABD, gl, size,
                                       0, 0, gl.nx, gl.ny, -100., 30., 20.))
    assert rel(gl.calc_kG(), kG_num) < 1e-12
    kG0_gl = gl.calc_kG().toarray()
    assert np.allclose(kG0_gl[np.ix_(w, w)], vk.calc_kG().toarray()[np.ix_(w, w)],
                       rtol=0, atol=1e-12*np.abs(kG0_gl).max())


def test_linear_static_equal_von_karman():
    us = []
    for model in (GL, VK):
        s = plate(model)
        s.add_pressure_load(1.e3)
        k0 = s.calc_kC()
        fext = s.calc_fext()
        us.append(solve(k0, fext, silent=True))
    assert np.allclose(us[0], us[1], rtol=0, atol=1e-14*np.abs(us[1]).max())


@pytest.mark.parametrize('stack', [(0, 90, -45, 45, 45, -45, 90, 0), (30, -45, 0, 90)])
def test_linear_buckling_equal_von_karman(stack):
    r"""With the constant stress of the analytical ``fkG0``, and with the
    stress of a linear static solution in ``fkG_num``; the ``u-u`` and
    ``v-v`` blocks of the geometric stiffness of ``plate_clpt_gl`` change the
    eigenvalue of the unsymmetric laminate only at the order of the membrane
    strain, and not at all that of the symmetric laminate"""
    eig = {}
    for model in (GL, VK):
        s = plate(model, stack=stack, m=10, n=10)
        s.Nxx = -1.
        eig[model, 'fkG0'] = lb(s.calc_kC(), s.calc_kG(), silent=True)[0][0]
        s.Nxx = 0.
        # the same compression from a load at x = a, linear static solution
        s.x2u = 1.
        s.add_distr_load_fixed_x(s.a, funcx=lambda y: -1., funcy=None,
                                 funcz=None, cte=True)
        k0 = s.calc_kC()
        c0 = solve(k0, s.calc_fext(), silent=True)
        eig[model, 'fkG_num'] = lb(k0, s.calc_kG(c=c0), silent=True)[0][0]
    print(eig)
    tol = 1e-12 if stack[0] == 0 else 1e-5
    for kind in ('fkG0', 'fkG_num'):
        assert np.isclose(eig[GL, kind], eig[VK, kind], rtol=tol)


def test_natural_frequencies_equal_von_karman():
    om = {}
    for model in (GL, VK):
        s = plate(model)
        s.Nxx = -60.
        k = s.calc_kC() + s.calc_kG()
        eigvals = freq(k, s.calc_kM(), sparse_solver=True, silent=True)[0]
        om[model] = np.sqrt(-eigvals[:5])
    assert np.allclose(om[GL], om[VK], rtol=1e-6)


@pytest.mark.parametrize('p, wtol', [(1.e2, 1e-5), (1.e3, 1e-4), (2.e4, 2e-3)])
def test_small_deflection_agrees_with_von_karman(p, wtol):
    r"""Simply supported plate under a uniform pressure: at ``p = 0.1, 1``
    and ``20 kPa`` the centre deflection is about 1.1, 3.4 and 11 times the
    thickness, and the two models differ by about 1e-6, 2e-5 and 8e-4,
    the quadratic terms of the in-plane gradients being of the order of the
    square of the membrane strains"""
    res = {}
    for model in (GL, VK):
        s = plate(model)
        s.add_pressure_load(p, cte=False)
        an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
        an.initialInc = 0.25
        an.maxInc = 0.25
        an.static(NLgeom=True, silent=True)
        assert np.isclose(an.increments[-1], 1.)
        c = an.cs[-1]
        w = s.uvw(c, xs=np.array([s.a/2]), ys=np.array([s.b/2]))[1]['w'].max()
        res[model] = w
    h = 8*0.125e-3
    print(p, res, res[GL]/h)
    assert np.isclose(res[GL], res[VK], rtol=wtol)


def elastica(alpha):
    r"""Tip deflection ``(w, u)/L`` of the inextensible cantilever under a tip
    force of constant direction, normal to the undeformed axis, ``alpha =
    P L^2/EI``: `\theta'' = -\alpha \cos\theta`, `\theta(0) = 0`,
    `\theta'(1) = 0`, solved by shooting from the tip"""
    def shoot(th_tip):
        f = lambda s, y: [y[1], -alpha*np.cos(y[0])]
        return solve_ivp(f, [1., 0.], [th_tip, 0.], rtol=1e-11, atol=1e-13,
                         dense_output=True)
    th = brentq(lambda t: shoot(t).y[0, -1], 1e-9, 1.5)
    s = np.linspace(0, 1, 4001)
    t = shoot(th).sol(s)[0]
    return np.trapezoid(np.sin(t), s), np.trapezoid(1 - np.cos(t), s), th


def cantilever_strip(model, direction, alpha, inc):
    r"""Tip deflection ``(delta, -u)/L`` of a strip ``L = 1``, ``b = 0.04``,
    ``t = 2e-3``, isotropic with ``nu = 0``, clamped at ``x = 0``, under a
    tip force of constant direction, ``z`` out of plane or ``y`` in plane,
    of ``alpha = P L^2/EI``, averaged over the tip"""
    L, b, t, E = 1., 0.04, 2e-3, 70e9
    EI = E*t**3*b/12 if direction == 'out-of-plane' else E*t*b**3/12
    P = alpha*EI/L**2
    s = Shell(model=model, a=L, b=b, stack=[0], plyt=t, laminaprop=(E, 0.),
              m=14, n=4)
    for edge in ('x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, edge + f, 1.)
            setattr(s, edge + f + 'r', 1.)
    s.x1u = s.x1v = s.x1w = s.x1wr = 0.
    if direction == 'out-of-plane':
        s.add_point_load(L, b/2, 0., 0., P, cte=False)
    else:
        s.add_point_load(L, b/2, 0., P, 0., cte=False)
    an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    an.initialInc = inc
    an.maxInc = inc
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    f = s.uvw(an.cs[-1], xs=np.full(5, L), ys=np.linspace(0, b, 5))[1]
    defl = f['w'].mean() if direction == 'out-of-plane' else f['v'].mean()
    return defl/L, -f['u'].mean()/L


def test_cantilever_strip_in_plane_against_elastica():
    r"""In-plane bending, ``alpha = 2``, 0.78 rad at the tip, is carried by
    the membrane strains, exact for large rotations: the elastica gives
    ``v = 0.4935 L``, ``u = -0.1606 L``, against ``2/3 L`` and ``0`` of the
    linear beam, which the von Kármán plate reproduces, its membrane strains
    being linear in ``u`` and ``v``"""
    w_ref, u_ref, th_ref = elastica(2.)
    gl = cantilever_strip(GL, 'in-plane', 2., 0.1)
    vk = cantilever_strip(VK, 'in-plane', 2., 0.5)
    print('elastica', (w_ref, u_ref), GL, gl, VK, vk)
    assert np.isclose(gl[0], w_ref, rtol=0.005)
    assert np.isclose(gl[1], u_ref, rtol=0.01)
    assert np.isclose(vk[0], 2/3, rtol=0.005)
    assert abs(vk[1]) < 1e-10


def test_cantilever_strip_out_of_plane_follows_the_linear_curvature():
    r"""Out-of-plane bending, ``alpha = 1``: the changes of curvature
    ``-w,xx`` are measured in the fixed frame, so the bending energy is that
    of the linear beam, and the Green-Lagrange membrane strains are kept
    zero by the shortening ``u``. The deflection is then that of the linear
    beam, ``w = alpha/3 L``, with ``u = -int (1 - sqrt(1 - w,x^2)) dx``,
    instead of ``w = 0.3017 L``, ``u = -0.0564 L`` of the elastica: the
    limitation of the model for large rotations carried by plate bending,
    see ``theory/shells/plate_clpt_gl/plate_clpt_gl.md``"""
    alpha = 1.
    x = np.linspace(0, 1, 20001)
    w_x = alpha*(x - x**2/2)
    u_ref = np.trapezoid(1 - np.sqrt(1 - w_x**2), x)
    gl = cantilever_strip(GL, 'out-of-plane', alpha, 0.1)
    print('linear beam', (alpha/3, u_ref), 'elastica', elastica(alpha), GL, gl)
    assert np.isclose(gl[0], alpha/3, rtol=0.005)
    assert np.isclose(gl[1], u_ref, rtol=0.01)
