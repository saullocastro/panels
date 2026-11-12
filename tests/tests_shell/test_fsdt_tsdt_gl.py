r"""Verification of the FSDT and TSDT models with Green-Lagrange membrane strains

The models ``plate_fsdt_gl``, ``plate_tsdt_gl``, ``cylshell_fsdt_gl`` and
``cylshell_tsdt_gl`` have the Green-Lagrange membrane strains of
``plate_clpt_gl`` and ``cylshell_clpt_gl`` and the linear bending and
transverse shear strains of ``plate_fsdt_donnell``, ``plate_tsdt_donnell``,
``cylshell_fsdt_sanders`` and ``cylshell_tsdt_sanders``. The checks:

1. the linear kernels are those of the linear models, and the analytical
   geometric stiffness matrix equals the numerically integrated one;
2. a finite rigid rotation of a free plate, with constant rotations
   `\phi_x = -w_{,x}`, `\phi_y = -w_{,y}`, gives zero strains and zero
   internal forces; a finite rigid rotation of a free cylindrical panel
   gives zero membrane strains;
3. an in-plane cantilever strip bent through 0.78 rad, against the elastica
   integrated numerically by ``test_plate_clpt_gl.elastica``;
4. column (Euler) buckling of a long pinned tube, against the Euler load
   `\pi^2 E I/L^2`, as for ``cylshell_clpt_gl``.

The module also provides the hinged roof of Sabir and Lock, benchmark 3.8 of
Sze, Liu and Lo (2004), :func:`hinged_roof`, its path traced with the Riks
method, :func:`roof_path`, and its critical points,
:func:`limit_and_minimum`, for other modules.

The consistency of the tangent stiffness matrix with the internal force
vector, of ``calc_kTv`` and of the follower pressure are checked with the
other models in ``test_tangent_consistency.py``.

Notes
-----
The membrane strains of the plates are `\varepsilon_{xx} = u_{,x} +
\tfrac{1}{2}(u_{,x}^2 + v_{,x}^2 + w_{,x}^2)`, `\varepsilon_{yy} = v_{,y} +
\tfrac{1}{2}(u_{,y}^2 + v_{,y}^2 + w_{,y}^2)`, `\gamma_{xy} = u_{,y} +
v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} + w_{,x} w_{,y}`, those of the
cylinders the strains of the Flügge-Lur'e-Byrne non-linear theory as given
by Amabili [1]_, Eqs. (1.118)-(1.120). For a rigid rotation `\boldsymbol{R}`
of the plate the displacement gradients are the first two columns of
`\boldsymbol{R} - \boldsymbol{I}`, the membrane strains vanish, and the
bending and transverse shear strains vanish with the constant rotations
`\phi_x = -w_{,x}`, `\phi_y = -w_{,y}`; the third-order strains of the
TSDT contain `w_{,xx}`, `w_{,yy}`, `w_{,xy}`, zero as well.

The in-plane cantilever strip, `\alpha = P L^2/EI = 2`, has the tip
deflection `v = 0.4935 L` and shortening `u = -0.1606 L` of the elastica,
`\theta'' = -\alpha \cos\theta`; the bending in the plane of the strip is carried by the membrane
strains, so that the shear deformation of the FSDT does not enter.

The Euler load of a pinned tube of radius `R`, wall thickness `h` and
length `L`, per unit length of the circumference, is `N_E = \pi^2 E R^2
h/(2 L^2)`; the work of the axial force on the beam mode, `w = W(x)
\cos\theta`, `v = -W(x) \sin\theta`, needs the term `v_{,x}^2/2` of the
Green-Lagrange strain `\varepsilon_{xx}`, absent from
``cylshell_fsdt_sanders``, which gives about `2 N_E`.

The hinged roof [2]_, Sec. 3.8, Fig. 9a: `R = 2540`, `L = 508`, `\theta =
0.1` rad, `E = 3102.75`, `\nu = 0.3`, `h = 12.7`, straight edges hinged,
curved edges free, a point load at the centre; the quarter is modelled
with the boundary conditions of symmetry and the central deflection is
prescribed through a stiff spring and traced with the Riks method, as in
``test_cylshell_clpt_tl.py``; the rotations have `\phi_x = \phi_{y,x} = 0`
on the plane of symmetry `x = L/2`, `\phi_y = \phi_{x,y} = 0` on `y = R
\theta`, and are free on the hinged edge.

References
----------
.. [1] Amabili, M., "Nonlinear Vibrations and Stability of Shells and
   Plates", Cambridge University Press, 2008.
   https://doi.org/10.1017/CBO9780511619694
.. [2] Sze, K. Y., Liu, X. H., and Lo, S. H., "Popular benchmark problems
   for geometric nonlinear analysis of shells", Finite Elements in
   Analysis and Design, 40(11), 1551-1569, 2004.
   https://doi.org/10.1016/j.finel.2003.11.001

"""
import os
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from structsolve import Analysis, lb, solve

from panels.shell import Shell
from panels.models import fsdt_tsdt_field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_plate_clpt_gl import elastica

LINEAR = {'plate_fsdt_gl': 'plate_fsdt_donnell',
          'plate_tsdt_gl': 'plate_tsdt_donnell',
          'cylshell_fsdt_gl': 'cylshell_fsdt_sanders',
          'cylshell_tsdt_gl': 'cylshell_tsdt_sanders'}
PLATES = ['plate_fsdt_gl', 'plate_tsdt_gl']
CYLINDERS = ['cylshell_fsdt_gl', 'cylshell_tsdt_gl']
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
LAMINAPROP = (142.5e9, 142.5e9/20, 0.25, 0.5*142.5e9/20, 0.5*142.5e9/20,
              0.5*142.5e9/20)
E = 70.e9
NU = 0.3


def set_flags(s, edges, fields=FIELDS, **values):
    for e in edges:
        for f in fields:
            for k, val in values.items():
                setattr(s, e + f + k, val)


def panel(model, free=False, m=8, n=8):
    r"""Unsymmetric laminate, simply supported or free; a cylindrical panel
    of `R = 0.4`, `\theta_0 = 0.5` for the cylinders"""
    cyl = model.startswith('cylshell')
    s = Shell(model=model, a=0.3, b=0.2, r=0.4 if cyl else None,
              stack=[30, -45, 0, 90], plyt=0.125e-3, laminaprop=LAMINAPROP,
              rho=1.6e3, m=m, n=n)
    set_flags(s, ('x1', 'x2', 'y1', 'y2'), **{'': 1., 'r': 1.})
    if not free:
        set_flags(s, ('x1', 'x2', 'y1', 'y2'), fields=('w',), **{'': 0.})
        s.x1u = 0.
        s.y1v = 0.
    s._rebuild()
    return s


@pytest.mark.parametrize('model', list(LINEAR))
def test_linear_kernels_are_those_of_the_linear_model(model):
    r"""Constitutive stiffness and internal forces at zero displacements and
    the mass matrix equal to those of the linear model; analytical and
    numerical geometric stiffness matrices of a constant membrane state
    equal, and different from those of the linear model, which lack the
    in-plane blocks"""
    shells = {}
    for mdl in (model, LINEAR[model]):
        s = panel(mdl)
        s.Nxx, s.Nyy, s.Nxy = -1.3, 0.7, 0.4
        shells[mdl] = s
    g, li = shells[model], shells[LINEAR[model]]
    c0 = np.zeros(g.get_size())
    kC = g.calc_kC(c=c0, NLgeom=True)
    kC_lin = li.calc_kC(c=c0, NLgeom=True)
    assert abs(kC - kC_lin).max() < 1.e-12*abs(kC_lin).max()
    assert abs(g.calc_kC() - li.calc_kC()).max() == 0.
    assert abs(g.calc_kM() - li.calc_kM()).max() == 0.
    kG0 = g.calc_kG()
    kGnum = g.calc_kG(c=c0)
    scale = abs(kG0).max()
    assert abs(kG0 - kGnum).max() < 1.e-10*scale
    assert abs(kG0 - li.calc_kG()).max() > 1.e-3*scale
    # small displacements: internal forces of the linear model to second
    # order
    rng = np.random.default_rng(0)
    c = 1.e-6*rng.standard_normal(g.get_size())
    f, f_lin = g.calc_fint(c), li.calc_kC() @ c
    assert np.linalg.norm(f - f_lin) < 1.e-4*np.linalg.norm(f_lin)


def plate_rigid_rotation(s, axis, angle):
    r"""Ritz constants of ``u = (R - I) X`` with the constant rotations
    ``phix = -w,x``, ``phiy = -w,y``, least squares on a grid of more points
    than terms along each direction (exact, the approximation functions
    containing the linear ones)"""
    xs = np.linspace(0, s.a, 17)
    ys = np.linspace(0, s.b, 17)
    X, Y = [np.ascontiguousarray(v.ravel()) for v in np.meshgrid(xs, ys)]
    size = s.get_size()
    A = np.zeros((5*X.size, size))
    for k in range(size):
        e = np.zeros(size)
        e[k] = 1.
        for i, fld in enumerate(fsdt_tsdt_field.fuvw(e, s, X, Y, 1)):
            A[i::5, k] = fld
    R = Rotation.from_rotvec(np.asarray(axis)/np.linalg.norm(axis)*angle).as_matrix()
    G = R - np.eye(3)
    U = np.zeros((X.size, 5))
    U[:, :3] = (G@np.vstack([X, Y, 0*X])).T
    U[:, 3] = -G[2, 0]
    U[:, 4] = -G[2, 1]
    c = np.linalg.lstsq(A, U.ravel(), rcond=None)[0]
    assert np.abs(A@c - U.ravel()).max() < 1e-12
    return c


AXES = {'in-plane z': (0, 0, 1), 'out-of-plane x': (1, 0, 0),
        'out-of-plane y': (0, 1, 0), 'oblique': (1, 2, 3)}


@pytest.mark.parametrize('model', PLATES)
@pytest.mark.parametrize('axis', list(AXES))
@pytest.mark.parametrize('angle', [0.5, 1.0])
def test_plate_rigid_rotation_gives_zero_strains(model, axis, angle):
    r"""Finite rigid rotation of a free plate: all the strains and the
    internal forces vanish to the accuracy of the fit, whereas the von
    Kármán membrane strains of the linear model are of the order of the
    angle squared"""
    s = panel(model, free=True)
    c = plate_rigid_rotation(s, AXES[axis], angle)
    _, fields = s.strain(c, gridx=7, gridy=7, NLgeom=True)
    # strains at the faces z = +-h/2: the curvatures times h/2, the third-
    # order terms times (h/2)^3 and the second-order shear terms times
    # (h/2)^2
    h2 = sum(s.plyts)/2
    scale = {'kxx': h2, 'kyy': h2, 'kxy': h2, 'kxx3': h2**3, 'kyy3': h2**3,
             'kxy3': h2**3, 'gyz2': h2**2, 'gxz2': h2**2}
    names = fsdt_tsdt_field.strain_names(s)
    assert max(scale.get(k, 1.)*np.abs(fields[k]).max() for k in names) < 1.e-12
    fint = s.calc_fint(c)
    assert np.linalg.norm(fint) < 1.e-9*np.linalg.norm(s.calc_kC() @ c)
    sl = panel(LINEAR[model], free=True)
    _, fl = sl.strain(c, gridx=7, gridy=7, NLgeom=True)
    assert max(np.abs(fl[k]).max() for k in ('exx', 'eyy', 'gxy')) > 1.e-3


def cylinder_rigid_rotation(s, axis, angle):
    r"""Ritz constants of the finite rigid rotation ``(Q - I) X`` of the
    panel about the global axis ``axis`` through the axis of the cylinder,
    least squares on a grid, the rotations ``phix``, ``phiy`` zero

    The panel is `X = (x, R \sin\theta, R \cos\theta)`, `\theta` measured
    from its centre line, `v` along `(0, \cos\theta, -\sin\theta)` and `w`
    along `(0, \sin\theta, \cos\theta)`.

    """
    Q = Rotation.from_rotvec(angle*np.asarray(axis, dtype=float)).as_matrix()
    theta0 = s.b/s.r
    size = s.get_size()
    G, rhs = [], []
    for x in np.linspace(0, s.a, 19):
        for y in np.linspace(0, s.b, 19):
            th = y/s.r - theta0/2
            X = np.array([x, s.r*np.sin(th), s.r*np.cos(th)])
            U = (Q - np.eye(3)) @ X
            g = np.zeros((5, size))
            fsdt_tsdt_field.fg(g, x, y, s)
            G += [g[0], g[1], g[2], g[3], g[4]]
            rhs += [U[0], U @ np.array([0, np.cos(th), -np.sin(th)]),
                    U @ np.array([0, np.sin(th), np.cos(th)]), 0., 0.]
    return np.linalg.lstsq(np.array(G), np.array(rhs), rcond=None)[0]


@pytest.mark.parametrize('model', CYLINDERS)
@pytest.mark.parametrize('axis', [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
def test_cylinder_rigid_rotation_gives_zero_membrane_strains(model, axis):
    r"""Finite rigid rotation of 1 rad of a free panel, `R = 0.4`, `L =
    0.3`, `\theta_0 = 0.5`, about the axis of the cylinder (`x`) and the two
    axes normal to it: the membrane strains vanish to the accuracy of the
    fit, those of the linear (Sanders) model are of the order of the angle
    squared. The bending strains, linear, are not checked."""
    res = {}
    for mdl in (model, LINEAR[model]):
        s = panel(mdl, free=True, m=12, n=12)
        c = cylinder_rigid_rotation(s, axis, 1.)
        _, fields = s.strain(c, gridx=9, gridy=9, NLgeom=True)
        res[mdl] = max(np.abs(fields[k]).max() for k in ('exx', 'eyy', 'gxy'))
    assert res[model] < 1.e-9
    assert res[LINEAR[model]] > 1.e-3


def cantilever_strip(model, alpha, inc):
    r"""Tip deflection ``(v, u)/L`` of a strip ``L = 1``, ``b = 0.04``,
    ``t = 2e-3``, isotropic with ``nu = 0``, clamped at ``x = 0``, under a
    tip force along ``y``, in its plane, of ``alpha = P L^2/EI``, averaged
    over the tip"""
    L, b, t, E_ = 1., 0.04, 2e-3, 70e9
    EI = E_*t*b**3/12
    P = alpha*EI/L**2
    s = Shell(model=model, a=L, b=b, stack=[0], plyt=t, laminaprop=(E_, 0.),
              m=14, n=4)
    set_flags(s, ('x2', 'y1', 'y2'), **{'': 1., 'r': 1.})
    set_flags(s, ('x1', ), **{'': 0., 'r': 1.})
    s.x1wr = 0.
    s._rebuild()
    s.add_point_load(L, b/2, 0., P, 0., cte=False)
    an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    an.initialInc = inc
    an.maxInc = inc
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    f = s.uvw(an.cs[-1], xs=np.full(5, L), ys=np.linspace(0, b, 5))[1]
    return f['v'].mean()/L, -f['u'].mean()/L


@pytest.mark.parametrize('model', PLATES)
def test_cantilever_strip_in_plane_against_elastica(model):
    r"""In-plane bending of a strip, ``alpha = 2``, 0.78 rad at the tip,
    carried by the membrane strains: the elastica gives ``v = 0.4935 L``,
    ``u = -0.1606 L``, against ``2/3 L`` and ``0`` of the linear beam, which
    the von Kármán plate reproduces"""
    v_ref, u_ref, _ = elastica(2.)
    v, u = cantilever_strip(model, 2., 0.1)
    assert np.isclose(v, v_ref, rtol=0.005)
    assert np.isclose(u, u_ref, rtol=0.01)
    v, u = cantilever_strip(LINEAR[model], 2., 0.5)
    assert np.isclose(v, 2/3, rtol=0.005)
    assert abs(u) < 1e-10


def half_tube(model, R, h, L, m, n):
    r"""Half of a tube, `0 \le \theta \le \pi`, pinned at both ends

    The ends `x = 0, L` have `v = w = \phi_y = 0`, `u` and `\phi_x` free
    (diaphragms that let the cross-section rotate), and the edges `\theta =
    0, \pi` are the symmetry planes of the bending of the tube in the plane
    `\theta = 0`: `v = \phi_y = 0`, `u_{,y} = w_{,y} = \phi_{x,y} = 0`.

    """
    s = Shell(model=model, a=L, b=np.pi*R, r=R, stack=[0.], plyt=h,
              laminaprop=(E, NU), rho=2700., m=m, n=n)
    set_flags(s, ('x1', 'x2'), **{'': 1., 'r': 1.})
    set_flags(s, ('x1', 'x2'), fields=('v', 'w', 'phiy'), **{'': 0.})
    set_flags(s, ('y1', 'y2'), fields=('u', 'w', 'phix'), **{'': 1., 'r': 0.})
    set_flags(s, ('y1', 'y2'), fields=('v', 'phiy'), **{'': 0., 'r': 1.})
    s._rebuild()
    return s


def euler_ratio(model, LR, Rh=100., m=10, n=8):
    r"""Critical axial load of the pinned tube over `N_E` and the ratio
    `w(L/2, \pi)/w(L/2, 0)` of the mode, `-1` for the beam mode; the rigid
    axial translation is removed by a penalty on `u` at `x = L/2`, `\theta =
    0`"""
    R = 1.
    h = R/Rh
    L = LR*R
    s = half_tube(model, R, h, L, m, n)
    s.Nxx = -1.
    K = s.calc_kC() + csr_matrix(s.calc_stiffness_point_constraint(
        L/2, 0., u=True, v=False, w=False, kuvw=E*h))
    eigvals, eigvecs = lb(K, s.calc_kG(), silent=True, num_eigvalues=4)
    NE = np.pi**2*E*R**2*h/(2*L**2)
    c = np.ascontiguousarray(eigvecs[:, 0].real)
    _, fields = s.uvw(c, xs=np.array([L/2, L/2]), ys=np.array([0., np.pi*R]))
    w = np.ravel(fields['w'])
    return eigvals[0].real/NE, w[1]/w[0]


@pytest.mark.parametrize('model, ref', [
    ('cylshell_fsdt_gl', 0.9890986),
    ('cylshell_tsdt_gl', 0.9890986),
    ('cylshell_fsdt_sanders', 1.984334),
])
def test_euler_load_of_a_pinned_tube(model, ref):
    r"""Column buckling of a pinned tube, `R/h = 100`, `L/R = 50`: the beam
    mode at the Euler load with the Green-Lagrange membrane strains, at
    about `2 N_E` with ``cylshell_fsdt_sanders``, as for the CLPT models
    (``cylshell_clpt_gl``: 0.989099 `N_E`, ``cylshell_clpt_sanders``:
    1.984334 `N_E`, ``test_cylshell_clpt_gl.py``): the transverse shear does
    not change the beam mode of the thin tube. Stored values of 10 x 8
    terms."""
    ratio, wr = euler_ratio(model, 50.)
    assert np.isclose(wr, -1., atol=1e-3)
    target = 2. if 'sanders' in model else 1.
    assert np.isclose(ratio, target, rtol=0.02)
    if ref is not None:
        assert np.isclose(ratio, ref, rtol=1e-4)


# Hinged roof, [2] Sec. 3.8 and Fig. 9a
ROOF_R = 2540.
ROOF_A = 508.
ROOF_B = 0.2*ROOF_R
ROOF_E = 3102.75
ROOF_NU = 0.3
PMAX = 3000.


def hinged_roof(model, h, m):
    r"""Quarter of the hinged roof, the boundary conditions of
    ``test_cylshell_clpt_tl.hinged_roof`` with, for the rotations, `\phi_x
    = \phi_{y,x} = 0` at `x = L` and `\phi_y = \phi_{x,y} = 0` at `y = R
    \theta`"""
    s = Shell(model=model, a=ROOF_A/2, b=ROOF_B/2, r=ROOF_R, stack=[0.],
              plyt=h, laminaprop=(ROOF_E, ROOF_NU), m=m, n=m)
    set_flags(s, ('x1', 'x2', 'y1', 'y2'), **{'': 1., 'r': 1.})
    s.y1u = s.y1v = s.y1w = 0.
    s.x2u = s.x2vr = s.x2wr = 0.
    s.y2v = s.y2ur = s.y2wr = 0.
    s.x2phix = s.x2phiyr = 0.
    s.y2phiy = s.y2phixr = 0.
    s._rebuild()
    return s


def roof_path(model, h, m, wmax=30., max_inc=0.01, max_steps=1000,
              kfac=100.):
    r"""Reaction ``P`` (whole roof) and central deflection ``wc`` along the
    path traced with the Riks method under a central deflection prescribed
    through a spring, as ``test_cylshell_clpt_tl.roof_path``"""
    s = hinged_roof(model, h, m)
    g = np.zeros((5, s.get_size()))
    fsdt_tsdt_field.fg(g, s.a, s.b, s)
    e = g[2].copy()
    c1 = solve(s.calc_kC(), -e, silent=True)
    K = kfac/abs(e @ c1)
    delta = -wmax
    Ke = csr_matrix(K*np.outer(e, e))
    an = Analysis(lambda inc=1., silent=True: inc*K*delta*e,
                  lambda c, silent=True: s.calc_fint(c) + K*e*(e @ c),
                  lambda c=None, NLgeom=False, silent=True:
                      s.calc_kC(c=c, NLgeom=NLgeom) + Ke,
                  s.calc_kG)
    an.NL_method = 'arc_length_riks'
    an.initialInc = an.maxInc = max_inc
    an.maxArcLength = 1.e6
    an.final_load_factor = None
    an.stop_criterion = lambda an: (-(e @ an.cs[-1]) >= wmax
                                    or len(an.cs) >= max_steps)
    an.static(NLgeom=True, silent=True)
    wc = np.array([e @ c for c in an.cs])
    lbd = np.asarray(an.increments)
    return 4*K*(wc - lbd*delta), wc


def limit_and_minimum(P, wc):
    r"""``(P, -w_C)`` at the first maximum of ``P`` and the following
    minimum"""
    w = -np.r_[0., wc]
    P = np.r_[0., P]
    d = np.diff(P)
    i1 = int(np.argmax(d < 0))
    i2 = i1 + int(np.argmax(np.diff(P[i1:]) > 0))
    return (P[i1], w[i1]), (P[i2], w[i2])


