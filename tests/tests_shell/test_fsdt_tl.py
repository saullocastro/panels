r"""Verification of the total-Lagrangian FSDT models

The models ``plate_fsdt_tl`` and ``cylshell_fsdt_tl`` have the
Green-Lagrange membrane strains of ``plate_fsdt_gl`` and
``cylshell_fsdt_gl`` and the curvatures and transverse shear strains of the
second-order director of the moderate rotation theory of Schmidt and Reddy
(1988) and Palmerio et al. (1990), with an inextensible normal. The checks:

1. the linear kernels are those of ``plate_fsdt_donnell`` and
   ``cylshell_fsdt_sanders``, and the numerical geometric stiffness matrix
   of a constant membrane state is the analytical one of the GL models;
2. a finite rigid rotation of a free plate, with the rotations the
   components of the rotated normal, gives zero membrane and bending strains
   and transverse shear strains of the fifth order in the angle;
3. a finite rigid rotation of a free cylindrical panel gives zero membrane
   strains and bending and transverse shear strains of the fourth order in
   the angle, against the second order for ``cylshell_fsdt_gl``;
4. column (Euler) buckling of a long pinned tube, against the Euler load,
   with the value of ``cylshell_fsdt_gl``;
5. a cantilever strip bent out of its plane through moderate rotations,
   against the elastica, which ``plate_fsdt_gl`` does not follow.

The consistency of the tangent stiffness matrix with the internal force
vector, of ``calc_kTv`` and of the follower pressure are checked with the
other models in ``test_tangent_consistency.py``.

Notes
-----
The deformed position of the point at `z` is `\mathbf{X} + z \mathbf{d}`
with the second-order director `\mathbf{d} = (\psi_x, \psi_y, 1 -
(\psi_x^2 + \psi_y^2)/2)` in the local basis, `\psi_x = \phi_x`, `\psi_y =
\phi_y + v/r` (`1/r = 0` for the plate), the displacement field of [1]_,
Eq. (16), with the transverse normal strain of [2]_, Eq. (20), set to
zero. The curvatures `\mathbf{X}_{,\alpha} \cdot \mathbf{d}_{,\beta}` and
transverse shear strains `\mathbf{X}_{,\alpha} \cdot \mathbf{d}` change
under a rigid rotation `\boldsymbol{R}` only through `|\mathbf{d}|^2 - 1 =
(\psi_x^2 + \psi_y^2)^2/4`: with `\psi` the components of the rotated
normal `\boldsymbol{R} \mathbf{e}_z`, the director differs from it by
`O(t^4)` along `\mathbf{e}_z`, so that the transverse shear strains of the
plate are `O(t^5)` and, the normal of the cylinder varying along `y`, its
curvatures `O(t^4)`; the linear curvatures of ``cylshell_fsdt_gl`` are of
`O(t^2)`, those of Sanders vanishing only for rotations about the axis.

The Euler load of a pinned tube is that of ``test_fsdt_tsdt_gl.py``, `N_E =
\pi^2 E R^2 h/(2 L^2)`: linear buckling uses the analytical kernels of
``cylshell_fsdt_gl``.

The cantilever strip, `L = 1`, `b = 0.04`, `t = 2 \times 10^{-3}`,
isotropic with `\nu = 0`, clamped at `x = 0`, under a tip force of constant
direction normal to its plane, `\alpha = P L^2/EI`, follows the
inextensible elastica, `\theta'' = -\alpha \cos\theta` (solved by shooting,
``test_plate_clpt_gl.elastica``): at `\alpha = 1` the tip rotates by 0.46
rad, `w = 0.3017 L`, `u = -0.0564 L`, against `w = L/3` of the linear beam,
which the linear curvatures of ``plate_fsdt_gl`` reproduce, as those of
``plate_clpt_gl``. The second-order director is 0.5 % too long at 0.46
rad; at `\alpha = 3`, 0.99 rad, the deflection of ``plate_fsdt_tl`` is 1.8 %
below that of the elastica (``plate_fsdt_gl``: +21 %).

References
----------
.. [1] Schmidt, R., and Reddy, J. N., "A Refined Small Strain and Moderate
   Rotation Theory of Elastic Anisotropic Shells", Journal of Applied
   Mechanics, Vol. 55, No. 3, 1988, pp. 611-617.
   https://doi.org/10.1115/1.3125837
.. [2] Palmerio, A. F., Reddy, J. N., and Schmidt, R., "On a moderate
   rotation theory of laminated anisotropic shells - Part 1. Theory",
   International Journal of Non-Linear Mechanics, Vol. 25, No. 6, 1990,
   pp. 687-700. https://doi.org/10.1016/0020-7462(90)90007-V

"""
import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from structsolve import Analysis

from panels.shell import Shell
from panels.models import fsdt_tsdt_field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_plate_clpt_gl import elastica
from test_fsdt_tsdt_gl import set_flags, panel, euler_ratio

LINEAR = {'plate_fsdt_tl': 'plate_fsdt_donnell',
          'cylshell_fsdt_tl': 'cylshell_fsdt_sanders'}
GL = {'plate_fsdt_tl': 'plate_fsdt_gl',
      'cylshell_fsdt_tl': 'cylshell_fsdt_gl'}
STRAINS = ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz')


@pytest.mark.slow
@pytest.mark.parametrize('model', list(LINEAR))
def test_linear_kernels_are_those_of_the_linear_model(model):
    r"""Constitutive stiffness at zero displacements and mass matrix equal
    to those of the linear model; the numerical geometric stiffness matrix
    of a constant membrane state at zero displacements equal to the
    analytical one of the GL model, which the model uses"""
    t, li, gl = panel(model), panel(LINEAR[model]), panel(GL[model])
    for s in (t, li, gl):
        s.Nxx, s.Nyy, s.Nxy = -1.3, 0.7, 0.4
    c0 = np.zeros(t.get_size())
    kC = t.calc_kC(c=c0, NLgeom=True)
    kC_lin = li.calc_kC(c=c0, NLgeom=True)
    assert abs(kC - kC_lin).max() < 1.e-12*abs(kC_lin).max()
    assert abs(t.calc_kC() - li.calc_kC()).max() == 0.
    assert abs(t.calc_kM() - li.calc_kM()).max() == 0.
    kG0 = t.calc_kG()
    assert abs(kG0 - gl.calc_kG()).max() == 0.
    kGnum = t.calc_kG(c=c0)
    assert abs(kG0 - kGnum).max() < 1.e-10*abs(kG0).max()
    rng = np.random.default_rng(0)
    c = 1.e-6*rng.standard_normal(t.get_size())
    f, f_lin = t.calc_fint(c), li.calc_kC() @ c
    assert np.linalg.norm(f - f_lin) < 1.e-4*np.linalg.norm(f_lin)


def plate_rigid_rotation(s, axis, angle):
    r"""Ritz constants of ``u = (R - I) X`` with the constant rotations
    ``phix``, ``phiy`` the components of ``R e_z``, least squares on a grid
    (exact, the approximation functions containing the linear ones)"""
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
    U = np.zeros((X.size, 5))
    U[:, :3] = ((R - np.eye(3))@np.vstack([X, Y, 0*X])).T
    U[:, 3] = R[0, 2]
    U[:, 4] = R[1, 2]
    c = np.linalg.lstsq(A, U.ravel(), rcond=None)[0]
    assert np.abs(A@c - U.ravel()).max() < 1e-12
    return c


@pytest.mark.slow
@pytest.mark.parametrize('axis', [(1, 0, 0), (0, 1, 0), (1, 2, 3)])
def test_plate_rigid_rotation(axis):
    r"""Rigid rotation of a free plate by 0.2 and 0.1 rad: membrane and
    bending strains zero to the accuracy of the fit, transverse shear
    strains of the fifth order in the angle (ratio about 32)"""
    s = panel('plate_fsdt_tl', free=True)
    shear = []
    for angle in (0.2, 0.1):
        c = plate_rigid_rotation(s, axis, angle)
        _, f = s.strain(c, gridx=9, gridy=9, NLgeom=True)
        assert max(np.abs(f[k]).max() for k in STRAINS[:6]) < 1.e-10
        shear.append(max(np.abs(f[k]).max() for k in ('gyz', 'gxz')))
    assert 4.7 < np.log2(shear[0]/shear[1]) < 5.3, shear
    assert shear[0] < 1e-4


def cylinder_rigid_rotation(s, axis, angle):
    r"""Ritz constants of the finite rigid rotation ``(Q - I) X`` of the
    panel about the global axis ``axis`` through the axis of the cylinder,
    with the rotations of the normal ``psix = phix`` and ``psiy = phiy +
    v/r`` (the field ``phiy`` of ``fsdt_tsdt_field.fg``) the components of
    the rotated normal, least squares on a grid

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
            ey = np.array([0, np.cos(th), -np.sin(th)])
            ez = np.array([0, np.sin(th), np.cos(th)])
            g = np.zeros((5, size))
            fsdt_tsdt_field.fg(g, x, y, s)
            G += [g[0], g[1], g[2], g[3], g[4]]
            rhs += [U[0], U @ ey, U @ ez, (Q @ ez)[0], (Q @ ez) @ ey]
    return np.linalg.lstsq(np.array(G), np.array(rhs), rcond=None)[0]


def cylinder_rotation_strains(model, axis, angle):
    r"""Largest membrane strain and largest bending or transverse shear
    strain of the rigid rotation, free panel `R = 0.4`, `L = 0.3`,
    `\theta_0 = 0.5`, 12 x 12 terms"""
    s = panel(model, free=True, m=12, n=12)
    c = cylinder_rigid_rotation(s, axis, angle)
    _, f = s.strain(c, gridx=9, gridy=9, NLgeom=True)
    return (max(np.abs(f[k]).max() for k in STRAINS[:3]),
            max(np.abs(f[k]).max() for k in STRAINS[3:]))


@pytest.mark.parametrize('axis', [(0, 1, 0), (0, 0, 1)])
def test_cylinder_rigid_rotation(axis):
    r"""Rigid rotation of a free cylindrical panel by 0.2 and 0.1 rad about
    the two axes normal to the axis of the cylinder: zero membrane strains;
    bending and transverse shear strains of the fourth order in the angle
    (ratio about 16), against the second order (ratio about 4) for the
    linear curvatures of ``cylshell_fsdt_gl`` with the same fields"""
    tl, gl = [], []
    for angle in (0.2, 0.1):
        mem, ben = cylinder_rotation_strains('cylshell_fsdt_tl', axis, angle)
        assert mem < 1e-9
        tl.append(ben)
        gl.append(cylinder_rotation_strains('cylshell_fsdt_gl', axis, angle)[1])
    assert np.log2(tl[0]/tl[1]) > 3.6, tl
    assert 1.8 < np.log2(gl[0]/gl[1]) < 2.2, gl
    assert tl[1] < 0.05*gl[1]


def test_euler_load_of_a_pinned_tube():
    r"""Column buckling of a pinned tube, `R/h = 100`, `L/R = 50`, 10 x 8
    terms: the beam mode at 0.9890986 `N_E`, the value of
    ``cylshell_fsdt_gl`` (``test_fsdt_tsdt_gl.py``), whose analytical kernels
    linear buckling uses"""
    ratio, wr = euler_ratio('cylshell_fsdt_tl', 50.)
    assert np.isclose(wr, -1., atol=1e-3)
    assert np.isclose(ratio, 0.9890986, rtol=1e-4)


def cantilever_strip(model, alpha, inc):
    r"""Tip deflection ``(w, -u)/L`` of a strip ``L = 1``, ``b = 0.04``,
    ``t = 2e-3``, isotropic with ``nu = 0``, clamped at ``x = 0``, under a
    tip force along ``z`` of ``alpha = P L^2/EI``, averaged over the tip"""
    L, b, t, E_ = 1., 0.04, 2e-3, 70e9
    EI = E_*t**3*b/12
    P = alpha*EI/L**2
    s = Shell(model=model, a=L, b=b, stack=[0], plyt=t, laminaprop=(E_, 0.),
              m=14, n=4)
    set_flags(s, ('x2', 'y1', 'y2'), **{'': 1., 'r': 1.})
    set_flags(s, ('x1', ), **{'': 0., 'r': 1.})
    s._rebuild()
    s.add_point_load(L, b/2, 0., 0., P, cte=False)
    an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    an.initialInc = inc
    an.maxInc = inc
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    f = s.uvw(an.cs[-1], xs=np.full(5, L), ys=np.linspace(0, b, 5))[1]
    return f['w'].mean()/L, -f['u'].mean()/L


@pytest.mark.slow
def test_cantilever_strip_out_of_plane_against_elastica():
    r"""Out-of-plane bending, ``alpha = 1``, tip rotation 0.46 rad: the
    elastica gives ``w = 0.3017 L``, ``u = -0.0564 L``; ``plate_fsdt_tl``
    follows it within -0.27 % and -0.59 % (second-order director),
    ``plate_fsdt_gl`` gives the deflection of the linear beam, ``w = L/3``,
    +10.5 %. Stored values of ``plate_fsdt_tl``, 14 x 4 terms: ``w =
    0.3009111 L``, ``u = -0.0561018 L``"""
    w_ref, u_ref, _ = elastica(1.)
    w, u = cantilever_strip('plate_fsdt_tl', 1., 0.1)
    assert np.isclose(w, w_ref, rtol=0.005)
    assert np.isclose(u, u_ref, rtol=0.01)
    assert np.isclose(w, 0.3009111, rtol=1e-5)
    assert np.isclose(u, 0.0561018, rtol=1e-5)
    w, u = cantilever_strip('plate_fsdt_gl', 1., 0.1)
    assert np.isclose(w, 1/3, rtol=0.001)
