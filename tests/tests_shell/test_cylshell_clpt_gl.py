r"""Verification of the cylinder with Green-Lagrange membrane strains, ``cylshell_clpt_gl``

The model has the Green-Lagrange membrane strains of the mid-surface and the
changes of curvature of ``cylshell_clpt_sanders``. The checks:

1. column (Euler) buckling of a long pinned tube, against the Euler load
   `\pi^2 E I/L^2` of the tube as a beam: ``cylshell_clpt_gl`` gives the
   Euler load, ``cylshell_clpt_sanders`` about twice it, the term `N_{xx}
   v_{,x}^2` of the geometric stiffness being absent from its strains;
2. axial buckling of panels supported by shear diaphragms, against Batdorf
   (1947) as given by NASA SP-8007-2020/REV 2, Sec. 4.1.1.1, and against
   ``cylshell_clpt_sanders``: the models agree where the circumferential
   waves are short;
3. finite rigid rotations of a free panel, about the axis of the cylinder
   and about the two axes normal to it, give zero membrane strains, which
   the Sanders strains do not;
4. the linear kernels (constitutive stiffness, mass) are those of
   ``cylshell_clpt_sanders``, and the analytical geometric stiffness matrix
   equals the numerically integrated one;
5. the connections of :class:`.MultiDomain` include the term `v/r` of the
   rotation of the normal, as for ``cylshell_clpt_sanders``.

The consistency of the tangent stiffness matrix with the internal force
vector (central differences and Taylor test), of ``calc_kTv`` and of the
follower pressure are checked with the other models in
``test_tangent_consistency.py``, ``test_tangent_product.py`` and
``test_follower_pressure.py``, and the connection energies in
``tests/multidomain/test_bf_sanders.py``.

Notes
-----
The membrane strains of the model are

.. math::

    \varepsilon_{xx} = u_{,x} + \tfrac{1}{2}(u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2), \quad
    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \tfrac{1}{2}\left(u_{,y}^2 +
    \left(v_{,y} + \frac{w}{r}\right)^2 + \left(w_{,y} -
    \frac{v}{r}\right)^2\right)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x}\left(v_{,y} +
    \frac{w}{r}\right) + w_{,x}\left(w_{,y} - \frac{v}{r}\right)

those of the Flügge-Lur'e-Byrne non-linear theory as given by Amabili
[1]_, Sec. 1.3.6, Eqs. (1.118)-(1.120). For the bending of a tube as a beam,
`w = W(x) \cos\theta`, `v = -W(x) \sin\theta`, the work of the axial force
`N_{xx}` is `\frac{1}{2} N_{xx} \int (v_{,x}^2 + w_{,x}^2) dA`, twice the
`\frac{1}{2} N_{xx} \int w_{,x}^2 dA` of the Sanders strains of
``cylshell_clpt_sanders``, which omit the rotation about the normal that
Sanders [2]_ keeps. The Euler load of a pinned tube of radius `R`, wall
thickness `h` and length `L`, `I = \pi R^3 h`, per unit length of the
circumference, is `N_E = \pi^2 E I/(2 \pi R L^2) = \pi^2 E R^2 h/(2 L^2)`.

The classical buckling load of an axially compressed cylindrical panel
supported by shear diaphragms is that of Batdorf [3]_, the minimum over the
Navier modes `(m, j)` of the Donnell equations,

.. math::

    N_{cr} = \min_{m, j} \left( D \frac{(\alpha^2 + \beta^2)^2}{\alpha^2} +
    \frac{E h}{R^2} \frac{\alpha^2}{(\alpha^2 + \beta^2)^2} \right), \qquad
    \alpha = \frac{m \pi}{L}, \quad \beta = \frac{j \pi}{b}

with `D = E h^3/(12(1 - \nu^2))`.

References
----------
.. [1] Amabili, M., "Nonlinear Vibrations and Stability of Shells and
   Plates", Cambridge University Press, 2008.
   https://doi.org/10.1017/CBO9780511619694
.. [2] Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
   Applied Mathematics, 21(1), 21-36, 1963.
   https://doi.org/10.1090/qam/147023
.. [3] Batdorf, S. B., NACA TR 874, 1947, and NASA SP-8007-2020/REV 2, Sec.
   4.1.1.1.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from structsolve import lb

from panels.shell import Shell
from panels.models.clpt_field import fg

GL = 'cylshell_clpt_gl'
SANDERS = 'cylshell_clpt_sanders'
DONNELL = 'cylshell_clpt_donnell'
E = 70.e9
NU = 0.3


def isotropic(model, R, h, L, b, m, n):
    s = Shell(model=model, a=L, b=b, r=R, stack=[0.], plyt=h,
              laminaprop=(E, NU), rho=2700., m=m, n=n)
    return s


def flags(s, edges, **values):
    for e in edges:
        for k, val in values.items():
            setattr(s, e + k, val)


def half_tube(model, R, h, L, m, n):
    r"""Half of a tube, `0 \le \theta \le \pi`, pinned at both ends

    The ends `x = 0, L` have `v = w = 0`, `u` free (diaphragms that let the
    cross-section rotate), and the edges `\theta = 0, \pi` are the
    symmetry planes of the bending of the tube in the plane `\theta = 0`:
    `v = 0`, `u_{,y} = w_{,y} = 0`.

    """
    s = isotropic(model, R, h, L, np.pi*R, m, n)
    flags(s, ('x1', 'x2'), u=1., ur=1., v=0., vr=1., w=0., wr=1.)
    flags(s, ('y1', 'y2'), u=1., ur=0., v=0., vr=1., w=1., wr=0.)
    s._rebuild()
    return s


def euler_ratio(model, LR, Rh=100., m=10, n=8):
    r"""Critical axial load of the pinned tube over `N_E`, and the ratios
    `w(L/2, \pi)/w(L/2, 0)` and `v(L/2, \pi/2)/w(L/2, 0)` of the mode,
    `-1` for the bending of the tube as a beam

    The rigid axial translation is removed by a penalty on `u` at `x = L/2`,
    `\theta = 0`, where the beam mode has `u = 0`.

    """
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
    xs = np.array([L/2, L/2, L/2])
    ys = np.array([0., np.pi*R, np.pi*R/2])
    _, fields = s.uvw(c, xs=xs, ys=ys)
    w = np.ravel(fields['w'])
    v = np.ravel(fields['v'])
    return eigvals[0].real/NE, w[1]/w[0], v[2]/w[0]


def sd_panel(model, R, h, L, theta0, m, n):
    r"""Panel supported by shear diaphragms on its four edges"""
    s = isotropic(model, R, h, L, R*theta0, m, n)
    flags(s, ('x1', 'x2'), u=1., v=0., w=0.)
    flags(s, ('y1', 'y2'), u=0., v=1., w=0.)
    s._rebuild()
    return s


def batdorf(R, h, L, b):
    r"""Batdorf's buckling load of the SD panel, `N_{cr} R/(E h^2)`"""
    D = E*h**3/(12*(1 - NU**2))
    best = np.inf
    for m in range(1, 40):
        for j in range(1, 40):
            alpha = m*np.pi/L
            beta = j*np.pi/b
            k2 = alpha**2 + beta**2
            best = min(best, D*k2**2/alpha**2 + E*h/R**2*alpha**2/k2**2)
    return best*R/(E*h**2)


def sd_panel_load(model, theta0, m, n, Rh=100., LR=2.):
    r"""Axial buckling load of the SD panel, `N_{cr} R/(E h^2)`"""
    R = 1.
    h = R/Rh
    s = sd_panel(model, R, h, LR*R, theta0, m, n)
    s.Nxx = -1.
    eigvals, _ = lb(s.calc_kC(), s.calc_kG(), silent=True)
    return eigvals[0].real*R/(E*h**2)


def free_panel(model, R=0.4, h=1.e-3, L=0.3, theta0=0.5, m=12, n=12):
    s = isotropic(model, R, h, L, R*theta0, m, n)
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, e + f, 1.)
            setattr(s, e + f + 'r', 1.)
    s._rebuild()
    return s


def rigid_rotation_constants(s, axis, angle):
    r"""Ritz constants of the finite rigid rotation ``(Q - I) X`` of the
    panel, ``Q`` of the angle ``angle`` about the global axis ``axis``
    through the axis of the cylinder, least squares on a grid

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
            fg(g, x, y, s)
            G += [g[0], g[1], g[2]]
            rhs += [U[0], U @ np.array([0, np.cos(th), -np.sin(th)]),
                    U @ np.array([0, np.sin(th), np.cos(th)])]
    c = np.linalg.lstsq(np.array(G), np.array(rhs), rcond=None)[0]
    return c


def membrane_strain_max(model, axis, angle):
    s = free_panel(model)
    c = rigid_rotation_constants(s, axis, angle)
    _, fields = s.strain(c, gridx=9, gridy=9, NLgeom=True)
    return max(np.abs(fields[k]).max() for k in ('exx', 'eyy', 'gxy'))


@pytest.mark.parametrize('LR, ref_gl, ref_sanders', [
    (50., 0.989099, 1.984334),
    (80., 0.995764, 1.993967),
])
def test_euler_load_of_a_pinned_tube(LR, ref_gl, ref_sanders):
    r"""Column buckling of a pinned tube, `R/h = 100`, `L/R = 50` and `80`

    The critical mode is the bending of the tube as a beam, `w(\theta = \pi)
    = -w(\theta = 0)` and `v(\theta = \pi/2) = -w(\theta = 0)`, and the
    critical load is the Euler load `N_E` with ``cylshell_clpt_gl``, about
    `2 N_E` with ``cylshell_clpt_sanders``. The stored values are those of
    10 x 8 terms; 14 x 10 terms change them by less than 1e-4.

    """
    gl, wr, vr = euler_ratio(GL, LR)
    assert np.isclose(wr, -1., atol=1e-3)
    assert np.isclose(vr, -1., atol=1e-2)
    assert np.isclose(gl, 1., atol=0.02)
    assert np.isclose(gl, ref_gl, rtol=1e-4)
    sanders, wr, vr = euler_ratio(SANDERS, LR)
    assert np.isclose(wr, -1., atol=1e-3)
    assert np.isclose(sanders, 2., atol=0.04)
    assert np.isclose(sanders, ref_sanders, rtol=1e-4)


@pytest.mark.parametrize('axis', [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
@pytest.mark.parametrize('angle', [0.5, 1.0])
def test_rigid_rotation_gives_zero_membrane_strains(axis, angle):
    r"""Finite rigid rotation of a free panel, `R = 0.4`, `L = 0.3`,
    `\theta_0 = 0.5`, about the axis of the cylinder (`x`) and the two
    axes normal to it

    The membrane strains of ``cylshell_clpt_gl`` vanish to the accuracy of
    the least-squares fit of the rotation, those of
    ``cylshell_clpt_sanders`` are of the order of the angle squared (of its
    fourth power for the rotation about the axis, which has no rotation
    about the normal). The changes of curvature, linear, are not checked.

    """
    gl = membrane_strain_max(GL, axis, angle)
    sanders = membrane_strain_max(SANDERS, axis, angle)
    assert gl < 1.e-9
    assert sanders > 1.e-3


def test_linear_kernels_are_those_of_sanders():
    r"""Linear constitutive stiffness and mass matrices equal to those of
    ``cylshell_clpt_sanders``, analytical and numerical geometric stiffness
    matrices of a constant membrane state equal, and different from those
    of ``cylshell_clpt_sanders``"""
    shells = {}
    for model in (GL, SANDERS):
        s = sd_panel(model, 0.4, 1.e-3, 0.3, 0.5, 8, 8)
        s.Nxx, s.Nyy, s.Nxy = -1.3, 0.7, 0.4
        shells[model] = s
    g, sa = shells[GL], shells[SANDERS]
    assert abs(g.calc_kC() - sa.calc_kC()).max() == 0.
    assert abs(g.calc_kM() - sa.calc_kM()).max() == 0.
    kG0 = g.calc_kG()
    kGnum = g.calc_kG(c=np.zeros(g.get_size()))
    scale = abs(kG0).max()
    assert abs(kG0 - kGnum).max() < 1.e-10*scale
    assert abs(kG0 - sa.calc_kG()).max() > 1.e-3*scale


@pytest.mark.parametrize('theta0, m, ref_gl, ref_sanders', [
    (np.pi, 20, 0.560946, 0.585590),
    (2*np.pi/9, 16, 0.594587, 0.600760),
])
def test_axial_buckling_of_sd_panels(theta0, m, ref_gl, ref_sanders):
    r"""Panels of 180 and 40 degrees, `R/h = 100`, `L/R = 2`, supported by
    shear diaphragms, uniform `N_{xx}`, against Batdorf (1947) and
    ``cylshell_clpt_sanders``

    The parameter is `N_{cr} R/(E h^2)`, 0.60523 with Batdorf's Donnell
    equations for both panels, which ``cylshell_clpt_donnell`` reproduces
    (``test_sanders_kinematics.py``). The wider the circumferential waves,
    the larger the in-plane rotations `v_{,x}` against `w_{,x}`: the
    Green-Lagrange strains are 4.2 % below the Sanders strains for the
    panel of 180 degrees and 1.0 % below for that of 40 degrees, with
    shorter waves. The stored values are those of ``m x m`` terms; 24 x 24
    and 20 x 20 terms give the same six digits.

    """
    R = 1.
    ref = batdorf(R, R/100, 2*R, R*theta0)
    assert np.isclose(ref, 0.60523, rtol=1e-4)
    gl = sd_panel_load(GL, theta0, m, m)
    sanders = sd_panel_load(SANDERS, theta0, m, m)
    assert np.isclose(gl, ref_gl, rtol=1e-5)
    assert np.isclose(sanders, ref_sanders, rtol=1e-5)
    assert gl < sanders < ref
    if theta0 < 1.:
        assert np.isclose(gl, sanders, rtol=0.011)
        assert np.isclose(gl, ref, rtol=0.02)
