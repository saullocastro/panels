r"""Verification of the total-Lagrangian Kirchhoff-Love cylinder, ``cylshell_clpt_tl``

The model has the Green-Lagrange membrane strains of ``cylshell_clpt_gl`` and
the changes of curvature of the deformed normal, whose linear part is that
of ``cylshell_clpt_sanders``. The checks:

1. finite rigid rotations of a free panel, about the axis of the cylinder
   and about the two axes normal to it, give zero strains, membrane and
   bending, and zero internal forces; ``cylshell_clpt_gl`` has zero membrane
   strains but not zero changes of curvature;
2. a bent panel rotated rigidly keeps its six strains, whereas the changes
   of curvature of ``cylshell_clpt_gl`` change;
3. the linear constitutive stiffness of the numerical kernels is that of
   ``cylshell_clpt_sanders`` and ``cylshell_clpt_gl``, and the geometric
   stiffness of a constant membrane state that of ``cylshell_clpt_gl``;
4. the hinged cylindrical roof of Sabir and Lock under a central point load,
   benchmark 3.8 of Sze, Liu and Lo (2004), with 5 x 5 terms: limit load of
   the thick roof (snap-through), against Table 9a, and the path of the
   thin roof through its snap-back, traced with the Riks method, against
   values of the code.

The consistency of the tangent stiffness matrix with the internal force
vector, of ``calc_kTv`` and of the follower pressure are checked with the
other models in ``test_tangent_consistency.py``, ``test_tangent_product.py``
and ``test_follower_pressure.py``, and the connections in
``tests/multidomain/test_bf_sanders.py``.

Notes
-----
With `\boldsymbol{X}` the deformed mid-surface and `\boldsymbol{a} =
\boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y}` its area vector, the
changes of curvature of the model are

.. math::

    \kappa_{xx} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,xx} \qquad
    \kappa_{yy} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,yy} - \frac{1}{r} -
    \frac{\varepsilon_{xx} + 2 \varepsilon_{yy}}{r} \qquad
    \kappa_{xy} = -2 \boldsymbol{a} \cdot \boldsymbol{X}_{,xy} -
    \frac{\gamma_{xy}}{2 r}

built from `\boldsymbol{a}`, `\boldsymbol{X}_{,\alpha\beta}` and the
Green-Lagrange membrane strains, which do not change when a deformed state
is rotated rigidly, `\boldsymbol{X} \to \boldsymbol{Q} \boldsymbol{X}`; for
a rigid motion of the undeformed cylinder `\boldsymbol{a} \cdot
\boldsymbol{X}_{,\alpha\beta}` is the second fundamental form of the
cylinder, `-1/r` for `yy` and zero otherwise, see :mod:`panels.models`. The
linear changes of curvature of ``cylshell_clpt_gl``, those of Sanders [1]_,
are exact for small rotations only.

The hinged roof, [2]_, Sec. 3.8 and Fig. 9a: radius `R = 2540`, axial
length `2L = 508` with free curved edges, half-angle `\theta = 0.1` rad,
straight edges hinged (`u = v = w = 0`), `E = 3102.75`, `\nu = 0.3`, `h =
12.7` and `6.35`, central point load `P`, `P_{max} = 3000`. The reference
values are the load-deflection paths of ABAQUS S4R elements of Tables 9a
and 9d, `P/P_{max}` against the central deflection `-W_C`.

References
----------
.. [1] Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
   Applied Mathematics, 21(1), 21-36, 1963.
   https://doi.org/10.1090/qam/147023
.. [2] Sze, K.Y., Liu, X.H., Lo, S.H., 2004. Popular benchmark problems for
   geometric nonlinear analysis of shells. Finite Elements in Analysis and
   Design 40, 1551-1569. https://doi.org/10.1016/j.finel.2003.11.001

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.transform import Rotation
from structsolve import Analysis, solve

from panels.shell import Shell
from panels.models.clpt_field import fg

TL = 'cylshell_clpt_tl'
GL = 'cylshell_clpt_gl'
SANDERS = 'cylshell_clpt_sanders'
DONNELL = 'cylshell_clpt_donnell'

AXES = {'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}

# Hinged roof, [2] Sec. 3.8 and Fig. 9a
ROOF_R = 2540.
ROOF_A = 508.
ROOF_B = 0.2*ROOF_R
ROOF_E = 3102.75
ROOF_NU = 0.3
PMAX = 3000.

#: [2] Table 9a, h = 12.7: (P/Pmax, -W_C) along the path, read column by
#: column
SZE_9A = np.array([
    (0.0877, 0.693), (0.1980, 1.638), (0.3473, 3.087), (0.4686, 4.477),
    (0.5647, 5.802), (0.6381, 7.057), (0.6908, 8.237), (0.7246, 9.339),
    (0.7412, 10.358), (0.7421, 11.293), (0.7286, 12.141), (0.7023, 12.903),
    (0.6649, 13.583), (0.6182, 14.188), (0.5643, 14.728), (0.5055, 15.217),
    (0.4442, 15.676), (0.3830, 16.125), (0.3245, 16.590), (0.2717, 17.094),
    (0.2272, 17.657), (0.1940, 18.299), (0.1750, 19.028), (0.1729, 19.852),
    (0.1905, 20.771), (0.2303, 21.780), (0.2950, 22.875), (0.3871, 24.049),
    (0.4443, 24.663), (0.5093, 25.293), (0.5826, 25.940), (0.6644, 26.601),
    (0.7551, 27.276), (0.8549, 27.964), (0.9643, 28.663), (1.0835, 29.374)])

#: [2] Table 9d, h = 6.35: (P/Pmax, -W_C) along the path, read column by
#: column; the deflection decreases from 17.008 to 14.451 (snap-back)
SZE_9D = np.array([
    (0.0517, 1.846), (0.1182, 5.271), (0.1583, 8.257), (0.1837, 10.799),
    (0.1914, 11.904), (0.1953, 12.892), (0.1950, 13.752), (0.1901, 14.472),
    (0.1806, 15.050), (0.1671, 15.501), (0.1323, 16.145), (0.0923, 16.602),
    (0.0504, 16.915), (0.0083, 17.008), (-0.0312, 16.697), (-0.0622, 15.780),
    (-0.0739, 15.206), (-0.0861, 14.767), (-0.1001, 14.520),
    (-0.1142, 14.451), (-0.1247, 14.862), (-0.1288, 15.778),
    (-0.1271, 16.961), (-0.1196, 18.320), (-0.1055, 19.817),
    (-0.0825, 21.420), (-0.0484, 23.100), (-0.0006, 24.824),
    (0.0626, 26.565), (0.1427, 28.302), (0.2403, 30.023), (0.3559, 31.720),
    (0.4898, 33.388), (0.6417, 35.024), (0.8114, 36.626), (1.0313, 38.450)])


def hinged_roof(model, h, m, n=None):
    r"""Quarter of the hinged roof, ``0 <= x <= L``, ``0 <= y <= R theta``,
    with the hinged straight edge at ``y = 0`` (``u = v = w = 0``), the free
    curved edge at ``x = 0`` and the planes of symmetry at ``x = L`` (``u =
    v,x = w,x = 0``) and ``y = R theta`` (``v = u,y = w,y = 0``); the centre
    of the roof is the corner ``(L, R theta)``"""
    n = m if n is None else n
    s = Shell(model=model, a=ROOF_A/2, b=ROOF_B/2, r=ROOF_R, stack=[0.],
              plyt=h, laminaprop=(ROOF_E, ROOF_NU), m=m, n=n)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, edge + f, 1.)
            setattr(s, edge + f + 'r', 1.)
    s.y1u = s.y1v = s.y1w = 0.
    s.x2u = s.x2vr = s.x2wr = 0.
    s.y2v = s.y2ur = s.y2wr = 0.
    s._rebuild()
    return s


def centre_vector(s):
    r"""``e`` such that the central deflection is ``w_C = e @ c``"""
    g = np.zeros((5, s.get_size()))
    fg(g, s.a, s.b, s)
    return g[2].copy()


def roof_path(model, h, m, n=None, wmax=None, max_inc=0.01, max_steps=1000,
              kfac=100.):
    r"""Load-deflection path of the centre of the hinged roof under a
    prescribed central deflection, traced with the Riks method

    The deflection `w_C` of the centre is prescribed through a spring of
    stiffness ``K = kfac`` times the linear stiffness of the centre, whose
    other end moves by `\lambda \delta`, `\delta = -w_{max}`, `\lambda` the
    load factor of the Riks method: the external force vector is `\lambda
    K \delta \boldsymbol{e}`, and the spring adds `K \boldsymbol{e}
    \boldsymbol{e}^\top` to the stiffness and `K \boldsymbol{e}
    (\boldsymbol{e}^\top \boldsymbol{c})` to the internal forces. The
    reaction on the roof is the force of the spring, `P = 4 K (w_C -
    \lambda \delta)`, positive downwards, for the four quarters. The path
    is traced until `-w_C \ge w_{max}`, by default 30 for ``h > 10`` and 39
    otherwise. Returns the arrays ``P``, ``wc`` and ``lbd``.

    """
    s = hinged_roof(model, h, m, n)
    e = centre_vector(s)
    c1 = solve(s.calc_kC(), -e, silent=True)
    K = kfac/abs(e @ c1)
    wmax = (30. if h > 10 else 39.) if wmax is None else wmax
    delta = -wmax
    Ke = csr_matrix(K*np.outer(e, e))

    def calc_fext(inc=1., silent=True):
        return inc*K*delta*e

    def calc_fint(c, silent=True):
        return s.calc_fint(c) + K*e*(e @ c)

    def calc_kC(c=None, NLgeom=False, silent=True):
        return s.calc_kC(c=c, NLgeom=NLgeom) + Ke

    an = Analysis(calc_fext, calc_fint, calc_kC, s.calc_kG)
    an.NL_method = 'arc_length_riks'
    an.initialInc = an.maxInc = max_inc
    an.maxArcLength = 1.e6
    an.final_load_factor = None
    an.stop_criterion = lambda an: (-(e @ an.cs[-1]) >= wmax
                                    or len(an.cs) >= max_steps)
    an.static(NLgeom=True, silent=True)
    wc = np.array([e @ c for c in an.cs])
    lbd = np.asarray(an.increments)
    P = 4*K*(wc - lbd*delta)
    return P, wc, lbd


def free_panel(model, m=12, n=12):
    r"""Free panel, `R = 0.4`, `L = 0.3`, `\theta_0 = 0.5`, quasi-isotropic
    laminate"""
    s = Shell(model=model, a=0.3, b=0.4*0.5, r=0.4, m=m, n=n,
              stack=[0, 45, -45, 90, 90, -45, 45, 0], plyt=0.125e-3,
              laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9))
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, edge + f, 1.)
            setattr(s, edge + f + 'r', 1.)
    s._rebuild()
    return s


def local_frame(s, y):
    r"""Global position of the undeformed mid-surface at ``y``, without
    ``x``, and the local basis ``e_y``, ``e_z``, with the global axes ``z``
    through the middle of the panel"""
    th = y/s.r - s.b/s.r/2
    X0 = np.array([0., s.r*np.sin(th), s.r*np.cos(th)])
    ey = np.array([0., np.cos(th), -np.sin(th)])
    ez = np.array([0., np.sin(th), np.cos(th)])
    return X0, ey, ez


def fit(s, disp):
    r"""Ritz constants of the least-squares fit of the local displacements
    of ``disp(x, y)``, a function returning the global displacement vector
    of the point ``(x, y)``"""
    size = s.get_size()
    G, rhs = [], []
    g = np.zeros((5, size))
    for x in np.linspace(0, s.a, 2*s.m + 3):
        for y in np.linspace(0, s.b, 2*s.n + 3):
            _, ey, ez = local_frame(s, y)
            U = disp(x, y)
            fg(g, x, y, s)
            G += [g[0].copy(), g[1].copy(), g[2].copy()]
            rhs += [U[0], U @ ey, U @ ez]
    return np.linalg.lstsq(np.array(G), np.array(rhs), rcond=None)[0]


def rigid(s, axis, angle):
    r"""Ritz constants of the rigid rotation ``angle`` about ``axis``
    through the middle of the panel"""
    Q = Rotation.from_rotvec(angle*np.asarray(axis, dtype=float)).as_matrix()

    def disp(x, y):
        X0, _, _ = local_frame(s, y)
        X = X0 + np.array([x - s.a/2, 0., 0.])
        return (Q - np.eye(3)) @ X

    return fit(s, disp)


def bent(s, axis, angle, amp=0.02):
    r"""Ritz constants of a bent state, ``w = amp (x/a)^2 (1 - y/b) y/b``
    along the normal plus ``u = amp/4 (x/a) (y/b)``, and of the same state
    rotated rigidly by ``angle`` about ``axis``"""
    Q = Rotation.from_rotvec(angle*np.asarray(axis, dtype=float)).as_matrix()

    def deformed(x, y):
        X0, ey, ez = local_frame(s, y)
        X0 = X0 + np.array([x - s.a/2, 0., 0.])
        w = amp*(x/s.a)**2*(1 - y/s.b)*y/s.b
        u = amp/4*(x/s.a)*(y/s.b)
        return X0, X0 + w*ez + u*np.array([1., 0., 0.])

    c0 = fit(s, lambda x, y: np.subtract(*deformed(x, y)[::-1]))
    c1 = fit(s, lambda x, y: Q @ deformed(x, y)[1] - deformed(x, y)[0])
    return c0, c1


def strains(s, c, grid=9):
    _, fields = s.strain(c, gridx=grid, gridy=grid, NLgeom=True)
    return {k: np.ravel(fields[k]) for k in ('exx', 'eyy', 'gxy', 'kxx',
                                             'kyy', 'kxy')}


@pytest.mark.parametrize('axis', list(AXES))
@pytest.mark.parametrize('angle', [0.5, 1.0])
def test_rigid_rotation_gives_zero_strains(axis, angle):
    r"""Finite rigid rotation of a free panel about the axis of the
    cylinder (`x`) and the two axes normal to it

    The six strains of ``cylshell_clpt_tl`` vanish to the accuracy of the
    least-squares fit of the rotation, the membrane strains to 1e-9 and the
    changes of curvature to 1e-7 of `1/r`, as does the internal force
    vector, to 1e-7 of the linear one of the same constants. Those of
    ``cylshell_clpt_gl``: the membrane strains vanish, the changes of
    curvature do not about the axes normal to the axis of the cylinder.

    """
    s = free_panel(TL)
    c = rigid(s, AXES[axis], angle)
    e = strains(s, c)
    assert max(np.abs(e[k]).max() for k in ('exx', 'eyy', 'gxy')) < 1.e-9
    assert max(np.abs(e[k]).max() for k in ('kxx', 'kyy', 'kxy'))*s.r < 1.e-7
    fint = s.calc_fint(c)
    flin = s.calc_kC() @ c
    assert np.linalg.norm(fint) < 1.e-7*np.linalg.norm(flin)
    g = free_panel(GL)
    eg = strains(g, c)
    assert max(np.abs(eg[k]).max() for k in ('exx', 'eyy', 'gxy')) < 1.e-9
    kg = max(np.abs(eg[k]).max() for k in ('kxx', 'kyy', 'kxy'))*s.r
    if axis == 'x':
        assert kg < 1.e-7
    else:
        assert kg > 1.e-2


@pytest.mark.parametrize('axis', ['y', 'z'])
@pytest.mark.parametrize('angle', [0.5, 1.0])
def test_bent_panel_rotated_keeps_its_strains(axis, angle):
    r"""A bent panel rotated rigidly by 0.5 and 1 rad about the axes normal
    to the axis of the cylinder keeps its six strains with
    ``cylshell_clpt_tl``, to 1e-6 of the largest; the changes of curvature
    of ``cylshell_clpt_gl`` change by more than 1 %"""
    s = free_panel(TL)
    c0, c1 = bent(s, AXES[axis], angle)
    e0, e1 = strains(s, c0), strains(s, c1)
    for k in e0:
        scale = np.abs(e0[k]).max()
        assert np.abs(e1[k] - e0[k]).max() < 1.e-6*scale, k
    g = free_panel(GL)
    eg0, eg1 = strains(g, c0), strains(g, c1)
    dk = max(np.abs(eg1[k] - eg0[k]).max()/np.abs(eg0[k]).max()
             for k in ('kxx', 'kyy', 'kxy'))
    assert dk > 1.e-2


def test_linear_kernels_are_those_of_sanders():
    r"""The numerical constitutive stiffness at ``c = 0`` equals the
    analytical one of ``cylshell_clpt_sanders`` and ``cylshell_clpt_gl``,
    and the geometric stiffness of a constant membrane state at ``c = 0``
    that of ``cylshell_clpt_gl``"""
    t, g = free_panel(TL, 6, 6), free_panel(GL, 6, 6)
    for s in (t, g):
        s.x1u = s.x1v = s.x1w = 0.
        s.Nxx, s.Nyy, s.Nxy = -1.3, 0.7, 0.4
        s._rebuild()
    zero = np.zeros(t.get_size())
    kC0 = g.calc_kC()
    kCt = t.calc_kC(c=zero, NLgeom=True)
    scale = abs(kC0).max()
    assert abs(kCt - kC0).max() < 1.e-10*scale
    kGt = t.calc_kG(c=zero)
    kGg = g.calc_kG(c=zero)
    scale = abs(kGg).max()
    assert abs(kGt - kGg).max() < 1.e-10*scale


def first_turn(f, start=0, down=True):
    r"""Index of the first local maximum (``down=True``) or minimum of
    ``f`` from ``start``"""
    d = np.diff(f[start:])
    return start + int(np.argmax(d < 0) if down else np.argmax(d > 0))


def critical_points(P, wc):
    r"""``(P, -w_C)`` at the first maximum of ``P``, the following minimum
    of ``P``, the first maximum of ``-w_C`` and the following minimum of
    ``-w_C``, at the converged increments"""
    w = -np.r_[0., wc]
    P = np.r_[0., P]
    i1 = first_turn(P)
    i2 = first_turn(P, i1, down=False)
    i3 = first_turn(w)
    i4 = first_turn(w, i3, down=False)
    return [(P[i], w[i]) for i in (i1, i2, i3, i4)]


@pytest.mark.slow
def test_hinged_roof_snap_through():
    r"""Hinged roof, `h = 12.7`, quarter with 5 x 5 terms, central
    deflection prescribed up to `-w_C = 22` and traced with the Riks method
    in increments of at most 0.02: the limit load `P_1` and the following
    minimum `P_2` of the reaction exist, `P_1` is within 0.5 % of 2226.3,
    [2], Table 9a (S4R elements), and the stored values are those of the
    code with these terms, at the converged increments, without
    refinement."""
    P, wc, lbd = roof_path(TL, 12.7, 5, wmax=22., max_inc=0.02)
    (P1, w1), (P2, w2), _, _ = critical_points(P, wc)
    assert -wc[-1] >= 22.
    assert P2 < 0.3*P1
    assert np.isclose(P1, 2226.3, rtol=0.005)
    assert np.allclose([P1, w1, P2, w2], [2225.939, 10.47973, 508.4248,
                                          19.23811], rtol=1e-4)


@pytest.mark.slow
def test_hinged_roof_snap_back():
    r"""Hinged roof, `h = 6.35`, quarter with 5 x 5 terms, central
    deflection prescribed up to `-w_C = 20` and traced with the Riks method
    in increments of at most 0.02 through the snap-back: the deflection
    grows to `w_3`, turns back to `w_4` and grows again, and the prescribed
    deflection `\lambda \delta` turns back along the snap-back; the limit
    load `P_1`, the minimum load `P_2` and the turns `w_3`, `w_4` are values
    of the code with these terms, at the converged increments, without
    refinement."""
    P, wc, lbd = roof_path(TL, 6.35, 5, wmax=20., max_inc=0.02)
    (P1, w1), (P2, w2), (P3, w3), (P4, w4) = critical_points(P, wc)
    assert -wc[-1] >= 20.
    # the prescribed displacement turns back with the snap-back
    assert np.any(np.diff(lbd) < 0)
    assert w4 < w3
    assert np.allclose([P1, w1, P2, w2, w3, w4],
                       [607.2394, 13.22098, -384.2222, 16.55691, 17.05234,
                        14.00960], rtol=1e-4)
