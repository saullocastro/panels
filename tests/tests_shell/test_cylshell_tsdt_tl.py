r"""Verification of the total-Lagrangian TSDT cylinder, ``cylshell_tsdt_tl``

The model has the Green-Lagrange membrane strains of ``cylshell_tsdt_gl``,
the curvatures and transverse shear strains of the second-order director
of ``cylshell_fsdt_tl``, and third-order strains built with the director
and the normal of the deformed mid-surface. The checks:

1. the linear kernels are those of ``cylshell_tsdt_sanders``, and the
   numerical geometric stiffness matrix of a constant membrane state is the
   analytical one of ``cylshell_tsdt_gl``;
2. a finite rigid rotation of a free cylindrical panel gives zero membrane
   strains and the other strains of the fourth order in the angle, against
   the second order for the linear curvatures of ``cylshell_tsdt_gl``;
3. the transverse shear strains at the faces vanish at any state;
4. column (Euler) buckling of a long pinned tube, against the Euler load;
5. the hinged cylindrical roof of Sabir and Lock under a central point
   load, benchmark 3.8 of Sze, Liu and Lo (2004): limit load and following
   minimum of the snap-through;
6. the tangent stiffness matrix is the derivative of the internal force
   vector, also with a follower pressure, and ``calc_kTv`` is its product
   by a vector;
7. an annular web of ``annular_tsdt_tl`` tied to the inner surface of the
   cylinder with ``'BFxcte'``: with small models, the frequencies of the
   same web clamped directly, above the roots of the clamped-free annulus;
   and rigid-body motions of a free cylinder with two webs without strain
   energy.

Notes
-----
The deformed position of the point at `z` is `\mathbf{X} + z \mathbf{d} +
z^3 \mathbf{q}`, `\mathbf{q} = -c_1 (\mathbf{d} - \mathbf{N})`, with the
second-order director `\mathbf{d} = (\psi_x, \psi_y, 1 - (\psi_x^2 +
\psi_y^2)/2)` in the local basis, `\psi_x = \phi_x`, `\psi_y = \phi_y +
v/r`, of [1]_, Eq. (16), with the transverse normal strain of [2]_, Eq.
(20), set to zero, the normal `\mathbf{N} = \mathbf{X}_{,x} \times
\mathbf{X}_{,y}` and `c_1 = 4/(3 h^2)`; `\kappa^{(3)}_{yy}` has the term
`-c_1 (\varepsilon_{xx} + \varepsilon_{yy})/r` with the Green-Lagrange
membrane strains, so that the linear part is that of Reddy's TSDT with
Sanders' kinematics. The strains change under a rigid rotation only through
`|\mathbf{d}|^2 - 1`, `O(t^4)`.

The Euler load of a pinned tube of radius `R`, wall thickness `h` and
length `L`, per unit length of the circumference, is `N_E = \pi^2 E R^2
h/(2 L^2)` (``test_fsdt_tsdt_gl.py``); linear buckling uses the analytical
kernels of ``cylshell_tsdt_gl``.

The hinged roof [3]_, Sec. 3.8, Fig. 9a: `R = 2540`, `L = 508`, `\theta =
0.1` rad, `E = 3102.75`, `\nu = 0.3`, `h = 12.7`, straight edges hinged,
curved edges free, a point load at the centre, modelled as in
``test_fsdt_tsdt_gl.py`` (quarter, symmetry, central deflection prescribed
through a stiff spring, Riks method). Table 9a of [3]_ (S4R elements): limit
load 2226.3 at `-W_C = 11.293`, following minimum 518.7 at 19.852.

The web on the cylinder is the small one of ``test_annular_cylinder.py``
(``check_web_clamped``): `h/r_o = 0.001`, `\beta = 0.4`, `m = 4`, `n = 6`
terms for the half cylinders and the half webs, the web clamped outside by
the cylinder clamped at `x = 0` and free inside. Its five lowest
`\lambda^2` and those of the same web clamped directly on its outer edge
are values of the code: they agree to 6.2e-10 with the null space
(tolerance 1e-8) and differ by 0.20 % to 0.29 % with the penalty
connection and its default constants (tolerance 5e-3); with a
``cylshell_tsdt_sanders`` cylinder the connected web gives the same values
to the last digit (hpc12). The directly clamped web is a Rayleigh-Ritz
approximation of the annulus clamped outside and free inside, bounded from
below by the roots of the frequency equation of Leissa (1969), Eqs.
(1.11)-(1.13) and (1.17), and above them by 0.37 % (`n = 0`) to 6.83 % (`n
= 2`): ``EXCESS`` of ``test_annular_cylinder.py``, 7 %, is a regression
bound of the small model.

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
.. [3] Sze, K. Y., Liu, X. H., and Lo, S. H., "Popular benchmark problems
   for geometric nonlinear analysis of shells", Finite Elements in
   Analysis and Design, Vol. 40, No. 11, 2004, pp. 1551-1569.
   https://doi.org/10.1016/j.finel.2003.11.001

"""
import os
import sys

import numpy as np
import pytest

from panels.multidomain.connections import nullspace

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'multidomain'))
from test_fsdt_tsdt_gl import panel, euler_ratio, roof_path, limit_and_minimum
from test_fsdt_tl import cylinder_rigid_rotation
from test_plate_tsdt_tl import check_tangent
import test_annular_cylinder as tac

MODEL = 'cylshell_tsdt_tl'
STRAINS = ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'kxx3', 'kyy3', 'kxy3',
           'gyz', 'gxz', 'gyz2', 'gxz2')


def test_linear_kernels_are_those_of_the_linear_model():
    r"""Constitutive stiffness at zero displacements and mass matrix equal
    to those of ``cylshell_tsdt_sanders``; the numerical geometric stiffness
    matrix of a constant membrane state at zero displacements equal to the
    analytical one of ``cylshell_tsdt_gl``, which the model uses; internal
    forces of small displacements equal to ``KC0 c``"""
    t, li, gl = [panel(m) for m in (MODEL, 'cylshell_tsdt_sanders',
                                    'cylshell_tsdt_gl')]
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


def rotation_strains(model, axis, angle):
    r"""Largest membrane strain and largest other strain of the rigid
    rotation, free panel `R = 0.4`, `L = 0.3`, `\theta_0 = 0.5`, 12 x 12
    terms"""
    s = panel(model, free=True, m=12, n=12)
    c = cylinder_rigid_rotation(s, axis, angle)
    _, f = s.strain(c, gridx=9, gridy=9, NLgeom=True)
    return (max(np.abs(f[k]).max() for k in STRAINS[:3]),
            max(np.abs(f[k]).max() for k in STRAINS[3:]))


@pytest.mark.parametrize('axis', [(0, 1, 0), (0, 0, 1)])
def test_rigid_rotation(axis):
    r"""Rigid rotation of a free cylindrical panel by 0.2 and 0.1 rad about
    the two axes normal to the axis of the cylinder: zero membrane strains;
    the other strains of the fourth order in the angle (ratio about 16),
    against the second order (ratio about 4) for ``cylshell_tsdt_gl`` with
    the same fields"""
    tl, gl = [], []
    for angle in (0.2, 0.1):
        mem, other = rotation_strains(MODEL, axis, angle)
        assert mem < 1e-9
        tl.append(other)
        gl.append(rotation_strains('cylshell_tsdt_gl', axis, angle)[1])
    assert np.log2(tl[0]/tl[1]) > 3.6, tl
    assert 1.8 < np.log2(gl[0]/gl[1]) < 2.2, gl
    assert tl[1] < 0.05*gl[1]


def test_traction_free_faces():
    r"""At a random state, the transverse shear strains at the faces `z =
    \pm h/2`, ``g + g2/(3 c1)``, vanish at every point, while ``g`` does
    not"""
    s = panel(MODEL, m=6, n=5)
    rng = np.random.default_rng(1)
    c = 2e-4*rng.standard_normal(s.get_size())
    _, f = s.strain(c, gridx=7, gridy=7, NLgeom=True)
    h = sum(s.plyts)
    c1 = 4/(3*h**2)
    for g, g2 in (('gyz', 'gyz2'), ('gxz', 'gxz2')):
        assert np.abs(f[g]).max() > 1e-4
        assert np.abs(f[g] + f[g2]/(3*c1)).max() < 1e-12*np.abs(f[g]).max()


def test_euler_load_of_a_pinned_tube():
    r"""Column buckling of a pinned tube, `R/h = 100`, `L/R = 50`, 10 x 8
    terms: the beam mode at 0.9890986 `N_E`, the value of
    ``cylshell_tsdt_gl`` (``test_fsdt_tsdt_gl.py``), whose analytical kernels
    linear buckling uses"""
    ratio, wr = euler_ratio(MODEL, 50.)
    assert np.isclose(wr, -1., atol=1e-3)
    assert np.isclose(ratio, 0.9890986, rtol=1e-4)


PMAX = 3000.


def test_hinged_roof_snap_through():
    r"""Hinged roof, `h = 12.7`, quarter with 8 x 8 terms: the limit load
    and the following minimum of the reaction against [3], Table 9a, 2226.3
    at `-W_C = 11.293` and 518.7 at 19.852 (S4R elements): 2218.14 at
    10.902 (-0.37 %, -3.5 %) and 507.21 at 19.455 (-2.2 %, -2.0 %);
    ``cylshell_fsdt_gl`` with 8 x 8 terms gives 2217.70 at 10.907 and
    493.22 at 19.417 (``test_fsdt_tsdt_gl.py``). The stored values are those
    of the converged increments, without refinement."""
    P, wc = roof_path(MODEL, 12.7, 8)
    (P1, w1), (P2, w2) = limit_and_minimum(P, wc)
    assert -wc[-1] >= 30.
    assert np.isclose(P1, 2226.3, rtol=0.005)
    assert np.isclose(w1, 11.293, rtol=0.05)
    assert np.isclose(P2, 518.7, rtol=0.06)
    assert np.isclose(w2, 19.852, rtol=0.03)
    assert np.allclose([P1, w1, P2, w2], [2218.145, 10.90227, 507.2120,
                                          19.45469], rtol=1e-5)


def test_tangent_consistency():
    r"""``calc_kC(NLgeom=True) + calc_kG`` is the derivative of
    ``calc_fint`` (error falling by ten per decade of the step), also with
    a follower pressure, and ``calc_kTv`` its product by a vector"""
    check_tangent(MODEL)


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_annular_web_on_the_inner_surface(conn_method):
    r"""Small web of ``annular_tsdt_tl``, `h/r_o = 0.001`, its outer edge
    tied to the inner surface (``face='lower'``) of the small cylinder
    clamped at `x = 0`: the five lowest frequency parameters `\lambda^2`
    against those of the same web clamped directly on its outer edge, and
    the latter between the roots of the clamped-free annulus and these
    roots plus the regression bound ``EXCESS``, see ``check_web_clamped``
    of ``test_annular_cylinder.py``"""
    tac.check_web_clamped(MODEL, 'annular_tsdt_tl', 'lower', conn_method)


def test_rigid_body_motions_with_webs():
    r"""Free cylinder with two webs of ``annular_tsdt_tl`` on its faces
    (null-space method): the translation along the axis and the rotation
    about it give no strain energy and satisfy the constraints"""
    md, cyls, webs, conn_webs = tac._frames(MODEL, conn_method=None,
                                            web='annular_tsdt_tl', faces=True)
    K = md.calc_kC(silent=True)
    size = md.get_size()
    for c in tac._rigid(md, cyls, webs):
        assert abs(c @ (K @ c)) < 1e-10*abs(K.diagonal()).max()*(c @ c)
        for ci in md.conn:
            B = nullspace.constraint_matrix(ci, size)
            assert np.abs(B @ c).max() < 1e-10*np.abs(c).max()
