r"""Verification of the total-Lagrangian TSDT plate, ``plate_tsdt_tl``

The model has the Green-Lagrange membrane strains of ``plate_tsdt_gl``, the
curvatures and transverse shear strains of the second-order director of
``plate_fsdt_tl``, and third-order strains built with the director and the
normal of the deformed mid-surface. The checks:

1. the linear kernels are those of ``plate_tsdt_donnell``, and the
   numerical geometric stiffness matrix of a constant membrane state is the
   analytical one of ``plate_tsdt_gl``;
2. a finite rigid rotation of a free plate, with the rotations the
   components of the rotated normal, gives zero membrane, bending and
   third-order bending strains and transverse shear strains of the fifth
   order in the angle; the internal forces of a rotation about an oblique
   axis are of the fifth order, those of ``plate_tsdt_gl`` of the second;
3. the transverse shear strains at the faces, `\gamma + (h^2/4)
   \gamma^{(2)}`, vanish at any state;
4. a thin cantilever strip bent out of its plane through moderate
   rotations, against the elastica, which ``plate_tsdt_gl`` does not
   follow;
5. a thick strip, `L/h = 5`, against the geometrically exact (Reissner)
   beam;
6. the tangent stiffness matrix is the derivative of the internal force
   vector, also with a follower pressure, and ``calc_kTv`` is its product
   by a vector.

Notes
-----
The deformed position of the point at `z` is `\mathbf{X} + z \mathbf{d} +
z^3 \mathbf{q}`, `\mathbf{q} = -c_1 (\mathbf{d} - \mathbf{N})`, with the
second-order director `\mathbf{d} = (\phi_x, \phi_y, 1 - (\phi_x^2 +
\phi_y^2)/2)` of [1]_, Eq. (16), with the transverse normal strain of [2]_,
Eq. (20), set to zero, the normal `\mathbf{N} = \mathbf{X}_{,x} \times
\mathbf{X}_{,y}` and `c_1 = 4/(3 h^2)`, whose linear part is the field of
Reddy [3]_. All strains are dot products of vectors that rotate with the
body; under a rigid rotation they change only through `|\mathbf{d}|^2 - 1
= (\phi_x^2 + \phi_y^2)^2/4`, `O(t^4)`, and the transverse shear strains of
the undeformed plate are `O(t^5)`. Since `\mathbf{X}_{,\alpha} \cdot
\mathbf{N} = 0`, `\gamma^{(2)}_\alpha = -3 c_1 \gamma_\alpha` and the shear
strain `\gamma_\alpha (1 - 4 z^2/h^2)` vanishes at `z = \pm h/2`.

The cantilever strips, `L = 1`, isotropic with `\nu = 0`, clamped at `x =
0` (`w_{,x} = 0` too), are loaded at the tip by a force of constant
direction normal to their plane, `\alpha = P L^2/EI`. The thin strip, `b =
0.04`, `h = 0.002`, follows the inextensible elastica, `\theta'' = -\alpha
\cos\theta` (solved by shooting, ``test_plate_clpt_gl.elastica``): at
`\alpha = 1` the tip rotates by 0.46 rad, `w = 0.3017208 L`, `-u =
0.0564332 L`, against `w = L/3` of the linear beam, which the linear
curvatures of ``plate_tsdt_gl`` reproduce. The thick strip, `b = h = 0.2`,
is compared with the geometrically exact beam of Reissner [4]_, `\mathbf{r}'
= (1 + e) \mathbf{a}_1 + g \mathbf{a}_3`, `e = P \sin\theta/EA`, `g = P
\cos\theta/kGA`, `\theta' = M/EI`, `M' = -P x'`, with the shear stiffness
`kGA` such that the linear tip deflection of the beam, `P L^3/(3 EI) + P
L/kGA`, is that of the model.

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
.. [3] Reddy, J. N., "Mechanics of Laminated Composite Plates and Shells:
   Theory and Analysis", 2nd ed., CRC Press, 2003.
   https://doi.org/10.1201/b12409
.. [4] Reissner, E., "On one-dimensional finite-strain beam theory: the
   plane problem", Zeitschrift für angewandte Mathematik und Physik, Vol.
   23, 1972, pp. 795-804. https://doi.org/10.1007/BF01602645

"""
import os
import sys

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from structsolve import Analysis

from panels.shell import Shell

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_plate_clpt_gl import elastica
from test_fsdt_tsdt_gl import set_flags
from test_fsdt_tl import plate_rigid_rotation
from test_tangent_consistency import make_shell, random_state
from test_follower_pressure import add_pressure

MODEL = 'plate_tsdt_tl'
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
STRAINS = ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'kxx3', 'kyy3', 'kxy3',
           'gyz', 'gxz', 'gyz2', 'gxz2')


def thick_plate(model, free=False, m=8, n=8):
    r"""Unsymmetric laminate, ``a = 0.3``, ``b = 0.2``, ``a/h = 6``, simply
    supported or free"""
    s = Shell(model=model, a=0.3, b=0.2, stack=[30, -45, 0, 90], plyt=0.0125,
              laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9), m=m, n=n)
    set_flags(s, ('x1', 'x2', 'y1', 'y2'), FIELDS, **{'': 1., 'r': 1.})
    if not free:
        set_flags(s, ('x1', 'x2', 'y1', 'y2'), ('w',), **{'': 0.})
        s.x1u = 0.
        s.y1v = 0.
    s._rebuild()
    return s


@pytest.mark.slow
def test_linear_kernels_are_those_of_the_linear_model():
    r"""Constitutive stiffness at zero displacements and mass matrix equal
    to those of ``plate_tsdt_donnell``; the numerical geometric stiffness
    matrix of a constant membrane state at zero displacements equal to the
    analytical one of ``plate_tsdt_gl``, which the model uses; internal
    forces of small displacements equal to ``KC0 c``"""
    t, li, gl = [thick_plate(m) for m in (MODEL, 'plate_tsdt_donnell',
                                          'plate_tsdt_gl')]
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


@pytest.mark.slow
@pytest.mark.parametrize('axis', [(1, 0, 0), (0, 1, 0), (1, 2, 3)])
def test_rigid_rotation_strains(axis):
    r"""Rigid rotation of a free plate by 0.2 and 0.1 rad: membrane,
    bending and third-order bending strains zero to the accuracy of the
    fit, transverse shear strains of the fifth order in the angle (ratio
    about 32)"""
    s = thick_plate(MODEL, free=True)
    shear = []
    for angle in (0.2, 0.1):
        c = plate_rigid_rotation(s, axis, angle)
        _, f = s.strain(c, gridx=9, gridy=9, NLgeom=True)
        assert max(np.abs(f[k]).max() for k in STRAINS[:9]) < 1.e-8
        shear.append(max(np.abs(f[k]).max() for k in STRAINS[9:]))
    assert 4.7 < np.log2(shear[0]/shear[1]) < 5.3, shear


def rotation_residual(model, axis, angle):
    r"""``|fint(c)|/|KC0 c|`` of the rigid rotation ``c`` of a free plate"""
    s = thick_plate(model, free=True)
    c = plate_rigid_rotation(s, axis, angle)
    return (np.linalg.norm(s.calc_fint(c))
            /np.linalg.norm(s.calc_kC(c=np.zeros_like(c), NLgeom=True) @ c))


@pytest.mark.slow
def test_rigid_rotation_internal_forces():
    r"""Rotation about the oblique axis ``(1, 2, 3)`` by 0.2 and 0.1 rad,
    ``KC0 c`` being of `O(t^2)`: the residual is of `O(t^3)` (internal
    forces of `O(t^5)`, ratio about 8), that of ``plate_tsdt_gl``, whose
    linear curvatures and third-order strains do not rotate, of `O(1)`.
    Stored values: 4.250e-6 and 5.253e-7; ``plate_tsdt_gl`` 3.283e-2"""
    r = [rotation_residual(MODEL, (1, 2, 3), t) for t in (0.2, 0.1)]
    assert np.log2(r[0]/r[1]) > 2.7, r
    assert np.allclose(r, [4.250e-6, 5.253e-7], rtol=2e-3)
    r_gl = rotation_residual('plate_tsdt_gl', (1, 2, 3), 0.2)
    assert np.isclose(r_gl, 3.283e-2, rtol=2e-3)


def test_traction_free_faces():
    r"""At a random state of rotations of about 0.3 rad, the transverse
    shear strains at the faces `z = \pm h/2`, ``g + g2/(3 c1)``, vanish at
    every point, while ``g`` does not"""
    s = thick_plate(MODEL, m=6, n=5)
    rng = np.random.default_rng(1)
    c = 0.02*rng.standard_normal(s.get_size())
    _, f = s.strain(c, gridx=7, gridy=7, NLgeom=True)
    h = sum(s.plyts)
    c1 = 4/(3*h**2)
    for g, g2 in (('gyz', 'gyz2'), ('gxz', 'gxz2')):
        assert np.abs(f[g]).max() > 1e-3
        assert np.abs(f[g] + f[g2]/(3*c1)).max() < 1e-12*np.abs(f[g]).max()


def strip(model, L, b, h, alpha, inc, NLgeom=True, m=14, n=4, E=70e9):
    r"""Tip ``(w, -u)/L``, mean over the tip, of a strip clamped at ``x =
    0`` (``w,x = 0`` too) under a tip force along ``z``, ``alpha = P
    L^2/EI``, ``nu = 0``"""
    P = alpha*E*h**3*b/12/L**2
    s = Shell(model=model, a=L, b=b, stack=[0], plyt=h, laminaprop=(E, 0.),
              m=m, n=n)
    set_flags(s, ('x2', 'y1', 'y2'), FIELDS, **{'': 1., 'r': 1.})
    set_flags(s, ('x1', ), FIELDS, **{'': 0., 'r': 1.})
    s.x1wr = 0.
    s._rebuild()
    for y in np.linspace(0, b, 5):
        s.add_point_load(L, y, 0., 0., P/5, cte=False)
    an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    an.initialInc = an.maxInc = inc
    an.static(NLgeom=NLgeom, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    f = s.uvw(an.cs[-1], xs=np.full(5, L), ys=np.linspace(0, b, 5))[1]
    return f['w'].mean()/L, -f['u'].mean()/L


@pytest.mark.slow
def test_thin_cantilever_strip_against_elastica():
    r"""Thin strip, ``alpha = 1``, tip rotation 0.46 rad: the elastica gives
    ``w = 0.3017208 L``, ``-u = 0.0564332 L``; ``plate_tsdt_tl`` follows it
    within -0.22 % and -0.47 % (second-order director), ``plate_tsdt_gl``
    gives the deflection of the linear beam, ``w = L/3``, +10.5 %. Stored
    values of ``plate_tsdt_tl``, 14 x 4 terms: ``w = 0.3010715 L``, ``-u =
    0.0561673 L``"""
    w_ref, u_ref, _ = elastica(1.)
    assert np.isclose(w_ref, 0.3017208, rtol=1e-6)
    w, u = strip(MODEL, 1., 0.04, 2e-3, 1., 0.1)
    assert np.isclose(w, w_ref, rtol=0.003)
    assert np.isclose(u, u_ref, rtol=0.006)
    assert np.isclose(w, 0.3010715, rtol=1e-5)
    assert np.isclose(u, 0.0561673, rtol=1e-5)
    w, u = strip('plate_tsdt_gl', 1., 0.04, 2e-3, 1., 0.1)
    assert np.isclose(w, 1/3, rtol=0.001)


def reissner(P, L, EI, EA, kGA):
    r"""Tip ``(w, -u)/L`` and rotation of the geometrically exact beam of
    Reissner, clamped at ``s = 0``, ``M(L) = 0``, under the tip force ``P``
    of constant direction normal to the undeformed axis"""
    def rhs(s, y):
        th, M, x, z = y
        e, g = P*np.sin(th)/EA, P*np.cos(th)/kGA
        xp = (1 + e)*np.cos(th) - g*np.sin(th)
        zp = (1 + e)*np.sin(th) + g*np.cos(th)
        return [M/EI, -P*xp, xp, zp]

    def end(M0):
        return solve_ivp(rhs, [0, L], [0., M0, 0., 0.], rtol=1e-11, atol=1e-14)
    M0 = brentq(lambda M0: end(M0).y[1, -1], 1e-6*P*L, 1.0001*P*L)
    y = end(M0).y[:, -1]
    return y[3]/L, (L - y[2])/L, y[0]


@pytest.mark.slow
def test_thick_strip_against_reissner_beam():
    r"""Thick strip, ``L = 1``, ``b = h = 0.2``, ``alpha = 3``, tip
    rotation 0.98 rad, against the Reissner beam with the shear stiffness
    of the linear deflection of the model (``kGA = 0.8353 GA``): ``w``
    within -0.81 %, ``-u`` within -0.20 %. Stored values, 14 x 4 terms:
    ``w = 0.614051 L``, ``-u = 0.257283 L``"""
    L, b, h, E = 1., 0.2, 0.2, 70e9
    EI, EA = E*h**3*b/12, E*h*b
    P = 3*EI/L**2
    wl, _ = strip(MODEL, L, b, h, 3., 0.05, NLgeom=False)
    kGA = P*L/(wl*L - P*L**3/(3*EI))
    assert np.isclose(kGA/(E/2*b*h), 0.8353, rtol=1e-3)
    w_ref, u_ref, _ = reissner(P, L, EI, EA, kGA)
    w, u = strip(MODEL, L, b, h, 3., 0.05)
    assert np.isclose(w, w_ref, rtol=0.01)
    assert np.isclose(u, u_ref, rtol=0.005)
    assert np.allclose([w, u], [0.614051, 0.257283], rtol=1e-5)


def taylor_errors(fint, KT, c, d):
    r"""``|fint(c + h d) - fint(c) - h KT d|/|h KT d|`` for ``h = 1e-2``
    to ``1e-6``"""
    f0 = fint(c)
    KTd = KT(c) @ d
    return [np.linalg.norm(fint(c + h*d) - f0 - h*KTd)/(h*np.linalg.norm(KTd))
            for h in (1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6)]


def check_tangent(model):
    r"""Taylor test of ``calc_kT`` against ``calc_fint`` at a random state,
    the same with a follower pressure (residual ``fint - fext``), and
    ``calc_kTv`` against ``calc_kT``; shell of
    ``test_tangent_consistency.make_shell``"""
    s = make_shell(model)
    rng = np.random.default_rng(0)
    c, d = random_state(s, rng), random_state(s, rng)
    fint = lambda cc: np.asarray(s.calc_fint(c=np.ascontiguousarray(cc)))
    KT = lambda cc: s.calc_kT(c=np.ascontiguousarray(cc)).toarray()
    errors = taylor_errors(fint, KT, c, d)
    for prev, err in zip(errors[:-1], errors[1:]):
        assert err < 0.2*prev, errors
    assert errors[-1] < 1.e-4, errors
    kTv = s.calc_kTv(c, d)
    assert np.allclose(kTv, KT(c) @ d, rtol=1e-10, atol=1e-10*np.abs(kTv).max())
    add_pressure(s, 'constant', follower='quadratic', scale=100.)
    inc = 0.7
    fres = lambda cc: np.asarray(s.calc_fint(np.ascontiguousarray(cc), inc=inc))
    KTf = lambda cc: s.calc_kT(c=np.ascontiguousarray(cc), inc=inc).toarray()
    errors = taylor_errors(fres, KTf, c, d)
    for prev, err in zip(errors[:-1], errors[1:]):
        assert err < 0.2*prev, errors


def test_tangent_consistency():
    r"""``calc_kC(NLgeom=True) + calc_kG`` is the derivative of
    ``calc_fint`` (error falling by ten per decade of the step), also with
    a follower pressure, and ``calc_kTv`` its product by a vector"""
    check_tangent(MODEL)
