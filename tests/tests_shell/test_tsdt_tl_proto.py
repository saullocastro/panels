r"""Prototypes of the total-Lagrangian TSDT, options A and B (temporary)

Verifies the kernels of ``'plate_tsdt_tlA_proto'`` (third-order strains
linear) and ``'plate_tsdt_tlB_proto'`` (third-order strains built with the
second-order director and the normal of the deformed mid-surface), see
``theory/shells/plate_tsdt_tl_A`` and ``_B``: linear kernels equal to those
of ``'plate_tsdt_donnell'``, tangent stiffness matrix equal to the
derivative of the internal force vector, and the internal force vector of
a finite rigid rotation of a free plate.

Notes
-----
Under a rigid rotation `t` about an oblique axis the third-order transverse
shear strain of option A, `-3 c_1 (\phi_\alpha + w_{,\alpha})`, is of
`O(t^2)`, so that the internal forces are of `O(t^2)` relative to the
linear ones; option B, whose strains are those of the field `\mathbf{X} +
z \mathbf{d} + z^3 \mathbf{q}`, leaves only the error of the second-order
director, `|\mathbf{d}|^2 - 1 = O(t^4)`.

"""
import os
import sys

import numpy as np
import pytest

from panels.shell import Shell

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdt_tsdt_gl import set_flags
from test_fsdt_tl import plate_rigid_rotation

MODELS = ('plate_tsdt_tlA_proto', 'plate_tsdt_tlB_proto')


def thick_plate(model, free=False, m=8, n=8):
    r"""Unsymmetric laminate, ``a/h = 6``, simply supported or free"""
    s = Shell(model=model, a=0.3, b=0.2, stack=[30, -45, 0, 90], plyt=0.0125,
              laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9), m=m, n=n)
    set_flags(s, ('x1', 'x2', 'y1', 'y2'), ('u', 'v', 'w', 'phix', 'phiy'),
              **{'': 1., 'r': 1.})
    if not free:
        set_flags(s, ('x1', 'x2', 'y1', 'y2'), ('w',), **{'': 0.})
        s.x1u = 0.
        s.y1v = 0.
    s._rebuild()
    return s


@pytest.mark.parametrize('model', MODELS)
def test_linear_kernels_are_those_of_plate_tsdt_donnell(model):
    r"""``KC0`` at zero displacements equal to that of the linear model and
    the internal forces of small displacements equal to ``KC0 c``"""
    t, li = thick_plate(model), thick_plate('plate_tsdt_donnell')
    c0 = np.zeros(t.get_size())
    kC = t.calc_kC(c=c0, NLgeom=True)
    kC_lin = li.calc_kC()
    assert abs(kC - kC_lin).max() < 1.e-12*abs(kC_lin).max()
    rng = np.random.default_rng(0)
    c = 1.e-6*rng.standard_normal(t.get_size())
    f, f_lin = t.calc_fint(c), kC_lin @ c
    assert np.linalg.norm(f - f_lin) < 1.e-4*np.linalg.norm(f_lin)


@pytest.mark.parametrize('model', MODELS)
def test_tangent_is_derivative_of_fint(model):
    r"""``calc_kC(NLgeom=True) + calc_kG`` against central differences of
    ``calc_fint`` at a random state of rotations about 0.3 rad"""
    s = thick_plate(model, m=6, n=5)
    rng = np.random.default_rng(1)
    c = 0.02*rng.standard_normal(s.get_size())
    KT = (s.calc_kC(c=c, NLgeom=True) + s.calc_kG(c=c, NLgeom=True)).toarray()
    h = 1.e-7
    for k in rng.choice(s.get_size(), 12, replace=False):
        dc = np.zeros_like(c)
        dc[k] = h
        fd = (s.calc_fint(c + dc) - s.calc_fint(c - dc))/(2*h)
        assert np.allclose(fd, KT[:, k], rtol=1e-5, atol=1e-6*abs(KT).max())


def rotation_residual(model, axis, angle):
    r"""``|fint(c)|/|KC0 c|`` of the rigid rotation ``c`` of a free plate"""
    s = thick_plate(model, free=True)
    c = plate_rigid_rotation(s, axis, angle)
    return (np.linalg.norm(s.calc_fint(c))
            /np.linalg.norm(s.calc_kC(c=np.zeros_like(c), NLgeom=True) @ c))


@pytest.mark.parametrize('axis', [(1, 2, 3)])
def test_rigid_rotation_residual(axis):
    r"""Oblique axis, 0.2 and 0.1 rad: residual of option A of `O(t^2)`,
    of option B of `O(t^4)` or smaller (ratios about 4 and 16 or more), and
    the residual of option B at 0.2 rad below a fiftieth of that of A"""
    rA = [rotation_residual(MODELS[0], axis, t) for t in (0.2, 0.1)]
    rB = [rotation_residual(MODELS[1], axis, t) for t in (0.2, 0.1)]
    assert 1.7 < np.log2(rA[0]/rA[1]) < 2.3, rA
    assert np.log2(rB[0]/rB[1]) > 3.5, rB
    assert rB[0] < rA[0]/50, (rA, rB)
