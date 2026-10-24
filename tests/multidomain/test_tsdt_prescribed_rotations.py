r"""Prescribed rotations of TSDT panels with laminate offsets

The rotations ``phix`` and ``phiy`` prescribed with
:meth:`.Shell.add_point_pd`, :meth:`.Shell.add_distr_pd_fixed_x` and
:meth:`.Shell.add_distr_pd_fixed_y`, imposed with ``MultiDomain(...,
pd_method='null-space')``, are the rotations of the normal at the reference
surface `z = 0`,

.. math::

    \theta_x = u_{,z}\big|_{z=0} = \phi_x - \frac{4 d^2}{h^2} (\phi_x +
    w_{,x}) \qquad
    \theta_y = v_{,z}\big|_{z=0} = \Phi_y - \frac{4 d^2}{h^2} (\phi_y +
    w_{,y})

with the offset `d` and the thickness `h` of the panel, the same rotations
tied by the connections, see :mod:`panels.multidomain.connections.kCsdt`.
`\phi_x, \phi_y` are the rotations at the mid-surface of the TSDT field
`u(z) = u + z \phi_x - c_1 [(z - d)^3 + d^3] (\phi_x + w_{,x})`, and `\phi
+ w_{,x}` is the transverse shear strain ``gxz`` (or ``gyz``) of
:meth:`.Shell.strain`. For the FSDT, and for the TSDT with `d = 0`,
`\theta = \phi`.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest

from panels.shell import Shell
import panels.modelDB as modelDB
from panels.multidomain import MultiDomain
from panels.multidomain.connections import nullspace


E2 = 1.e9
LAMINAPROP = (10*E2, E2, 0.25, 0.6*E2, 0.6*E2, 0.5*E2)
STACK = [0, 90, 0, 90]
H = 0.04
THETA = 2.e-3
PHIX, PHIY = 3, 4


def _shell(model, offset, m=8, n=8):
    r"""Panel with `u = v = w = 0` at the edges by the boundary condition
    flags and the rotations free"""
    s = Shell(model=model, a=0.6, b=0.4, r=1. if 'cylshell' in model else
              None, stack=STACK, plyt=H/len(STACK), laminaprop=LAMINAPROP,
              offset=offset, m=m, n=n)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for field in nullspace.FIELDS:
            setattr(s, edge + field, 0. if field in 'uvw' else 1.)
            setattr(s, edge + field + 'r', 1.)
    s._rebuild()
    return s


def _alpha(p):
    return 4*p.offset**2/H**2 if 'tsdt' in p.model else 0.


def _rotations(p, c, xs, ys):
    r"""`\theta_x, \theta_y` at the reference surface and `\phi_x`,
    `\Phi_y` of ``p``, from :meth:`.Shell.uvw` and :meth:`.Shell.strain`"""
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    ys = np.atleast_1d(np.asarray(ys, dtype=float))
    cp = np.ascontiguousarray(c[p.col_start:p.col_end])
    p.uvw(cp, xs=xs, ys=ys)
    phix = p.fields['phix'].ravel().copy()
    phiy = p.fields['phiy'].ravel().copy()
    p.strain(cp, xs=xs, ys=ys, NLgeom=False)
    gxz, gyz = p.fields['gxz'].ravel(), p.fields['gyz'].ravel()
    a = _alpha(p)
    return phix - a*gxz, phiy - a*gyz, phix, phiy


@pytest.mark.parametrize('model', ['plate_tsdt_donnell',
                                   'cylshell_tsdt_sanders',
                                   'plate_fsdt_donnell'])
@pytest.mark.parametrize('offset', [0., 0.3*H, -0.4*H, 0.5*H])
def test_rows_are_the_rotations_at_the_reference_surface(model, offset):
    r"""The rows of the prescribed ``'phix'`` and ``'phiy'`` applied to any
    Ritz constants give `\theta_x` and `\theta_y` evaluated from the field;
    for the FSDT and for `d = 0` they are the rows of ``fg``, as before"""
    p = _shell(model, offset)
    p.col_start = 0
    p.col_end = size = p.get_size()
    c = np.random.default_rng(1).standard_normal(size)
    fg = modelDB.db[model]['field'].fg
    g = np.zeros((5, size))
    for x, y in ((0.1, 0.2), (0., 0.13), (0.6, 0.3), (0.27, 0.)):
        thx, thy, _, _ = _rotations(p, c, x, y)
        Bx = nullspace.pd_constraint_matrix(p, PHIX, x, y, size).toarray()[0]
        By = nullspace.pd_constraint_matrix(p, PHIY, x, y, size).toarray()[0]
        scale = np.abs(c).max()*1e3
        assert np.isclose(Bx @ c, thx[0], rtol=1e-10, atol=1e-12*scale)
        assert np.isclose(By @ c, thy[0], rtol=1e-10, atol=1e-12*scale)
        if _alpha(p) == 0:
            fg(g, x, y, p)
            assert np.allclose(Bx, g[PHIX], rtol=0, atol=1e-12)
            assert np.allclose(By, g[PHIY], rtol=0, atol=1e-12)


def _prescribe(p, kind, comp, value):
    r"""Prescribed rotation at a point, or along the edge `x = 0` (for
    ``'phix'``) or `y = 0` (for ``'phiy'``) with the value `4 \theta s (L -
    s)/L^2`, zero at the corners, where `w_{,x} = w_{,y} = 0` and for `d =
    \pm h/2` the rotation at `z = 0`, `-w_{,x}` or `-w_{,y}` (plus `v/r`),
    cannot take another value; returns the points and the prescribed
    values"""
    if kind == 'point':
        if comp == PHIX:
            p.add_point_pd(0.2, 0.15, None, None, None, None, None, None,
                           kphix=1., phixp=value)
        else:
            p.add_point_pd(0.2, 0.15, None, None, None, None, None, None,
                           kphiy=1., phiyp=value)
        return np.array([0.2]), np.array([0.15]), np.array([value])
    s = np.linspace(0., 1., 9)
    if comp == PHIX:
        func = lambda y: 4*value*y*(p.b - y)/p.b**2
        p.add_distr_pd_fixed_x(0., kphix=1., funcphix=func)
        return np.zeros_like(s), s*p.b, func(s*p.b)
    func = lambda x: 4*value*x*(p.a - x)/p.a**2
    p.add_distr_pd_fixed_y(0., kphiy=1., funcphiy=func)
    return s*p.a, np.zeros_like(s), func(s*p.a)


@pytest.mark.parametrize('model', ['plate_tsdt_donnell',
                                   'cylshell_tsdt_sanders'])
@pytest.mark.parametrize('kind', ['point', 'line'])
@pytest.mark.parametrize('comp', [PHIX, PHIY])
@pytest.mark.parametrize('offset', [0.3*H, -0.4*H, 0.5*H])
def test_prescribed_rotation_with_offset(model, kind, comp, offset):
    r"""Pressure loaded panel with a prescribed rotation, at a point or
    along an edge: the rotation at the reference surface evaluated from the
    field is the prescribed value, not the rotation `\phi` at the
    mid-surface, which differs by `\alpha \gamma` with the transverse shear
    strain `\gamma \neq 0`"""
    p = _shell(model, offset)
    p.add_pressure_load(-2.e5)
    xs, ys, values = _prescribe(p, kind, comp, THETA)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    c = md.linear_static()
    thx, thy, phix, phiy = _rotations(p, c, xs, ys)
    theta, phi = (thx, phix) if comp == PHIX else (thy, phiy)
    assert np.allclose(theta, values, rtol=0, atol=1e-8*THETA)
    # the shear strain is not zero at the prescription, phi != theta
    assert np.abs(phi - values).max() > 1e-3*THETA


@pytest.mark.parametrize('model', ['plate_tsdt_donnell', 'plate_fsdt_donnell'])
@pytest.mark.parametrize('kind', ['point', 'line'])
@pytest.mark.parametrize('comp', [PHIX, PHIY])
def test_prescribed_rotation_without_offset_is_phi(model, kind, comp):
    r"""For the TSDT with `d = 0` and for the FSDT with any offset the
    prescribed rotation is `\phi`, as in the previous versions"""
    offset = 0. if 'tsdt' in model else 0.3*H
    p = _shell(model, offset)
    p.add_pressure_load(-2.e5)
    xs, ys, values = _prescribe(p, kind, comp, THETA)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    c = md.linear_static()
    _, _, phix, phiy = _rotations(p, c, xs, ys)
    phi = phix if comp == PHIX else phiy
    assert np.allclose(phi, values, rtol=0, atol=1e-8*THETA)
