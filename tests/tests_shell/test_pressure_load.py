r"""Pressure loads of :meth:`.Shell.add_pressure_load`

References:

- Timoshenko, S. and Woinowsky-Krieger, S., "Theory of plates and shells",
  2nd ed., McGraw-Hill, 1959, Table 8, p. 120: simply supported square plate
  under a uniform pressure, central deflection `w = 0.00406 q a^4/D`, whose
  Navier series is `0.00406235 q a^4/D`.

- Pagano, N. J., "Exact solutions for rectangular bidirectional composites
  and sandwich plates", J. Compos. Mater., 4, 20-34, 1970, as in
  ``test_fsdt_tsdt_plates.py``.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.special import roots_legendre
from structsolve import solve

from panels import modelDB
from panels.shell import Shell

CLPT = 'plate_clpt_donnell'
FSDT = 'plate_fsdt_donnell'
TSDT = 'plate_tsdt_donnell'
CYL = 'cylshell_fsdt_sanders'

E = 70.e9
NU = 0.3


def make_shell(model=CLPT, a=1., b=1., h=0.01, m=10, n=10, r=None,
               stack=(0,), laminaprop=(E, NU), **kwargs):
    s = Shell(a=a, b=b, r=r, stack=list(stack), plyt=h/len(stack),
              laminaprop=laminaprop, m=m, n=n, model=model, **kwargs)
    s._rebuild()
    return s


def fext_by_quadrature(s, func, x1=0., x2=None, y1=0., y2=None, npts=None):
    r"""Independent integration of `\int p(x, y) g_w dx dy` over a patch"""
    x2 = s.a if x2 is None else x2
    y2 = s.b if y2 is None else y2
    fg = modelDB.db[s.model]['field'].fg
    size = s.get_size()
    fext = np.zeros(size)
    g = np.zeros((5, size))
    points, weights = roots_legendre(npts or 2*max(s.m, s.n))
    for xi, wx in zip(points, weights):
        x = x1 + (xi + 1)*(x2 - x1)/2
        for eta, wy in zip(points, weights):
            y = y1 + (eta + 1)*(y2 - y1)/2
            fg(g, x, y, s)
            fext += wx*wy*(x2 - x1)/2*(y2 - y1)/2*func(x, y)*g[2]
    return fext


def central_deflection(s, fext):
    c = solve(s.calc_kC(), fext, silent=True)
    _, fields = s.uvw(c, xs=np.array([s.a/2]), ys=np.array([s.b/2]))
    return fields['w'][0]


@pytest.mark.parametrize('model', [CLPT, FSDT, TSDT, CYL])
def test_against_independent_quadrature(model):
    s = make_shell(model, r=(2. if model == CYL else None))
    q0 = -3.
    func = lambda x, y: q0*np.sin(np.pi*x/s.a)*np.sin(np.pi*y/s.b)
    s.add_pressure_load(func)
    assert np.allclose(s.calc_fext(), fext_by_quadrature(s, func),
                       rtol=1e-12, atol=1e-14)
    s.clear_loads()
    s.add_pressure_load(q0)
    assert np.allclose(s.calc_fext(),
                       fext_by_quadrature(s, lambda x, y: q0),
                       rtol=1e-12, atol=1e-14)


def test_uniform_pressure_timoshenko():
    r"""SSSS isotropic square plate, `w = 0.00406235 q a^4/D`"""
    a = 1.
    h = 0.01
    q = 1.e3
    D = E*h**3/(12*(1 - NU**2))
    s = make_shell(CLPT, a=a, b=a, h=h, m=15, n=15)
    s.add_pressure_load(q)
    w = central_deflection(s, s.calc_fext())
    assert np.isclose(w, 0.00406235*q*a**4/D, rtol=1e-4)


def test_constant_equals_function():
    s = make_shell()
    s.add_pressure_load(2.5)
    f1 = s.calc_fext()
    s.clear_loads()
    s.add_pressure_load(lambda x, y: 2.5)
    assert np.allclose(s.calc_fext(), f1, rtol=1e-14, atol=0)


def test_patches_add_up():
    s = make_shell()
    s.add_pressure_load(-1.)
    full = s.calc_fext()
    s.clear_loads()
    for x1, x2 in ((0., 0.3), (0.3, 1.)):
        for y1, y2 in ((None, 0.55), (0.55, None)):
            s.add_pressure_load(-1., x1=x1, x2=x2, y1=y1, y2=y2)
    assert np.allclose(s.calc_fext(), full, rtol=1e-12, atol=1e-14)


def test_patch_against_independent_quadrature():
    s = make_shell(FSDT)
    func = lambda x, y: 1. + x*y**2
    s.add_pressure_load(func, x1=0.2, x2=0.45, y1=0.6, y2=0.9)
    ref = fext_by_quadrature(s, func, 0.2, 0.45, 0.6, 0.9)
    assert np.allclose(s.calc_fext(), ref, rtol=1e-12, atol=1e-14)


def test_increment():
    s = make_shell()
    s.add_pressure_load(1., cte=True)
    s.add_pressure_load(-2., x1=0.1, x2=0.4, cte=False)
    f0 = s.calc_fext(inc=0.)
    f1 = s.calc_fext(inc=1.)
    f3 = s.calc_fext(inc=3.)
    assert np.allclose(f3 - f0, 3*(f1 - f0), rtol=1e-12, atol=1e-14)
    s.clear_loads()
    s.add_pressure_load(1.)
    assert np.allclose(s.calc_fext(inc=0.), f0, rtol=1e-14, atol=0)
    s.clear_loads()
    assert s.pressure_loads == [] and s.pressure_loads_inc == []
    assert np.all(s.calc_fext() == 0)


def test_partial_domain():
    r"""The patch is intersected with the integration domain"""
    s = make_shell()
    s.x1 = 0.5
    s.add_pressure_load(1.)
    s.add_pressure_load(1., x1=0.1, x2=0.7)
    ref = (fext_by_quadrature(s, lambda x, y: 1., x1=0.5)
           + fext_by_quadrature(s, lambda x, y: 1., x1=0.5, x2=0.7))
    assert np.allclose(s.calc_fext(), ref, rtol=1e-12, atol=1e-14)
    s.clear_loads()
    s.add_pressure_load(1., x1=0.1, x2=0.4)
    assert np.all(s.calc_fext() == 0)


def test_external_pressure_on_cylinder_is_inwards():
    r"""`w` is positive outwards, so an external pressure is negative"""
    s = make_shell(CYL, a=2., b=1., r=1.5, h=0.01)
    s.add_pressure_load(-1.e3)
    assert central_deflection(s, s.calc_fext()) < 0


def test_pagano_sinusoidal_pressure():
    r"""Same as ``test_bending_symmetric_cross_ply_pagano`` for `a/h = 10`"""
    E2 = 1.e9
    a = 1.
    h = 0.1
    q0 = 1.e3
    laminaprop = (25*E2, E2, 0.25, 0.5*E2, 0.5*E2, 0.2*E2)
    refs = {TSDT: 0.7147, FSDT: 0.6628, CLPT: 0.43125}
    for model, ref in refs.items():
        s = make_shell(model, a=a, b=a, h=h, m=12, n=12, stack=[0, 90, 90, 0],
                       laminaprop=laminaprop, fsdt_shear_correction=5/6)
        for edge in ('x1', 'x2'):
            setattr(s, edge + 'u', 1.)
            setattr(s, edge + 'phix', 1.)
            setattr(s, edge + 'phiy', 0.)
        for edge in ('y1', 'y2'):
            setattr(s, edge + 'v', 1.)
            setattr(s, edge + 'phix', 0.)
            setattr(s, edge + 'phiy', 1.)
        s._rebuild()
        s.add_pressure_load(lambda x, y: q0*np.sin(np.pi*x/a)*np.sin(np.pi*y/a))
        wbar = 100*central_deflection(s, s.calc_fext())*E2*h**3/(q0*a**4)
        assert np.isclose(wbar, ref, rtol=2e-4), model


def test_validation():
    s = make_shell()
    with pytest.raises(ValueError, match='outside'):
        s.add_pressure_load(1., x2=1.5)
    with pytest.raises(ValueError, match='outside'):
        s.add_pressure_load(1., y1=-0.1)
    with pytest.raises(ValueError, match='x1 < x2'):
        s.add_pressure_load(1., x1=0.5, x2=0.2)
    with pytest.raises(ValueError, match='y1 < y2'):
        s.add_pressure_load(1., y1=0.5, y2=0.5)
    with pytest.raises(ValueError, match='p must be'):
        s.add_pressure_load(None)
    with pytest.raises((TypeError, ValueError)):
        s.add_pressure_load('a')
