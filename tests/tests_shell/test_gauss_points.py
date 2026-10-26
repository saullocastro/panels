r"""Default numbers of integration points ``nx`` and ``ny``

They follow ``m`` and ``n``, ``gauss_points(m) = max(2*m - 1, 7)``, the
smallest number of Gauss-Legendre points that integrates exactly the
integrands of degree `4 \max(m - 1, 3)` of the internal force vector and of
the tangent stiffness matrix, unless a value is assigned.
"""
import sys
sys.path.append('../..')

import io

import numpy as np
import pytest

from panels.shell import Shell, gauss_points, load


LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)


def shell(model, m, n):
    s = Shell(a=0.3, b=0.2, r=0.4 if 'cylshell' in model else None, m=m,
              n=n, model=model, stack=[0, 45, -45, 90], plyt=0.2e-3,
              laminaprop=LAMINAPROP)
    for e in ('x1', 'x2', 'y1', 'y2'):
        for d in ('u', 'v', 'w', 'phix', 'phiy'):
            for r in ('', 'r'):
                setattr(s, e + d + r, 1.)
    return s


def test_follow_m_n():
    assert [gauss_points(m) for m in (1, 2, 3, 4, 5, 11)] == [7, 7, 7, 7, 9, 21]
    s = Shell(m=8, n=3)
    assert (s.nx, s.ny) == (15, 7)
    s.m = 20
    assert s.nx == 39
    # an assigned value is kept, the rebuild raising it to m
    s = shell('plate_clpt_donnell', 8, 8)
    s.nx = 12
    s.m = 15
    s.calc_kC()
    assert (s.nx, s.ny) == (15, 15)
    s.nx = None
    assert s.nx == 29
    assert Shell(m=4, nx=30).nx == 30


def test_json_io():
    s = shell('plate_clpt_donnell', 8, 5)
    s.ny = 12
    buf = io.BytesIO()
    s.save(buf)
    buf.seek(0)
    s2 = load(buf)
    assert (s2.nx, s2.ny) == (15, 12)
    s2.m, s2.n = 10, 10
    assert (s2.nx, s2.ny) == (19, 12)


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_clpt_sanders',
                                   'cylshell_tsdt_sanders'])
@pytest.mark.parametrize('m', [3, 6])
def test_exact_and_minimal(model, m):
    r"""The default ``nx`` integrates `\boldsymbol{F}_{int}` and `\boldsymbol{K}_T` exactly,
    ``nx - 1`` points do not; ``2*m = 6`` points for ``m = 3`` did not"""
    s = shell(model, m, 4)
    rng = np.random.default_rng(0)
    #NOTE large displacements, such that the terms of degree 4 p dominate
    c = 10.*rng.standard_normal(s.get_size())
    ny = s.ny
    nx = s.nx
    assert nx == max(2*m - 1, 7)
    ref = s.calc_fint(c, nx=nx + 6, ny=ny)
    kref = s.calc_kT(c=c, nx=nx + 6, ny=ny).toarray()
    tol = 1e-12*np.abs(ref).max()
    assert np.allclose(s.calc_fint(c), ref, rtol=0, atol=tol)
    assert np.allclose(s.calc_kT(c=c).toarray(), kref, rtol=0,
                       atol=1e-12*np.abs(kref).max())
    assert not np.allclose(s.calc_fint(c, nx=nx - 1, ny=ny), ref, rtol=0,
                           atol=tol)
