r"""Base-flange connection ``'BFycte'`` with Sanders-Koiter skins

The rotation penalty of ``'BFycte'`` uses the rotation of the normal of each
panel about `x`, `\omega = w_{,y} - v/r` for the Sanders-Koiter kinematics
and `\omega = w_{,y}` for flat plates and the Donnell kinematics, see
:mod:`panels.multidomain.connections.kCBFycte`.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import kCBFycte


LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
R = 0.4


def _panels(base_model, seed=None):
    base = Shell(group='base', model=base_model, x0=0, y0=0, a=0.6, b=0.3,
                 m=6, n=7, stack=[0, 45, -45, 90], plyt=0.2e-3,
                 laminaprop=LAMINAPROP)
    if 'cylshell' in base_model:
        base.r = R
    flange = Shell(group='flange', model='plate_clpt_donnell', x0=0, y0=0,
                   a=0.6, b=0.05, m=6, n=5, stack=[0, 90, 90, 0],
                   plyt=0.2e-3, laminaprop=LAMINAPROP)
    rng = np.random.default_rng(seed)
    for p in (base, flange):
        for e in ('x1', 'x2', 'y1', 'y2'):
            for d in ('u', 'v', 'w'):
                for r in ('', 'r'):
                    val = float(rng.integers(0, 2)) if seed is not None else 1.
                    setattr(p, e + d + r, val)
    return base, flange


def _field(p, c, xs, ys):
    p.uvw(np.ascontiguousarray(c[p.col_start:p.col_end]), xs=xs, ys=ys)
    return {k: np.ravel(p.fields[k]) for k in ('u', 'v', 'w', 'phix', 'phiy')}


@pytest.mark.parametrize('base_model', ['plate_clpt_donnell',
                                        'cylshell_clpt_donnell',
                                        'cylshell_clpt_sanders'])
def test_energy(base_model):
    r"""`\{c\}^T [K] \{c\}` against the line integral of `k_t [(u_1 -
    u_2)^2 + (v_1 - w_2)^2 + (w_1 + v_2)^2] + k_r (\omega_1 - \omega_2)^2`,
    with `\omega = -\phi_y` evaluated from the displacement field"""
    base, flange = _panels(base_model, seed=3)
    ycte1, ycte2 = 0.4*base.b, 0.
    kt, kr = 1.3e7, 2.1e3
    md = MultiDomain(panels=[base, flange],
                     conn=[dict(p1=base, p2=flange, func='BFycte', ycte1=ycte1,
                                ycte2=ycte2, kt=kt, kr=kr)])
    K = md.get_kC_conn().toarray()
    rng = np.random.default_rng(0)
    xi, wx = np.polynomial.legendre.leggauss(30)
    xs = (xi + 1)*base.a/2
    for _ in range(3):
        c = rng.standard_normal(md.get_size())
        f1 = _field(base, c, xs, np.full_like(xs, ycte1))
        f2 = _field(flange, c, xs, np.full_like(xs, ycte2))
        e = (kt*((f1['u'] - f2['u'])**2 + (f1['v'] - f2['w'])**2
                 + (f1['w'] + f2['v'])**2)
             + kr*(f1['phiy'] - f2['phiy'])**2)
        ref = np.sum(wx*e)*base.a/2
        assert np.isclose(c @ K @ c, ref, rtol=1e-11)


def test_donnell_and_plates_unchanged():
    base, flange = _panels('cylshell_clpt_donnell', seed=5)
    md = MultiDomain(panels=[base, flange], conn=[])
    size = md.get_size()
    args = (1.e7, 1.e3)
    K = (kCBFycte.fkCBFycte11(*args, base, 0., size, base.row_start,
                              base.col_start)
         + kCBFycte.fkCBFycte12(*args, base, flange, 0., 0., size,
                                base.row_start, flange.col_start)
         + kCBFycte.fkCBFycte22(*args, base, flange, 0., size,
                                flange.row_start, flange.col_start))
    conn = [dict(p1=base, p2=flange, func='BFycte', ycte1=0., ycte2=0.,
                 kt=args[0], kr=args[1])]
    md = MultiDomain(panels=[base, flange], conn=conn)
    assert np.array_equal(finalize_symmetric_matrix(K).toarray(),
                          md.get_kC_conn().toarray())


def _fit(p, targets, gridx=12, gridy=10):
    """Ritz constants of ``p`` that reproduce the fields ``targets``"""
    xs, ys = np.meshgrid(np.linspace(0, p.a, gridx), np.linspace(0, p.b, gridy))
    xs, ys = xs.ravel(), ys.ravel()
    size = p.get_size()
    N = np.zeros((3*xs.size, size))
    for i in range(size):
        e = np.zeros(size)
        e[i] = 1.
        p.uvw(e, xs=xs, ys=ys)
        N[:, i] = np.concatenate([np.ravel(p.fields[k]) for k in 'uvw'])
    t = np.concatenate([targets[k](xs, ys) for k in 'uvw'])
    c, *_ = np.linalg.lstsq(N, t, rcond=None)
    assert np.allclose(N @ c, t, atol=1e-10*np.abs(t).max())
    return c


def test_rigid_rotation_about_the_axis():
    r"""Rigid-body rotation `\theta` of a Sanders-Koiter skin with a flange
    about the axis of the cylinder

    The skin has `v_1 = r \theta`, the flange, whose axis `y_2` points
    along `-z_1` from `y_2 = 0`, has `w_2 = (r - y_2) \theta`. Both rotate
    by `\omega = -\theta`, and neither the panels nor the connection store
    energy. Without the term `v_1/r` of the rotation of the skin, the
    rotation penalty gives `k_r a \theta^2`
    """
    base, flange = _panels('cylshell_clpt_sanders')
    ycte1, ycte2 = base.b/2, 0.
    kt, kr = 1.e8, 1.e4
    conn = [dict(p1=base, p2=flange, func='BFycte', ycte1=ycte1, ycte2=ycte2,
                 kt=kt, kr=kr)]
    md = MultiDomain(panels=[base, flange], conn=conn)
    theta = 1.e-3
    zero = lambda x, y: 0.*x
    c = np.zeros(md.get_size())
    c[base.col_start:base.col_end] = _fit(base, dict(
        u=zero, v=lambda x, y: R*theta + 0.*x, w=zero))
    c[flange.col_start:flange.col_end] = _fit(flange, dict(
        u=zero, v=zero, w=lambda x, y: (R - y)*theta))
    kC = md.calc_kC(silent=True)
    K_conn = md.get_kC_conn()
    U_conn = 0.5*c @ K_conn @ c
    U_total = 0.5*c @ kC @ c
    U_ref = 0.5*kr*base.a*theta**2
    assert abs(U_conn) < 1e-9*U_ref
    assert abs(U_total) < 1e-9*U_ref

    # the rotation penalty of the Donnell rotation w,y
    size = md.get_size()
    args = (kt, kr)
    K_old = finalize_symmetric_matrix(
        kCBFycte.fkCBFycte11(*args, base, ycte1, size, base.row_start,
                             base.col_start)
        + kCBFycte.fkCBFycte12(*args, base, flange, ycte1, ycte2, size,
                               base.row_start, flange.col_start)
        + kCBFycte.fkCBFycte22(*args, base, flange, ycte2, size,
                               flange.row_start, flange.col_start))
    assert np.isclose(0.5*c @ K_old @ c, U_ref, rtol=1e-8)
