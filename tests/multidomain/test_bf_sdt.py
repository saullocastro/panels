r"""Base-flange connections ``'BFycte'`` and ``'BFxcte'``

For the models based on the classical laminated plate theory (CLPT) the
rotation penalty is on `-w_{,y}` (``'BFycte'``) and `-w_{,x}`
(``'BFxcte'``), for the models based on shear deformation theories (FSDT and
TSDT) on the rotations `\phi_y` and `\phi_x`, see
:mod:`panels.multidomain.connections.kCBFycte` and
:mod:`panels.multidomain.connections.kCBFxcte`.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.linalg import eigh
from structsolve.sparseutils import remove_null_cols

from panels.shell import Shell
from panels.multidomain import MultiDomain


LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
SDT = ['plate_fsdt_donnell', 'plate_tsdt_donnell']
FIELDS = {3: 'uvw', 5: ('u', 'v', 'w', 'phix', 'phiy')}


def _flags(p, seed):
    rng = np.random.default_rng(seed)
    names = ['u', 'v', 'w'] + (['phix', 'phiy'] if 'dt' in p.model else [])
    for e in ('x1', 'x2', 'y1', 'y2'):
        for d in names:
            for r in ('', 'r'):
                val = float(rng.integers(0, 2)) if seed is not None else 1.
                setattr(p, e + d + r, val)


def _panels(model, func, seed=None, plyt=0.2e-3, flags=True):
    kw = dict(model=model, x0=0, y0=0, plyt=plyt, laminaprop=LAMINAPROP)
    if func == 'BFycte':
        base = Shell(group='base', a=0.6, b=0.3, m=6, n=7,
                     stack=[0, 45, -45, 90], **kw)
        flange = Shell(group='flange', a=0.6, b=0.05, m=6, n=5,
                       stack=[0, 90, 90, 0], **kw)
    else:
        base = Shell(group='base', a=0.3, b=0.6, m=7, n=6,
                     stack=[0, 45, -45, 90], **kw)
        flange = Shell(group='flange', a=0.05, b=0.6, m=5, n=6,
                       stack=[0, 90, 90, 0], **kw)
    if flags:
        _flags(base, seed)
        _flags(flange, None if seed is None else seed + 1)
    return base, flange


def _field(p, c, xs, ys):
    p.uvw(np.ascontiguousarray(c[p.col_start:p.col_end]), xs=xs, ys=ys)
    return {k: np.ravel(p.fields[k]) for k in ('u', 'v', 'w', 'phix', 'phiy')}


def _conn(func, base, flange, cte1, cte2, **kw):
    cte = func[2:]
    return dict(p1=base, p2=flange, func=func, **{cte + '1': cte1,
                cte + '2': cte2}, **kw)


@pytest.mark.parametrize('order', ['base first', 'flange first'])
@pytest.mark.parametrize('func', ['BFycte', 'BFxcte'])
@pytest.mark.parametrize('model', ['plate_clpt_donnell'] + SDT)
def test_energy(model, func, order):
    r"""`\{c\}^T [K] \{c\}` against the line integral of the penalty energy
    evaluated from the displacement field, the rotation being `\phi_y` or
    `\phi_x`, which are `-w_{,y}` and `-w_{,x}` for the CLPT"""
    base, flange = _panels(model, func, seed=3)
    kt, kr = 1.3e7, 2.1e3
    L = base.a if func == 'BFycte' else base.b
    cte1 = 0.4*(base.b if func == 'BFycte' else base.a)
    cte2 = 0.
    panels = [base, flange] if order == 'base first' else [flange, base]
    md = MultiDomain(panels=panels, conn=[_conn(func, base, flange, cte1,
                                                cte2, kt=kt, kr=kr)])
    K = md.get_kC_conn().toarray()
    xi, wg = np.polynomial.legendre.leggauss(30)
    s = (xi + 1)*L/2
    rng = np.random.default_rng(0)
    for _ in range(3):
        c = rng.standard_normal(md.get_size())
        if func == 'BFycte':
            f1 = _field(base, c, s, np.full_like(s, cte1))
            f2 = _field(flange, c, s, np.full_like(s, cte2))
            e = (kt*((f1['u'] - f2['u'])**2 + (f1['v'] - f2['w'])**2
                     + (f1['w'] + f2['v'])**2)
                 + kr*(f1['phiy'] - f2['phiy'])**2)
        else:
            f1 = _field(base, c, np.full_like(s, cte1), s)
            f2 = _field(flange, c, np.full_like(s, cte2), s)
            e = (kt*((f1['u'] - f2['w'])**2 + (f1['v'] - f2['v'])**2
                     + (f1['w'] + f2['u'])**2)
                 + kr*(f1['phix'] - f2['phix'])**2)
        ref = np.sum(wg*e)*L/2
        assert np.isclose(c @ K @ c, ref, rtol=1e-11)


def _fit(p, targets, gridx=12, gridy=10):
    """Ritz constants of ``p`` that reproduce the fields ``targets``"""
    xs, ys = np.meshgrid(np.linspace(0, p.a, gridx), np.linspace(0, p.b, gridy))
    xs, ys = xs.ravel(), ys.ravel()
    size = p.get_size()
    fields = FIELDS[size//(p.m*p.n)]
    N = np.zeros((len(fields)*xs.size, size))
    for i in range(size):
        e = np.zeros(size)
        e[i] = 1.
        p.uvw(e, xs=xs, ys=ys)
        N[:, i] = np.concatenate([np.ravel(p.fields[k]) for k in fields])
    t = np.concatenate([targets.get(k, lambda x, y: 0.*x)(xs, ys) for k in fields])
    c, *_ = np.linalg.lstsq(N, t, rcond=None)
    assert np.allclose(N @ c, t, atol=1e-10*np.abs(t).max())
    return c


@pytest.mark.parametrize('order', ['base first', 'flange first'])
@pytest.mark.parametrize('func', ['BFycte', 'BFxcte'])
@pytest.mark.parametrize('model', ['plate_clpt_donnell'] + SDT)
def test_rigid_rotation_about_the_connection(model, func, order):
    r"""Rigid-body rotation `\theta` of the T-joint about the axis of the
    connection, which stores no energy

    With the flange along `-z_1` from its edge `y_2 = 0` (``'BFycte'``) or
    `x_2 = 0` (``'BFxcte'``), the base has `w_1 = \theta (y - y_{cte1})`
    and the flange `w_2 = \theta y_2`, or `w_1 = \theta (x - x_{cte1})` and
    `w_2 = \theta x_2`, and both rotate by `\phi = -\theta`
    """
    base, flange = _panels(model, func)
    theta = 1.e-3
    sdt = model in SDT
    if func == 'BFycte':
        cte1 = base.b/2
        tb = dict(w=lambda x, y: theta*(y - cte1))
        tf = dict(w=lambda x, y: theta*y)
        if sdt:
            tb['phiy'] = tf['phiy'] = lambda x, y: -theta + 0.*x
    else:
        cte1 = base.a/2
        tb = dict(w=lambda x, y: theta*(x - cte1))
        tf = dict(w=lambda x, y: theta*x)
        if sdt:
            tb['phix'] = tf['phix'] = lambda x, y: -theta + 0.*x
    kt, kr = 1.e8, 1.e4
    panels = [base, flange] if order == 'base first' else [flange, base]
    md = MultiDomain(panels=panels, conn=[_conn(func, base, flange, cte1, 0.,
                                                kt=kt, kr=kr)])
    c = np.zeros(md.get_size())
    c[base.col_start:base.col_end] = _fit(base, tb)
    c[flange.col_start:flange.col_end] = _fit(flange, tf)
    U_ref = 0.5*kr*(base.a if func == 'BFycte' else base.b)*theta**2
    assert abs(0.5*c @ md.get_kC_conn() @ c) < 1e-9*U_ref
    assert abs(0.5*c @ md.calc_kC(silent=True) @ c) < 1e-9*U_ref


def _freq(model, func):
    r"""First natural frequency of a simply supported plate with a blade at
    mid-width or mid-length, with the free edge of the blade at `y_2 = b_2`
    or `x_2 = a_2`, and the other edges simply supported"""
    base, flange = _panels(model, func, plyt=0.1e-3, flags=False)
    base.m = base.n = flange.m = flange.n = 10
    for p in (base, flange):
        p.rho = 1600.
    #NOTE the edge of the blade at the connection is only tied by the penalty
    edges = ('y1', 'y2') if func == 'BFycte' else ('x1', 'x2')
    for e in edges:
        for d in ['u', 'v', 'w'] + (['phix', 'phiy'] if model in SDT else []):
            for r in ('', 'r'):
                setattr(flange, e + d + r, 1.)
    cte1 = base.b/2 if func == 'BFycte' else base.a/2
    md = MultiDomain(panels=[base, flange],
                     conn=[_conn(func, base, flange, cte1, 0.)])
    K, _ = remove_null_cols(md.calc_kC(silent=True))
    M, _ = remove_null_cols(md.calc_kM(silent=True))
    return np.sqrt(eigh(K.toarray(), M.toarray(), eigvals_only=True,
                        subset_by_index=[0, 0])[0])


@pytest.mark.parametrize('func', ['BFycte', 'BFxcte'])
def test_thin_blade_stiffened_plate_clpt_fsdt_tsdt(func):
    r"""For thin laminates, `a/h = 750` for the plate, the FSDT and TSDT
    give the natural frequency of the CLPT, 243 Hz for ``'BFycte'`` and 244
    Hz for ``'BFxcte'``, against 86 Hz without the blade"""
    f_clpt = _freq('plate_clpt_donnell', func)
    assert f_clpt > 2.5*86.
    for model in SDT:
        assert np.isclose(_freq(model, func), f_clpt, rtol=1e-4)
