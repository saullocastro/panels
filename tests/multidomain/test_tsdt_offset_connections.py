r"""Connections of TSDT panels with laminate offsets

The connected panels share the reference surface `z = 0`, where the
connections tie `u, v, w` and the rotations of the normal at `z = 0`,

.. math::

    \theta_x = \phi_x - \frac{4 d^2}{h^2} (\phi_x + w_{,x}) \qquad
    \theta_y = \Phi_y - \frac{4 d^2}{h^2} (\phi_y + w_{,y})

with the offset `d` and the thickness `h` of each panel, `\phi_x, \phi_y`
being the rotations at the mid-surface of the TSDT field `u(z) = u + z
\phi_x - c_1 [(z - d)^3 + d^3] (\phi_x + w_{,x})`, see
:mod:`panels.multidomain.connections.kCsdt`. `\phi + w_{,x}` is the
transverse shear strain ``gxz`` (or ``gyz``) of :meth:`.Shell.strain`.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.linalg import eigh
from structsolve import solve
from structsolve.sparseutils import remove_null_cols

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import calc_kt_kr, kCsdt


TSDT = 'plate_tsdt_donnell'
E2 = 1.e9
LAMINAPROP = (10*E2, E2, 0.25, 0.6*E2, 0.6*E2, 0.5*E2)
STACK = [0, 90, 0, 90]
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
H = 0.1


def _plate(a, b, offset, bcs, x0=0., m=10, n=10, model=TSDT):
    r"""Plate with the edges `x = 0, y = 0, x = a, y = b` simply supported,
    ``'S'``, or free, ``'F'``"""
    s = Shell(group='plate', x0=x0, y0=0., model=model, a=a, b=b, stack=STACK,
              plyt=H/len(STACK), laminaprop=LAMINAPROP, rho=1500.,
              offset=offset, m=m, n=n)
    for edge, kind in zip(('x1', 'y1', 'x2', 'y2'), bcs):
        tangential = 'v' if edge.startswith('x') else 'u'
        for field in FIELDS:
            setattr(s, edge + field, 1.)
            setattr(s, edge + field + 'r', 1.)
        if kind == 'S':
            setattr(s, edge + tangential, 0.)
            setattr(s, edge + 'w', 0.)
    s._rebuild()
    return s


def _two_plates(d1, d2, factor=1.e3, conn_method='penalty', xs=0.4, m=10):
    p1 = _plate(xs, 1., d1, 'SSFS', m=m, n=m)
    p2 = _plate(1. - xs, 1., d2, 'FSSS', x0=xs, m=m, n=m)
    kt, kr = calc_kt_kr(p1, p2, 'xcte')
    conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0.,
                 kt=factor*kt, kr=factor*kr)]
    return MultiDomain([p1, p2], conn, conn_method=conn_method), p1, p2


def _frequencies(K, M, k):
    K, _ = remove_null_cols(K)
    M, _ = remove_null_cols(M)
    return np.sqrt(eigh(K.toarray(), M.toarray(), eigvals_only=True,
                        subset_by_index=[0, k - 1]))


def _rotations(p, c, xs, ys):
    r"""`\theta_x, \theta_y` at the reference surface and `\phi_x`,
    `\phi_y` of ``p``, from :meth:`.Shell.uvw` and :meth:`.Shell.strain`"""
    cp = np.ascontiguousarray(c[p.col_start:p.col_end])
    p.uvw(cp, xs=xs, ys=ys)
    phix, phiy = p.fields['phix'].ravel().copy(), p.fields['phiy'].ravel().copy()
    p.strain(cp, xs=xs, ys=ys, NLgeom=False)
    gxz, gyz = p.fields['gxz'].ravel(), p.fields['gyz'].ravel()
    alpha = 4*p.offset**2/H**2
    return phix - alpha*gxz, phiy - alpha*gyz, phix, phiy


@pytest.mark.parametrize('conn_method', ['penalty', 'null-space'])
def test_same_offset_split_plate_equals_single_plate(conn_method):
    r"""Two panels with the same offset reproduce the single panel"""
    d = 0.3*H
    md, p1, p2 = _two_plates(d, d, conn_method=conn_method, m=14)
    if conn_method == 'null-space':
        K, M = md.reduce(md.calc_kC()), md.reduce(md.calc_kM(silent=True))
    else:
        K, M = md.calc_kC(), md.calc_kM(silent=True)
    omegas = _frequencies(K, M, 6)
    single = _plate(1., 1., d, 'SSSS', m=18, n=18)
    ref = _frequencies(single.calc_kC(), single.calc_kM(), 6)
    assert np.allclose(omegas, ref, rtol=1e-4)


@pytest.mark.parametrize('d2', [0.3*H, 0.5*H, -0.5*H])
def test_reference_rotation_continuous_with_different_offsets(d2):
    r"""With the null-space method the rotations at the common reference
    surface are continuous across the edge, not those at the mid-surfaces,
    which differ by `-\alpha \gamma_{xz}` for `d_1 \neq d_2`"""
    md, p1, p2 = _two_plates(0., d2, conn_method='null-space')
    p1.add_pressure_load(-1.e4)
    p2.add_pressure_load(-1.e4)
    K = md.reduce(md.calc_kC())
    c = md.expand(solve(K, md.reduce(md.calc_fext()), silent=True))
    ys = np.linspace(0.05, 0.95, 7)
    th1x, th1y, phi1x, _ = _rotations(p1, c, np.full_like(ys, p1.a), ys)
    th2x, th2y, phi2x, _ = _rotations(p2, c, np.zeros_like(ys), ys)
    scale = np.abs(th1x).max()
    assert scale > 0
    assert np.allclose(th1x, th2x, rtol=0, atol=1e-9*scale)
    assert np.allclose(th1y, th2y, rtol=0, atol=1e-9*scale)
    # the rotations at the mid-surfaces are not continuous
    assert np.abs(phi1x - phi2x).max() > 1e-3*scale


def test_reference_rotation_at_a_face():
    r"""At `d = \pm h/2` the transverse shear vanishes at `z = 0`, and
    `\theta_x = -w_{,x}`, `\theta_y = -w_{,y}`"""
    for d in (0.5*H, -0.5*H):
        p = _plate(0.4, 1., d, 'SSSS')
        assert kCsdt._rotation_x(p) == [(3, 0, 0, 0.), (2, 1, 0, -1.)]
        assert kCsdt._rotation_y(p) == [(4, 0, 0, 0.), (2, 0, 1, -1.)]
        p.model = 'plate_fsdt_donnell'
        assert kCsdt._rotation_x(p) == [(3, 0, 0, 1.)]


def test_offset_zero_unchanged():
    r"""Without offset `\theta = \phi`, the connection of previous versions"""
    p = _plate(0.4, 1., 0., 'SSSS')
    assert kCsdt._rotation_x(p) == [(3, 0, 0, 1.)]
    assert kCsdt._rotation_y(p) == [(4, 0, 0, 1.)]


@pytest.mark.parametrize('d1,d2', [(0., 0.5*H), (0.2*H, -0.4*H)])
def test_edge_penalty_energy(d1, d2):
    r"""`\{\bar{c}\}^\top [K] \{\bar{c}\}` of ``'SSxcte'`` against the line integral of the
    penalty energy evaluated from the fields"""
    md, p1, p2 = _two_plates(d1, d2)
    kt, kr = 1.3e7, 2.1e3
    md.conn[0].update(kt=kt, kr=kr)
    K = md.get_kC_conn().toarray()
    xi, wg = np.polynomial.legendre.leggauss(30)
    ys = (xi + 1)/2
    rng = np.random.default_rng(0)
    for _ in range(3):
        c = rng.standard_normal(md.get_size())
        f = []
        for p, x in ((p1, p1.a), (p2, 0.)):
            cp = np.ascontiguousarray(c[p.col_start:p.col_end])
            xs = np.full_like(ys, x)
            p.uvw(cp, xs=xs, ys=ys)
            out = {k: p.fields[k].ravel().copy() for k in ('u', 'v', 'w')}
            out['thx'], out['thy'] = _rotations(p, c, xs, ys)[:2]
            p.strain(cp, xs=xs, ys=ys, NLgeom=False)
            out['wx'] = p.fields['gxz'].ravel() - _rotations(p, c, xs, ys)[2]
            f.append(out)
        e = (kt*sum((f[0][k] - f[1][k])**2 for k in ('u', 'v', 'w'))
             + kr*sum((f[0][k] - f[1][k])**2 for k in ('thx', 'thy', 'wx')))
        ref = np.sum(wg*e)/2
        assert np.isclose(c @ K @ c, ref, rtol=1e-9)


@pytest.mark.parametrize('func', ['BFycte', 'BFxcte'])
def test_base_flange_penalty_energy_with_offsets(func):
    r"""The base-flange kernels penalize `\theta` with the offset of each
    panel, the base with its reference surface on its upper face"""
    kw = dict(model=TSDT, x0=0, y0=0, plyt=0.2e-3, laminaprop=LAMINAPROP)
    if func == 'BFycte':
        base = Shell(group='base', a=0.6, b=0.3, m=6, n=7, stack=[0, 45, -45, 90],
                     offset=-0.4e-3, **kw)
        flange = Shell(group='flange', a=0.6, b=0.05, m=6, n=5,
                       stack=[0, 90, 90, 0], offset=0.1e-3, **kw)
    else:
        base = Shell(group='base', a=0.3, b=0.6, m=7, n=6, stack=[0, 45, -45, 90],
                     offset=-0.4e-3, **kw)
        flange = Shell(group='flange', a=0.05, b=0.6, m=5, n=6,
                       stack=[0, 90, 90, 0], offset=0.1e-3, **kw)
    rng = np.random.default_rng(3)
    for p in (base, flange):
        for e in ('x1', 'x2', 'y1', 'y2'):
            for d in FIELDS:
                for r in ('', 'r'):
                    setattr(p, e + d + r, float(rng.integers(0, 2)))
        p._rebuild()
    kt, kr = 1.3e7, 2.1e3
    cte = func[2:]
    L = base.a if func == 'BFycte' else base.b
    cte1 = 0.4*(base.b if func == 'BFycte' else base.a)
    md = MultiDomain([base, flange], [dict(p1=base, p2=flange, func=func,
                     **{cte + '1': cte1, cte + '2': 0.}, kt=kt, kr=kr)],
                     conn_method='penalty')
    K = md.get_kC_conn().toarray()
    xi, wg = np.polynomial.legendre.leggauss(30)
    s = (xi + 1)*L/2
    for _ in range(3):
        c = rng.standard_normal(md.get_size())
        f = []
        for p, v in ((base, cte1), (flange, 0.)):
            xs, ys = (s, np.full_like(s, v)) if func == 'BFycte' else (np.full_like(s, v), s)
            cp = np.ascontiguousarray(c[p.col_start:p.col_end])
            p.uvw(cp, xs=xs, ys=ys)
            out = {k: p.fields[k].ravel().copy() for k in ('u', 'v', 'w')}
            out['thx'], out['thy'] = _rotations_t(p, c, xs, ys)
            f.append(out)
        f1, f2 = f
        if func == 'BFycte':
            e = (kt*((f1['u'] - f2['u'])**2 + (f1['v'] - f2['w'])**2
                     + (f1['w'] + f2['v'])**2) + kr*(f1['thy'] - f2['thy'])**2)
        else:
            e = (kt*((f1['u'] - f2['w'])**2 + (f1['v'] - f2['v'])**2
                     + (f1['w'] + f2['u'])**2) + kr*(f1['thx'] - f2['thx'])**2)
        ref = np.sum(wg*e)*L/2
        assert np.isclose(c @ K @ c, ref, rtol=1e-10)


def _rotations_t(p, c, xs, ys):
    r"""As :func:`._rotations`, with the thickness of ``p``"""
    cp = np.ascontiguousarray(c[p.col_start:p.col_end])
    p.uvw(cp, xs=xs, ys=ys)
    phix, phiy = p.fields['phix'].ravel().copy(), p.fields['phiy'].ravel().copy()
    p.strain(cp, xs=xs, ys=ys, NLgeom=False)
    alpha = 4*p.offset**2/sum(p.plyts)**2
    return phix - alpha*p.fields['gxz'].ravel(), phiy - alpha*p.fields['gyz'].ravel()
