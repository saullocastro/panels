r"""Skin-base connections of panels with laminate offsets

The laminate of a panel occupies `d - h/2 \le z \le d + h/2`, `z` being
measured from its reference surface and `d` its offset, ``Shell.offset``.
The connections ``'SB'`` and ``'SB_TSL'`` connect the lower face of ``p1``,
`z = d_1 - h_1/2`, to the upper face of ``p2``, `z = d_2 + h_2/2`, see
:func:`.kCsdt._sb_distances`:

- the distances from the reference surfaces to the interface follow the
  offsets
- two laminates bonded with ``'SB'`` are the single laminate with both
  stacking sequences for any offsets, with the CLPT and the FSDT, imposed
  exactly or with penalties
- the penalty energy of ``'SB_TSL'`` is that of the separations between the
  faces in contact, evaluated from the fields of both panels

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.special import roots_legendre

from panels.multidomain import MultiDomain
from panels.multidomain.connections import calc_kt_kr, kCsdt

from test_multidomain_fsdt_tsdt import (CLPT, FSDT, frequencies, make_plate,
        orthotropic)
from test_sb_tsl import _panels


def test_sb_distances():
    top = make_plate(CLPT, 1., 1., 0.02, [0, 90], orthotropic(10))
    bot = make_plate(CLPT, 1., 1., 0.03, [0, 90, 0], orthotropic(10))
    assert np.allclose(kCsdt._sb_distances(top, bot), (0.01, 0.015))
    # base on the reference surface of the skin
    top.offset = 0.025
    assert np.allclose(kCsdt._sb_distances(top, bot), (-0.015, 0.015))
    top.offset = 0.004
    bot.offset = -0.007
    assert np.allclose(kCsdt._sb_distances(top, bot), (0.006, 0.008))


#NOTE offsets of the top and bottom laminates as fractions of the thickness
#     h of the single laminate, each bonded laminate having h/2: without
#     offsets, on the reference surface of the bottom laminate, with the
#     reference surface at the interface, on the reference surface of the top
#     laminate, and arbitrary offsets
OFFSETS = [(0., 0.), (0.5, 0.), (0.25, -0.25), (0., -0.5), (0.3, -0.2)]


def _bonded(model, offsets, conn_method):
    r"""The laminates `[0/90]` over `[0/90]`, `a/h = 5`, connected by
    ``'SB'`` with the rotations tied, and the single laminate `[0/90]_2`"""
    a = 1.
    h = a/5
    laminaprop = orthotropic(10)
    top = make_plate(model, a, a, h/2, [0, 90], laminaprop)
    bot = make_plate(model, a, a, h/2, [0, 90], laminaprop)
    for p, d in zip((top, bot), offsets):
        p.offset = d*h
        p._rebuild()
    conn = dict(p1=top, p2=bot, func='SB')
    if conn_method == 'penalty':
        kt, _ = calc_kt_kr(top, bot, 'bot-top')
        conn['kt'] = 1.e5*kt
        conn['kr'] = 1.e5*kt*h**2
    else:
        conn['kr'] = 1.
    md = MultiDomain([top, bot], [conn], conn_method=conn_method)
    single = make_plate(model, a, a, h, [0, 90, 0, 90], laminaprop)
    return md, single


@pytest.mark.parametrize('offsets', OFFSETS)
@pytest.mark.parametrize('model', [CLPT, FSDT])
@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_skin_base_with_offsets_equals_single_laminate(conn_method, model,
                                                       offsets):
    r"""The frequencies do not depend on the reference surfaces: the edges
    are simply supported (SS-1), with the tangential displacement restrained,
    the same at any `z` since `w_{,s} = 0` (and `\phi_s = 0` for the FSDT)
    along the edge"""
    md, single = _bonded(model, offsets, conn_method)
    K = md.calc_kC()
    M = md.calc_kM(silent=True)
    if conn_method == 'null-space':
        K, M = md.reduce(K), md.reduce(M)
    omegas = frequencies(K, M, k=8)
    ref = frequencies(single.calc_kC(), single.calc_kM(), k=8)
    rtol = 1e-10 if conn_method == 'null-space' else 2e-5
    assert np.allclose(omegas, ref, rtol=rtol)


@pytest.mark.parametrize('use_kernels', [False, True])
@pytest.mark.parametrize('offsets', [(0., 0.), (1.4, -0.6), (-0.3, 0.9)])
def test_sb_tsl_energy_with_offsets(offsets, use_kernels):
    r"""`\{c\}^T [K] \{c\} = \int k_o (\Delta_u^2 + \Delta_v^2 +
    \Delta_w^2) dA` of the pristine interface, with the separations between
    the lower face of the top panel and the upper face of the bottom panel,
    `u(z) = u + z \phi_x` with `\phi_x = -w_{,x}`, evaluated from the
    fields"""
    rng, top, bot = _panels(7, flags_random=True)
    for p, d in zip((top, bot), offsets):
        p.offset = d
        p._rebuild()
    nx, ny = 20, 12
    k_o, tau_o, G1c = 5e4, 87., 1.12
    conn = [dict(p1=top, p2=bot, func='SB_TSL', tsl_type='bilinear',
                 nr_x_gauss=nx, nr_y_gauss=ny, k_o=k_o, tau_o=tau_o, G1c=G1c,
                 use_kernels=use_kernels)]
    assy = MultiDomain(panels=[bot, top], conn=conn, conn_method='penalty')
    size = assy.get_size()
    c = rng.standard_normal(size)
    rt = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=top)
    rb = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=bot)
    # normal separations far below the onset of damage, tau_o/k_o
    c *= 1e-2*(tau_o/k_o)/np.max(np.abs(rt['w'][0] - rb['w'][0]))
    K = assy.get_kC_conn(c=c)

    rt = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=top)
    rb = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=bot)
    g = lambda r, k: np.asarray(r[k][0])
    zt = top.offset - sum(top.plyts)/2.
    zb = bot.offset + sum(bot.plyts)/2.
    du = g(rt, 'u') + zt*g(rt, 'phix') - g(rb, 'u') - zb*g(rb, 'phix')
    dv = g(rt, 'v') + zt*g(rt, 'phiy') - g(rb, 'v') - zb*g(rb, 'phiy')
    dw = g(rt, 'w') - g(rb, 'w')
    _, wx = roots_legendre(nx)
    _, wy = roots_legendre(ny)
    W = np.outer(wy, wx)*top.a*top.b/4
    E = np.sum(W*k_o*(du**2 + dv**2 + dw**2))
    assert np.isclose(c @ (K @ c), E, rtol=1e-10)
