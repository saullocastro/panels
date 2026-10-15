import numpy as np
import pytest

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import (calc_kw_tsl, tsl_resolution,
        tsl_margin, tsl_discretization)


def _assy(nx, ny, k_o=5e4, tau_o=87., G1c=1.12):
    lam = (133e3, 11e3, 0.3, 5e3, 5e3, 5e3)
    top = Shell(group='top', x0=0, y0=0, a=17., b=25., m=6, n=4, plyt=0.14,
                stack=[0]*12, laminaprop=lam)
    bot = Shell(group='bot', x0=0, y0=0, a=17., b=25., m=5, n=5, plyt=0.14,
                stack=[0]*9, laminaprop=lam)
    for p in (top, bot):
        for e in ('x1', 'x2', 'y1', 'y2'):
            for d in ('u', 'v', 'w'):
                setattr(p, e + d, 1.)
                setattr(p, e + d + 'r', 1.)
    conn = [dict(p1=top, p2=bot, func='SB_TSL', tsl_type='bilinear',
                 nr_x_gauss=nx, nr_y_gauss=ny, k_o=k_o, tau_o=tau_o, G1c=G1c)]
    assy = MultiDomain(panels=[bot, top], conn=conn, conn_method='penalty')
    return assy, conn, top, bot


def _state(assy, top, bot, nx, ny, max_separation, seed=1):
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(assy.get_size())
    rt = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=top)
    rb = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=bot)
    return c*max_separation/np.max(rt['w'][0] - rb['w'][0])


def test_more_than_304_gauss_points():
    """The Gauss-Legendre rules have no upper limit"""
    nx, ny = 320, 8
    assy, conn, top, bot = _assy(nx, ny)
    # separations in the softening range
    c = _state(assy, top, bot, nx, ny, 0.015)
    res = assy.calc_results(c=c, eval_panel=top, vec='w', nr_x_gauss=nx,
                            nr_y_gauss=ny)
    uvw = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=top)
    assert np.asarray(res['w'][0]).shape == (ny, nx)
    assert np.allclose(res['w'][0], uvw['w'][0], rtol=1e-12, atol=0)
    assy.update_TSL_history(np.zeros((ny, nx)))
    _, dmg, _, _ = assy.calc_k_dmg(c=c, pA=top, pB=bot, nr_x_gauss=nx,
            nr_y_gauss=ny, tsl_type='bilinear', prev_max_dmg_index=None,
            k_i=5e4, tau_o=87., G1c=1.12)
    assert dmg.shape == (ny, nx)
    assert 0 < dmg.max() < 1
    assy.update_TSL_history(dmg)
    K_products = assy.get_kC_conn(c=c).toarray()
    conn[0]['use_kernels'] = True
    K_kernels = assy.get_kC_conn(c=c).toarray()
    assert np.allclose(K_products, K_kernels, rtol=1e-12,
                       atol=1e-12*abs(K_kernels).max())


def test_tsl_resolution_reference_dcb():
    """Arm of the reference DCB of the convergence study, see
    doc/source/cohesive_zone.rst"""
    lam = (133150., 10950., 0.316, 5190., 5190., 5190.)
    arm = Shell(group='top', x0=0, y0=0, a=24., b=25., m=25, n=10,
                plyt=0.14, stack=[0]*15, laminaprop=lam)
    lamA, lcz = tsl_resolution(arm, 1e5, 87., 1.12)
    lamB, lczB = tsl_resolution(arm, 2e5, 87., 1.12)
    # symmetric DCB, lambda = (k_o/(2 D11))**0.25
    assert np.isclose(lamA, 0.8335, atol=5e-4)
    assert np.isclose(lamB, 0.9912, atol=5e-4)
    assert np.isclose(lamA, (1e5/(2*arm.lam.D[0, 0]))**0.25, rtol=1e-12)
    assert tsl_resolution(arm, 1e5, 87., 1.12, p2=arm) == (lamA, lcz)
    assert np.isclose(lcz, 1.426, atol=5e-4)
    assert lczB == lcz
    assert np.isclose(tsl_margin(lamA, lcz), 2*np.pi/lamA + 3*lcz)
    # 25.0 terms and 168 points asked for over 24 mm
    assert tsl_discretization(lamA, lcz, 24.) == (25, 170)
    assert tsl_discretization(lamA, lcz, 24., m_max=30) == (26, 170)
    # no upper limit for the Gauss points
    assert tsl_discretization(lamA, lcz/3, 24.) == (25, 510)
    # at least 4 points per term
    assert tsl_discretization(lamA, 100., 24.) == (25, 100)


def test_tsl_resolution_two_panels():
    r"""`4 \lambda^4 = k_o (1/D_{11}^{(1)} + 1/D_{11}^{(2)})`, which tends to
    the rate of one panel on a rigid foundation, `(k_o/(4 D_{11}))^{1/4}`,
    when the other panel is much stiffer"""
    lam = (133150., 10950., 0.316, 5190., 5190., 5190.)
    kw = dict(group='top', x0=0, y0=0, a=24., b=25., m=8, n=8, plyt=0.14,
              laminaprop=lam)
    p1 = Shell(stack=[0]*15, **kw)
    p2 = Shell(stack=[0]*9, **kw)
    stiff = Shell(stack=[0]*600, **kw)
    lam12, _ = tsl_resolution(p1, 1e5, 87., 1.12, p2=p2)
    D1, D2 = p1.lam.D[0, 0], p2.lam.D[0, 0]
    assert np.isclose(lam12, (1e5*(1/D1 + 1/D2)/4)**0.25, rtol=1e-12)
    assert np.isclose(lam12, tsl_resolution(p2, 1e5, 87., 1.12, p2=p1)[0],
                      rtol=1e-14)
    lam_rigid, _ = tsl_resolution(p1, 1e5, 87., 1.12, p2=stiff)
    assert np.isclose(lam_rigid, (1e5/(4*D1))**0.25, rtol=1e-5)


def test_calc_kw_tsl_parameters():
    del_d = np.array([[-1e-3, 0., 1e-3, 5e-3, 1.]])
    with pytest.raises(ValueError, match='required'):
        calc_kw_tsl(None, del_d=del_d, tau_o=87., G1c=1.12)
    with pytest.raises(ValueError, match='not available'):
        calc_kw_tsl(None, tsl_type='linear', k_i=5e4, del_d=del_d, tau_o=87.,
                    G1c=1.12)
    kw, d = calc_kw_tsl(None, k_i=5e4, del_d=del_d, tau_o=87., G1c=1.12)
    del_o, del_f = 87./5e4, 2*1.12/87.
    d_ref = np.array([[0., 0., 0., del_f*(5e-3 - del_o)/((del_f - del_o)*5e-3), 1.]])
    assert np.allclose(d, d_ref)
    # negative separations keep the pristine stiffness
    assert np.allclose(kw, 5e4*(1 - d_ref))


def test_resolution_warning(capsys):
    nx, ny = 20, 8
    assy, conn, top, bot = _assy(nx, ny)
    kw = dict(pA=top, pB=bot, nr_x_gauss=nx, nr_y_gauss=ny, tsl_type='bilinear',
              prev_max_dmg_index=None, k_i=5e4, tau_o=87., G1c=1.12)
    # no warning while no point is damaged
    assy.calc_k_dmg(c=np.zeros(assy.get_size()), **kw)
    assert 'WARNING' not in capsys.readouterr().out
    c = _state(assy, top, bot, nx, ny, 0.015)
    assy.calc_k_dmg(c=c, **kw)
    out = capsys.readouterr().out
    assert 'terms per decay length' in out
    assert 'points per cohesive zone length' in out
    # only once
    assy.calc_k_dmg(c=c, **kw)
    assert 'WARNING' not in capsys.readouterr().out


def test_penalties_of_the_connections_at_90_degrees():
    r"""``'xcte-ycte'`` connects the edge `x` = constant of ``p1`` to an edge
    `y` = constant of ``p2``, the harmonic means of `A_{11}` of ``p1`` and
    `A_{22}` of ``p2``, and conversely for ``'ycte-xcte'``"""
    from panels.shell import Shell
    from panels.multidomain.connections import calc_kt_kr
    kw = dict(model='plate_clpt_donnell', a=0.5, b=0.3, plyt=0.2e-3,
              laminaprop=(140e9, 9e9, 0.3, 5e9, 5e9, 3e9), m=4, n=4)
    p1 = Shell(stack=[0, 0, 90], **kw)
    p2 = Shell(stack=[90, 45, -45, 0], **kw)
    h = p1.lam.h + p2.lam.h
    A11, A22 = p1.lam.A[0, 0], p2.lam.A[1, 1]
    D11, D22 = p1.lam.D[0, 0], p2.lam.D[1, 1]
    kt, kr = calc_kt_kr(p1, p2, 'xcte-ycte')
    assert np.isclose(kt, 4*A11*A22/((A11 + A22)*h))
    assert np.isclose(kr, 4*D11*D22/((D11 + D22)*h))
    assert np.allclose(calc_kt_kr(p2, p1, 'ycte-xcte'), (kt, kr))
