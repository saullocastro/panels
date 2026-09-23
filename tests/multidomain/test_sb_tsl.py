import numpy as np
import pytest
from scipy.special import roots_legendre
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import kCSB_dmg


def _panels(seed, flags_random):
    rng = np.random.default_rng(seed)
    lam = (133e3, 11e3, 0.3, 5e3, 5e3, 5e3)
    top = Shell(group='top', x0=0, y0=0, a=17., b=25., m=6, n=4, plyt=0.14,
                stack=[0]*12, laminaprop=lam)
    bot = Shell(group='bot', x0=0, y0=0, a=17., b=25., m=5, n=5, plyt=0.14,
                stack=[0]*9, laminaprop=lam)
    for p in (top, bot):
        for e in ('x1', 'x2', 'y1', 'y2'):
            for d in ('u', 'v', 'w'):
                for r in ('', 'r'):
                    val = float(rng.integers(0, 2)) if flags_random else 1.
                    setattr(p, e + d + r, val)
    return rng, top, bot


def test_kCSB_dmg_energy():
    rng, top, bot = _panels(1, flags_random=True)
    # bot first in the assembly, the 12 block is transposed
    assy = MultiDomain(panels=[bot, top], conn=[])
    size = assy.get_size()
    nx, ny = 20, 12
    kw = rng.uniform(0.5, 2., (ny, nx))
    dt = sum(top.plyts)/2.
    db = sum(bot.plyts)/2.
    K = kCSB_dmg.fkCSB11_dmg(dt, top, size, top.row_start, top.col_start, nx, ny, kw)
    K = K + kCSB_dmg.fkCSB12_dmg(dt, db, top, bot, size, top.row_start,
                                bot.col_start, nx, ny, kw).T
    K = K + kCSB_dmg.fkCSB22_dmg(db, top, bot, size, bot.row_start,
                                bot.col_start, nx, ny, kw)
    K = finalize_symmetric_matrix(K)
    c = rng.standard_normal(size)

    _, wx = roots_legendre(nx)
    _, wy = roots_legendre(ny)
    W = np.outer(wy, wx)*top.a*top.b/4
    rt = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=top)
    rb = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=bot)
    g = lambda r, k: np.asarray(r[k][0])
    # phix = -w,x and phiy = -w,y
    du = g(rt, 'u') - dt*g(rt, 'phix') - g(rb, 'u') - db*g(rb, 'phix')
    dv = g(rt, 'v') - dt*g(rt, 'phiy') - g(rb, 'v') - db*g(rb, 'phiy')
    dw = g(rt, 'w') - g(rb, 'w')
    E = np.sum(W*kw*(du**2 + dv**2 + dw**2))
    assert np.isclose(c @ (K @ c), E, rtol=1e-12)


def test_kC_TSL_matrix_products_vs_kernels():
    for order in ('bot-top', 'top-bot'):
        rng, top, bot = _panels(5, flags_random=True)
        nx, ny = 20, 12
        k_o, tau_o, G1c = 5e4, 87., 1.12
        conn = [dict(p1=top, p2=bot, func='SB_TSL', tsl_type='bilinear',
                     nr_x_gauss=nx, nr_y_gauss=ny, k_o=k_o, tau_o=tau_o,
                     G1c=G1c)]
        panels = [bot, top] if order == 'bot-top' else [top, bot]
        assy = MultiDomain(panels=panels, conn=conn)
        size = assy.get_size()
        c = rng.standard_normal(size)
        rt = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=top)
        rb = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=bot)
        c *= 0.015/np.max(rt['w'][0] - rb['w'][0])
        # damage history with pristine, softening and failed points
        prev = rng.uniform(0., 1., (ny, nx))
        prev[prev < 0.3] = 0.
        prev[prev > 0.9] = 1.
        assy.update_TSL_history(prev)
        K_products = assy.get_kC_conn(c=c).toarray()
        conn[0]['use_kernels'] = True
        K_kernels = assy.get_kC_conn(c=c).toarray()
        assert np.allclose(K_products, K_kernels, rtol=1e-12,
                           atol=1e-12*abs(K_kernels).max())


def test_calc_kT_TSL_finite_differences():
    rng, top, bot = _panels(3, flags_random=False)
    nx, ny = 20, 12
    k_o, tau_o, G1c = 5e4, 87., 1.12
    conn = [dict(p1=top, p2=bot, func='SB_TSL', tsl_type='bilinear',
                 nr_x_gauss=nx, nr_y_gauss=ny, k_o=k_o, tau_o=tau_o, G1c=G1c)]
    assy = MultiDomain(panels=[bot, top], conn=conn)
    size = assy.get_size()
    c = rng.standard_normal(size)
    rt = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=top)
    rb = assy.uvw(c, None, nr_x_gauss=nx, nr_y_gauss=ny, eval_panel=bot)
    # separations inside the softening range of the law
    c *= 0.015/np.max(rt['w'][0] - rb['w'][0])

    assy.update_TSL_history(np.zeros((ny, nx)))
    _, _, _, dcurr = assy.calc_k_dmg(c=c, pA=top, pB=bot, nr_x_gauss=nx,
            nr_y_gauss=ny, tsl_type='bilinear', prev_max_dmg_index=None,
            k_i=k_o, tau_o=tau_o, G1c=G1c)
    # damage history below the current damage in part of the domain, and
    # above it elsewhere, where the damage does not grow
    prev = 0.5*dcurr
    prev[:, :nx//3] = np.minimum(dcurr[:, :nx//3] + 0.1, 1.)
    assy.update_TSL_history(prev)

    Ks = assy.get_kC_conn(c=c)
    kD = assy.calc_kT_TSL(c=c)
    assert kD.nnz > 0
    F = lambda cc: assy.get_kC_conn(c=cc) @ cc
    for _ in range(2):
        e = rng.standard_normal(size)
        h = 1e-6*np.linalg.norm(c)/np.linalg.norm(e)
        fd = (F(c + h*e) - F(c - h*e))/(2*h)
        err = np.linalg.norm((Ks + kD) @ e - fd)/np.linalg.norm(fd)
        assert err < 1e-6
        err_secant = np.linalg.norm(Ks @ e - fd)/np.linalg.norm(fd)
        assert err_secant > 1e-2


if __name__ == '__main__':
    test_kCSB_dmg_energy()
    test_calc_kT_TSL_finite_differences()


def test_panels_must_share_the_area():
    """The kernels and the matrix products integrate over the domain of the
    top panel, the bottom panel must have the same dimensions"""
    for use_kernels in (False, True):
        _, top, bot = _panels(2, flags_random=False)
        bot.a = top.a/2
        conn = [dict(p1=top, p2=bot, func='SB_TSL', tsl_type='bilinear',
                     nr_x_gauss=10, nr_y_gauss=8, k_o=5e4, tau_o=87.,
                     G1c=1.12, use_kernels=use_kernels)]
        assy = MultiDomain(panels=[top, bot], conn=conn)
        c = np.zeros(assy.get_size())
        with pytest.raises(ValueError, match='same dimensions'):
            assy.get_kC_conn(c=c)
