r"""Skin-base connections of curved panels with Sanders-Koiter kinematics

With the Sanders-Koiter kinematics of a cylinder, `v(z) = v + z (v/r -
w_{,y})` for the classical laminated plate theory and `v(z) = v + z (\phi_y +
v/r)` for the FSDT, a rigid rotation about the axis of the cylinder, `v = r
\Omega` at the reference surface of each panel, moves the lower face of
``p1`` and the upper face of ``p2`` together when the radius of each panel
is that of its reference surface, `r_1 - d^t = r_2 + d^b`. The connections
``'SB'``, with the penalty and the null-space methods, and ``'SB_TSL'``,
with the matrix products and with the kernels of ``kCSB_dmg.pyx``, must then
have no separation and no energy. With the Donnell kinematics the term `v/r`
is neglected, and a uniform `v` of two panels with the same radius has no
separation.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import kCsdt, nullspace


LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
R = 0.5
OMEGA = 1e-3


def _bonded(model, seed=None):
    r"""Two curved panels bonded with a skin-base connection, the radius of
    ``bot`` that of its reference surface, `r_t - d^t - d^b`, for the
    Sanders-Koiter kinematics, and the same radius for the Donnell
    kinematics"""
    def panel(group, r, stack):
        p = Shell(group=group, model=model, x0=0, y0=0, a=0.2, b=0.1, r=r,
                  m=6, n=5, stack=stack, plyt=0.2e-3, laminaprop=LAMINAPROP)
        rng = np.random.default_rng(seed)
        for e in ('x1', 'x2', 'y1', 'y2'):
            for d in ('u', 'v', 'w', 'phix', 'phiy'):
                for s in ('', 'r'):
                    #NOTE v free at all the edges, the rigid rotation
                    #     v = r Omega must be representable
                    val = (1. if seed is None or d == 'v'
                           else float(rng.integers(0, 2)))
                    setattr(p, e + d + s, val)
        return p
    top = panel('top', R, [0, 90, 0])
    bot = panel('bot', R, [0, 45, -45, 90])
    dt, db = kCsdt._sb_distances(top, bot)
    if 'sanders' in model:
        bot.r = R - dt - db
    return top, bot


def _rigid_rotation(md, p, nx=8, ny=8):
    r"""Ritz constants of ``p`` with `v = r \Omega` and, for the FSDT,
    `\phi_y = 0`, the other fields being zero"""
    N, _ = md._gauss_field_operators(p, nx, ny)
    #NOTE u, v, w only; the solution of minimum norm has phix = phiy = 0
    #     for the FSDT, whose Ritz constants of phix and phiy do not enter
    #     u, v, w
    A = np.vstack(N[:3])
    rhs = np.zeros(A.shape[0])
    rhs[N.shape[1]:2*N.shape[1]] = p.r*OMEGA
    c, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    assert np.allclose(A @ c, rhs, rtol=0, atol=1e-12*p.r*OMEGA)
    return c


def _state(md, top, bot):
    c = np.zeros(md.get_size())
    for p in (top, bot):
        c[p.col_start:p.col_end] = _rigid_rotation(md, p)
    return c


MODELS = ['cylshell_clpt_sanders', 'cylshell_fsdt_sanders',
          'cylshell_clpt_donnell']


@pytest.mark.parametrize('model', MODELS)
def test_sb_penalty_rigid_rotation(model):
    top, bot = _bonded(model, seed=2)
    kt = 1e12
    #NOTE the rotation penalty applies only to the FSDT
    kr = 0. if 'clpt' in model else 1e6
    conn = [dict(p1=top, p2=bot, func='SB', kt=kt, kr=kr)]
    md = MultiDomain([bot, top], conn, conn_method='penalty')
    c = _state(md, top, bot)
    K = md.get_kC_conn(conn=conn)
    E = c @ (K @ c)
    # energy of a separation equal to the displacement r*Omega
    E_ref = kt*(R*OMEGA)**2*top.a*top.b
    assert abs(E) < 1e-12*E_ref


@pytest.mark.parametrize('model', MODELS)
def test_sb_null_space_rigid_rotation(model):
    top, bot = _bonded(model, seed=4)
    conn = dict(p1=top, p2=bot, func='SB', kr=0. if 'clpt' in model else 1.)
    md = MultiDomain([top, bot], [conn])
    c = _state(md, top, bot)
    B = nullspace.constraint_matrix(conn, md.get_size())
    assert np.abs(B @ c).max() < 1e-13*R*OMEGA*np.abs(B).max()
    # the rigid rotation is an admissible configuration: it is in the range
    # of the basis of the null space
    Z = md.get_Z()
    c_r = np.linalg.lstsq(Z.toarray(), c, rcond=None)[0]
    assert np.allclose(Z @ c_r, c, rtol=0, atol=1e-9*np.abs(c).max())


@pytest.mark.parametrize('use_kernels', [False, True])
@pytest.mark.parametrize('model', ['cylshell_clpt_sanders',
                                   'cylshell_clpt_donnell'])
def test_sb_tsl_rigid_rotation(model, use_kernels):
    top, bot = _bonded(model)
    k_o = 1e12
    conn = [dict(p1=top, p2=bot, func='SB_TSL', tsl_type='bilinear',
                 k_o=k_o, tau_o=60e6, G1c=1e3, nr_x_gauss=12, nr_y_gauss=12,
                 use_kernels=use_kernels)]
    md = MultiDomain([top, bot], conn, conn_method='penalty')
    c = _state(md, top, bot)
    _, Bv, _, _, dofs = md._tsl_operators(top, bot, 12, 12)
    assert np.abs(Bv @ c[dofs]).max() < 1e-13*R*OMEGA
    K = md.get_kC_conn(conn=conn, c=np.zeros_like(c))
    E_ref = k_o*(R*OMEGA)**2*top.a*top.b
    assert abs(c @ (K @ c)) < 1e-12*E_ref


@pytest.mark.parametrize('model', ['cylshell_clpt_sanders',
                                   'cylshell_clpt_donnell'])
def test_sb_tsl_kernels_equal_matrix_products(model):
    r"""The kernels of ``kCSB_dmg.pyx`` and the matrix products of
    ``MultiDomain._kC_TSL`` give the same matrix, also with the term `v/r`
    of the Sanders-Koiter kinematics, which the kernels missed before, a
    relative difference of 1.2e-3"""
    top, bot = _bonded(model, seed=7)
    rng = np.random.default_rng(1)
    nx, ny = 20, 12
    conn = [dict(p1=top, p2=bot, func='SB_TSL', tsl_type='bilinear',
                 k_o=1e12, tau_o=60e6, G1c=1e3, nr_x_gauss=nx,
                 nr_y_gauss=ny)]
    md = MultiDomain([bot, top], conn, conn_method='penalty')
    prev = rng.uniform(0., 1., (ny, nx))
    prev[prev < 0.3] = 0.
    prev[prev > 0.9] = 1.
    md.update_TSL_history(prev)
    c = np.zeros(md.get_size())
    K_products = md.get_kC_conn(c=c).toarray()
    conn[0]['use_kernels'] = True
    K_kernels = md.get_kC_conn(c=c).toarray()
    assert np.allclose(K_products, K_kernels, rtol=0,
                       atol=1e-13*abs(K_kernels).max())
