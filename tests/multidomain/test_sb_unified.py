r"""Skin-base connection ``'SB'``, one operator for every model

For every model, ``'SB'`` imposes `u_1(z_1) = u_2(z_2)`, `v_1(z_1) =
v_2(z_2)` and `w_1 = w_2`, with `z_1 = d_1 - h_1/2` the lower face of ``p1``
and `z_2 = d_2 + h_2/2` the upper face of ``p2``, each panel with its own
kinematics, see :mod:`panels.multidomain.connections.skin_base`:

- the constraints of the null-space method are the in-plane displacements
  at the thickness coordinates `z_1` and `z_2` of
  :func:`.nullspace.pd_rows`, and `w`
- the penalty matrix gives the energy `k_t \int (\Delta_u^2 + \Delta_v^2 +
  \Delta_w^2) dA` of the same quantities
- the separations of ``'SB_TSL'`` are the same quantities, and a pristine
  interface has the matrix of ``'SB'``
- the penalty method converges to the null-space method with `1/k_t`
- the deprecated block functions ``fkCSB11``, ``fkCSB12`` and ``fkCSB22``
  are the blocks of :func:`.skin_base.fkCSB`

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import (calc_kt_kr, kCsdt, nullspace,
        skin_base, fkCSB11, fkCSB12, fkCSB22)

from test_bf_sdt import _flags, LAMINAPROP
from test_multidomain_fsdt_tsdt import frequencies, set_edge


MODELS = ['plate_clpt_donnell', 'cylshell_clpt_sanders',
          'cylshell_clpt_donnell', 'plate_fsdt_donnell',
          'cylshell_fsdt_sanders', 'plate_tsdt_donnell',
          'cylshell_tsdt_sanders']


def _pair(model, seed, offsets=(0.3e-3, -0.2e-3)):
    r"""``top`` over ``bot`` with different numbers of terms, thicknesses,
    offsets and random boundary condition flags, the radius of ``bot`` that
    of its reference surface for the cylindrical shells"""
    kw = dict(model=model, x0=0, y0=0, a=0.4, b=0.3, plyt=0.2e-3,
              laminaprop=LAMINAPROP)
    if 'cylshell' in model:
        kw['r'] = 0.8
    top = Shell(group='top', m=6, n=5, stack=[0, 45, -45, 90],
                offset=offsets[0], **kw)
    bot = Shell(group='bot', m=5, n=6, stack=[0, 90, 0], offset=offsets[1],
                **kw)
    if 'cylshell' in model:
        dt, db = kCsdt._sb_distances(top, bot)
        bot.r = top.r - dt - db
        bot._rebuild()
    _flags(top, seed)
    _flags(bot, seed + 1)
    return top, bot


def _face_rows(md, top, bot, xis, etas):
    r"""Rows of `u(z), v(z)` at the faces in contact and of `w`, of ``top``
    minus those of ``bot``, at the points ``(xis, etas)``"""
    size = md.get_size()
    zt = top.offset - sum(top.plyts)/2.
    zb = bot.offset + sum(bot.plyts)/2.
    rows = []
    for comp in (nullspace.U, nullspace.V, nullspace.W):
        Bt = nullspace.pd_rows(top, comp, xis, etas, zt, top.col_start, size)
        Bb = nullspace.pd_rows(bot, comp, xis, etas, zb, bot.col_start, size)
        rows.append((Bt - Bb).toarray())
    return rows


@pytest.mark.parametrize('order', ['top first', 'bot first'])
@pytest.mark.parametrize('model', MODELS)
def test_null_space_rows_are_the_face_displacements(model, order):
    top, bot = _pair(model, seed=3)
    conn = dict(p1=top, p2=bot, func='SB')
    md = MultiDomain([top, bot] if order == 'top first' else [bot, top],
                     [conn])
    B = nullspace.constraint_matrix(conn, md.get_size()).toarray()
    xis = leggauss(max(top.m, bot.m, 4))[0]
    etas = leggauss(max(top.n, bot.n, 4))[0]
    ref = np.vstack(_face_rows(md, top, bot, xis, etas))
    assert B.shape == ref.shape
    assert np.allclose(B, ref, rtol=0, atol=1e-14*np.abs(ref).max())


@pytest.mark.parametrize('order', ['top first', 'bot first'])
@pytest.mark.parametrize('model', MODELS)
def test_penalty_energy_of_the_face_displacements(model, order):
    r"""`\{c\}^T [K] \{c\} = k_t \int (\Delta_u^2 + \Delta_v^2 +
    \Delta_w^2) dA`, integrated with a Gauss-Legendre rule exact for the
    polynomials of the approximation"""
    top, bot = _pair(model, seed=5)
    kt = 1.7e9
    conn = dict(p1=top, p2=bot, func='SB', kt=kt)
    md = MultiDomain([top, bot] if order == 'top first' else [bot, top],
                     [conn], conn_method='penalty')
    K = md.get_kC_conn(conn=[conn])
    c = np.random.default_rng(1).standard_normal(md.get_size())
    xis, wxi = leggauss(8)
    etas, weta = leggauss(8)
    weights = np.outer(weta, wxi).ravel()*top.a*top.b/4
    E = sum(np.sum(weights*(B @ c)**2)
            for B in _face_rows(md, top, bot, xis, etas))
    assert np.isclose(c @ (K @ c), kt*E, rtol=1e-11)


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_clpt_sanders'])
def test_sb_tsl_separations_are_the_sb_quantities(model):
    r"""The operators of ``'SB_TSL'`` are the rows of ``'SB'`` at the same
    Gauss-Legendre points, and the pristine interface, `k_o` at every point,
    has the matrix of ``'SB'`` with `k_t = k_o`"""
    top, bot = _pair(model, seed=9)
    k_o = 3.e12
    nx, ny = max(top.m, bot.m, 4), max(top.n, bot.n, 4)
    tsl = dict(p1=top, p2=bot, func='SB_TSL', tsl_type='bilinear', k_o=k_o,
               tau_o=60e6, G1c=1e3, nr_x_gauss=nx, nr_y_gauss=ny)
    md = MultiDomain([bot, top], [tsl], conn_method='penalty')
    size = md.get_size()
    Bu, Bv, Bw, _, dofs = md._tsl_operators(top, bot, nx, ny)
    B = nullspace.constraint_matrix(dict(p1=top, p2=bot, func='SB'),
                                    size).toarray()[:, dofs]
    npts = nx*ny
    for i, Bt in enumerate((Bu, Bv, Bw)):
        Bsb = B[i*npts:(i + 1)*npts]
        assert np.allclose(Bt, Bsb, rtol=0, atol=1e-13*np.abs(Bsb).max())
    K_tsl = finalize_symmetric_matrix(md.get_kC_conn(c=np.zeros(size)))
    K_sb = finalize_symmetric_matrix(skin_base.fkCSB(k_o, top, bot, size))
    K_tsl, K_sb = K_tsl.toarray(), K_sb.toarray()
    assert np.allclose(K_tsl, K_sb, rtol=0, atol=1e-12*np.abs(K_sb).max())


def _bonded(model, factor):
    r"""Laminates `[0/90/0]` over `[90/0/90]` with offsets, simply
    supported, the radius of ``bot`` that of its reference surface"""
    kw = dict(model=model, x0=0, y0=0, a=0.4, b=0.3, m=8, n=8, plyt=0.4e-3,
              laminaprop=LAMINAPROP, rho=1.6e3)
    if 'cylshell' in model:
        kw['r'] = 0.8
    top = Shell(group='top', stack=[0, 90, 0], offset=0.2e-3, **kw)
    bot = Shell(group='bot', stack=[90, 0, 90], offset=-0.1e-3, **kw)
    if 'cylshell' in model:
        dt, db = kCsdt._sb_distances(top, bot)
        bot.r = top.r - dt - db
    for p in (top, bot):
        for edge in ('x1', 'x2', 'y1', 'y2'):
            set_edge(p, edge, 'S')
        p._rebuild()
    conn = dict(p1=top, p2=bot, func='SB')
    if factor is None:
        return MultiDomain([top, bot], [conn])
    conn['kt'] = factor*calc_kt_kr(top, bot, 'bot-top')[0]
    return MultiDomain([top, bot], [conn], conn_method='penalty')


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_clpt_sanders',
                                   'plate_fsdt_donnell'])
def test_penalty_converges_to_null_space(model):
    r"""The error of the penalty method decreases with `1/k_t`"""
    md = _bonded(model, None)
    ref = frequencies(md.reduce(md.calc_kC()),
                      md.reduce(md.calc_kM(silent=True)), k=4)
    errors = []
    for factor in (1e1, 1e2, 1e3):
        md = _bonded(model, factor)
        omegas = frequencies(md.calc_kC(), md.calc_kM(silent=True), k=4)
        assert np.all(omegas < ref*(1 + 1e-10))
        errors.append(np.abs(omegas/ref - 1).max())
    for e1, e2 in zip(errors[:-1], errors[1:]):
        assert 9 < e1/e2 < 11


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_clpt_sanders'])
def test_deprecated_blocks(model):
    top, bot = _pair(model, seed=4)
    md = MultiDomain([top, bot], [], conn_method='penalty')
    size = md.get_size()
    kt = 2.3e9
    with pytest.warns(DeprecationWarning):
        K = (fkCSB11(kt, 0., top, size, top.row_start, top.col_start)
             + fkCSB12(kt, 0., top, bot, size, top.row_start, bot.col_start)
             + fkCSB22(kt, top, bot, size, bot.row_start, bot.col_start))
    K_ref = skin_base.fkCSB(kt, top, bot, size)
    assert np.allclose(K.toarray(), K_ref.toarray(), rtol=0,
                       atol=1e-14*abs(K_ref).max())
