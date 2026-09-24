r"""Multi-domain plates with the first-order (FSDT) and Reddy's third-order
(TSDT) shear deformation theories

The penalty connections of :mod:`panels.multidomain.connections.kCsdt` are
verified against single-domain models and against the literature:

- ``'SSxcte'`` and ``'SSycte'``: a plate divided in several domains must give
  the results of the same plate modelled with a single domain, here the
  frequencies of Noor (1973) and the deflection of Pagano (1970), with the
  single-domain references of ``tests/tests_shell/test_fsdt_tsdt_plates.py``.

- ``'SB'``: two laminates connected over their area with the rotations tied
  must behave like a single laminate with both stacking sequences, which for
  the FSDT is exact. Without the rotations tied, the FSDT laminates rotate
  independently like in a layerwise theory and the connected laminates are
  more flexible than the single laminate. The TSDT enforces zero transverse
  shear strains at the faces of each laminate, including the interface, and
  the connected laminates are stiffer than the single laminate. In both cases
  the difference vanishes with `(h/a)^2`.

The FSDT uses the shear correction factor `k = 5/6` of the references,
instead of the default ``Shell.fsdt_shear_correction = 'rohwer'``.

References:

- Noor, A. K., "Free vibrations of multilayered composite plates", AIAA J.,
  11(7), 1038-1039, 1973, with the FSDT (`k = 5/6`) and the TSDT of Reddy,
  J. N., "A simple higher-order theory for laminated composite plates", J.
  Appl. Mech., 51, 745-752, 1984, as reproduced in Table 3 of Adim, B.,
  Daouadji, T. H. and Rabahi, A., Int. J. Adv. Struct. Eng., 8, 103-117,
  2016.

- Pagano, N. J., "Exact solutions for rectangular bidirectional composites
  and sandwich plates", J. Compos. Mater., 4, 20-34, 1970, with the TSDT of
  Reddy (1984) and the FSDT of Reddy and Chao (1981), as reproduced in Table
  1 of Natarajan, S., Nguyen-Xuan, H., Ferreira, A. J. M. and Carrera, E.,
  arXiv:1312.4032.

The default penalty constants of :func:`.calc_kt_kr` give, for these thick
plates, errors of a few percent, as for the models based on the classical
laminated plate theory, therefore the penalties are raised here.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.special import roots_legendre
from structsolve import solve
from structsolve.sparseutils import remove_null_cols

from panels import modelDB
from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import calc_kt_kr

E2 = 1.e9
RHO = 1500.

CLPT = 'plate_clpt_donnell'
FSDT = 'plate_fsdt_donnell'
TSDT = 'plate_tsdt_donnell'
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')


def orthotropic(E1E2, G12=0.6, G13=0.6, G23=0.5, nu12=0.25):
    return (E1E2*E2, E2, nu12, G12*E2, G13*E2, G23*E2)


def set_edge(s, edge, kind):
    r"""Simply supported (SS-1) edge for ``kind='S'``, free for ``'F'``"""
    tangential = 'v' if edge.startswith('x') else 'u'
    rot_tangential = 'phiy' if edge.startswith('x') else 'phix'
    for field in FIELDS:
        setattr(s, edge + field, 1.)
        setattr(s, edge + field + 'r', 1.)
    if kind == 'S':
        setattr(s, edge + tangential, 0.)
        setattr(s, edge + 'w', 0.)
        setattr(s, edge + rot_tangential, 0.)
    elif kind != 'F':
        raise ValueError(kind)


def make_plate(model, a, b, h, stack, laminaprop, bcs='SSSS', m=10, n=10,
               x0=0., y0=0.):
    """Plate with the edges x = 0, y = 0, x = a and y = b given by ``bcs``"""
    s = Shell(group='plate', x0=x0, y0=y0)
    s.model = model
    s.a = a
    s.b = b
    s.stack = stack
    s.plyt = h/len(stack)
    s.laminaprop = laminaprop
    s.rho = RHO
    s.m = m
    s.n = n
    # the k = 5/6 of the references, which is also needed by the 'SB'
    # connection to reproduce a single laminate, instead of the default
    # 'rohwer', which depends on the stacking sequence of each laminate
    s.fsdt_shear_correction = 5/6
    for edge, kind in zip(('x1', 'y1', 'x2', 'y2'), bcs):
        set_edge(s, edge, kind)
    s._rebuild()
    return s


def penalties(p1, p2, connection_type, factor):
    kt, kr = calc_kt_kr(p1, p2, connection_type)
    return dict(kt=factor*kt, kr=factor*kr)


def domains(model, a, h, stack, laminaprop, layout, xs=0.4, ys=0.6, m=10,
            n=10, factor=1.e3):
    r"""Square plate divided in domains, with all the outer edges simply
    supported

    ``layout`` is ``'x'`` for two domains divided at `x = x_s a`, ``'y'``
    for two domains divided at `y = y_s a`, or ``'2x2'`` for four domains
    divided at both, with an interior corner.

    """
    def plate(x0, y0, dx, dy, bcs):
        return make_plate(model, dx, dy, h, stack, laminaprop, bcs, m=m, n=n,
                          x0=x0, y0=y0)
    if layout == 'x':
        p1 = plate(0, 0, xs*a, a, 'SSFS')
        p2 = plate(xs*a, 0, (1 - xs)*a, a, 'FSSS')
        conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0,
                     **penalties(p1, p2, 'xcte', factor))]
        return [p1, p2], conn
    if layout == 'y':
        p1 = plate(0, 0, a, ys*a, 'SSSF')
        p2 = plate(0, ys*a, a, (1 - ys)*a, 'SFSS')
        conn = [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0,
                     **penalties(p1, p2, 'ycte', factor))]
        return [p1, p2], conn
    assert layout == '2x2'
    p00 = plate(0, 0, xs*a, ys*a, 'SSFF')
    p10 = plate(xs*a, 0, (1 - xs)*a, ys*a, 'FSSF')
    p01 = plate(0, ys*a, xs*a, (1 - ys)*a, 'SFFS')
    p11 = plate(xs*a, ys*a, (1 - xs)*a, (1 - ys)*a, 'FFSS')
    conn = []
    for pA, pB in ((p00, p10), (p01, p11)):
        conn.append(dict(p1=pA, p2=pB, func='SSxcte', xcte1=pA.a, xcte2=0,
                         **penalties(pA, pB, 'xcte', factor)))
    for pA, pB in ((p00, p01), (p10, p11)):
        conn.append(dict(p1=pA, p2=pB, func='SSycte', ycte1=pA.b, ycte2=0,
                         **penalties(pA, pB, 'ycte', factor)))
    return [p00, p10, p01, p11], conn


def frequencies(K, M, k):
    K, _ = remove_null_cols(K)
    M, _ = remove_null_cols(M)
    #NOTE the dense symmetric solver, see test_fsdt_tsdt_plates.py
    eigvals = eigh(K.toarray(), M.toarray(), eigvals_only=True,
                   subset_by_index=[0, k - 1])
    return np.sqrt(eigvals)


def sinusoidal_load(s, q0, a, size):
    r"""External force vector of the pressure `q_0 \sin(\pi x/a) \sin(\pi
    y/a)` over the domain of ``s``, in the global coordinates of the plate"""
    fg = modelDB.db[s.model]['field'].fg
    fext = np.zeros(size)
    g = np.zeros((5, s.get_size()))
    points, weights = roots_legendre(2*max(s.m, s.n))
    for xi, wx in zip(points, weights):
        x = (xi + 1)*s.a/2
        for eta, wy in zip(points, weights):
            y = (eta + 1)*s.b/2
            g[:] = 0
            fg(g, x, y, s)
            q = q0*np.sin(np.pi*(s.x0 + x)/a)*np.sin(np.pi*(s.y0 + y)/a)
            fext[s.col_start:s.col_end] += wx*wy*(s.a/2)*(s.b/2)*q*g[2]
    return fext


# Noor (1973), [0/90]_2 with E1/E2 = 10 and a/h = 5, 3D elasticity 8.1445
NOOR_1973 = {TSDT: 8.1954, FSDT: 8.2246}


@pytest.mark.parametrize('layout', ['x', 'y', '2x2'])
@pytest.mark.parametrize('model', [FSDT, TSDT])
def test_frequencies_divided_plate_noor(model, layout):
    r"""Noor (1973), `\bar\omega = \omega a^2/h \sqrt{\rho/E_2}`, with the
    first modes also compared with the single-domain plate"""
    a = 1.
    h = a/5
    stack = [0, 90, 0, 90]
    laminaprop = orthotropic(10)
    panels, conn = domains(model, a, h, stack, laminaprop, layout)
    md = MultiDomain(panels, conn, conn_method='penalty')
    omegas = frequencies(md.calc_kC(), md.calc_kM(silent=True), k=8)
    scale = a**2/h*np.sqrt(RHO/E2)
    assert np.isclose(omegas[0]*scale, NOOR_1973[model], rtol=1e-4)

    single = make_plate(model, a, a, h, stack, laminaprop, m=14, n=14)
    ref = frequencies(single.calc_kC(), single.calc_kM(), k=8)
    assert np.allclose(omegas, ref, rtol=1e-4)


# Pagano (1970), [0/90/90/0] with a/h = 10, 3D elasticity 0.7430
PAGANO = {TSDT: 0.7147, FSDT: 0.6628}


@pytest.mark.parametrize('model', [FSDT, TSDT])
def test_bending_divided_plate_pagano(model):
    r"""Pagano (1970), sinusoidal pressure and `\bar{w} = 100 w E_2 h^3/(q_0
    a^4)` at the center of the plate, which is the interior corner of four
    domains

    The stress resultants of the domains, including the transverse shear
    forces, are also compared with those of the single-domain plate.
    """
    a = 1.
    h = a/10
    q0 = 1.e3
    stack = [0, 90, 90, 0]
    laminaprop = orthotropic(25, G12=0.5, G13=0.5, G23=0.2)
    panels, conn = domains(model, a, h, stack, laminaprop, '2x2', xs=0.5,
                           ys=0.5)
    md = MultiDomain(panels, conn, conn_method='penalty')
    size = md.get_size()
    fext = sum(sinusoidal_load(p, q0, a, size) for p in panels)
    c = solve(md.calc_kC(), fext, silent=True)
    for p in panels:
        res = md.uvw(c, group=None, eval_panel=p, gridx=2, gridy=2)
        # the corner of each domain at the center of the plate
        w = res['w'][0][int(p.y0 == 0), int(p.x0 == 0)]
        wbar = 100*w*E2*h**3/(q0*a**4)
        # the FSDT reference, 0.6628, is one unit of the last digit above its
        # Navier solution with k = 5/6, 0.66271
        assert np.isclose(wbar, PAGANO[model], rtol=2e-4)

    single = make_plate(model, a, a, h, stack, laminaprop, m=14, n=14)
    single.x0 = single.y0 = 0.
    single.col_start, single.col_end = 0, single.get_size()
    c_single = solve(single.calc_kC(), sinusoidal_load(single, q0, a,
                     single.get_size()), silent=True)
    p = panels[0]
    res = md.stress(c, group=None, eval_panel=p, gridx=4, gridy=4)
    xs, ys = res['x'][0], res['y'][0]
    _, ref = single.stress(c_single, xs=xs + p.x0, ys=ys + p.y0)
    names = single._stress_names()
    assert set(names) <= set(res)
    assert 'Qx' in names and 'Qy' in names
    for name in ('Nxx', 'Mxx', 'Myy', 'Mxy', 'Qx', 'Qy'):
        scale = np.abs(ref[name]).max()
        assert np.allclose(res[name][0], ref[name], atol=2e-3*scale), name


def sb_plates(model, ah, stack_top, stack_bot, factor, kr_factor=None, m=10,
              n=10):
    r"""Two laminates connected over their area, ``stack_top`` over
    ``stack_bot``, and the single laminate with both stacking sequences"""
    a = 1.
    num_plies = len(stack_top) + len(stack_bot)
    plyt = a/ah/num_plies
    laminaprop = orthotropic(10)
    top = make_plate(model, a, a, plyt*len(stack_top), stack_top, laminaprop,
                     m=m, n=n)
    bot = make_plate(model, a, a, plyt*len(stack_bot), stack_bot, laminaprop,
                     m=m, n=n)
    kt, _ = calc_kt_kr(top, bot, 'bot-top')
    conn = dict(p1=top, p2=bot, func='SB', kt=factor*kt)
    if kr_factor is not None:
        h = a/ah
        conn['kr'] = kr_factor*factor*kt*h**2
    md = MultiDomain([top, bot], [conn], conn_method='penalty')
    #NOTE the first ply of a stacking sequence is at the bottom
    single = make_plate(model, a, a, a/ah, list(stack_bot) + list(stack_top),
                        laminaprop, m=m, n=n)
    return md, single


@pytest.mark.parametrize('model', [CLPT, FSDT])
def test_skin_base_equals_single_laminate(model):
    r"""The laminates `[0/90]` over `[0/90]`, with the rotations tied, give
    the single laminate `[0/90]_2`, for the FSDT the plate of Noor (1973)
    with `E_1/E_2 = 10` and `a/h = 5`"""
    ah = 5
    md, single = sb_plates(model, ah, [0, 90], [0, 90], factor=1.e5,
                           kr_factor=1.)
    omegas = frequencies(md.calc_kC(), md.calc_kM(silent=True), k=8)
    ref = frequencies(single.calc_kC(), single.calc_kM(), k=8)
    assert np.allclose(omegas, ref, rtol=2e-5)
    if model == FSDT:
        # a = 1
        scale = ah*np.sqrt(RHO/E2)
        assert np.isclose(omegas[0]*scale, NOOR_1973[FSDT], rtol=1e-4)


@pytest.mark.parametrize('model', [FSDT, TSDT])
def test_skin_base_layerwise_converges_to_single_laminate(model):
    r"""Laminates connected only at the interface

    The FSDT laminates rotate independently and are more flexible than the
    single laminate, the TSDT laminates have zero transverse shear strains at
    the interface and are stiffer. The difference to the single laminate
    decreases with `(h/a)^2`.
    """
    diffs = []
    for ah in (20, 40):
        md, single = sb_plates(model, ah, [0, 90], [90, 0, 90], factor=1.e5)
        omegas = frequencies(md.calc_kC(), md.calc_kM(silent=True), k=4)
        ref = frequencies(single.calc_kC(), single.calc_kM(), k=4)
        diff = omegas[0]/ref[0] - 1
        if model == FSDT:
            assert diff < 0
        else:
            assert diff > 0
        diffs.append(abs(diff))
    assert diffs[0] < 5e-3
    assert 3.5 < diffs[0]/diffs[1] < 4.5


def test_skin_base_rotation_penalty_stiffens():
    r"""Tying the rotations can only stiffen the connected laminates"""
    free, _ = sb_plates(FSDT, 10, [0, 90], [90, 0, 90], factor=1.e3)
    tied, single = sb_plates(FSDT, 10, [0, 90], [90, 0, 90], factor=1.e3,
                             kr_factor=1.)
    w_free = frequencies(free.calc_kC(), free.calc_kM(silent=True), k=6)
    w_tied = frequencies(tied.calc_kC(), tied.calc_kM(silent=True), k=6)
    # the in-plane modes are not affected by the rotations
    assert np.all(w_free <= w_tied*(1 + 1e-8))
    assert np.all(w_free[:2] < w_tied[:2]*(1 - 1e-4))


#NOTE 'BFycte' and 'BFxcte' are tested in test_bf_sdt.py
@pytest.mark.parametrize('func', ['SB_TSL'])
def test_unsupported_connections(func):
    p1 = make_plate(FSDT, 0.5, 0.5, 0.01, [0, 90], orthotropic(10), m=4, n=4)
    p2 = make_plate(FSDT, 0.5, 0.5, 0.01, [0, 90], orthotropic(10), m=4, n=4)
    conn = [dict(p1=p1, p2=p2, func=func, xcte1=p1.a, xcte2=0, ycte1=p1.b,
                 ycte2=0)]
    md = MultiDomain([p1, p2], conn, conn_method='penalty')
    with pytest.raises(NotImplementedError, match=func):
        md.get_kC_conn()


def test_connection_between_different_theories():
    p1 = make_plate(CLPT, 0.5, 0.5, 0.01, [0, 90], orthotropic(10), m=4, n=4)
    p2 = make_plate(FSDT, 0.5, 0.5, 0.01, [0, 90], orthotropic(10), m=4, n=4)
    md = MultiDomain([p1, p2], conn_method='penalty')
    assert md.get_size() == (3 + 5)*4*4
    conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0)]
    with pytest.raises(NotImplementedError, match='different number of DOFs'):
        md.get_kC_conn(conn)
