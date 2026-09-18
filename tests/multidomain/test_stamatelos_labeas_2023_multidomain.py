r"""Stamatelos and Labeas (2023) with multi-domain plate models only

Reference
---------
D. G. Stamatelos, G. N. Labeas, *Buckling Analysis of Laminated Stiffened
Plates with Material Anisotropy Using the Rayleigh-Ritz Approach*,
Computation 2023, 11, 110. https://doi.org/10.3390/computation11060110

This is the :class:`.MultiDomain` counterpart of
``tests/tests_stiffpanelbay/test_stamatelos_labeas_2023.py``, whose published
values, laminates and helpers are imported, so that both modules compare
against a single source of truth. The single-domain :class:`.StiffPanelBay`
model of that module models every blade with the beam formulation of
:class:`.BladeStiff1D`. Here every skin strip and every blade is a
``plate_clpt_donnell`` domain, and each connection strategy of
:meth:`.MultiDomain.get_kC_conn` is exercised:

- ``'SSycte'`` and ``'SSxcte'``: the skin is split in strips, along `y` and
  along `x`, and must recover the single-domain skin.
- ``'BFycte'``: the blade is a plate domain standing on the skin, attached
  either at the common edge of two skin strips or along an interior line of
  a single skin domain. Tables 4 and 5 and Figure 7 are reproduced.
- ``'BFxcte'``: the same stiffened plate, rotated by 90 degrees, must give
  the ``'BFycte'`` result to machine precision.
- ``'SB'`` and ``'SB_TSL'``: the unsymmetric skin of Section 3.1.2 is split
  in two sub-laminates bonded over their whole area, which must recover the
  monolithic laminate, including its bending-extension coupling.

The penalty constants default to :func:`.calc_kt_kr`. For these thin skins
the default rotation penalty is soft and leaves a converged error of about
2 %, therefore the tests that check agreement to more digits raise the
penalties by ``PENALTY_FACTOR`` through the keys ``'kt'`` and ``'kr'`` of
the connection dictionaries.

"""
import os
import sys

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.sparse import coo_matrix, csr_matrix

import panels.bardell as bardell
from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import calc_kt_kr
from structsolve import lb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'tests_stiffpanelbay'))
import test_stamatelos_labeas_2023 as spb
from test_stamatelos_labeas_2023 import (PAPER, LAMINATES, PLY_T,
        LAMINAPROP_311, A_311, B_311, LAMINAPROP_312, STACK_312, T_312, A_312,
        B_312, H_321, A_322, B_322, N_STIFF_322, STIFF_STACK_322)

PENALTY_FACTOR = 100.


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def plate(a, b, stack, m, n, laminaprop=LAMINAPROP_312, **kwargs):
    """Simply supported plate domain, as in ``spb.make_skin``"""
    return Shell(a=a, b=b, r=None, stack=stack, plyt=PLY_T,
                 laminaprop=laminaprop, rho=1600., model='plate_clpt_donnell',
                 m=m, n=n, **kwargs)


def release(p, *edges):
    """Free the three translations of the given edges

    The rotations are already free by default, so a released edge is free.
    A connected edge must be released, since the connection is what then
    enforces the compatibility.

    """
    for edge in edges:
        for dof in ('u', 'v', 'w'):
            setattr(p, edge + dof, 1.)


def penalties(p1, p2, connection_type, factor):
    """Connection keys ``'kt'`` and ``'kr'``, scaled from the defaults"""
    kt, kr = calc_kt_kr(p1, p2, connection_type)
    return dict(kt=factor*kt, kr=factor*kr)


def kG_linear(p, N, phi, s0, s_total, size, direction='x', npts=80):
    r"""kG of one domain for a load varying linearly across the plate

    With ``direction='x'`` the load is ``Nxx(y) = N*(1 - phi*y/s_total)``,
    the paper's Eq. (18), with ``y = s0`` at the ``y1`` edge of the domain.
    With ``direction='y'`` it is ``Nyy(x) = N*(1 - phi*x/s_total)``, with
    ``x = s0`` at the ``x1`` edge. For ``s0 = 0`` and ``s_total = b`` and
    ``direction='x'`` this is ``spb.kG_linear``, only placed at the rows and
    columns of the domain within the assembly.

    """
    m, n, a, b = p.m, p.n, p.a, p.b
    xi, wt = leggauss(npts)
    if direction == 'x':
        dfx = np.array([bardell.calc_vec_fp(x, p.x1w, p.x1wr, p.x2w, p.x2wr)[:m]
                        for x in xi])
        fy = np.array([bardell.calc_vec_f(e, p.y1w, p.y1wr, p.y2w, p.y2wr)[:n]
                       for e in xi])
        Fx = (dfx.T*wt) @ dfx
        Fy = (fy.T*(wt*(1. - phi*(s0 + b*(1. + xi)/2.)/s_total))) @ fy
        scale = b/a
    elif direction == 'y':
        fx = np.array([bardell.calc_vec_f(x, p.x1w, p.x1wr, p.x2w, p.x2wr)[:m]
                       for x in xi])
        dfy = np.array([bardell.calc_vec_fp(e, p.y1w, p.y1wr, p.y2w, p.y2wr)[:n]
                        for e in xi])
        Fx = (fx.T*(wt*(1. - phi*(s0 + a*(1. + xi)/2.)/s_total))) @ fx
        Fy = (dfy.T*wt) @ dfy
        scale = a/b
    else:
        raise ValueError('direction must be "x" or "y"')
    blk = N*scale*np.einsum('ik,jl->jilk', Fx, Fy).reshape(m*n, m*n)
    idx = p.row_start + 3*np.arange(m*n) + 2
    rows, cols = np.meshgrid(idx, idx, indexing='ij')
    return coo_matrix((blk.ravel(), (rows.ravel(), cols.ravel())),
                      shape=(size, size)).tocsr()


def buckling(md, kG, num_eigvalues=6):
    kC = md.calc_kC(silent=True)
    return lb(csr_matrix(kC), csr_matrix(kG), silent=True,
              num_eigvalues=num_eigvalues)


def strips(a, b, stack, nstrip, m, n, along, factor, laminaprop=LAMINAPROP_312):
    """Skin split in ``nstrip`` equal strips, connected along ``along``"""
    if along == 'y':
        edges = np.linspace(0, b, nstrip + 1)
        skins = [plate(a, y2 - y1, stack, m, n, laminaprop, x0=0., y0=y1,
                       group='skin') for y1, y2 in zip(edges[:-1], edges[1:])]
        conn = [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0.,
                     **penalties(p1, p2, 'ycte', factor))
                for p1, p2 in zip(skins[:-1], skins[1:])]
        for p1, p2 in zip(skins[:-1], skins[1:]):
            release(p1, 'y2')
            release(p2, 'y1')
    else:
        edges = np.linspace(0, a, nstrip + 1)
        skins = [plate(x2 - x1, b, stack, m, n, laminaprop, x0=x1, y0=0.,
                       group='skin') for x1, x2 in zip(edges[:-1], edges[1:])]
        conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0.,
                     **penalties(p1, p2, 'xcte', factor))
                for p1, p2 in zip(skins[:-1], skins[1:])]
        for p1, p2 in zip(skins[:-1], skins[1:]):
            release(p1, 'x2')
            release(p2, 'x1')
    return skins, conn


def blade(length, height, stack, m, n, along, laminaprop=LAMINAPROP_312,
          t_skin=T_312):
    r"""Blade stiffener as a plate domain

    With ``along='x'`` the blade is parallel to `x` and is attached by its
    edge ``y = 0`` with ``'BFycte'``, with ``along='y'`` it is parallel to
    `y` and attached by its edge ``x = 0`` with ``'BFxcte'``.

    The connections tie the blade root to the skin mid-surface, so the
    domain is extended by ``t_skin/2``: it then reaches the same tip as a
    blade of height ``height`` standing on the skin surface, which is the
    geometry of the paper and of :class:`.BladeStiff1D`, and has practically
    the same second moment of area about the skin mid-surface.

    Every edge along the blade is free. The ends are simply supported like
    the skin, but free to move along the blade axis: forcing that axial
    displacement to zero over the whole height would clamp the blade against
    bending, which the skin does not do.

    """
    h = height + t_skin/2
    if along == 'x':
        f = plate(length, h, stack, m, n, laminaprop, x0=0., y0=0.,
                  group='blade')
        release(f, 'y1', 'y2')
        f.x1u = f.x2u = 1.
    else:
        f = plate(h, length, stack, m, n, laminaprop, x0=0., y0=0.,
                  group='blade')
        release(f, 'x1', 'x2')
        f.y1v = f.y2v = 1.
    return f


def rotate(stack):
    return [theta + 90 for theta in stack]


def blade_axial_load(load_blade):
    """Blade ``Nxx`` per unit skin ``Nxx``, the paper's Eq. (8)

    Same as ``spb.buckling(loaded_stiffeners=True)``: the axial force of the
    beam, ``E1*H_321``, spread over the height of the plate domain.

    """
    if not load_blade:
        return 0.
    skin = spb.make_skin(A_312, B_312, STACK_312, PLY_T, LAMINAPROP_312)
    st = spb.add_stiffeners(skin, 1, H_321, STACK_312, PLY_T,
                            LAMINAPROP_312)[0]
    Et_skin = spb.Et_reduced(STACK_312, PLY_T, LAMINAPROP_312)
    return -st.E1*H_321/Et_skin/(H_321 + T_312/2)


def single_blade(variant, phi, load_blade, factor=1.):
    """Section 3.2.1: one blade at mid-width, returns the eigenvalues

    ``variant`` is one of:

    - ``'edge'``: two skin strips, ``'SSycte'`` between them and the blade
      attached with ``'BFycte'`` at their common edge
    - ``'interior'``: a single skin domain and the blade attached with
      ``'BFycte'`` along the line ``y = b/2``
    - ``'rotated'``: the ``'edge'`` model rotated by 90 degrees, with
      ``'SSxcte'`` and ``'BFxcte'``

    """
    if variant == 'rotated':
        skins, conn = strips(B_312, A_312, rotate(STACK_312), 2, 10, 12,
                             along='x', factor=factor)
        f = blade(A_312, H_321, rotate(STACK_312), 6, 12, along='y')
        conn.append(dict(p1=skins[0], p2=f, func='BFxcte', xcte1=skins[0].a,
                         xcte2=0., **penalties(skins[0], f, 'xcte', factor)))
        f.Nyy = blade_axial_load(load_blade)
    else:
        if variant == 'edge':
            skins, conn = strips(A_312, B_312, STACK_312, 2, 12, 10, along='y',
                                 factor=factor)
            ycte1 = skins[0].b
        else:
            skins = [plate(A_312, B_312, STACK_312, 16, 16, x0=0., y0=0.,
                           group='skin')]
            conn = []
            ycte1 = B_312/2
        f = blade(A_312, H_321, STACK_312, 12, 6, along='x')
        conn.append(dict(p1=skins[0], p2=f, func='BFycte', ycte1=ycte1,
                         ycte2=0., **penalties(skins[0], f, 'ycte', factor)))
        f.Nxx = blade_axial_load(load_blade)
    md = MultiDomain(skins + [f], conn)
    size = md.get_size()
    if variant == 'rotated':
        kG = sum(kG_linear(p, -1., phi, p.x0, B_312, size, direction='y')
                 for p in skins)
    else:
        kG = sum(kG_linear(p, -1., phi, p.y0, B_312, size) for p in skins)
    kG = kG + f.calc_kG(size=size, row0=f.row_start, col0=f.col_start,
                        silent=True, finalize=True)
    eigvals, _ = buckling(md, kG)
    return eigvals


# --------------------------------------------------------------------------
# 'SSycte' and 'SSxcte': the skin split in strips
# --------------------------------------------------------------------------
@pytest.mark.parametrize('along', ['y', 'x'])
def test_strips_recover_the_single_domain_skin(along):
    """Section 3.1.2 plate, triangular load, three strips

    With the default penalties the strips are too flexible by up to 2 %,
    and the result does not improve with more terms. Raising the penalties
    recovers the single-domain eigenvalues to about 1e-4.

    """
    skin = spb.make_skin(A_312, B_312, STACK_312, PLY_T, LAMINAPROP_312)
    ref, _ = spb.buckling(skin, phi=1.)
    m, n = (12, 8) if along == 'y' else (8, 12)
    for factor, rtol in ((1., 2.5e-2), (PENALTY_FACTOR, 5e-4)):
        skins, conn = strips(A_312, B_312, STACK_312, 3, m, n, along, factor)
        md = MultiDomain(skins, conn)
        size = md.get_size()
        kG = sum(kG_linear(p, -1., 1., p.y0, B_312, size) for p in skins)
        eigvals, _ = buckling(md, kG)
        rel = eigvals[:3]/ref[:3] - 1
        assert np.all(rel <= 1e-6), (factor, rel)       # penalties are soft
        assert np.all(np.abs(rel) < rtol), (factor, rel)


def test_table3_all_three_modes_with_strips():
    """Table 3 through ``'SSycte'`` strips, with the calibration of the
    single-domain test and nothing re-fitted"""
    scale, _ = spb.table3_scale()
    skins, conn = strips(A_312, B_312, STACK_312, 3, 12, 8, 'y',
                         PENALTY_FACTOR)
    md = MultiDomain(skins, conn)
    size = md.get_size()
    kG = sum(kG_linear(p, -1., 1., p.y0, B_312, size) for p in skins)
    eigvals, _ = buckling(md, kG)
    for i, mode in enumerate(['(1, 1)', '(2, 1)', '(3, 1)']):
        got = eigvals[i]/1e3*scale
        assert np.isclose(got, PAPER['table3'][mode]['rr'], rtol=5e-3), mode


@pytest.mark.parametrize('name', list(LAMINATES))
def test_table2_laminates_with_strips_along_the_load(name):
    """Table 2 split in three ``'SSxcte'`` strips across the loading
    direction, loaded edges clamped, against the single domain of the
    StiffPanelBay test"""
    skin = spb.make_skin(A_311, B_311, LAMINATES[name], PLY_T, LAMINAPROP_311,
                         clamped_loaded_edges=True)
    ref, _ = spb.buckling(skin)
    skins, conn = strips(A_311, B_311, LAMINATES[name], 3, 8, 12, 'x',
                         PENALTY_FACTOR, laminaprop=LAMINAPROP_311)
    skins[0].x1wr = 0.
    skins[-1].x2wr = 0.
    md = MultiDomain(skins, conn)
    size = md.get_size()
    kG = sum(kG_linear(p, -1., 0., 0., B_311, size) for p in skins)
    eigvals, _ = buckling(md, kG)
    assert np.isclose(eigvals[0], ref[0], rtol=1e-3)


# --------------------------------------------------------------------------
# 'BFycte' and 'BFxcte': Tables 4 and 5, Section 3.2.1
# --------------------------------------------------------------------------
@pytest.mark.parametrize('variant', ['edge', 'interior'])
@pytest.mark.parametrize('table, load', [('table4', 'uniform'),
                                         ('table4', 'triangular'),
                                         ('table5', 'uniform'),
                                         ('table5', 'triangular')])
def test_tables4_and_5_blade_as_a_plate(table, load, variant):
    """One blade of height ``9*t_skin`` modelled as a plate domain

    With the scale factor of Table 3 and nothing re-fitted, all eight cases
    are within 5 % of the published Rayleigh-Ritz values, as the beam model
    of the StiffPanelBay test is. The two models of the blade agree within
    4 %, the plate being the stiffer one.

    """
    scale, _ = spb.table3_scale()
    phi = 0. if load == 'uniform' else 1.
    loaded = table == 'table5'
    eigvals = single_blade(variant, phi, loaded)
    got = eigvals[0]/1e3*scale
    assert np.isclose(got, PAPER[table][load]['rr'], rtol=5e-2)

    skin = spb.make_skin(A_312, B_312, STACK_312, PLY_T, LAMINAPROP_312)
    sts = spb.add_stiffeners(skin, 1, H_321, STACK_312, PLY_T, LAMINAPROP_312)
    ref, _ = spb.buckling(skin, sts, phi=phi, loaded_stiffeners=loaded,
                          Et_skin=spb.Et_reduced(STACK_312, PLY_T,
                                                 LAMINAPROP_312))
    assert np.isclose(eigvals[0], ref[0], rtol=4.5e-2)


@pytest.mark.parametrize('phi, load_blade', [(0., False), (1., False),
                                             (0., True), (1., True)])
def test_bfxcte_is_bfycte_rotated(phi, load_blade):
    """The rotated model, with ``'SSxcte'`` and ``'BFxcte'``, must give the
    ``'SSycte'`` and ``'BFycte'`` eigenvalues to machine precision"""
    ycte = single_blade('edge', phi, load_blade)
    xcte = single_blade('rotated', phi, load_blade)
    assert np.allclose(xcte[:3], ycte[:3], rtol=1e-9)


# --------------------------------------------------------------------------
# 'BFycte': Figure 7, Section 3.2.2
# --------------------------------------------------------------------------
def figure7_multidomain(height, m=12, n=6, mf=12, nf=4):
    """Ten blades on the 500 mm plate, eleven skin strips, returns P_cr in N"""
    stack = LAMINATES['[0_3/90_3]s']
    t_skin = len(stack)*PLY_T
    skins, conn = strips(A_322, B_322, stack, N_STIFF_322 + 1, m, n, 'y', 1.,
                         laminaprop=LAMINAPROP_311)
    blades = []
    if height > 0:
        for p in skins[:-1]:
            f = blade(A_322, height, STIFF_STACK_322, mf, nf, 'x',
                      laminaprop=LAMINAPROP_311, t_skin=t_skin)
            conn.append(dict(p1=p, p2=f, func='BFycte', ycte1=p.b, ycte2=0.))
            blades.append(f)
    md = MultiDomain(skins + blades, conn)
    size = md.get_size()
    kG = sum(kG_linear(p, -1., 0., p.y0, B_322, size) for p in skins)
    eigvals, _ = buckling(md, kG, num_eigvalues=4)
    return eigvals[0]*B_322


@pytest.mark.parametrize('height', [0., 4.e-3, 8.e-3])
def test_figure7_ten_blades_match_bladestiff1d(height):
    """Ten plate-domain blades against ten :class:`.BladeStiff1D` beams

    In this global mode the blades bend with the skin, and the two blade
    models agree within 1.1 %, while the load grows four times from the
    unstiffened plate to the 8 mm blades.

    """
    stack = LAMINATES['[0_3/90_3]s']
    skin = spb.make_skin(A_322, B_322, stack, PLY_T, LAMINAPROP_311)
    sts = () if height == 0. else spb.add_stiffeners(
        skin, N_STIFF_322, height, STIFF_STACK_322, PLY_T, LAMINAPROP_311)
    ref, _ = spb.buckling(skin, sts)
    P_ref = ref[0]*B_322
    P_md = figure7_multidomain(height)
    assert np.isclose(P_md, P_ref, rtol=1.5e-2)


# --------------------------------------------------------------------------
# 'SB' and 'SB_TSL': the unsymmetric skin split in two bonded sub-laminates
# --------------------------------------------------------------------------
N_BOTTOM = 2            # plies of the bottom sub-laminate, stack[0] is at -z
M_SB = N_SB = 12


def sb_split(func, order, kt=None):
    """Top ``[90/0/0]`` bonded over its whole area onto bottom ``[0/90]``

    Returns the eigenvalues under the triangular load. ``order`` is the
    order of the two domains in the assembly, which must not matter.

    """
    top = plate(A_312, B_312, STACK_312[N_BOTTOM:], M_SB, N_SB, x0=0., y0=0.,
                group='top')
    bot = plate(A_312, B_312, STACK_312[:N_BOTTOM], M_SB, N_SB, x0=0., y0=0.,
                group='bottom')
    # the bottom sub-laminate only follows the top one
    release(bot, 'x1', 'x2', 'y1', 'y2')
    if kt is None:
        kt = PENALTY_FACTOR*calc_kt_kr(top, bot, 'bot-top')[0]
    if func == 'SB':
        conn = [dict(p1=top, p2=bot, func='SB', kt=kt)]
        c = None
    else:
        #NOTE a pristine interface: the separation is zero and the
        #     stiffness of the traction-separation law is k_o everywhere
        conn = [dict(p1=top, p2=bot, func='SB_TSL', tsl_type='bilinear',
                     k_o=kt, tau_o=1e30, G1c=1e30, nr_x_gauss=2*M_SB,
                     nr_y_gauss=2*N_SB)]
    md = MultiDomain([top, bot] if order == 'top first' else [bot, top],
                     conn)
    size = md.get_size()
    if func == 'SB_TSL':
        c = np.zeros(size)
    kC = md.calc_kC(c=c, silent=True)
    #NOTE w is common to both, so the whole load can be given to the top
    kG = kG_linear(top, -1., 1., 0., B_312, size)
    eigvals, _ = lb(csr_matrix(kC), kG, silent=True, num_eigvalues=4)
    return eigvals


def sb_monolithic():
    """The same laminate, referred to the mid-surface of the top part

    The skin is simply supported with ``u = v = 0`` on its reference
    surface, which in the split model is the mid-surface of the top
    sub-laminate; ``offset`` moves the reference surface of the monolithic
    laminate there.

    """
    h_bottom = N_BOTTOM*PLY_T
    mono = plate(A_312, B_312, STACK_312, M_SB, N_SB, offset=-h_bottom/2)
    mono.row_start = 0
    size = mono.get_size()
    eigvals, _ = lb(mono.calc_kC(silent=True),
                    kG_linear(mono, -1., 1., 0., B_312, size), silent=True,
                    num_eigvalues=4)
    return eigvals


@pytest.mark.parametrize('func', ['SB', 'SB_TSL'])
@pytest.mark.parametrize('order', ['top first', 'bottom first'])
def test_bonded_sublaminates_recover_the_laminate(func, order):
    """The bending-extension coupling of ``[0/90/90/0/0]`` must be rebuilt
    by the offset ``dsb`` of the connection, whichever domain comes first"""
    ref = sb_monolithic()
    eigvals = sb_split(func, order)
    assert np.allclose(eigvals[:3], ref[:3], rtol=1e-3)


def test_sb_old_hardcoded_penalty_was_far_too_soft():
    """Up to panels 0.6.9 ``'SB'`` ignored :func:`.calc_kt_kr` and used
    ``kt = 2e5``, a value in N/mm**3. In SI units the sub-laminates were
    practically disconnected and the buckling load was 84 % too low"""
    ref = sb_monolithic()
    eigvals = sb_split('SB', 'top first', kt=2e5)
    assert eigvals[0]/ref[0] < 0.2


def test_sb_panels_must_share_the_area():
    top = plate(A_312, B_312, STACK_312[N_BOTTOM:], 6, 6)
    bot = plate(A_312/2, B_312, STACK_312[:N_BOTTOM], 6, 6)
    md = MultiDomain([top, bot], [dict(p1=top, p2=bot, func='SB')])
    with pytest.raises(ValueError, match='same dimensions'):
        md.calc_kC(silent=True)


# --------------------------------------------------------------------------
# MultiDomain must not modify the connection dictionaries
# --------------------------------------------------------------------------
def test_connection_order_and_repeated_calls():
    """``p1`` after ``p2`` in the assembly, and ``ycte1 != ycte2``

    Up to panels 0.6.9 the coordinates were swapped inside the user's
    dictionary, so every second call to :meth:`.MultiDomain.get_kC_conn`, for
    instance ``calc_kC()`` followed by ``calc_kT()``, used them swapped back.

    """
    skins, conn = strips(A_312, B_312, STACK_312, 2, 8, 6, 'y', 1.)
    reversed_conn = [dict(p1=skins[1], p2=skins[0], func='SSycte', ycte1=0.,
                          ycte2=skins[0].b)]
    before = dict(reversed_conn[0])
    md_ref = MultiDomain(skins, conn)
    k_ref = md_ref.get_kC_conn().toarray()
    md = MultiDomain(skins, reversed_conn)
    k1 = md.get_kC_conn().toarray()
    k2 = md.get_kC_conn().toarray()
    assert reversed_conn[0] == before
    scale = np.abs(k_ref).max()
    assert np.abs(k1 - k_ref).max() < 1e-12*scale
    assert np.abs(k2 - k_ref).max() < 1e-12*scale


def test_connection_to_itself_is_rejected():
    p = plate(A_312, B_312, STACK_312, 6, 6)
    md = MultiDomain([p], [dict(p1=p, p2=p, func='SSycte', ycte1=0.,
                                ycte2=p.b)])
    with pytest.raises(ValueError, match='different panels'):
        md.get_kC_conn()


if __name__ == '__main__':
    pytest.main(['-v', __file__])
