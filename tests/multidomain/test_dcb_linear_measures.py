r"""Linear solutions of the DCB model of the thesis of D'Souza (2024)

Values of the tables of the sections "Load measured at the loaded edge" and
"Penalty stiffness of the edge connections" of doc/source/cohesive_zone.rst.

"""
import numpy as np
from scipy.special import roots_legendre
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import calc_kt_kr, fkCld_xcte


def dcb_thesis_linear(k_o, edge_penalty_factor=1., a2=0.015, wp=0.5,
                      nr_terms=(10, 6, 8, 6), nr_x_gauss=60, nr_y_gauss=20):
    r"""Linear solution of the DCB model of the thesis, per mm of opening

    The DCB of ``test_dcb_damage.py`` (``dcb_damage_prop_no_f_kcrack``):
    `L = 65` mm, `b = 25` mm, `a_0 = 48` mm, arms `[0]_{15}` of 0.14 mm,
    three domains per arm (cohesive domain of 17 mm, connecting domain of
    length ``a2``, free arm), the tip of the top arm loaded by the
    prescribed displacement ``wp`` imposed with `k_w = 10^6`, the tip of the
    bottom arm simply supported and the rotation of the far edge of the top
    cohesive domain constrained. ``nr_terms`` gives `m, n` of the cohesive
    domains and of the other domains, and ``edge_penalty_factor`` multiplies
    the default penalties of :func:`.calc_kt_kr` of the ``'SSxcte'``
    connections. ``wp`` must keep the interface below the damage onset,
    where the solution is linear in ``wp``; the results are divided by
    ``wp``.

    Returns a dict with, per mm of ``wp``: ``R``, the reaction of
    :meth:`.MultiDomain.reaction_line_pd_xcte`; ``F``, the area integral of
    the traction with the full separation,
    :meth:`.MultiDomain.force_out_plane_damage`; ``F_corr``, the same with
    the corrected separation of :meth:`.MultiDomain.correct_separation`;
    ``F_Qx``, the line integral of `Q_x` of
    :meth:`.MultiDomain.force_out_plane` at the second last of 100 points
    along `x` of the free arm; ``del_max``, the largest normal separation;
    and, not divided by ``wp``, ``del_o``, the separation at damage onset.

    """
    m_tsl, n_tsl, m, n = nr_terms
    tau_o, G1c = 87., 1.12
    E1 = (138300. + 128000.)/2.
    E2 = (10400. + 11500.)/2.
    G12 = 5190.
    nu12 = 0.316
    laminaprop = (E1, E2, nu12, G12, G12, G12)
    a, b, a0 = 65., 25., 48.
    a1 = a - a0
    kw = dict(plyt=0.14, stack=[0]*15, laminaprop=laminaprop)
    top1 = Shell(group='top', x0=0, y0=0, a=a1, b=b, m=m_tsl, n=n_tsl, **kw)
    top2 = Shell(group='top', x0=a1, y0=0, a=a2, b=b, m=m, n=n, **kw)
    top3 = Shell(group='top', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, **kw)
    bot1 = Shell(group='bot', x0=0, y0=0, a=a1, b=b, m=m_tsl, n=n_tsl, **kw)
    bot2 = Shell(group='bot', x0=a1, y0=0, a=a2, b=b, m=m, n=n, **kw)
    bot3 = Shell(group='bot', x0=a1+a2, y0=0, a=a-a1-a2, b=b, m=m, n=n, **kw)
    for p in (top1, top2, top3, bot1, bot2, bot3):
        for e in ('x1', 'x2', 'y1', 'y2'):
            for d in ('u', 'v', 'w'):
                setattr(p, e + d, 1.)
                setattr(p, e + d + 'r', 1.)
    top1.x1wr = 0.
    bot3.x2u = bot3.x2v = bot3.x2w = 0.
    conn = [
        dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0),
        dict(p1=top2, p2=top3, func='SSxcte', xcte1=top2.a, xcte2=0),
        dict(p1=bot1, p2=bot2, func='SSxcte', xcte1=bot1.a, xcte2=0),
        dict(p1=bot2, p2=bot3, func='SSxcte', xcte1=bot2.a, xcte2=0),
        dict(p1=top1, p2=bot1, func='SB_TSL', tsl_type='bilinear',
             nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss, tau_o=tau_o,
             G1c=G1c, k_o=k_o),
    ]
    for conni in conn[:4]:
        kt, kr = calc_kt_kr(conni['p1'], conni['p2'], 'xcte')
        conni['kt'] = edge_penalty_factor*kt
        conni['kr'] = edge_penalty_factor*kr
    assy = MultiDomain(panels=[bot1, bot2, bot3, top1, top2, top3],
                       conn=conn, conn_method='penalty')
    size = assy.get_size()
    # pristine interface
    assy.update_TSL_history(curr_max_dmg_index=np.zeros((nr_y_gauss,
                                                         nr_x_gauss)))
    kC = assy.calc_kC(c=np.zeros(size))
    k_w = 1e6
    kCp = finalize_symmetric_matrix(fkCld_xcte(0., 0., k_w, top3, top3.a,
                                    size, top3.row_start, top3.col_start))
    top3.add_distr_pd_fixed_x(top3.a, None, None, k_w, funcw=lambda y: wp)
    c = solve(kC + kCp, assy.calc_fext(), silent=True)

    R = assy.reaction_line_pd_xcte(c, top3, top3.a, k_w, lambda y: wp)
    F = assy.force_out_plane_damage(conn=conn, c=c)
    kw_tsl, dmg, del_corr, _ = assy.calc_k_dmg(c=c, pA=top1, pB=bot1,
            nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss,
            tsl_type='bilinear', prev_max_dmg_index=assy.dmg_index,
            k_i=k_o, tau_o=tau_o, G1c=G1c)
    assert np.all(dmg == 0), 'the interface must stay below damage onset'
    _, wx = roots_legendre(nr_x_gauss)
    _, wy = roots_legendre(nr_y_gauss)
    F_corr = np.einsum('i,j,ij->', wy, wx, kw_tsl*del_corr)*top1.a*top1.b/4
    F_Qx = assy.force_out_plane(c, None, eval_panel=top3, gridx=100,
                                nr_y_gauss=128)
    return dict(R=R/wp, F=F/wp, F_corr=F_corr/wp, F_Qx=F_Qx/wp,
                del_max=del_corr.max()/wp, del_o=tau_o/k_o)




def test_load_measures():
    # tip displacement of 1 mm, default edge penalties; columns: k_o (N/mm^3),
    # reaction, int k Delta dA, int k Delta^corr dA, int Q_x dy (N)
    table = [
        (1e4, 24.99, 24.99, 388.1, 24.42),
        (5e4, 25.87, 25.87, 593.1, 25.28),
        (2e5, 26.42, 26.42, 839.4, 25.82),
    ]
    for k_o, R, F, F_corr, F_Qx in table:
        res = dcb_thesis_linear(k_o, wp=0.5)
        assert np.isclose(res['R'], R, rtol=5e-4)
        assert np.isclose(res['F'], F, rtol=5e-4)
        assert np.isclose(res['F_corr'], F_corr, rtol=5e-4)
        assert np.isclose(res['F_Qx'], F_Qx, rtol=5e-4)
        # equilibrium of the loaded arm, the traction with the full
        # separation is the reaction
        assert np.isclose(res['F'], res['R'], rtol=1e-4)
    # with k_o = 2e5 an opening of 1 mm is beyond the damage onset, hence
    # the linear solution at 0.5 mm, scaled
    assert res['del_max'] > res['del_o']


def test_edge_penalty_factor():
    # k_o = 2e5 N/mm^3; columns: factor on the default edge penalties,
    # P/delta (N/mm), Delta_max/delta
    table = [
        (1, 26.42, 5.07e-4),
        (10, 32.13, 6.26e-4),
        (100, 32.84, 6.42e-4),
        (1000, 32.92, 6.43e-4),
    ]
    res = {}
    for factor, P, del_max in table:
        res[factor] = dcb_thesis_linear(2e5, edge_penalty_factor=factor)
        assert np.isclose(res[factor]['R'], P, rtol=5e-4)
        assert np.isclose(res[factor]['del_max'], del_max, rtol=1e-3)
    # with the default penalties, the stiffness is 20% lower and the
    # separation at the crack front lags the imposed displacement by 21%
    assert np.isclose(1 - res[1]['R']/res[1000]['R'], 0.20, atol=0.005)
    assert np.isclose(1 - res[1]['del_max']/res[1000]['del_max'], 0.21,
                      atol=0.005)
    # connecting domain of 5 mm instead of 0.015 mm
    assert np.isclose(dcb_thesis_linear(2e5, a2=5.)['R'], 26.94, rtol=5e-4)
    assert np.isclose(dcb_thesis_linear(2e5, edge_penalty_factor=100,
                                        a2=5.)['R'], 32.85, rtol=5e-4)
