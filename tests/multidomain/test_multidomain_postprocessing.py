import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from structsolve import solve

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import calc_kt_kr


E = 70e9
nu = 0.3
t = 2e-3
N = 1e4 # applied tension, N/m
B = 0.3


@pytest.fixture(scope='module')
def tension():
    """Two plates in series under uniform tension ``N``

    ::

        u = 0  ________________________
         |    |          |            |
         |    |    p1    |     p2     | --> N
         |    |__________|____________|

    The stress state is exactly uniform, so the post-processing of every
    panel must give ``Nxx = N``, ``exx = N/(E t)`` and ``eyy = -nu exx``.

    """
    def plate(x0, a):
        p = Shell(group='g', x0=x0, y0=0, a=a, b=B, r=0., m=8, n=6,
                  stack=[0], plyt=t, laminaprop=(E, nu),
                  model='plate_clpt_donnell')
        for edge in ('x1', 'x2', 'y1', 'y2'):
            for dof in ('u', 'v', 'w'):
                setattr(p, edge + dof, 1.)
                setattr(p, edge + dof + 'r', 1.)
        return p

    p1 = plate(0., 0.4)
    p2 = plate(0.4, 0.6)
    p1.x1u = p1.x1w = p1.x1wr = 0.
    p2.x2w = 0.
    # removing the rigid body motion along y
    p1.add_point_pd(0., 0., 0., 0., 1e9, 0., 0., 0.)
    p2.add_distr_load_fixed_x(p2.a, funcx=lambda y: N)

    kt, kr = calc_kt_kr(p1, p2, 'xcte')
    conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0.,
                 kt=kt, kr=kr)]
    md = MultiDomain([p1, p2], conn, conn_method='penalty')
    c = solve(md.calc_kC(), md.calc_fext(), silent=True)
    return md, c


def test_uniform_stress_and_strain(tension):
    md, c = tension
    s = md.stress(c, 'g', gridx=7, gridy=5)
    e = md.strain(c, 'g', gridx=7, gridy=5)
    exx = N/(E*t)
    for i in range(2):
        assert s['x'][i].shape == (5, 7)
        assert np.allclose(s['Nxx'][i], N, rtol=1e-8)
        assert np.allclose(s['Nyy'][i], 0., atol=1e-8*N)
        assert np.allclose(s['Nxy'][i], 0., atol=1e-8*N)
        assert np.allclose(e['exx'][i], exx, rtol=1e-8)
        assert np.allclose(e['eyy'][i], -nu*exx, rtol=1e-8)
        assert np.allclose(e['kxx'][i], 0., atol=1e-8*exx)


def test_strain_single_panel_at_gauss_points(tension):
    md, c = tension
    p2 = md.panels[1]
    e = md.strain(c, None, eval_panel=p2, nr_x_gauss=4, nr_y_gauss=3)
    assert len(e['exx']) == 1
    assert e['exx'][0].shape == (3, 4)
    x = np.sort(np.unique(e['x'][0]))
    assert np.allclose(x, p2.a/2*(np.polynomial.legendre.leggauss(4)[0] + 1))
    assert np.allclose(e['exx'][0], N/(E*t), rtol=1e-8)


@pytest.mark.parametrize('vec', ['w', 'exx', 'Nxx'])
def test_calc_results(tension, vec):
    md, c = tension
    res = md.calc_results(c, 'g', vec=vec, gridx=6, gridy=4)
    assert len(res[vec]) == 2
    assert res[vec][0].shape == (4, 6)


def test_calc_results_invalid_input(tension):
    md, c = tension
    with pytest.raises(ValueError, match='not a valid vec'):
        md.calc_results(c, 'g', vec='xyz')
    with pytest.raises(ValueError, match='304'):
        md.calc_results(c, 'g', vec='w', nr_x_gauss=305)
    with pytest.raises(ValueError, match='304'):
        md.calc_results(c, 'g', vec='w', nr_y_gauss=305)


@pytest.mark.parametrize('kwargs, match', [
    (dict(), 'x_cte or y_cte'),
    (dict(x_cte_force=0.3, y_cte_force=0.1), 'single line'),
    (dict(x_cte_force=0.3), 'nr_y_gauss'),
    (dict(y_cte_force=0.1), 'nr_x_gauss'),
    (dict(x_cte_force=0.3, nr_y_gauss=305), '304'),
    (dict(y_cte_force=0.1, nr_x_gauss=305), '304'),
])
def test_force_invalid_input(tension, kwargs, match):
    md, c = tension
    with pytest.raises(ValueError, match=match):
        md.force(c, 'g', eval_panel=md.panels[1], **kwargs)


def test_force_is_independent_of_the_number_of_gauss_points(tension):
    md, c = tension
    p2 = md.panels[1]
    F = [md.force(c, 'g', eval_panel=p2, x_cte_force=0.3, nr_y_gauss=ng,
                  gridx=11)['Fxx'] for ng in (4, 10)]
    assert np.isclose(F[0], F[1])


@pytest.mark.parametrize('panel', [0, 1])
def test_force_equals_applied_load(tension, panel):
    """The section force along x = cte is the applied load N*b"""
    md, c = tension
    p = md.panels[panel]
    res = md.force(c, 'g', eval_panel=p, x_cte_force=p.a/2, nr_y_gauss=4,
                   gridx=11)
    assert np.isclose(res['Fxx'], N*B)
    assert np.isclose(res['Fxy'], 0., atol=1e-8*N*B)
    assert np.isclose(res['Fyy'], 0., atol=1e-8*N*B)


def test_force_along_y_cte(tension):
    """Along y = cte the integral of Nxx is N times the panel length"""
    md, c = tension
    p2 = md.panels[1]
    res = md.force(c, 'g', eval_panel=p2, y_cte_force=B/2, nr_x_gauss=5,
                   gridy=11)
    assert np.isclose(res['Fxx'], N*p2.a)


def test_force_of_group_uses_its_first_panel(tension):
    md, c = tension
    res = md.calc_results(c, 'g', vec='Fxx', eval_panel=None, x_cte_force=0.2,
                          nr_y_gauss=4, gridx=11)
    assert np.isclose(res['Fxx'], N*B)


def test_force_out_plane_is_zero_under_in_plane_load(tension):
    md, c = tension
    p2 = md.panels[1]
    F = md.force_out_plane(c, 'g', eval_panel=p2, nr_y_gauss=6, gridx=11)
    assert abs(F) < 1e-6*N*B


def test_plot_options(tension, tmp_path):
    md, c = tension
    filename = tmp_path / 'md.png'
    ax, data = md.plot(c, 'g', vec='exx', gridx=6, gridy=4, num_levels=5,
                 identify=True, show_boundaries=True, colorbar=True,
                 cbar_title='exx', display_zero=True, invert_y=True,
                 title='title', clean=False, colormap='not_a_colormap',
                 texts=[dict(x=0.1, y=0.1, s='text')], dpi=20,
                 filename=str(filename))
    assert filename.exists()
    assert ax.get_title() == 'title'
    assert np.isclose(data['vecmax'], N/(E*t))
    plt.close('all')


def test_plot_from_results_and_single_panel(tension):
    md, c = tension
    res = md.calc_results(c, 'g', vec='u', gridx=6, gridy=4)
    fig, ax = plt.subplots()
    assert md.plot(res=res, group='g', vec='u', ax=ax, num_levels=5)[0] is ax
    res = md.calc_results(c, None, vec='u', gridx=6, gridy=4,
                          eval_panel=md.panels[0])
    assert md.plot(res=res, vec='u', ax=ax, num_levels=5,
                   eval_panel=md.panels[0], show_boundaries=True,
                   flip_plot=True)[0] is ax
    plt.close(fig)


def test_plot_invalid_input(tension):
    md, c = tension
    with pytest.raises(ValueError, match='"c" or "res"'):
        md.plot(group='g')
    with pytest.raises(ValueError, match='Axes'):
        md.plot(c, 'g', ax='axes', gridx=6, gridy=4)
    plt.close('all')


@pytest.mark.parametrize('method', ['calc_kC', 'calc_kG', 'calc_kM'])
def test_assembly_requires_row_and_col_start(tension, method):
    md, c = tension
    p = md.panels[0]
    row_start = p.row_start
    try:
        p.row_start = None
        with pytest.raises(ValueError, match='row_start'):
            getattr(md, method)(silent=True)
    finally:
        p.row_start = row_start


@pytest.mark.parametrize('method', ['calc_fint', 'calc_fext'])
def test_force_vectors_require_col_start(tension, method):
    md, c = tension
    p = md.panels[0]
    col_start = p.col_start
    try:
        p.col_start = None
        args = (c,) if method == 'calc_fint' else ()
        with pytest.raises(ValueError, match='col_start'):
            getattr(md, method)(*args, silent=True)
    finally:
        p.col_start = col_start


def test_connection_to_itself_is_rejected(tension):
    md, c = tension
    p1 = md.panels[0]
    conn = [dict(p1=p1, p2=p1, func='SSxcte', xcte1=0., xcte2=p1.a,
                 kt=1., kr=1.)]
    with pytest.raises(ValueError, match='two different panels'):
        md.get_kC_conn(conn=conn)


def test_get_kC_conn_without_connections():
    p = Shell(a=1., b=1., r=0., m=4, n=4, stack=[0], plyt=1e-3,
              laminaprop=(E, nu), model='plate_clpt_donnell')
    md = MultiDomain([p], conn_method='penalty')
    with pytest.raises(RuntimeError, match='No connectivity'):
        md.get_kC_conn()


def test_unknown_connection_function(tension):
    md, c = tension
    p1, p2 = md.panels
    conn = [dict(p1=p1, p2=p2, func='XYZ')]
    with pytest.raises(ValueError, match='not recognized'):
        md.get_kC_conn(conn=conn)
