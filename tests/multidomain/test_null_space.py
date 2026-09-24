r"""Connections imposed exactly with the null-space method

With ``MultiDomain(..., conn_method='null-space')`` the connections are the
constraints `[B] \{c\} = \{0\}`, eliminated with a basis `[T]` of the null
space of `[B]`, see :mod:`panels.multidomain.connections.nullspace`:

- the constraints are those of the penalty kernels: for every connection,
  model and random boundary condition flags, the vectors `[T] \{c_r\}` have
  zero penalty energy, and they are all the vectors with zero penalty energy
- the solution is the limit of the penalty method for penalty constants
  going to infinity
- two laminates connected by ``'SB'`` are exactly the single laminate with
  both stacking sequences, also for the non-linear analyses, and a plate
  divided in domains gives the single-domain plate up to the convergence of
  the approximation, without any penalty constant

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.linalg import eigh
from structsolve import Analysis, solve

from panels.shell import Shell
from panels.multidomain import (MultiDomain, cylinder_compression_lb_Nxx_cte,
        cylinder_blade_stiffened_compression_lb_Nxx_from_static)
from panels.multidomain.connections import nullspace
from panels.multidomain.tstiff2d_1stiff_freq import tstiff2d_1stiff_freq

from test_bf_sdt import _flags, _conn, LAMINAPROP
from test_multidomain_fsdt_tsdt import (CLPT, FSDT, TSDT, NOOR_1973, RHO, E2,
        PAGANO, orthotropic, domains, frequencies, make_plate, sb_plates,
        sinusoidal_load)


MODELS = [CLPT, 'cylshell_clpt_sanders', FSDT, TSDT, 'cylshell_fsdt_sanders']
FUNCS = ['SSxcte', 'SSycte', 'BFycte', 'BFxcte', 'SB']


def _pair(model, func, seed):
    r"""Two panels with random boundary condition flags and a different
    number of terms, connected by ``func``"""
    kw = dict(x0=0, y0=0, plyt=0.2e-3, laminaprop=LAMINAPROP)
    kw1 = dict(kw, model=model, stack=[0, 45, -45, 90])
    kw1.update(dict(r=0.8) if 'cylshell' in model else {})
    kw2 = dict(kw1, stack=[0, 90, 90, 0])
    if func == 'SSxcte':
        p1 = Shell(a=0.4, b=0.3, m=6, n=7, **kw1)
        p2 = Shell(a=0.5, b=0.3, m=7, n=7, **kw2)
        conn = dict(p1=p1, p2=p2, func=func, xcte1=p1.a, xcte2=0.)
    elif func == 'SSycte':
        p1 = Shell(a=0.4, b=0.3, m=6, n=7, **kw1)
        p2 = Shell(a=0.4, b=0.2, m=6, n=5, **kw2)
        conn = dict(p1=p1, p2=p2, func=func, ycte1=p1.b, ycte2=0.)
    elif func == 'SB':
        p1 = Shell(a=0.4, b=0.3, m=6, n=6, **kw1)
        p2 = Shell(a=0.4, b=0.3, m=5, n=7, **kw2)
        conn = dict(p1=p1, p2=p2, func=func, kr=1.)
    else:
        #NOTE a flat flange, with the base at an interior line
        kw2 = dict(kw, stack=[0, 90, 90, 0],
                   model=model.replace('cylshell', 'plate').replace('sanders', 'donnell'))
        if func == 'BFycte':
            p1 = Shell(a=0.6, b=0.3, m=6, n=7, **kw1)
            p2 = Shell(a=0.6, b=0.05, m=6, n=5, **kw2)
            cte1 = 0.4*p1.b
        else:
            p1 = Shell(a=0.3, b=0.6, m=7, n=6, **kw1)
            p2 = Shell(a=0.05, b=0.6, m=5, n=6, **kw2)
            cte1 = 0.4*p1.a
        conn = _conn(func, p1, p2, cte1, 0.)
    _flags(p1, seed)
    _flags(p2, seed + 1)
    return [p1, p2], conn


@pytest.mark.parametrize('order', ['p1 first', 'p2 first'])
@pytest.mark.parametrize('func', FUNCS)
@pytest.mark.parametrize('model', MODELS)
def test_same_constraints_as_penalty(model, func, order):
    r"""The null space of the penalty matrix is the span of `[T]`

    The vectors `[T] \{c_r\}` have zero penalty energy, `[K_{pen}] [T] = 0`,
    and the penalty matrix has as many zero eigenvalues as `[T]` has
    columns, with the boundary condition flags removing different terms of
    the two panels.
    """
    for seed in (1, 7):
        panels, conn = _pair(model, func, seed)
        if order == 'p2 first':
            panels = panels[::-1]
        md = MultiDomain(panels, [conn], conn_method='null-space')
        T = md.get_T()
        assert T.shape[0] == md.get_size()
        K = MultiDomain(panels, [dict(conn, kt=1., kr=1.)],
                        conn_method='penalty').get_kC_conn()
        K = K.toarray()
        assert np.abs(K @ T).max() < 1e-12*np.abs(K).max()
        ev = eigh(K, eigvals_only=True)
        #NOTE the zero eigenvalues are at the round-off level, 1e-16, and
        #     the smallest penalized mode has 1e-13 in these cases
        assert np.sum(ev < 1e-14*ev.max()) == T.shape[1]


def _divided_plate_matrices(model, layout, conn_method):
    a = 1.
    h = a/5
    panels, conn = domains(model, a, h, [0, 90, 0, 90], orthotropic(10),
                           layout, factor=1.)
    md = MultiDomain(panels, conn, conn_method=conn_method)
    return md, md.calc_kC(), md.calc_kM(silent=True)


def test_penalty_mode_is_unchanged():
    r"""With ``conn_method='penalty'`` `[T]` is the identity and the
    connections are in the stiffness matrix"""
    md, K, M = _divided_plate_matrices(FSDT, '2x2', 'penalty')
    T = md.get_T()
    assert T.shape == (md.get_size(),)*2
    assert np.array_equal(T.toarray(), np.eye(md.get_size()))
    assert md.get_kC_conn().nnz > 0


def test_connection_matrix_excludes_null_space_connections():
    md, K, M = _divided_plate_matrices(FSDT, '2x2', 'null-space')
    assert md.get_kC_conn().nnz == 0
    # the 4 domains of 5*10*10 terms, minus the independent constraints
    T = md.get_T()
    assert T.shape[0] == 2000
    assert T.shape[1] < 2000
    for connecti in md.conn:
        B = nullspace.constraint_matrix(connecti, md.get_size())
        assert np.abs(B @ T).max() < 1e-12*np.abs(B).max()


@pytest.mark.parametrize('layout', ['x', 'y', '2x2'])
@pytest.mark.parametrize('model', [FSDT, TSDT])
def test_frequencies_divided_plate_noor(model, layout):
    r"""Noor (1973), the divided plate gives the single-domain plate without
    any penalty constant, see ``test_multidomain_fsdt_tsdt.py``, where the
    penalty constants must be raised 1000 times to reach ``rtol=1e-4``"""
    md, K, M = _divided_plate_matrices(model, layout, 'null-space')
    omegas = frequencies(md.reduce(K), md.reduce(M), k=8)
    a = 1.
    h = a/5
    scale = a**2/h*np.sqrt(RHO/E2)
    assert np.isclose(omegas[0]*scale, NOOR_1973[model], rtol=1e-4)
    single = make_plate(model, a, a, h, [0, 90, 0, 90], orthotropic(10),
                        m=14, n=14)
    ref = frequencies(single.calc_kC(), single.calc_kM(), k=8)
    assert np.allclose(omegas, ref, rtol=1e-7)


@pytest.mark.parametrize('model', [FSDT, TSDT])
def test_bending_divided_plate_pagano(model):
    r"""Pagano (1970), deflection at the interior corner of four domains,
    solved with the reduced system and expanded back"""
    a = 1.
    h = a/10
    q0 = 1.e3
    panels, conn = domains(model, a, h, [0, 90, 90, 0],
                           orthotropic(25, G12=0.5, G13=0.5, G23=0.2), '2x2',
                           xs=0.5, ys=0.5)
    md = MultiDomain(panels, conn, conn_method='null-space')
    fext = sum(sinusoidal_load(p, q0, a, md.get_size()) for p in panels)
    c_r = solve(md.reduce(md.calc_kC()), md.reduce(fext), silent=True)
    c = md.expand(c_r)
    assert c.shape == (md.get_size(),)
    for p in panels:
        res = md.uvw(c, group=None, eval_panel=p, gridx=2, gridy=2)
        w = res['w'][0][int(p.y0 == 0), int(p.x0 == 0)]
        wbar = 100*w*E2*h**3/(q0*a**4)
        assert np.isclose(wbar, PAGANO[model], rtol=2e-4)


@pytest.mark.parametrize('model', [CLPT, FSDT])
def test_skin_base_equals_single_laminate(model):
    r"""The laminates `[0/90]` over `[0/90]` with ``'SB'`` are exactly the
    laminate `[0/90]_2`, the penalty method with 1e5 times the default
    penalty constant gives an error of 1e-5"""
    md, single = sb_plates(model, 5, [0, 90], [0, 90], factor=1.,
                           kr_factor=1.)
    md.conn_method = 'null-space'
    omegas = frequencies(md.reduce(md.calc_kC()),
                         md.reduce(md.calc_kM(silent=True)), k=8)
    ref = frequencies(single.calc_kC(), single.calc_kM(), k=8)
    assert np.allclose(omegas, ref, rtol=1e-10)


def test_skin_base_nonlinear_equals_single_laminate():
    r"""Large deflection of the laminates `[0/90]` over `[0/90]`, with the
    von Karman kinematics, with :meth:`.MultiDomain.get_reduced_functions`

    The in-plane displacements of the edges are only restrained in the
    tangential direction: restraining the normal one in both laminates would
    also restrain the rotation of the edge, since `u_{top} = u_{bot} - d_{sb}
    w_{,x}`.
    """
    a = 1.
    h = a/50
    P = 2.e5

    def plate(stack, h):
        s = make_plate(CLPT, a, a, h, stack, orthotropic(10), m=8, n=8)
        s.nx = s.ny = 16
        return s

    single = plate([0, 90, 0, 90], h)
    single.add_point_load(a/2, a/2, 0, 0, -P, cte=False)
    an = Analysis(single.calc_fext, single.calc_fint, single.calc_kC,
                  single.calc_kG)
    an.static(NLgeom=True, silent=True)
    w_ref = single.uvw(c=an.cs[-1], gridx=3, gridy=3)[1]['w'][1, 1]
    c_lin = solve(single.calc_kC(), single.calc_fext(), silent=True)
    w_lin = single.uvw(c=c_lin, gridx=3, gridy=3)[1]['w'][1, 1]
    # strongly non-linear
    assert abs(w_lin) > 5*abs(w_ref)

    top = plate([0, 90], h/2)
    bot = plate([0, 90], h/2)
    top.add_point_load(a/2, a/2, 0, 0, -P, cte=False)
    md = MultiDomain([top, bot], [dict(p1=top, p2=bot, func='SB')],
                     conn_method='null-space')
    an = Analysis(*md.get_reduced_functions())
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    c = md.expand(an.cs[-1])
    w = md.uvw(c, group=None, eval_panel=top, gridx=3, gridy=3)['w'][0][1, 1]
    assert np.isclose(w, w_ref, rtol=1e-10)


def test_limit_of_the_penalty_method():
    r"""The penalty method converges to the null-space method as the penalty
    constants increase, a plate divided in 2x2 domains with the CLPT

    The penalty method is more flexible and its error decreases with the
    inverse of the penalty constants, from 1.3% with the default ones.
    """
    a = 1.
    h = a/50
    panels, conn = domains(CLPT, a, h, [0, 90, 0, 90], orthotropic(10),
                           '2x2', factor=1.)
    md = MultiDomain(panels, conn, conn_method='null-space')
    ref = frequencies(md.reduce(md.calc_kC()),
                      md.reduce(md.calc_kM(silent=True)), k=4)
    errors = []
    for factor in (1., 10., 100., 1000.):
        panels, conn = domains(CLPT, a, h, [0, 90, 0, 90], orthotropic(10),
                               '2x2', factor=factor)
        md = MultiDomain(panels, conn, conn_method='penalty')
        omegas = frequencies(md.calc_kC(), md.calc_kM(silent=True), k=4)
        assert np.all(omegas < ref)
        errors.append(np.abs(omegas/ref - 1).max())
    assert np.isclose(errors[0], 0.013, rtol=0.05)
    for e1, e2 in zip(errors[:-1], errors[1:]):
        assert 9 < e1/e2 < 11


def test_cylinder_redundant_constraints():
    r"""Closed cylinder, the constraints at the ends of the edges, where the
    boundary conditions already remove the terms, are redundant

    The value of the penalty method is -47055.984, see ``test_cylinder.py``.
    """
    Nxxs = [-100.]*5
    md, eigvals, eigvecs = cylinder_compression_lb_Nxx_cte(height=0.5,
        r=0.25, plyt=0.125e-3, stack=[0, 45, -45, 90, -45, 45],
        laminaprop=LAMINAPROP, npanels=5, Nxxs=Nxxs, m=8, n=12,
        num_eigvalues=10, conn_method='null-space')
    assert eigvecs.shape[0] == md.get_size()
    assert md.T.shape == (1440, 1315)
    assert np.isclose(Nxxs[0]*eigvals[0], -47113.039, rtol=1e-5)


def test_cylinder_blade_stiffened_from_static():
    r"""The default penalty constants give -39993.838, 4% below the
    null-space method, see ``test_cylinder_blade_stiffened.py``, and 1000
    times higher constants give -41694.868"""
    Nxxs = [-100.]*5
    md, c, eigvals, eigvecs = \
        cylinder_blade_stiffened_compression_lb_Nxx_from_static(height=0.5,
        r=0.25, plyt=0.125e-3, stack=[0, 45, -45, 90, -45, 45],
        stack_blades=[[0, 90, 0]*4]*5, width_blades=[0.02]*5,
        laminaprop=LAMINAPROP, npanels=5, Nxxs_skin=Nxxs, Nxxs_blade=Nxxs,
        m=8, n=12, num_eigvalues=10, conn_method='null-space')
    assert c.shape == (md.get_size(),)
    assert np.isclose(Nxxs[0]*eigvals[0], -41696.682, rtol=1e-5)


def test_tstiff2d_freq():
    r"""T-stiffened panel with ``'SSxcte'``, ``'SSycte'``, ``'SB'`` and
    ``'BFycte'``, whose base and skin have a different number of terms, the
    penalty method gives 48.2733 with the default penalty constants and
    48.3858 with 100 times higher constants, see
    ``test_tstiff2d_assembly.py``"""
    assy, eigvals, eigvecs = tstiff2d_1stiff_freq(b=1., bb=0.2, bf=0.1, a=3.,
        ys=0.5, defect_a=0.1, rho=1.3e3, plyt=0.125e-3, laminaprop=LAMINAPROP,
        stack_skin=[0, 45, -45, 90, -45, 45, 0], stack_base=[0, 90, 0]*4,
        stack_flange=[0, 90, 0]*8, m=6, n=7, mb=5, nb=6, mf=6, nf=7,
        num_eigvalues=10, conn_method='null-space')
    assert np.isclose((-eigvals[0])**0.5, 48.3875, rtol=1e-5)


def test_T_is_reused_and_rebuilt():
    r"""`[T]` is rebuilt when a boundary condition flag changes

    The terms removed by the flags are zero columns of the matrices, and
    remain independent Ritz constants. Releasing `w` at the edge `y = 0` of
    the first domain adds a constraint at its corner with the connected
    edge.
    """
    md, K, M = _divided_plate_matrices(FSDT, 'x', 'null-space')
    T = md.get_T()
    assert md.get_T() is T
    md.panels[0].y1w = 1.
    T2 = md.get_T()
    assert T2 is not T
    assert T2.shape[1] == T.shape[1] - 1
    B = nullspace.constraint_matrix(md.conn[0], md.get_size())
    assert np.abs(B @ T2).max() < 1e-12*np.abs(B).max()
    assert np.abs(B @ T).max() > 1e-3*np.abs(B).max()


def test_default_is_null_space():
    r"""The connections are imposed exactly by default, the connection
    matrix is then empty"""
    panels, conn = domains(FSDT, 1., 0.2, [0, 90, 0, 90], orthotropic(10),
                           '2x2', factor=1.)
    md = MultiDomain(panels, conn)
    assert md.conn_method == 'null-space'
    assert md.get_kC_conn().nnz == 0
    assert md.get_T().shape[1] < md.get_size()


def test_invalid_conn_method():
    with pytest.raises(ValueError, match='conn_method'):
        MultiDomain([], conn_method='lagrange')


def test_unsupported_connection():
    p1 = make_plate(FSDT, 0.5, 0.5, 0.01, [0, 90], orthotropic(10), m=4, n=4)
    p2 = make_plate(FSDT, 0.5, 0.5, 0.01, [0, 90], orthotropic(10), m=4, n=4)
    with pytest.raises(NotImplementedError, match='SB_TSL'):
        nullspace.constraint_matrix(dict(p1=p1, p2=p2, func='SB_TSL'), 32)
