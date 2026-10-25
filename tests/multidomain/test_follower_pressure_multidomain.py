r"""Follower pressure loads on the panels of a :class:`.MultiDomain`

The assembly forwards the load factor ``inc`` to :meth:`.Shell.calc_fint`,
the configuration ``c`` to :meth:`.Shell.calc_fext`, and adds the
unsymmetric load stiffness :meth:`.Shell.calc_kCfollower` after the
symmetrization of the assembled matrices, see :meth:`.MultiDomain.calc_kC`.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from structsolve import Analysis

from panels.shell import Shell
from panels.multidomain import MultiDomain

LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)


def panel(model, a, b, m, n):
    kw = dict(model=model, stack=[0, 45, -45, 90], plyt=0.2e-3,
              laminaprop=LAMINAPROP, x0=0, y0=0)
    if 'cylshell' in model:
        kw['r'] = 0.8
    s = Shell(a=a, b=b, m=m, n=n, **kw)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for field in 'uvw':
            setattr(s, edge + field, 1.)
    s.x1u = s.y1v = 0.
    s.x1w = s.x2w = s.y1w = 0.
    s._rebuild()
    return s


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_fsdt_sanders'])
def test_single_panel_assembly_equals_shell(model):
    s = panel(model, 0.4, 0.3, 6, 6)
    s.add_pressure_load(-3.e3, follower='quadratic', cte=False)
    md = MultiDomain([s], [], conn_method='penalty')
    rng = np.random.default_rng(0)
    c = 1.e-3*rng.standard_normal(md.get_size())
    inc = 0.6
    assert np.allclose(md.calc_fint(c, inc=inc), s.calc_fint(c, inc=inc),
                       rtol=1e-13, atol=0)
    assert np.allclose(md.calc_fext(inc=inc, c=c), s.calc_fext(inc=inc, c=c),
                       rtol=1e-13, atol=0)
    KT = md.calc_kT(c=c, inc=inc).toarray()
    assert np.allclose(KT, s.calc_kT(c=c, inc=inc).toarray(), rtol=1e-12,
                       atol=1e-12*np.abs(KT).max())
    kC = md.calc_kC(c=c, NLgeom=True, inc=inc).toarray()
    assert np.allclose(kC, s.calc_kC(c=c, NLgeom=True, inc=inc).toarray(),
                       rtol=1e-12, atol=1e-12*np.abs(kC).max())
    # Newton-Raphson through the reduced functions, as for any assembly
    an = Analysis(*md.get_reduced_functions())
    an.static(NLgeom=True, silent=True)
    an_s = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    an_s.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    assert np.allclose(md.expand(an.cs[-1]), an_s.cs[-1], rtol=1e-8,
                       atol=1e-10*np.abs(an_s.cs[-1]).max())


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_clpt_sanders'])
def test_two_panels_tangent_is_jacobian(model):
    r"""Two panels connected along ``x = const`` with the null-space method,
    with follower pressures on both: the reduced tangent stiffness matrix
    is the Jacobian of the reduced residual"""
    p1 = panel(model, 0.4, 0.3, 5, 5)
    p2 = panel(model, 0.5, 0.3, 5, 5)
    p2.x1u = 1.
    p2._rebuild()
    p1.add_pressure_load(-3.e4, follower='quadratic', cte=False)
    p2.add_pressure_load(lambda x, y: 2.e4*(1 + x), x1=0.1, follower=True,
                         cte=False)
    md = MultiDomain([p1, p2], [dict(p1=p1, p2=p2, func='SSxcte',
                                     xcte1=p1.a, xcte2=0.)],
                     conn_method='null-space')
    calc_fext, calc_fint, calc_kC, calc_kG = md.get_reduced_functions()
    rng = np.random.default_rng(3)
    Z = md.get_Z()
    c = 1.e-3*rng.standard_normal(Z.shape[1])
    inc = 0.8
    K = (calc_kC(c=c, NLgeom=True, inc=inc)
         + calc_kG(c=c, NLgeom=True)).toarray()
    N = c.shape[0]
    J = np.empty((N, N))
    step = 1.e-9
    for j in range(N):
        e = np.zeros(N)
        e[j] = step
        J[:, j] = (calc_fint(c + e, inc=inc) - calc_fint(c - e, inc=inc))/(2*step)
    assert np.linalg.norm(K - J) <= 1.e-6*np.linalg.norm(J)
    # unsymmetric, the patch and the varying pressure of p2
    assert np.linalg.norm(K - K.T) > 1.e-8*np.linalg.norm(K)
    # the follower part is not lost in the assembly
    Kf = np.asarray((Z.T @ md._calc_kCfollower(Z @ c, inc, True) @ Z).todense())
    assert np.linalg.norm(K - J) <= 1.e-3*np.linalg.norm(Kf)


def test_closed_cylinder_load_stiffness_symmetric():
    r"""A uniform follower pressure on a closed cylinder with simply
    supported ends: the load stiffness of each panel is unsymmetric, the
    contour integrals along the connected edges cancel only where the
    displacements of both sides are equal, i.e. in the matrix reduced by the
    null-space method, but not in the matrix of the penalty method"""
    from panels.multidomain import create_cylinder
    for conn_method in ('null-space', 'penalty'):
        md, conns = create_cylinder(height=0.5, r=0.3, stack=[0, 45, -45, 90],
                                    plyt=0.125e-3, laminaprop=LAMINAPROP,
                                    npanels=4, m=6, n=6,
                                    conn_method=conn_method)
        size = md.get_size()
        K = 0
        for p in md.panels:
            p.add_pressure_load(-1.e4, follower=True)
            k = p.calc_kCfollower().toarray()
            assert np.linalg.norm(k - k.T) > 0.1*np.linalg.norm(k)
            K = K + p.calc_kCfollower(size=size, row0=p.row_start,
                                      col0=p.col_start)
        if conn_method == 'null-space':
            Z = md.get_Z(conns)
            K = (Z.T @ K @ Z).toarray()
            assert np.linalg.norm(K - K.T) < 1e-12*np.linalg.norm(K)
        else:
            K = K.toarray()
            assert np.linalg.norm(K - K.T) > 0.1*np.linalg.norm(K)
