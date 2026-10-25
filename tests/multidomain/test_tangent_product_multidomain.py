r"""Product of the tangent stiffness matrix of a :class:`.MultiDomain` by a
vector, :meth:`.MultiDomain.calc_kTv`, without assembly

It must be the product of the assembled `[K_T]`, ``calc_kC + calc_kG`` with
``NLgeom=True``, including the penalty connections and the follower loads,
and the derivative of :meth:`.MultiDomain.calc_fint`. With prescribed
displacements imposed exactly, the reduced ``calc_fext(c, lbd)`` uses it for
`\partial R_r/\partial \lambda = [Z]^T (\{F_{ext}\} - [K_T] \{c_{p1}\})`.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest

from panels.multidomain import MultiDomain
from test_follower_pressure_multidomain import panel

MODELS = ['plate_clpt_donnell', 'cylshell_clpt_donnell',
          'cylshell_clpt_sanders', 'plate_fsdt_donnell', 'plate_tsdt_donnell',
          'cylshell_fsdt_donnell', 'cylshell_fsdt_sanders',
          'cylshell_tsdt_donnell', 'cylshell_tsdt_sanders']


def rel(a, b):
    return np.linalg.norm(a - b)/np.linalg.norm(b)


def two_panels(model, conn_method, pd_method='penalty'):
    p1 = panel(model, 0.4, 0.3, 5, 5)
    p2 = panel(model, 0.5, 0.3, 5, 5)
    p2.x1u = 1.
    p2._rebuild()
    p1.add_pressure_load(-3.e4, follower='quadratic', cte=False)
    p2.add_pressure_load(lambda x, y: 2.e4*(1 + x), x1=0.1, follower=True,
                         cte=True)
    conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0.)]
    return MultiDomain([p1, p2], conn, conn_method=conn_method,
                       pd_method=pd_method)


def state(md, rng):
    return 1.e-3*rng.standard_normal(md.get_size())


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_product_equals_assembled_tangent(model, conn_method):
    md = two_panels(model, conn_method)
    rng = np.random.default_rng(1)
    c = state(md, rng)
    v = state(md, rng)
    inc = 0.7
    KT = (md.calc_kC(c=c, NLgeom=True, inc=inc)
          + md.calc_kG(c=c, NLgeom=True))
    assert rel(md.calc_kTv(c, v, inc=inc), KT @ v) < 1.e-11
    if conn_method == 'penalty':
        # the connections are in the product
        assert md.get_kC_conn().nnz > 0


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_tsdt_sanders'])
def test_product_is_derivative_of_fint(model):
    md = two_panels(model, 'penalty')
    rng = np.random.default_rng(2)
    c = state(md, rng)
    d = state(md, rng)
    inc = 0.7
    f0 = md.calc_fint(c, inc=inc)
    KTd = md.calc_kTv(c, d, inc=inc)
    steps = [1.e-1, 1.e-2, 1.e-3, 1.e-4]
    errors = [np.linalg.norm(md.calc_fint(c + h*d, inc=inc) - f0 - h*KTd)
              /np.linalg.norm(h*KTd) for h in steps]
    for prev, err in zip(errors[:-1], errors[1:]):
        assert err < 0.2*prev, errors


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_fsdt_sanders'])
def test_reduced_load_vector_with_prescribed_displacements(model):
    r"""`\partial R_r/\partial \lambda` from the product equals the one of
    the assembled tangent stiffness matrix"""
    md = two_panels(model, 'null-space', pd_method='null-space')
    p1, p2 = md.panels
    p1.add_distr_pd_fixed_x(0., ku=1., funcu=lambda y: 0.)
    p2.add_distr_pd_fixed_x(p2.a, ku=1., funcu=lambda y: -1.e-3*(1 + y),
                            cte=False)
    calc_fext = md.get_reduced_functions()[0]
    Z = md.get_Z()
    c_p0, c_p1 = md.get_cp()
    assert np.any(c_p1)
    rng = np.random.default_rng(5)
    c_r = 1.e-3*rng.standard_normal(Z.shape[1])
    lbd = 0.6
    c = Z @ c_r + c_p0 + lbd*c_p1
    KT = md.calc_kC(c=c, NLgeom=True, inc=lbd) + md.calc_kG(c=c, NLgeom=True)
    ref = Z.T @ (md.calc_fext(inc=1., c=c) - KT @ c_p1)
    assert rel(calc_fext(inc=1., c=c_r, lbd=lbd), ref) < 1.e-11
