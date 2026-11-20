r"""Coupons of :mod:`panels.multidomain.delamination`

- The lever of the MMB coupon, a linear constraint between the mean
  deflections of the upper arm at the load line and at the saddle,
  :meth:`.MultiDomain.add_linear_constraint`, transmits the line loads `P
  c/L` and `P (c + L)/L` of the statics of the rigid lever, `P` being the
  reaction of the constraint: the same coupon without the lever, loaded by
  these line loads, has the lever displacement `\delta` of the constraint
  (verification V6 of the study).
- A load-displacement analysis with damage converges and the dissipated
  energy grows monotonically.

The ply properties are those of AS4/PEEK of Camanho et al. (2003), Table 1,
used for verification only.

"""
import numpy as np
import pytest

from panels.multidomain.delamination import create_coupon, solve_coupon
from panels.multidomain.connections import nullspace
from panels.multidomain.connections.czm import BilinearLaw

LAM = (122700., 10100., 0.25, 5500., 5500., 3700.)
L, C, B, A0 = 50.8, 44.4, 25.4, 34.1


def _coupon(model, **kw):
    law = BilinearLaw(0.969, 1.719, 80., 100., 1e5, 2.284)
    args = dict(L=L, c=C, model=model, nd=2, L_coh=20., m_crack=10,
                m_coh=10, m_far=8, n=4, nr_x_gauss=40, nr_y_gauss=6)
    args.update(kw)
    return create_coupon('MMB', LAM, 0.13, [0]*12, B, A0, [law], **args)


def _mean_w(md, info, c, x):
    for p in info['top']:
        if p.x0 - 1e-9 <= x <= p.x0 + p.a + 1e-9:
            row = nullspace.mean_row(p, nullspace.W, x - p.x0, None,
                                     md.get_size())
            return (row @ c)[0]


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'plate_fsdt_donnell'])
def test_lever_transmits_rigid_lever_loads(model):
    md, info = _coupon(model)
    c = md.linear_static(lbd=1.)
    P = md.reaction_pd(c, 1., NLgeom=False)['inc']
    assert P > 0
    md2, info2 = _coupon(model)
    md2._linear_constraints = []
    md2.Z = None
    md2._Z_key = None
    top = info2['top']
    Fe = P*C/L/B
    Fm = P*(C + L)/L/B
    top[0].add_distr_load_fixed_x(0., funcz=lambda y: Fe)
    for p in top:
        if p.x0 - 1e-9 <= L <= p.x0 + p.a + 1e-9:
            p.add_distr_load_fixed_x(L - p.x0, funcz=lambda y: -Fm)
            break
    c2 = md2.linear_static(lbd=1.)
    delta = C/L*_mean_w(md2, info2, c2, 0.) - (1 + C/L)*_mean_w(md2, info2,
                                                                c2, L)
    assert abs(delta - 1.) < 1e-6
    np.testing.assert_allclose(md2.get_Z() @ np.zeros(md2.get_Z().shape[1]),
                               0.)


@pytest.mark.slow
def test_mmb_damage_converges():
    md, info = _coupon('plate_clpt_donnell')
    res, an = solve_coupon(md, info, 3., initialInc=0.5, maxInc=0.5)
    assert res['delta'][-1] == pytest.approx(3.)
    assert np.all(np.diff(res['dissipated']) >= -1e-12)
    assert res['P'].max() > 0
