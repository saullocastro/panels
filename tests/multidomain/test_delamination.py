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
- The DCB (both supports) and the ENF satisfy their prescribed opening or
  lever displacement and Clapeyron's theorem; the symmetric DCB deflects
  antisymmetrically; the rigid lever keeps the load line straight and is
  stiffer than the mean lever; penalty connections and a perfectly bonded
  far domain tend to the default coupon.
- :func:`.crack_length` follows the damage set by hand, and the control of
  the separation at the load line gives the curve of the load control.

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


# Tiny coupons of the checks below, each well under a second in a linear
# analysis
SMALL = dict(nd=1, L_coh=10., m_crack=6, m_coh=6, m_far=6, n=4,
             nr_y_gauss=4)


def _small(kind, **kw):
    law = BilinearLaw(0.969, 1.719, 80., 100., 1e5, 2.284)
    args = dict(SMALL)
    if kind == 'DCB':
        args['x_end'] = 60.
        a0 = 30.
    else:
        args.update(L=L, c=C)
        a0 = A0
    args.update(kw)
    return create_coupon(kind, LAM, 0.13, [0]*12, B, a0, [law], **args)


def _linear(md):
    c = md.linear_static(lbd=1.)
    return c, md.reaction_pd(c, 1., NLgeom=False)['inc']


def _w_mean(md, p, c, x, y=None):
    row = nullspace.mean_row(p, nullspace.W, x, y, md.get_size())
    return (row @ c)[0]


def test_create_coupon_input_errors():
    """Each invalid input raises its error: the kind of coupon, the MMB
    without lever length, the DCB without end, a saddle beyond the end of
    the coupon"""
    with pytest.raises(ValueError, match="kind must be 'DCB', 'ENF' or "
                                         "'MMB'"):
        _small('SLB')
    with pytest.raises(ValueError, match=r'L \(and c for the MMB\) are '
                                         'required'):
        _small('MMB', c=None)
    with pytest.raises(ValueError, match='x_end is required for the DCB'):
        _small('DCB', x_end=None)
    with pytest.raises(ValueError, match='outside the coupon'):
        _small('MMB', x_end=0.9*L)


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'plate_fsdt_donnell'])
@pytest.mark.parametrize('dcb_bc', ['hinge', 'symmetric'])
def test_dcb_clapeyron_and_symmetry(model, dcb_bc):
    r"""DCB in a linear analysis, with the supports ``dcb_bc``: the opening
    at the load line is the prescribed `\delta = 1`; the opening load `P`,
    the reaction of the incremented prescriptions, satisfies Clapeyron's
    theorem `P \delta = c^T K c` (property of a linear analysis loaded by
    prescribed displacements only); with ``dcb_bc='symmetric'`` the
    deflections of the arms are antisymmetric, `w^t(x) = -w^b(x)`
    (property of the symmetric supports). The cohesive domains have the
    default ``4*m_coh`` points along `x`."""
    md, info = _small('DCB', model=model, dcb_bc=dcb_bc)
    c, P = _linear(md)
    assert P > 0
    assert P == pytest.approx(c @ (md.calc_kC() @ c), rel=1e-8)
    top, bot = info['top'][0], info['bot'][0]
    opening = _w_mean(md, top, c, 0.) - _w_mean(md, bot, c, 0.)
    assert opening == pytest.approx(1., rel=1e-10)
    if dcb_bc == 'symmetric':
        for x in (0., 0.5*top.a, top.a):
            assert _w_mean(md, top, c, x) == pytest.approx(
                -_w_mean(md, bot, c, x), abs=1e-9)
    md.cohesive_interfaces()
    assert md._czm[id(info['cohesive'][0])].nr_x_gauss == 4*SMALL['m_coh']


def test_enf_is_mmb_without_lever_arm():
    r"""ENF: the lever length is zero, so the lever constraint prescribes
    the deflection of the upper arm at the saddle, `-\bar{w}^t(L) = \delta
    = 1`, and the load satisfies Clapeyron's theorem `P \delta = c^T K c`
    (property of a linear analysis loaded by prescribed displacements
    only)"""
    md, info = _small('ENF')
    assert info['c'] == 0.
    c, P = _linear(md)
    pm = [p for p in info['top'] if p.x0 - 1e-9 <= L <= p.x0 + p.a + 1e-9][0]
    assert -_w_mean(md, pm, c, L - pm.x0) == pytest.approx(1., rel=1e-10)
    assert P > 0
    assert P == pytest.approx(c @ (md.calc_kC() @ c), rel=1e-8)


def test_rigid_lever_uniform_and_stiffer():
    """MMB, ``lever='rigid'``: the deflection of the upper arm along the
    load line is uniform across the width (property of the constraints, to
    round-off), whereas it varies with ``lever='mean'``; the additional
    constraints cannot decrease the strain energy at the same lever
    displacement, so the lever load is larger (principle of minimum
    potential energy)"""
    loads = {}
    for lever in ('mean', 'rigid'):
        md, info = _small('MMB', lever=lever)
        c, loads[lever] = _linear(md)
        top = info['top'][0]
        ws = np.array([_w_mean(md, top, c, 0., y)
                       for y in np.linspace(0., B, 5)])
        spread = np.ptp(ws)/abs(ws.mean())
        if lever == 'rigid':
            assert spread < 1e-10
        else:
            assert spread > 1e-3
    assert loads['rigid'] > loads['mean']


def test_penalty_connections_and_far_bond_tend_to_exact():
    """MMB, lever load of a linear analysis compared with that of the
    default coupon, null-space connections and cohesive far domain (other
    formulations of the same coupon): with ``conn_method='penalty'`` the
    difference decreases as ``penalty_factor`` grows (penalty method), and
    with ``far='sb'`` the perfect bond of the far domain is slightly stiffer
    than the pristine cohesive bond, which tends to it, see
    test_czm.test_pristine_interface_tends_to_sb"""
    _, P0 = _linear(_small('MMB')[0])
    errs = []
    for factor in (1., 10., 100.):
        _, P = _linear(_small('MMB', conn_method='penalty',
                              penalty_factor=factor)[0])
        errs.append(abs(P - P0)/P0)
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 2e-3
    md, info = _small('MMB', far='sb')
    assert [ci['func'] for ci in md.conn].count('SB') == 1
    _, Psb = _linear(md)
    assert 0 < (Psb - P0)/P0 < 1e-3


def test_crack_length_from_damage():
    """:func:`.crack_length` against the damage set by hand: `a_0` before
    the cohesive interfaces exist; with the first cohesive domain failed
    everywhere and the second one failed across the whole width up to its
    Gauss-Legendre point ``k - 1`` along `x`, the position of that point;
    `a_0` with a threshold above the damage"""
    from panels.multidomain.delamination import crack_length
    md, info = _small('MMB', nd=2, L_coh=20.)
    assert crack_length(md, info) == A0
    md.cohesive_interfaces()
    i1, i2 = [md._czm[id(ci)] for ci in info['cohesive'][:2]]
    assert i1.p_top.x0 + i1.p_top.a == pytest.approx(i2.p_top.x0)
    i1.d[:] = 1.
    nx, ny = i2.nr_x_gauss, i2.nr_y_gauss
    k = 5
    d = np.zeros((ny, nx))
    d[:, :k] = 1.
    d[0, k] = 1.   # failed at one point across the width only
    i2.d[:] = d.ravel()
    a = crack_length(md, info)
    assert a == pytest.approx(i2.p_top.x0 + i2.x[k - 1], rel=1e-14)
    assert crack_length(md, info, threshold=1.5) == A0


def test_solve_coupon_control_error():
    """An unknown ``control`` of :func:`.solve_coupon` raises its error"""
    md, info = _small('DCB')
    with pytest.raises(ValueError, match="control must be 'load' or "
                                         "'load-line'"):
        solve_coupon(md, info, 1., control='arc')


def test_load_line_control_follows_load_control():
    r"""MMB, ``control='load-line'``: the controlled quantity ``psi @ c_r``
    of :func:`.load_line_control_vector` grows at every converged step, the
    analysis stops once `\delta` reaches ``delta_max`` or after
    ``max_steps``, and the lever load at the last `\delta`, with damage, is
    that of ``control='load'`` (other formulation of the same analysis;
    before the peak the damage grows monotonically at every point, so that
    it does not depend on the increments)"""
    from panels.multidomain.delamination import load_line_control_vector
    md, info = _small('MMB', nd=2, L_coh=20., m_coh=8, nr_x_gauss=24)
    psi = load_line_control_vector(md, info)
    q1, q = [], []
    kw = dict(initialInc=0.5, maxInc=0.5, control='load-line',
              modified_NR=False)
    res1, _ = solve_coupon(md, info, 1.2, max_steps=2, **kw,
                           record=lambda an, c, row: q1.append(row))
    assert len(q1) == len(res1['delta']) == 2
    res, _ = solve_coupon(md, info, 1.2, minInc=1e-2, **kw,
                          record=lambda an, c, row: q.append(psi @ an.cs[-1]))
    assert len(q) == len(res['delta']) == 3
    assert q[0] > 0 and np.all(np.diff(q) > 0)
    assert res['delta'][-1] >= 1.2 > res['delta'][-2]
    assert res['max_d'][-1] > 0
    delta = res['delta'][-1]
    ref, _ = solve_coupon(md, info, delta, initialInc=delta/4,
                          maxInc=delta/4)
    assert ref['P'][-1] == pytest.approx(res['P'][-1], rel=1e-6)


def test_linear_constraint_errors():
    """:meth:`.MultiDomain.add_linear_constraint` raises its errors: a panel
    outside the assembly, the prescribed displacements imposed with
    penalties, and a constraint that contradicts the prescriptions imposed
    before it, here the opening of the DCB at the load line, named by its
    label"""
    md, info = _small('DCB')
    top = info['top'][0]
    other = _small('DCB')[1]['top'][0]
    with pytest.raises(ValueError, match='not part of the assembly'):
        md.add_linear_constraint([(other, 'w', 0., None, 0., 1.)])
    md.add_linear_constraint([(top, 'w', 0., None, 0., 1.)], value=2.,
                             label='twice the opening')
    with pytest.raises(ValueError, match="The linear constraint 'twice the "
                       r"opening' \(cte=False\) is not consistent with the "
                       'constraints imposed before it'):
        md.linear_static(lbd=1.)
    from panels.multidomain import MultiDomain
    p1, p2 = info['top'][:2]
    mdp = MultiDomain([p1, p2], [dict(p1=p1, p2=p2, func='SSxcte',
                      xcte1=p1.a, xcte2=0.)], pd_method='penalty')
    mdp.add_linear_constraint([(p1, 'w', 0., None, 0., 1.)], value=1.)
    with pytest.raises(NotImplementedError, match='require '
                       'pd_method="null-space"'):
        mdp.get_Z()
