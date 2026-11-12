r"""Wing-box of :mod:`panels.multidomain.wingbox`

The defaults are the case High-AR-t5 of Elalfy, De Breuker and Castro [1]_.

- ``test_create_wingbox``: the domains, connections and models created by
  :func:`.create_wingbox`, and the errors of ``models``;
- ``test_wingbox_tip_deflection``: linear static tip deflection of two bays,
  with the penalty and the null-space connections, against the
  Euler-Bernoulli beam with the box section between the mid-surfaces;
- ``test_create_wingbox`` also checks the default ``conn_method``,
  ``'null-space'``;
- ``test_wingbox_tip_force_lb_default``: linear buckling with the default
  terms and the default null-space connections, against a regression value
  (the model is not converged);
- ``test_wingbox_models``: the spars of the model ``'plate_clpt_gl'``,
  the default, against ``'plate_clpt_donnell'``, selected with
  ``models``, with 2 stringers per skin, the default terms and penalty
  connections, and all the groups ``'plate_clpt_gl'`` against all
  ``'plate_clpt_tl'`` without stringers, against regression values.

The spars are ``'plate_clpt_gl'`` by default: the geometric stiffness of
their Green-Lagrange strains contains the work of the bending stresses of
the box on the vertical deflection of the spars, which the von Kármán
strains omit. Without stringers the first modes are local modes of the
upper skin and the model of the spars hardly changes the loads: with the
default terms and penalty connections 172.7826 N with these spars against
172.7828 N, with skins of 12 x 12 terms, spars of 12 x 7 and ribs of 6 x 8
and the null-space connections 176.4351 N against 176.4354 N.

Notes
-----
In a spar web the bending stress resultant `N_{xx}` of the box varies
linearly over the height with a zero resultant, so that the term
`\frac{1}{2} N_{xx} v_{,x}^2` of ``'plate_clpt_gl'`` does no work on a
vertical deflection `v` uniform over the height. The difference between the
two spar models is therefore small, 1.0e-5 (relative) with 2 stringers.

References
----------
.. [1] Elalfy, M. H., De Breuker, R. and Castro, S. G. P., "Beam-like models
   for the post-buckling behaviour of wingbox structures", preprint, Delft
   University of Technology.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest

from panels.multidomain import create_wingbox, wingbox_tip_force_lb


def test_create_wingbox():
    # 2 bays instead of 9, the number of domains being per bay
    md, info = create_wingbox(nbays=2)
    assert md.conn_method == 'null-space'
    assert md.conn is info['conn']
    # per bay 2 skins, 2 spars and a rib, the last one at the tip
    assert len(md.panels) == 2*5
    assert np.isclose(info['a'], 1.010/2)
    # each rib joined to the two spars with 'BFxycte'
    funcs = [conn['func'] for conn in info['conn']]
    assert funcs.count('BFxycte') == 2*2
    md, info = create_wingbox(nbays=2, rib_spar=False)
    assert 'BFxycte' not in [conn['func'] for conn in info['conn']]
    md, info = create_wingbox(nstr=2, nbays=2)
    # per bay 3 strips per skin, 2 spars, 2 stringers per skin, 3 rib strips
    assert len(md.panels) == 2*(2*3 + 2 + 2*2 + 3)
    assert np.isclose(info['ws'], 50.5e-3/3)
    md, info = create_wingbox(nstr=2, nbays=2, stringers=False, ribs=False)
    assert len(md.panels) == 2*(2*3 + 2)
    # the models of the domains, by default and from models
    defaults = dict(skin='plate_clpt_donnell', spar='plate_clpt_gl',
                    stringer='plate_clpt_gl', rib='plate_clpt_donnell')
    groups = dict(skin=('top', 'bot'), spar=('spars', ), stringer=('st', ),
                  rib=('ribs', ))
    for models, expected in ((None, defaults),
            (dict(skin='plate_clpt_tl', rib='plate_clpt_gl'),
             dict(defaults, skin='plate_clpt_tl', rib='plate_clpt_gl'))):
        info_m = create_wingbox(nstr=2, nbays=2, models=models)[1]
        assert info_m['models'] == expected
        bay = info_m['bays'][0]
        for group, keys in groups.items():
            assert all(p.model == expected[group]
                       for k in keys for p in bay[k])
    # unknown keys and models outside the plates of the classical
    # laminated plate theory
    for models in (dict(spars='plate_clpt_gl'),
                   dict(spar='plate_fsdt_donnell'),
                   dict(skin='cylshell_clpt_sanders')):
        with pytest.raises(ValueError):
            create_wingbox(nbays=2, models=models)
    # the clamped root
    for k in ('top', 'bot', 'spars'):
        for p in info['bays'][0][k]:
            assert p.x1u == p.x1v == p.x1w == p.x1wr == 0.
        for p in info['bays'][1][k]:
            assert p.x1u == p.x1v == p.x1w == p.x1wr == 1.


@pytest.mark.parametrize('conn_method', ['penalty', 'null-space'])
def test_wingbox_tip_deflection(conn_method):
    # linear static, 2 bays of the same span, against the Euler-Bernoulli
    # beam with the box section between the mid-surfaces
    L, W, H, ts, tsp, E = 1.010, 50.5e-3, 12.12e-3, 1.e-3, 5.e-3, 73.e9
    P = 1.
    md, info = create_wingbox(nbays=2, P=P, conn_method=conn_method)
    c = md.linear_static(silent=True)
    w = np.mean([md.uvw(c, None, gridx=3, gridy=3, eval_panel=p)['w'][0][:, -1]
                 for p in info['bays'][-1]['top']])
    I = 2*W*ts*(H/2)**2 + 2*tsp*H**3/12 + 2*W*ts**3/12
    w_beam = P*L**3/(3*E*I)
    # 0.893 mm, 1.5% below the beam
    assert abs(w/w_beam - 1) < 0.03, w/w_beam


def test_wingbox_tip_force_lb_default():
    # case High-AR-t5 without stringers, 9 bays, 45 domains, the default
    # terms and null-space connections, 6264 Ritz constants, 1 N at the tip;
    # regression value only, the model being not converged (10 % above
    # Abaqus), see test_wingbox_tip_force_lb
    md, c, eigvals, eigvecs = wingbox_tip_force_lb(num_eigvalues=4)
    print('first buckling tip force: %.4f N' % eigvals[0])
    assert md.conn_method == 'null-space'
    assert np.isclose(eigvals[0], 190.405, rtol=1e-4)
    # the mode is a local buckling of the upper skin of the root bay, in
    # compression
    root_top = md.panels[0]
    assert root_top.group == 'skin_top' and root_top.x0 == 0.
    v = abs(eigvecs[:, 0])
    assert v.argmax() in range(root_top.col_start, root_top.col_end)


def test_wingbox_models():
    # 2 stringers per skin, 9 bays, the default terms and penalty
    # connections, 12096 Ritz constants, 1 N at the tip; regression values
    # of the first buckling tip force with the two models of the spars
    ev = {}
    for spar_model in ('plate_clpt_gl', 'plate_clpt_donnell'):
        md, c, eigvals, eigvecs = wingbox_tip_force_lb(num_eigvalues=4,
                nstr=2, models=dict(spar=spar_model), conn_method='penalty')
        assert all(p.model == spar_model for p in md.panels
                   if p.group == 'spar')
        ev[spar_model] = eigvals[0]
    assert np.isclose(ev['plate_clpt_gl'], 436.0549, rtol=1e-5)
    assert np.isclose(ev['plate_clpt_donnell'], 436.0593, rtol=1e-5)
    # the 'plate_clpt_gl' spars lower the load by 1.0e-5 (relative)
    assert -2e-5 < ev['plate_clpt_gl']/ev['plate_clpt_donnell'] - 1 < 0
    # without stringers, all the groups 'plate_clpt_gl' or 'plate_clpt_tl':
    # the TL curvatures add the work of the pre-buckling bending moments,
    # -2.7e-6 (relative)
    for model in ('plate_clpt_gl', 'plate_clpt_tl'):
        models = dict(skin=model, spar=model, stringer=model, rib=model)
        md, c, eigvals, eigvecs = wingbox_tip_force_lb(num_eigvalues=4,
                models=models, conn_method='penalty')
        ev[model + ' all'] = eigvals[0]
    assert np.isclose(ev['plate_clpt_gl all'], 172.78252, rtol=1e-6)
    assert np.isclose(ev['plate_clpt_tl all'], 172.78206, rtol=1e-6)
    assert -1e-5 < ev['plate_clpt_tl all']/ev['plate_clpt_gl all'] - 1 < 0
