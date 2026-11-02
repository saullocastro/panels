r"""Wing-box of :mod:`panels.multidomain.wingbox`

Linear buckling under a tip force of the case High-AR-t5 of Elalfy, De
Breuker and Castro, against their Abaqus result, 173.11 N (Table 3, p. 8 of
the preprint).

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest

from panels.multidomain import create_wingbox, wingbox_tip_force_lb


ABAQUS_HIGH_AR_T5 = 173.11  # N, Elalfy et al., Table 3, p. 8


def test_create_wingbox():
    # 2 bays instead of 9, the number of domains being per bay
    md, info = create_wingbox(nbays=2)
    # per bay 2 skins, 2 spars and a rib, the last one at the tip
    assert len(md.panels) == 2*5
    assert np.isclose(info['a'], 1.010/2)
    md, info = create_wingbox(nstr=2, nbays=2)
    # per bay 3 strips per skin, 2 spars, 2 stringers per skin, 3 rib strips
    assert len(md.panels) == 2*(2*3 + 2 + 2*2 + 3)
    assert np.isclose(info['ws'], 50.5e-3/3)
    md, info = create_wingbox(nstr=2, nbays=2, stringers=False, ribs=False)
    assert len(md.panels) == 2*(2*3 + 2)
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


@pytest.mark.slow
def test_wingbox_tip_force_lb():
    # case High-AR-t5 without stringers, 9 bays, 45 domains, 6264 Ritz
    # constants, 1 N at the tip
    md, c, eigvals, eigvecs = wingbox_tip_force_lb(num_eigvalues=4)
    Pcr = eigvals[0]
    print('first buckling tip force: %.2f N, Abaqus %.2f N, error %.2f%%'
          % (Pcr, ABAQUS_HIGH_AR_T5, 100*(Pcr/ABAQUS_HIGH_AR_T5 - 1)))
    # 172.77 N, 0.20% below Abaqus
    assert np.isclose(Pcr, 172.77, rtol=1e-3)
    assert abs(Pcr/ABAQUS_HIGH_AR_T5 - 1) < 0.005
    # the mode is a local buckling of the upper skin of the root bay, in
    # compression
    root_top = md.panels[0]
    assert root_top.group == 'skin_top' and root_top.x0 == 0.
    v = abs(eigvecs[:, 0])
    assert v.argmax() in range(root_top.col_start, root_top.col_end)
