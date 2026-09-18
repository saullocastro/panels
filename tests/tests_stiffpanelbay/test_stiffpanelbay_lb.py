import sys
sys.path.append('../..')

import numpy as np
import pytest
from structsolve import lb

from panels.shell import Shell
from panels.stiffpanelbay import StiffPanelBay


laminaprop = (105.e9, 10.5e9, 0.25, 5.25e9, 5.25e9, 5.25e9)
stack = [0, 90, 90, 0, 0]
plyt = 0.134e-3
tskin = len(stack)*plyt


def _bay(nstiff, m=14, n=14):
    b = 100*tskin
    bay = StiffPanelBay()
    bay.a = 2*b
    bay.b = b
    bay.r = None
    bay.model = 'plate_clpt_donnell'
    bay.stack = stack
    bay.plyt = plyt
    bay.laminaprop = laminaprop
    bay.rho = 1600.
    bay.m = m
    bay.n = n

    ys = [(j + 1)*b/(nstiff + 1) for j in range(nstiff)]
    y_prev = 0.
    for y in ys + [b]:
        bay.add_panel(y1=y_prev, y2=y, Nxx=-1.)
        y_prev = y
    for y in ys:
        bay.add_bladestiff1d(ys=y, bf=9*tskin, fstack=stack, fplyt=plyt,
                             flaminaprop=laminaprop)
    return bay


def test_add_panel():
    """StiffPanelBay.add_panel() must build a usable Shell

    The laminate attributes have to reach the Shell constructor, which already
    calls Shell._rebuild().

    """
    bay = _bay(nstiff=0)
    assert len(bay.panels) == 1
    p = bay.panels[0]
    assert p.lam is not None
    assert p.stack == stack
    assert p.model == 'plate_clpt_donnell'
    assert p.rho == 1600.


def test_unstiffened_bay_matches_single_shell():
    """Splitting the skin into panels must not change the buckling load"""
    bay = _bay(nstiff=0)
    bay.panels = []
    b = bay.b
    for y1, y2 in zip([0, b/3, 2*b/3], [b/3, 2*b/3, b]):
        p = bay.add_panel(y1=y1, y2=y2, Nxx=-1.)
        p.nx = 4*p.m
        p.ny = 4*p.n
    eigvals_bay, _ = lb(bay.calc_kC(silent=True), bay.calc_kG(silent=True),
                        silent=True, num_eigvalues=4)

    s = Shell(a=bay.a, b=bay.b, r=None, stack=stack, plyt=plyt,
              laminaprop=laminaprop, model=bay.model, m=bay.m, n=bay.n)
    s.Nxx = -1.
    eigvals_shell, _ = lb(s.calc_kC(silent=True), s.calc_kG(silent=True),
                          silent=True, num_eigvalues=4)

    assert np.allclose(eigvals_bay[:3], eigvals_shell[:3], rtol=3e-3)


@pytest.mark.parametrize('b, edges', [(1.0, [0., 1/3, 2/3, 1.0]),
                                      (1.5, [0., 1.0, 1.5])])
def test_panel_edge_at_one_meter(b, edges):
    """A panel edge at ``y = 1.0`` is a coordinate like any other

    Up to panels 0.6.9 ``y2 = +1`` was also the sentinel meaning "full
    domain", so the panel ending at ``y = 1.0`` was integrated over the whole
    bay and its skin was counted more than once: the buckling load of the
    unit-width bay was 4.9 % too low.

    """
    m = n = 10
    bay = StiffPanelBay()
    bay.a = 2.
    bay.b = b
    bay.r = None
    bay.model = 'plate_clpt_donnell'
    bay.stack = [0, 90, 90, 0]
    bay.plyt = 0.5e-3
    bay.laminaprop = laminaprop
    bay.m = m
    bay.n = n
    for y1, y2 in zip(edges[:-1], edges[1:]):
        p = bay.add_panel(y1=y1, y2=y2, Nxx=-1.)
        p.nx = 4*m
        p.ny = 4*n
        assert p.is_partial_domain()
    eigvals_bay, _ = lb(bay.calc_kC(silent=True), bay.calc_kG(silent=True),
                        silent=True, num_eigvalues=4)

    s = Shell(a=bay.a, b=bay.b, r=None, stack=bay.stack, plyt=bay.plyt,
              laminaprop=laminaprop, model=bay.model, m=m, n=n)
    s.Nxx = -1.
    eigvals_shell, _ = lb(s.calc_kC(silent=True), s.calc_kG(silent=True),
                          silent=True, num_eigvalues=4)
    assert np.allclose(eigvals_bay[:3], eigvals_shell[:3], rtol=1e-6)


def test_bladestiff1d_increases_buckling_load():
    unstiffened, _ = lb(_bay(0).calc_kC(silent=True),
                        _bay(0).calc_kG(silent=True),
                        silent=True, num_eigvalues=4)
    bay = _bay(1)
    stiffened, _ = lb(bay.calc_kC(silent=True), bay.calc_kG(silent=True),
                      silent=True, num_eigvalues=4)
    assert stiffened[0] > 3*unstiffened[0]


if __name__ == '__main__':
    test_add_panel()
    test_unstiffened_bay_matches_single_shell()
    test_bladestiff1d_increases_buckling_load()
