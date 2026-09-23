import sys
sys.path.append('../..')

import numpy as np
import pytest

from panels.shell import Shell
from panels.models.plate_clpt_donnell_num import fkC_num as plate_kC
from panels.models.cylshell_clpt_donnell_num import fkC_num as cylshell_kC


def test_partial_domain_integration():
    m = 6
    n = 6
    s = Shell()
    s.model = 'plate_clpt_donnell'

    s.a = 7.
    s.b = 3.
    s.r = 1.e5
    s.stack = [90, 0, 90, 0]
    s.plyt = 1e-3*0.125
    E11 = 142.5e9
    E22 = E11/20
    G12 = G13 = G23 = 0.5*E22
    s.laminaprop = (E11, E22, 0.25, G12, G12, G12)
    s.m = m
    s.n = n
    s.nx = 2*m
    s.ny = 2*n
    s._rebuild()

    size = s.get_size()
    kCfull = plate_kC(np.zeros(size, float), s.lam.ABD, s, size, 0, 0, s.nx, s.ny, 0)
    s.x1 = 0
    s.x2 = s.a/3
    kC_1 = plate_kC(np.zeros(size, float), s.lam.ABD, s, size, 0, 0, s.nx, s.ny, 0)
    s.x1 = s.a/3
    s.x2 = s.a
    kC_2 = plate_kC(np.zeros(size, float), s.lam.ABD, s, size, 0, 0, s.nx, s.ny, 0)
    assert np.allclose(kCfull.toarray(), (kC_1 + kC_2).toarray())


@pytest.mark.parametrize('model', ['plate_clpt_donnell',
                                   'cylshell_clpt_sanders',
                                   'plate_fsdt_donnell',
                                   'plate_tsdt_donnell'])
def test_partial_domain_calc_matrices(model):
    """Shell.calc_kC/kG/kM must honour (x1, x2, y1, y2)

    The analytical closed-form matrices always integrate the full domain, so
    ``Shell.calc_*()`` has to switch to the numerically integrated ones
    whenever only part of the domain is to be integrated. Summing the
    matrices of adjacent strips must recover the full-domain matrices.

    """
    kwargs = dict(a=0.7, b=0.3, stack=[0, 45, -45, 90, 90, -45, 45, 0],
                  plyt=0.125e-3, rho=1600.,
                  laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
                  model=model, m=10, n=10)
    if model.startswith('cylshell'):
        kwargs['r'] = 0.5

    full = Shell(**kwargs)
    full.Nxx = -1.
    assert not full.is_partial_domain()
    kC_full = full.calc_kC(silent=True).toarray()
    kG_full = full.calc_kG(silent=True).toarray()
    kM_full = full.calc_kM(silent=True).toarray()

    kC = np.zeros_like(kC_full)
    kG = np.zeros_like(kG_full)
    kM = np.zeros_like(kM_full)
    edges = np.linspace(0, kwargs['b'], 4)
    for y1, y2 in zip(edges[:-1], edges[1:]):
        strip = Shell(**kwargs)
        strip.Nxx = -1.
        strip.y1 = y1
        strip.y2 = y2
        strip.nx = 4*strip.m
        strip.ny = 4*strip.n
        assert strip.is_partial_domain()
        kC += strip.calc_kC(silent=True).toarray()
        kG += strip.calc_kG(silent=True).toarray()
        kM += strip.calc_kM(silent=True).toarray()

    assert np.allclose(kC, kC_full)
    assert np.allclose(kG, kG_full)
    assert np.allclose(kM, kM_full)


if __name__ == '__main__':
    test_partial_domain_integration()
    for model in ('plate_clpt_donnell', 'cylshell_clpt_sanders',
                  'plate_fsdt_donnell', 'plate_tsdt_donnell'):
        test_partial_domain_calc_matrices(model)


