import numpy as np

from panels import Shell
from panels.models.plate_clpt_donnell_bardell_num import fkC_num as plate_kC
from panels.models.cylshell_clpt_donnell_bardell_num import fkC_num as cylshell_kC


def test_partial_domain_integration():
    m = 6
    n = 6
    s = Shell()
    s.model = 'plate_clpt_donnell_bardell'

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


if __name__ == '__main__':
    test_partial_domain_integration()


