from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from panels import Shell
from panels.models.plate_clpt_donnell_bardell_num import fkC_num


def test_partial_domain_integration():
    m = 6
    n = 6
    s = Shell()
    s.model = 'plate_clpt_donnell_bardell'

    s.a = 4.
    s.b = 1.
    s.r = 1.e5
    s.stack = [90, 0, 90, 0]
    s.plyt = 1e-3*0.125
    E11 = 142.5e9
    E22 = E11/20
    G12 = G13 = G23 = 0.5*E22
    s.laminaprop = (E11, E22, 0.25, G12, G12, G12)
    s.nx = m
    s.ny = n
    s.m = m
    s.n = n
    s._rebuild()

    size = s.get_size()
    kCfull = fkC_num(np.zeros(size, float), s.lam.ABD, s, size, 0, 0, s.nx, s.ny, 0)
    kC_1 = fkC_num(np.zeros(size, float), s.lam.ABD, s, size, 0, 0, s.nx, s.ny, 0, 0, s.a/2)
    kC_2 = fkC_num(np.zeros(size, float), s.lam.ABD, s, size, 0, 0, s.nx, s.ny, 0, s.a/2, s.a)
    assert np.allclose(kCfull.toarray(), (kC_1 + kC_2).toarray())


if __name__ == '__main__':
    test_partial_domain_integration()


