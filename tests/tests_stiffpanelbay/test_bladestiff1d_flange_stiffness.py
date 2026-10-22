r"""Stiffness properties of the 1D flange of :class:`.BladeStiff1D`

- torsion: the Saint-Venant torsional stiffness of a thin laminated strip of
  height `b_f`, `G J = 4 b_f D_{66}`, which is `G b_f h_f^3/3` for an
  isotropic flange;
- coupling `S_1` between the axial strain and the twist: the reduced
  stiffness `\bar Q_{16} - \bar Q_{12} \bar Q_{26}/\bar Q_{22}` of each ply
  with `\sigma_2 = 0` along the height of the flange.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.stiffener import modelDB
from panels.stiffpanelbay import StiffPanelBay


laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
stack = [0, 90, 90, 0, 0]
plyt = 0.134e-3
tskin = len(stack)*plyt
b = 100*tskin
bf = 9*tskin


def _stiffener(fstack, flaminaprop, fplyt=plyt):
    bay = StiffPanelBay()
    bay.a = 2*b
    bay.b = b
    bay.r = None
    bay.model = 'plate_clpt_donnell'
    bay.stack = stack
    bay.plyt = plyt
    bay.laminaprop = laminaprop
    bay.rho = 1600.
    bay.m = 6
    bay.n = 5
    bay.add_panel(0, b/3)
    bay.add_panel(b/3, b)
    bay.add_bladestiff1d(ys=b/3, bf=bf, fstack=fstack, fplyt=fplyt,
                         flaminaprop=flaminaprop)
    s = bay.bladestiff1ds[0]
    s._rebuild()
    return bay, s


def test_torsional_stiffness_isotropic_flange():
    r"""`G J = 4 b_f D_{66} = G b_f h_f^3/3` for an isotropic flange"""
    E, nu = 70e9, 0.3
    hf = 2e-3
    bay, s = _stiffener([0.], (E, nu), fplyt=hf)
    G = E/(2*(1 + nu))
    assert np.isclose(s.GJ, G*bf*hf**3/3, rtol=1e-12)


def test_torsional_stiffness_laminated_flange():
    fstack = [30, -60, 15, 45]
    bay, s = _stiffener(fstack, laminaprop)
    D66 = s.flam.D66
    assert D66 > 0
    assert np.isclose(s.GJ, 4*bf*D66, rtol=1e-12)


def test_kC_uses_the_torsional_stiffness():
    r"""The torsion term of ``fkCf`` is `G J/b_f`, the kernel multiplying
    the whole constitutive matrix of the flange by `b_f`"""
    bay, s = _stiffener([30, -60, 15, 45], laminaprop)
    size = bay.get_size()
    mod = modelDB.db[s.model]['matrices']

    def kernel(E1, F1, S1, J):
        return finalize_symmetric_matrix(mod.fkCf(s.ys, bay.a, bay.b, s.bf,
            s.dbf, E1, F1, S1, J, bay.m, bay.n,
            bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
            bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
            bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
            bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
            size=size, row0=0, col0=0)).toarray()

    s.calc_kC(size=size, silent=True)
    kC = s.kC.toarray()
    expected = kernel(s.E1, s.F1, s.S1, 0.) + s.GJ/s.bf*kernel(0., 0., 0., 1.)
    assert np.abs(kernel(0., 0., 0., 1.)).max() > 0
    assert np.allclose(kC, expected, rtol=1e-12, atol=1e-12*np.abs(kC).max())


def test_S1_condensed_reduced_stiffness():
    r"""With `\sigma_2 = 0` in each ply the stress `\sigma_1` along the
    flange is `(\bar Q_{11} - \bar Q_{12}^2/\bar Q_{22}) \varepsilon_1 +
    (\bar Q_{16} - \bar Q_{12} \bar Q_{26}/\bar Q_{22}) \gamma_{12}`"""
    fstack = [30, -60, 15, 45]
    bay, s = _stiffener(fstack, laminaprop)
    E1 = S1 = 0.
    faces = np.concatenate(([0.], np.cumsum([ply.h for ply in s.flam.plies])))
    for k, ply in enumerate(s.flam.plies):
        Q = np.array([[ply.q11L, ply.q12L, ply.q16L],
                      [ply.q12L, ply.q22L, ply.q26L],
                      [ply.q16L, ply.q26L, ply.q66L]])
        keep = [0, 2]
        Qc = Q[np.ix_(keep, keep)] - np.outer(Q[keep, 1], Q[1, keep])/Q[1, 1]
        yk = 0.5*(faces[k] + faces[k + 1])
        E1 += ply.h*Qc[0, 0]
        S1 += -yk*ply.h*Qc[0, 1]
    assert abs(s.flam.plies[0].q16L - s.flam.plies[0].q26L) > 1e-3*laminaprop[0]
    assert np.isclose(s.E1, E1, rtol=1e-12)
    assert np.isclose(s.S1, S1, rtol=1e-12)
