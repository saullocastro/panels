"""Snap-through of a hinged cylindrical shell traced with the Riks method

Classical arc-length benchmark of Sabir and Lock (1972): a shallow cylindrical
panel with the straight edges hinged and the curved edges free, loaded by a
point force at its center. The load-deflection curve has a limit point
followed by snap-through, after which the load drops and increases again.
Load control (Newton-Raphson) cannot pass the limit point, whereas the
arc-length method of structsolve traces the whole path until the load factor
reaches 1.

The Shell methods ``calc_fext``, ``calc_fint``, ``calc_kC`` and ``calc_kG``
are passed directly to :class:`structsolve.Analysis`.

Units: N, mm, MPa.
"""
import sys
sys.path.append('../..')

import numpy as np
from structsolve import Analysis

from panels.shell import Shell


def test_riks_hinged_cylindrical_shell():
    s = Shell()
    s.model = 'cylshell_clpt_donnell'
    R = 2540.
    theta = 0.2
    s.a = 508. # length of the free curved edges along x
    s.b = R*theta # length of the hinged straight edges along y
    s.r = R
    s.stack = [0.]
    s.plyt = 12.7
    s.laminaprop = (3102.75, 0.3) # isotropic
    s.m = 6
    s.n = 6
    s.nx = 2*s.m
    s.ny = 2*s.n
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for field in 'uvw':
            setattr(s, edge + field, 1.)
            setattr(s, edge + field + 'r', 1.)
    # hinged straight edges
    for edge in ('y1', 'y2'):
        for field in 'uvw':
            setattr(s, edge + field, 0.)

    # reference load larger than the limit load, such that the analysis must
    # pass the limit point and the snap-through before reaching a load factor
    # of 1
    Pref = 4000.
    s.add_point_load(s.a/2., s.b/2., 0., 0., -Pref, cte=False)

    an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    an.NL_method = 'arc_length_riks'
    an.initialInc = 0.05
    an.maxArcLength = 50.
    increments, cs = an.static(NLgeom=True, silent=True)

    lbds = np.asarray(increments)
    wc = np.array([s.uvw(c=c, gridx=3, gridy=3)[1]['w'][1, 1] for c in cs])
    for lbd, w in zip(lbds, wc):
        print('  load %7.1f N, center deflection %8.4f mm' % (lbd*Pref, w))

    assert np.isclose(lbds[-1], 1.)

    # every converged increment is in equilibrium
    fext = s.calc_fext()
    for lbd, c in zip(lbds, cs):
        fint = s.calc_fint(c=c)
        R = lbd*fext - fint
        assert np.linalg.norm(R) <= 1.e-5*max(np.linalg.norm(lbd*fext),
                                              np.linalg.norm(fint))

    # no snap-back for this thickness: the deflection increases monotonically
    assert np.all(np.diff(wc) < 0)

    # limit point: first converged increment followed by a load drop
    ilim = np.argmax(np.diff(lbds) < 0)
    assert 0 < ilim
    assert np.isclose(lbds[ilim]*Pref, 2280., rtol=0.02)
    assert np.isclose(wc[ilim], -8.83, rtol=0.1)

    # snap-through: after the limit point the load drops to a fraction of the
    # limit load and then increases again, up to the reference load
    imin = ilim + np.argmin(lbds[ilim:])
    assert ilim < imin < len(lbds) - 1
    assert lbds[imin] < 0.3*lbds[ilim]
    assert np.all(np.diff(lbds[:ilim + 1]) > 0)
    assert np.all(np.diff(lbds[ilim:imin + 1]) < 0)
    assert np.all(np.diff(lbds[imin:]) > 0)
    assert np.isclose(wc[-1], -29.1, rtol=0.02)


if __name__ == '__main__':
    test_riks_hinged_cylindrical_shell()
