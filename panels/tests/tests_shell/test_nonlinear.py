from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from panels import Shell


def test_nonlinear():
    m = 6
    n = 6
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell',
                  ]:
        s = Shell()
        s.model = model
        s.u1tx = 1
        s.u1rx = 1
        s.u2tx = 0
        s.u2rx = 1

        s.v1tx = 1
        s.v1rx = 1
        s.v2tx = 1
        s.v2rx = 1

        s.w1tx = 0
        s.w1rx = 1
        s.w2tx = 0
        s.w2rx = 1

        s.u1ty = 1
        s.u1ry = 1
        s.u2ty = 1
        s.u2ry = 1

        s.v1ty = 0
        s.v1ry = 1
        s.v2ty = 1
        s.v2ry = 1

        s.w1ty = 0
        s.w1ry = 1
        s.w2ty = 0
        s.w2ry = 1

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

        P = 25
        Nxx = P/s.b
        npts = 10000
        s.point_loads_inc = []
        for y in np.linspace(0, s.b, npts):
            s.point_loads_inc.append([0., y, P/(npts-1.), 0, 0])
        s.point_loads_inc[0][2] /= 2.
        s.point_loads_inc[-1][2] /= 2.
        s.point_loads.append([s.a/2., s.b/2., 0, 0, 0.001])

        s.analysis.maxArcLength = 8
        s.analysis.NL_method = 'arc_length_riks'
        s.analysis.modified_NR = True
        s.static(NLgeom=True, silent=False)
        analysis_arc_length = deepcopy(s.analysis)
        assert np.isclose(s.analysis.increments[-1], 0.5175975949998647, rtol=0.01)


        s.analysis.NL_method = 'NR'
        s.analysis.modified_NR = True
        s.static(NLgeom=True, silent=False)
        analysis_newton_raphson = deepcopy(s.analysis)
        assert np.isclose(s.analysis.increments[-1], 0.545674321, rtol=0.01)

        wmaxs = [s.uvw(c)[1]['w'].max() for c in analysis_newton_raphson.cs]
        plt.plot(wmaxs, analysis_newton_raphson.increments)

        wmaxs = [s.uvw(c)[1]['w'].max() for c in analysis_arc_length.cs]
        plt.plot(wmaxs, analysis_arc_length.increments, '--')

