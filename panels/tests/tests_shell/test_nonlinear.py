import sys
sys.path.append(r'C:\repositories\structsolve')
from copy import deepcopy

import numpy as np

from structsolve import Analysis
from panels import Shell


def test_nonlinear(plot=False):
    m = 6
    n = 6
    for model in [
            'plate_clpt_donnell_bardell',
            'cylshell_clpt_donnell_bardell',
                  ]:
        print('Testing model: %s' % model)
        s = Shell()
        an = Analysis(calc_fext=s.calc_fext,
                      calc_fint=s.calc_fint,
                      calc_kC=s.calc_kC,
                      calc_kG=s.calc_kG)

        s.model = model
        s.x1u = 1
        s.x2u = 0

        s.x1v = 1
        s.x2v = 1

        s.x1w = 0
        s.x1wr = 1
        s.x2w = 0
        s.x2wr = 1

        s.y1u = 1
        s.y2u = 1

        s.y1v = 0
        s.y2v = 1

        s.y1w = 0
        s.y1wr = 1
        s.y2w = 0
        s.y1wr = 1

        s.a = 4.
        s.b = 1.
        s.r = 1.e15
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

        P = 500
        Nxx = P/s.b
        # distributed axial load
        s.add_distr_load_fixed_x(0, funcx=None, funcy=lambda y: Nxx,
                funcz=None, cte=False)
        # perturbation load
        s.point_loads.append([s.a/2., s.b/2., 0, 0, 0.001])

        an.NL_method = 'NR'
        an.modified_NR = True
        an.static(NLgeom=True, silent=True)
        analysis_newton_raphson = deepcopy(an)
        print('    Newton-Raphson', an.increments[-1])
        assert np.isclose(an.increments[-1], 0.545674321, rtol=0.01)

        an.maxArcLength = 8
        an.NL_method = 'arc_length_riks'
        an.modified_NR = True
        an.static(NLgeom=True, silent=True)
        analysis_arc_length = deepcopy(an)
        print('    Arc-Length', an.increments[-1])
        assert np.isclose(an.increments[-1], 0.27193678, rtol=0.01)

        if plot:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            wmaxs = [s.uvw(c)[1]['w'].max() for c in analysis_newton_raphson.cs]
            plt.plot(wmaxs, analysis_newton_raphson.increments, label='NR')
            wmaxs = [s.uvw(c)[1]['w'].max() for c in analysis_arc_length.cs]
            plt.plot(wmaxs, analysis_arc_length.increments, '--',
                    label='Arc-Length')
            plt.legend()
            plt.show()

if __name__ == '__main__':
    test_nonlinear(plot=True)
