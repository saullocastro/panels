import numpy as np

from panels import Shell

def test_panel_field_outputs():
    m = 7
    n = 6
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell']:
        s = Shell()
        s.model = model
        s.u1tx = 1
        s.u1ty = 1
        s.u2ty = 1
        s.v1tx = 0
        s.v2tx = 0
        s.v1ty = 0
        s.v2ty = 0

        s.a = 2.
        s.b = 1.
        s.r = 1.e5
        s.stack = [0, -45, +45, 90, +45, -45, 0, 0]
        s.plyt = 1e-3*0.125
        s.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        s.nx = m
        s.ny = n
        s.m = m
        s.n = n

        P = 1000.
        npts = 100
        s.point_loads_inc = []
        for y in np.linspace(0, s.b, npts):
            s.point_loads_inc.append([0., y, P/(npts-1.), 0, 0])
        s.point_loads_inc[0][2] /= 2.
        s.point_loads_inc[-1][2] /= 2.

        s.static()
        c = s.analysis.cs[0]
        plot_mesh, stress = s.stress(c, gridx=50, gridy=50)
        _, strain = s.strain(c, gridx=50, gridy=50)


