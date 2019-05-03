import numpy as np

from structsolve import static
from panels import Shell

def test_panel_field_outputs(plot=False):
    m = 7
    n = 6
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell']:
        print('Testing model %s' % model)
        s = Shell()
        s.model = model
        s.x1u = 1
        s.y1u = 1
        s.y2u = 1
        s.x1v = 0
        s.x2v = 0
        s.y1v = 0
        s.y2v = 0

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
        Nxx = P/s.b
        s.add_distr_load_fixed_x(0, funcx=lambda y: Nxx, funcy=None,
                funcz=None, cte=False)
        increments, cs = static(s.calc_kC(), s.calc_fext(), silent=True)
        plot_mesh, fields = s.uvw(cs[0])
        plot_mesh, fields = s.stress(cs[0])
        _, fields = s.strain(cs[0])
        assert np.isclose(fields.get('w').max(), 0.00014, rtol=0.05)
        assert np.isclose(fields.get('Nxx').min(), -1011.4371924517162, rtol=0.001)
        assert np.isclose(fields.get('eyy').min(), -4.373275882508465e-07, rtol=0.001)

        if plot:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            x = plot_mesh['Ys']
            y = plot_mesh['Xs']
            field = fields['w']
            levels = np.linspace(field.min(), field.max(), 400)
            contour = plt.contourf(x, y, field, levels=levels)
            cbar = plt.colorbar(contour)
            plt.show()
            plt.cla()
            plt.contourf(x, y, fields['Nxx'])
            plt.show()
            plt.cla()
            plt.contourf(x, y, fields['eyy'])
            plt.show()


if __name__ == '__main__':
    test_panel_field_outputs(plot=True)
