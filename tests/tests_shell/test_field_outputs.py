import sys
sys.path.append('../../..')
import numpy as np

from structsolve import static
from panels import Shell


def test_panel_field_outputs():
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
        s.add_distr_load_fixed_x(0, funcx=lambda y: Nxx, funcy=None, funcz=None, cte=False)
        increments, cs = static(s.calc_kC(), s.calc_fext(), silent=True)
        _, fields = s.uvw(cs[0])
        _, fields = s.stress(cs[0])
        _, fields = s.strain(cs[0])
        assert np.isclose(fields.get('w').max(), 0.00014, rtol=0.05)
        assert np.isclose(fields.get('Nxx').min(), -1017.4018, rtol=0.001)
        assert np.isclose(fields.get('eyy').min(), -4.1451e-07, rtol=0.001)


if __name__ == '__main__':
    test_panel_field_outputs()
