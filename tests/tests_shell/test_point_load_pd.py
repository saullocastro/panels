import matplotlib
matplotlib.use('Agg')
import numpy as np
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels import Shell
from panels.multidomain.connections import calc_ku_kv_kw_point_pd
from panels.multidomain.connections import fkCpd
from panels.plot_shell import plot_shell


def test_nonlinear():
    m = 6
    n = 6
    for model in [
            'plate_clpt_donnell_bardell',
            'cylshell_clpt_donnell_bardell',
                  ]:
        print('Testing model: %s' % model)
        s = Shell()

        s.model = model
        s.y1w = s.y2w = s.y1wr = s.y2wr = 1
        s.x2w = s.x2wr = 1

        s.a = 3.
        s.b = 1.
        s.r = 1.e9
        s.stack = [90, 0, 90, 0]
        s.plyt = 1e-3*0.125
        E11 = 142.5e9
        E22 = E11/20
        G12 = G13 = G23 = 0.5*E22
        s.laminaprop = (E11, E22, 0.25, G12, G12, G12)
        s.m = m
        s.n = n

        k0 = s.calc_kC()
        ku, kv, kw = calc_ku_kv_kw_point_pd(s)
        kCp = fkCpd(0., 0., kw, s, s.a, s.b/2, s.get_size(), 0, 0)
        kCp = finalize_symmetric_matrix(kCp)
        k0 = k0 + kCp
        wp = 0.001
        s.add_point_pd(s.a, s.b/2, 0., 0., 0., 0., kw, wp)
        fext = s.calc_fext()
        c0 = solve(k0, fext)
        plot_shell(s, c0, vec='w', filename='test_point_load_pd.png', colorbar=True)
        assert np.isclose(s.fields['w'].max(), wp, rtol=0.001)


if __name__ == '__main__':
    test_nonlinear()
