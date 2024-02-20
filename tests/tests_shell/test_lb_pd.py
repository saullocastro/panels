import numpy as np

from structsolve import solve, lb
from structsolve.sparseutils import finalize_symmetric_matrix
from panels import Shell
from panels.multidomain.connections import calc_ku_kv_kw_line_pd_xcte, calc_ku_kv_kw_line_pd_ycte
from panels.multidomain.connections import fkCld_xcte, fkCld_ycte
from panels.plot_shell import plot_shell


def test_shell_pd_xcte():
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell']:
        print('Checking fkG_num for model {0}'.format(model))
        # ssss
        p = Shell()
        p.a = 8.
        p.b = 4.
        p.r = 1.e6
        p.stack = [0, 90, 90, 0, -45, +45]
        p.plyt = 1e-3*0.125
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        p.model = model

        p.m = 8
        p.n = 9

        p.x1ur = 1
        p.x2u = 0
        p.x2ur = 1

        p.y1u = 1
        p.y1ur = 1
        p.y2u = 1
        p.y2ur = 1

        p.x1v = 0
        p.x1vr = 1
        p.x2v = 1
        p.x2vr = 1
        p.y1v = 1
        p.y1vr = 1
        p.y2v = 1
        p.y2vr = 1

        # from pre-buckling static solution
        k0 = p.calc_kC()
        ku, kv, kw = calc_ku_kv_kw_line_pd_xcte(p)
        kCp = fkCld_xcte(ku, 0., 0., p, p.a, p.get_size(), 0, 0)
        kCp = finalize_symmetric_matrix(kCp)
        k0 = k0 + kCp
        p.add_distr_pd_fixed_x(p.a, ku, None, None, lambda y: -0.001, None, None)
        fext = p.calc_fext()
        c0 = solve(k0, fext)
        kG = p.calc_kG(c=c0)
        eigvals, eigvecs = lb(k0, kG, silent=True)
        print(eigvals)
        plot_shell(p, eigvecs[:, 0], vec='w', filename='test_lb_pd_xcte.png')
        #assert np.isclose(eigvals[0], 4.39212, atol=0.01, rtol=0)


def test_shell_pd_ycte():
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell']:
        print('Checking fkG_num for model {0}'.format(model))
        # ssss
        p = Shell()
        p.a = 8.
        p.b = 4.
        p.r = 1.e6
        p.stack = [0, 90, 90, 0, -45, +45]
        p.plyt = 1e-3*0.125
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        p.model = model

        p.m = 8
        p.n = 9

        p.x1u = 0
        p.x1ur = 1
        p.x2u = 1
        p.x2ur = 1

        p.y1u = 1
        p.y1ur = 1
        p.y2u = 1
        p.y2ur = 1

        p.x1v = 1
        p.x1vr = 1
        p.x2v = 1
        p.x2vr = 1
        p.y1v = 0
        p.y1vr = 1
        p.y2v = 0
        p.y2vr = 1

        # from pre-buckling static solution
        k0 = p.calc_kC()
        ku, kv, kw = calc_ku_kv_kw_line_pd_ycte(p)
        kCp = fkCld_ycte(0., kv, 0., p, p.b, p.get_size(), 0, 0)
        kCp = finalize_symmetric_matrix(kCp)
        k0 = k0 + kCp
        p.add_distr_pd_fixed_y(p.b, None, kv, None, None, lambda x: -0.001, None)
        fext = p.calc_fext()
        c0 = solve(k0, fext)
        kG = p.calc_kG(c=c0)
        eigvals, eigvecs = lb(k0, kG, silent=True)
        print(eigvals)
        plot_shell(p, eigvecs[:, 0], vec='w', filename='test_lb_pd_ycte.png')


if __name__ == '__main__':
    test_shell_pd_xcte()
    test_shell_pd_ycte()
