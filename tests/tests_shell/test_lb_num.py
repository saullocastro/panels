import numpy as np

from structsolve import solve, lb
from panels import Shell


def test_panel_fkG_num():
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell']:
        print('Checking fkG_num for model {0}'.format(model))
        # ssss
        p = Shell()
        p.a = 8.
        p.b = 4.
        p.r = 1.e8
        p.stack = [0, 90, 90, 0, -45, +45]
        p.plyt = 1e-3*0.125
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        p.model = model

        Nxx = -1.

        p.m = 8
        p.n = 9

        p.x1ur = 1
        p.x2u = 1
        p.x2ur = 1

        p.y1u = 1
        p.y1ur = 1
        p.y2u = 1
        p.y2ur = 1

        p.x1v = 0
        p.x1vr = 0
        p.x2v = 1
        p.x2vr = 1
        p.y1v = 1
        p.y1vr = 1
        p.y2v = 1
        p.y2vr = 1

        # from constant Nxx
        p.Nxx = Nxx
        k0 = p.calc_kC()
        kG0 = p.calc_kG()
        eigvals, eigvecs = lb(k0, kG0, silent=True)
        assert np.isclose(eigvals[0], 4.47698, atol=0.01, rtol=0)

        # from pre-buckling static solution
        p.Nxx = 0.
        p.add_distr_load_fixed_x(p.a, lambda y: Nxx, None, None)
        fext = p.calc_fext()
        c0 = solve(k0, fext)
        kG = p.calc_kG(c=c0)
        eigvals, eigvecs = lb(k0, kG, silent=True)
        assert np.isclose(eigvals[0], 4.39212, atol=0.01, rtol=0)


def test_panel_fkG_num_ABDnxny():
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell']:
        print('Checking fkG_num for model {0}'.format(model))
        # ssss
        p = Shell()
        p.a = 8.
        p.b = 4.
        p.r = 1.e8
        p.stack = [0, 90, 90, 0, -45, +45]
        p.plyt = 1e-3*0.125
        p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        p.model = model

        Nxx = -1.

        p.m = 8
        p.n = 9

        p.x1ur = 1
        p.x2u = 1
        p.x2ur = 1

        p.y1u = 1
        p.y1ur = 1
        p.y2u = 1
        p.y2ur = 1

        p.x1v = 0
        p.x1vr = 0
        p.x2v = 1
        p.x2vr = 1
        p.y1v = 1
        p.y1vr = 1
        p.y2v = 1
        p.y2vr = 1

        # from pre-buckling static solution
        p.Nxx = 0.
        p.add_distr_load_fixed_x(p.a, lambda y: Nxx, None, None)
        fext = p.calc_fext()
        k0 = p.calc_kC()
        c0 = solve(k0, fext)
        kG = p.calc_kG(c=c0)
        eigvals, eigvecs = lb(k0, kG, silent=True)
        assert np.isclose(eigvals[0], 4.39212, atol=0.01, rtol=0)

        nx = 9
        ny = 9
        ABDnxny = p.ABD
        kG = p.calc_kG(c=c0, ABDnxny=ABDnxny, nx=nx, ny=ny)
        eigvals, eigvecs = lb(k0, kG, silent=True)
        assert np.isclose(eigvals[0], 4.39212, atol=0.01, rtol=0)

        nx = 12
        ny = 10
        ABDnxny = np.array([[p.ABD]*ny]*nx)
        kG = p.calc_kG(c=c0, ABDnxny=ABDnxny, nx=nx, ny=ny)
        eigvals, eigvecs = lb(k0, kG, silent=True)
        assert np.isclose(eigvals[0], 4.39212, atol=0.01, rtol=0)


#TODO  test_panel_fkG_num_ckL()


if __name__ == '__main__':
    test_panel_fkG_num()
    test_panel_fkG_num_ABDnxny()
