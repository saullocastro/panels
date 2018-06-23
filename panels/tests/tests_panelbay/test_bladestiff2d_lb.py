import numpy as np
from structsolve import lb

from panels.panelbay import PanelBay


def test_bladestiff2d_lb():
    print('Testing linear buckling with BladeStiff2D')

    pb = PanelBay()
    pb.a = 2.
    pb.b = 1.
    pb.stack = [0, +45, -45, 90, -45, +45]
    pb.plyt = 1e-3*0.125
    pb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    pb.model = 'plate_clpt_donnell_bardell'
    pb.m = 17
    pb.n = 16

    Nxx = -50.
    pb.add_shell(y1=0, y2=pb.b/2., plyt=pb.plyt, Nxx=Nxx)
    pb.add_shell(y1=pb.b/2., y2=pb.b, plyt=pb.plyt, Nxx=Nxx)

    bb = pb.b/5.
    bf = bb
    stiff = pb.add_bladestiff2d(ys=pb.b/2., bf=bf, bb=bb,
                     fstack=[0, 90, 90, 0]*8,
                     fplyt=pb.plyt*1., flaminaprop=pb.laminaprop,
                     bstack=[0, 90, 90, 0]*4,
                     bplyt=pb.plyt*1., blaminaprop=pb.laminaprop,
                     mf=17, nf=16)

    kC = pb.calc_kC()
    kG = pb.calc_kG()
    eigvals, eigvecs = lb(kC, kG, silent=True)

    pb.plot_skin(eigvecs[:, 0], filename='tmp_test_bladestiff2d_lb_skin.png',
            colorbar=True)
    pb.plot_stiffener(eigvecs[:, 0], si=0, region='flange',
            filename='tmp_test_bladestiff2d_lb_stiff_flange.png', colorbar=True)

    calc = eigvals[0]*Nxx

    pb.plot_skin(eigvecs[:, 0], filename='tmp_test_bladestiff2d_lb_skin.png', colorbar=True, vec='w', clean=False)
    assert np.isclose(calc, -759.05689868085778, atol=0.0001, rtol=0.001)

