import numpy as np
from structsolve import freq

from panels.panelbay import PanelBay


def test_dynamic_with_BladeStiff2D():
    print('Testing dynamic analysis with BladeStiff2D')
    pb = PanelBay()
    pb.a = 2.
    pb.b = 1.
    pb.stack = [0, 90, 90, 0]
    pb.plyt = 1e-3*0.125
    pb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    pb.model = 'plate_clpt_donnell_bardell'
    pb.m = 10
    pb.n = 10
    pb.rho = 1.3e3

    pb.add_shell(y1=0, y2=pb.b/2., plyt=pb.plyt)
    pb.add_shell(y1=pb.b/2., y2=pb.b, plyt=pb.plyt)

    bb = pb.b/5.
    bf = bb
    stiff = pb.add_bladestiff2d(ys=pb.b/2., bf=bf, bb=bb,
                     fstack=[0, 90, 90, 0]*2,
                     fplyt=pb.plyt*1., flaminaprop=pb.laminaprop,
                     bstack=[0, 90, 90, 0]*1,
                     bplyt=pb.plyt*1., blaminaprop=pb.laminaprop,
                     mf=10, nf=10)

    kC = pb.calc_kC()
    M = pb.calc_kM()
    eigvals, eigvecs = freq(kC, M, silent=True)

    pb.plot_skin(eigvecs[:, 0], filename='tmp_test_bladestiff2d_dynamic_skin.png', colorbar=True)
    pb.plot_stiffener(eigvecs[:, 0], si=0, region='flange',
            filename='tmp_test_bladestiff2d_dynamic_stiff_flange.png', colorbar=True)

    assert np.isclose(eigvals[0], 29.881478223376853, atol=0.01, rtol=0.001)


if __name__ == '__main__':
    test_dynamic_with_BladeStiff2D()
