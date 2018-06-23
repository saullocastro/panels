import numpy as np
from structsolve import static

from panels.panelbay import PanelBay


def test_static_with_BladeStiff2D():
    print('Testing linear statics with BladeStiff2D')

    pb = PanelBay()
    pb.a = 2.
    pb.b = 1.
    pb.m = 12
    pb.n = 13
    pb.stack = [0, +45, -45, 90, -45, +45, 0]
    pb.plyt = 1e-3*0.125
    pb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)

    pb.add_shell(y1=0, y2=pb.b/2., plyt=pb.plyt)
    pb.add_shell(y1=pb.b/2., y2=pb.b, plyt=pb.plyt)

    bb = pb.b/5.
    bf = bb
    stiff = pb.add_bladestiff2d(ys=pb.b/2., bf=bf, bb=bb,
                     fstack=[0, 90, 90, 0]*8,
                     fplyt=pb.plyt*1., flaminaprop=pb.laminaprop,
                     bstack=[0, 90, 90, 0]*4,
                     bplyt=pb.plyt*1., blaminaprop=pb.laminaprop,
                     mf=17, nf=16)
    stiff.flange.point_loads.append([stiff.flange.a/2., stiff.flange.b, 0., 0., 1000.])

    kC = pb.calc_kC()
    fext = pb.calc_fext()
    inc, cs = static(kC, fext, silent=True)

    pb.uvw_skin(cs[0])
    wpanelmin = pb.fields['w'].min()
    #NOTE repeated call on purpose to evaluate if cs[0] is being messed up
    #     somewhere
    pb.uvw_skin(cs[0])
    wpanelmin = pb.fields['w'].min()
    pb.uvw_stiffener(cs[0], 0, region='flange')
    wflangemax = pb.fields['w'].max()
    #NOTE repeated call on purpose to evaluate if cs[0] is being messed up
    pb.uvw_stiffener(cs[0], 0, region='flange')
    wflangemax = pb.fields['w'].max()
    assert np.isclose(wpanelmin, -0.30581458201781481, atol=1.e-4, rtol=0.001)
    assert np.isclose(wflangemax, 0.331155797371884, atol=1.e-4, rtol=0.001)
    pb.plot_skin(cs[0], filename='tmp_test_bladestiff2d_static_skin.png', colorbar=True, vec='w', clean=False)
    pb.plot_stiffener(cs[0], si=0, region='flange',
            filename='tmp_test_bladestiff2d_stiff_static_flange.png', colorbar=True, clean=False)
