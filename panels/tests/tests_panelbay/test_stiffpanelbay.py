import numpy as np
from structsolve import freq, lb

from panels.panelbay import PanelBay


def test_freq_models():
    print('Testing frequency analysis for PanelBay with 2 plates')
    # From Table 4 of
    # Lee and Lee. "Vibration analysis of anisotropic plates with eccentric
    #    stiffeners". Computers & Structures, Vol. 57, No. 1, pp. 99-105,
    #    1995.
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell']:
        spb = PanelBay()
        spb.a = 0.5
        spb.b = 0.250
        spb.plyt = 0.00013
        spb.laminaprop = (128.e9, 11.e9, 0.25, 4.48e9, 1.53e9, 1.53e9)
        spb.stack = [0, -45, +45, 90, 90, +45, -45, 0]
        spb.model = model
        spb.r = 1.e6
        spb.alphadeg = 0.
        spb.rho = 1.5e3
        spb.m = 9
        spb.n = 10

        # clamping
        spb.w1rx = 0.
        spb.w2rx = 0.
        spb.w1ry = 0.
        spb.w2ry = 0.

        spb.add_shell(0, spb.b/2., plyt=spb.plyt)
        spb.add_shell(spb.b/2., spb.b, plyt=spb.plyt)

        kC = spb.calc_kC(silent=True)
        M = spb.calc_kM(silent=True)
        eigvals, eigvecs = freq(kC, M, silent=True)

        ref = [85.12907802-0.j, 134.16422850-0.j, 206.77295186-0.j,
                216.45992453-0.j, 252.24546171-0.j]
        assert np.allclose(eigvals[:5]/2/np.pi, ref, atol=0.1, rtol=0)


def test_lb_Stiffener1D():
    print('Testing linear buckling for PanelBay with a 1D Stiffener')
    spb = PanelBay()
    spb.a = 1.
    spb.b = 0.5
    spb.stack = [0, 90, 90, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'plate_clpt_donnell_bardell'
    spb.rho = 1.3e3
    spb.m = 15
    spb.n = 16

    spb.add_shell(y1=0, y2=spb.b/2., plyt=spb.plyt, Nxx=-1.)
    spb.add_shell(y1=spb.b/2., y2=spb.b, plyt=spb.plyt, Nxx_cte=1000.)

    spb.add_bladestiff1d(ys=spb.b/2., Fx=0., bf=0.05, fstack=[0, 90, 90, 0],
            fplyt=spb.plyt, flaminaprop=spb.laminaprop)

    kC = spb.calc_kC(silent=True)
    kG = spb.calc_kG(silent=True)
    eigvals, eigvecs = lb(kC, kG, silent=True)

    assert np.isclose(eigvals[0].real, 297.54633, atol=0.1, rtol=0)


def test_lb_Stiffener2D():
    print('Testing linear buckling for PanelBay with a 2D Stiffener')
    spb = PanelBay()
    spb.a = 1.
    spb.b = 0.5
    spb.stack = [0, 90, 90, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'plate_clpt_donnell_bardell'
    spb.rho = 1.3e3
    spb.m = 15
    spb.n = 16

    spb.add_shell(y1=0, y2=spb.b/2., plyt=spb.plyt, Nxx=-1.)
    spb.add_shell(y1=spb.b/2., y2=spb.b, plyt=spb.plyt, Nxx_cte=1000.)

    spb.add_bladestiff2d(ys=spb.b/2., m1=14, n1=11, bf=0.05,
                        fstack=[0, 90, 90, 0],
                        fplyt=spb.plyt, flaminaprop=spb.laminaprop)

    kC = spb.calc_kC(silent=True)
    kG = spb.calc_kG(silent=True)
    eigvals, eigvecs = lb(kC, kG, silent=True)

    assert np.isclose(eigvals[0].real, 301.0825234, atol=0.1, rtol=0)


def test_freq_Stiffener1D():
    print('Testing frequency analysis for PanelBay with a 1D Stiffener')
    spb = PanelBay()
    spb.a = 2.
    spb.b = 0.5
    spb.stack = [0, 90, 90, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'plate_clpt_donnell_bardell'
    spb.rho = 1.3e3
    spb.m = 15
    spb.n = 16

    spb.add_shell(y1=0, y2=spb.b/2., plyt=spb.plyt)
    spb.add_shell(y1=spb.b/2., y2=spb.b, plyt=spb.plyt)

    spb.add_bladestiff1d(ys=spb.b/2., Fx=0., bf=0.08, fstack=[0, 90, 90, 0]*5,
            fplyt=spb.plyt, flaminaprop=spb.laminaprop)

    kC = spb.calc_kC(silent=True)
    M = spb.calc_kM(silent=True)
    eigvals, eigvecs = freq(kC, M, silent=True, num_eigvalues=10)

    assert np.isclose(eigvals[0].real, 79.5906673583, atol=0.1, rtol=0)


def test_freq_Stiffener2D():
    print('Testing frequency analysis for PanelBay with a 2D Stiffener')
    spb = PanelBay()
    spb.a = 1.
    spb.b = 0.5
    spb.stack = [0, 90, 90, 0]
    spb.plyt = 1e-3*0.125
    spb.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    spb.model = 'plate_clpt_donnell_bardell'
    spb.rho = 1.3e3
    spb.m = 11
    spb.n = 12

    spb.add_shell(y1=0, y2=spb.b/2., plyt=spb.plyt)
    spb.add_shell(y1=spb.b/2., y2=spb.b, plyt=spb.plyt)

    spb.add_bladestiff2d(ys=spb.b/2., m1=14, n1=11, bf=0.08,
                        fstack=[0, 90, 90, 0]*5, fplyt=spb.plyt,
                        flaminaprop=spb.laminaprop)

    kC = spb.calc_kC(silent=True)
    M = spb.calc_kM(silent=True)
    eigvals, eigvecs = freq(kC, M, silent=True)

    assert np.isclose(eigvals[0].real, 150.5419226645058, atol=0.01, rtol=0)


def test_Lee_and_Lee_table4():
    print('Testing Lee and Lee Table 4')
    # Lee and Lee. "Vibration analysis of anisotropic plates with eccentric
    #    stiffeners". Computers & Structures, Vol. 57, No. 1, pp. 99-105,
    #    1995.
    models = (
        ('model4', 0.00208, 0.0060, 138.99917796302756),
        ('model5', 0.00260, 0.0075, 175.00597239286196),
        ('model7', 0.00364, 0.0105, 205.433509024))
    for model, hf, bf, value in models:
        spb = PanelBay()
        spb.model = 'plate_clpt_donnell_bardell'
        spb.rho = 1.500e3 # plate material density in kg / m^3
        spb.laminaprop = (128.e9, 11.e9, 0.25, 4.48e9, 1.53e9, 1.53e9)
        spb.stack = [0, -45, +45, 90, 90, +45, -45, 0]
        plyt = 0.00013
        spb.plyt = plyt
        spb.a = 0.5
        spb.b = 0.250
        spb.m = 14
        spb.n = 15
        hf = hf
        bf = bf
        n = int(hf/plyt)
        fstack = [0]*(n//4) + [90]*(n//4) + [90]*(n//4) + [0]*(n//4)
        # clamping
        spb.w1rx = 0.
        spb.w2rx = 0.
        spb.w1ry = 0.
        spb.w2ry = 0.

        spb.add_shell(y1=0, y2=spb.b/2.)
        spb.add_shell(y1=spb.b/2., y2=spb.b)
        spb.add_bladestiff1d(rho=spb.rho, ys=spb.b/2., bb=0., bf=bf,
                      fstack=fstack, fplyt=plyt, flaminaprop=spb.laminaprop)
        kC = spb.calc_kC(silent=True)
        M = spb.calc_kM(silent=True)
        eigvals, eigvecs = freq(kC, M, silent=True)

        herz = eigvals[0].real/2/np.pi
        assert np.isclose(herz, value, atol=0.001, rtol=0.001)
