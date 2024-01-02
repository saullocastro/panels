import numpy as np

from panels.multidomain import (
        tstiff2d_1stiff_freq,
        tstiff2d_1stiff_compression,
        tstiff2d_1stiff_flutter,
        )


def test_tstiff2d_1stiff_freq():
    print('Testing assembly function: tstiff2d_1stiff_freq')
    b = 1.
    bb = b/5.
    bf = bb/2.
    ys = b/2.
    assy, eigvals, eigvecs = tstiff2d_1stiff_freq(
        b=b,
        bb=bb,
        bf=bf,
        a=3.,
        ys=ys,
        defect_a=0.1,
        rho=1.3e3,
        plyt=0.125e-3,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        stack_skin=[0, 45, -45, 90, -45, 45, 0],
        stack_base=[0, 90, 0]*4,
        stack_flange=[0, 90, 0]*8,
        m=6, n=7,
        mb=5, nb=6,
        mf=6, nf=7,
        )
    omegan = (-eigvals[0])**0.5
    assert np.isclose(omegan, 48.2733, atol=0.001, rtol=0.001)


def test_tstiff2d_1stiff_compression():
    print('Testing assembly function: tstiff2d_1stiff_compression')
    b = 1.
    bb = b/5.
    bf = bb/2.
    ys = b/2.
    assy, eigvals, eigvecs = tstiff2d_1stiff_compression(
        b=b,
        bb=bb,
        bf=bf,
        a=3.,
        ys=ys,
        defect_a=0.6,
        rho=1.3e3,
        plyt=0.125e-3,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        stack_skin=[0, 45, -45, 90, -45, 45, 0],
        stack_base=[0, 90, 0]*4,
        stack_flange=[0, 90, 0]*8,
        m=8, n=7,
        mb=7, nb=6,
        mf=8, nf=6,
        run_static_case=False,
        Nxx_skin=-1.,
        Nxx_base=-1.,
        Nxx_flange=-1.,
        )
    assert np.isclose(eigvals[0], 142.65057725, rtol=0.001)

    assy, c, eigvals, eigvecs = tstiff2d_1stiff_compression(
        b=b,
        bb=bb,
        bf=bf,
        a=3.,
        ys=ys,
        defect_a=0.6,
        rho=1.3e3,
        plyt=0.125e-3,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        stack_skin=[0, 45, -45, 90, -45, 45, 0],
        stack_base=[0, 90, 0]*4,
        stack_flange=[0, 90, 0]*8,
        m=8, n=7,
        mb=7, nb=6,
        mf=8, nf=6,
        run_static_case=True,
        Nxx_skin=-1.,
        Nxx_base=-1.,
        Nxx_flange=-1.,
        )
    assert np.isclose(eigvals[0], 114.85854770879844, rtol=0.001)


def test_tstiff2d_1stiff_flutter():
    print('Testing assembly function: tstiff2d_1stiff_flutter')
    b = 1.
    bb = b/5.
    bf = bb/2.
    ys = b/2.
    assy, c, eigvals, eigvecs = tstiff2d_1stiff_flutter(
        b=b,
        bb=bb,
        bf=bf,
        a=3.,
        ys=ys,
        defect_a=0.1,
        rho=1.3e3,
        plyt=0.125e-3,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        stack_skin=[0, 45, -45, 90, -45, 45, 0],
        stack_base=[0, 90, 0]*4,
        stack_flange=[0, 90, 0]*8,
        m=6, n=7,
        mb=5, nb=6,
        mf=6, nf=7,
        air_speed=800.,
        rho_air=1500.,
        Mach=2.,
        speed_sound=343.,
        run_static_case=True,
        )
    omegan = (-eigvals[0])**0.5
    assert np.isclose(omegan, 958.7080550167907+0.j, rtol=0.001)

if __name__ == '__main__':
    test_tstiff2d_1stiff_freq()
    test_tstiff2d_1stiff_compression()
    test_tstiff2d_1stiff_flutter()
