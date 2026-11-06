r"""Assembly functions of a panel with one T stiffener

:func:`.tstiff2d_1stiff_freq`, :func:`.tstiff2d_1stiff_compression` (with
and without the pre-buckling static case) and
:func:`.tstiff2d_1stiff_flutter` build a skin, a stiffener base and a
flange in domains, with a debonded region of length ``defect_a*a`` between
skin and base, connected with penalties. The tests verify the first natural
frequency, the first buckling eigenvalue and the first eigenvalue of the
flutter analysis against regression values stored in the tests, and the
first natural frequencies of the curved panel against Castro and Donadon
(2017), Table 5.

"""
import sys
sys.path.append('../..')

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
        num_eigvalues=10, conn_method='penalty'
        )
    omegan = (-eigvals[0])**0.5
    assert np.isclose(omegan, 48.4404, atol=0.001, rtol=0.001)


def test_tstiff2d_1stiff_freq_castro_donadon_2017_table5():
    r"""First natural frequency of the curved panel, six defect lengths,
    against the semi-analytical results with `m = n = 8` of Castro and
    Donadon (2017), Table 5, within 0.6 %

    Stiffened panel #1 of their Table 2, `a/b = 0.5`, `b/b_b = 5`,
    `b = 1` m, `r = 10` m, `m = n = 8` for every domain, penalty
    connections, density 1300 kg/m^3.

    Notes
    -----
    The paper does not print the density; 1300 kg/m^3 is the value of
    the authors' models, used in every domain. The boundary conditions of
    :func:`.tstiff2d_1stiff_freq` are those of Fig. 8 of [1]_, p. 236. The
    computed frequencies lie +0.01 % to +0.51 % from the `m = n = 8` row of
    Table 5, p. 243, and 0.7 % to 1.0 % above its finite element row.

    References
    ----------
    .. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
       models to Address Linear Buckling and Vibration of Stiffened
       Composite Panels with Debonding Defect, Composite Structures, 2017.
       https://doi.org/10.1016/j.compstruct.2016.10.026

    """
    defects = [0., 0.207, 0.407, 0.607, 0.807, 0.907]
    table5 = [65.17, 64.98, 63.99, 61.59, 57.83, 55.69]  # Hz, m = n = 8
    for defect_a, ref in zip(defects, table5):
        assy, eigvals, eigvecs = tstiff2d_1stiff_freq(
            a=0.5, b=1., ys=0.5, bb=0.2, bf=0.1, r=10.,
            defect_a=defect_a,
            rho=1.3e3,
            plyt=0.125e-3,
            laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
            stack_skin=[0, 45, -45, 90, -45, 45, 0],
            stack_base=[0, 90, 0]*4,
            stack_flange=[0, 90, 0]*8,
            m=8, n=8, mb=8, nb=8, mf=8, nf=8,
            num_eigvalues=5, conn_method='penalty'
            )
        f1 = np.sqrt(-eigvals[0]).real/(2*np.pi)
        assert np.isclose(f1, ref, rtol=0.006)


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
        num_eigvalues=10, conn_method='penalty'
        )
    assert np.isclose(eigvals[0], 142.68, rtol=0.001)

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
        num_eigvalues=10, conn_method='penalty'
        )
    assert np.isclose(eigvals[0], 114.86, rtol=0.001)


def test_tstiff2d_1stiff_flutter():
    print('Testing assembly function: tstiff2d_1stiff_flutter')
    b = 1.
    bb = b/5.
    bf = bb/2.
    ys = b/2.
    assy, eigvals, eigvecs = tstiff2d_1stiff_flutter(
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
        air_speed=2*343.,
        rho_air=1.2,
        Mach=2,
        speed_sound=343.,
        run_static_case=False,
        num_eigvalues=10, conn_method='penalty'
        )
    omegan = (-eigvals[0])**0.5
    print(eigvals)
    assert np.isclose(omegan, 503.06-139.60j, rtol=0.001)

if __name__ == '__main__':
    test_tstiff2d_1stiff_freq()
    test_tstiff2d_1stiff_compression()
    test_tstiff2d_1stiff_flutter()
