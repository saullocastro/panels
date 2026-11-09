r"""Assembly functions of a panel with one T stiffener

:func:`.tstiff2d_1stiff_freq`, :func:`.tstiff2d_1stiff_compression` (with
and without the pre-buckling static case) and
:func:`.tstiff2d_1stiff_flutter` build a skin, a stiffener base and a
flange in domains, with a debonded region of length ``defect_a*a`` between
skin and base, connected with penalties. The tests verify the first natural
frequency, the first buckling eigenvalue and the first eigenvalue of the
flutter analysis against regression values stored in the tests, and the
first natural frequencies of the curved panel against Castro and Donadon
(2017), Table 5, and the ratios of the first natural frequency of the flat
panel with a defect to that without against their Fig. 15, digitised. The
in-plane loads of :func:`.tstiff2d_1stiff_flutter` are checked as constant
stress states and as edge loads of a static case, with the boundary
conditions of their Fig. 8.

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


def test_tstiff2d_1stiff_freq_castro_donadon_2017_fig15_flat():
    r"""Ratio of the first natural frequency of the flat panel with a
    defect to that without, six defect lengths, against the curve
    `a/b = 0.5`, `b/b_b = 5` of Castro and Donadon (2017), Fig. 15,
    digitised, within 0.01

    Stiffened panel #1 of their Table 2, `a/b = 0.5`, `b/b_b = 5`,
    `b = 1` m, flat, `m = n = 8` for every domain, penalty connections.

    Notes
    -----
    The flat panel appears in [1]_ only as the ratios
    `\omega_{CR}/\omega_{CR}^{\delta = 0}` of Fig. 15, p. 243, computed
    with `m = n = 8` (Sec. 7.3, p. 241); Table 5 lists the curved panel
    only. The geometry is that of Sec. 7, p. 236: `b = 1` m, `b_b = 2
    b_f`, the laminates of panel #1 of Table 2. The ratios do not depend on
    the density when it is the same in every domain (1300 kg/m^3 here, as
    in :func:`test_tstiff2d_1stiff_freq_castro_donadon_2017_table5`). The
    boundary conditions of :func:`.tstiff2d_1stiff_freq` are those of Fig.
    8 of [1]_, p. 236.

    The dotted curve `a/b = 0.5` of the panel `b/b_b = 5` was digitised
    from the raster image of the figure in the publisher's PDF, as the
    centroids of its dots, with the axes calibrated on the tick marks
    `\delta/a = 0, 0.5, 1` and `\omega_{CR}/\omega_{CR}^{\delta = 0} = 1.0,
    0.6, 0.2` (594 and 419 pixels per unit), and interpolated linearly
    between neighbouring dots; below `\delta/a = 0.4` the curve overlaps
    those of the other ratios `a/b` and is not used. The digitising
    uncertainty is about
    `\pm 0.004` in the ratio: one pixel of the tick calibration, `\pm
    0.0024`, half a pixel of the dot centroid, `\pm 0.0012`, and one pixel
    along `\delta/a`, `\pm 0.0006` on the steepest part of the curve. The
    computed ratios lie 0.002 to 0.005 below the digitised ones (0.995 to
    0.998 times them; with null-space connections 0.000 to 0.002 below);
    the tolerance of 0.01 covers the digitising uncertainty and a
    difference of the order of the 0.5 % between this function and the
    `m = n = 8` row of Table 5 of [1]_ for the curved panel.

    References
    ----------
    .. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
       models to Address Linear Buckling and Vibration of Stiffened
       Composite Panels with Debonding Defect, Composite Structures, 2017.
       https://doi.org/10.1016/j.compstruct.2016.10.026

    """
    defects = [0., 0.407, 0.507, 0.607, 0.707, 0.807, 0.907]
    fig15 = [1., 0.983, 0.973, 0.957, 0.935, 0.908, 0.877]  # digitised
    f = []
    for defect_a in defects:
        assy, eigvals, eigvecs = tstiff2d_1stiff_freq(
            a=0.5, b=1., ys=0.5, bb=0.2, bf=0.1, r=None,
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
        f.append(np.sqrt(-eigvals[0]).real/(2*np.pi))
    ratios = np.array(f)/f[0]
    assert np.allclose(ratios, fig15, rtol=0, atol=0.01)


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


FLUTTER_PANEL = dict(a=3., b=1., ys=0.5, bb=0.2, bf=0.1, defect_a=0.1,
        rho=1.3e3, plyt=0.125e-3,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        stack_skin=[0, 45, -45, 90, -45, 45, 0],
        stack_base=[0, 90, 0]*4, stack_flange=[0, 90, 0]*8,
        air_speed=2*343., rho_air=1.2, Mach=2, speed_sound=343.,
        m=6, n=7, mb=5, nb=6, mf=6, nf=7, num_eigvalues=5)
PARTS = ('skin', 'base', 'flange')
# sqrt(-Re lambda_1), rad/s, of the panel without loads
W0 = 487.4174


def _flutter_omega1(**kwargs):
    assy, eigvals, eigvecs = tstiff2d_1stiff_flutter(**FLUTTER_PANEL,
                                                     **kwargs)
    return assy, float(np.sqrt(-eigvals[0].real))


def test_tstiff2d_1stiff_flutter_constant_prestress():
    r"""In-plane loads of :func:`.tstiff2d_1stiff_flutter` as constant
    stress states, ``run_static_case=False``

    The panel of the function's Examples (null-space connections, the
    boundary conditions of Castro and Donadon 2017, Fig. 8). The test
    verifies that every domain of the skin, the base and the flange gets
    the ``Nxx``, ``Nyy`` and ``Nxy`` of its part; that
    `\sqrt{-\mathrm{Re}(\lambda_1)}` without loads is that of the Examples;
    that with `N_{xx} = -100` N/m in the three parts, and in each part
    alone, it equals the values obtained for these constant stress states
    by a separate script that sets the
    attributes ``Nxx`` of the domains of the assembly and calls
    :meth:`.MultiDomain.calc_kG` without ``c`` (486.3599, 486.6128,
    487.2331 and 487.3477 rad/s), within ``rtol=1e-6``; and that
    compression lowers and tension raises it, for `N_{xx}` and `N_{yy}`,
    and that `N_{xy}` changes it.

    """
    assy, w0 = _flutter_omega1(run_static_case=False)
    assert np.isclose(w0, W0, rtol=1e-6)

    loads = dict(Nxx_skin=-100., Nxx_base=-90., Nxx_flange=-80.,
                 Nyy_skin=-10., Nyy_base=-9., Nyy_flange=-8.,
                 Nxy_skin=5., Nxy_base=4., Nxy_flange=3.)
    assy, w = _flutter_omega1(run_static_case=False, **loads)
    for p in assy.panels:
        for comp in ('Nxx', 'Nyy', 'Nxy'):
            assert getattr(p, comp) == loads['%s_%s' % (comp, p.group)]

    for parts, ref in ((PARTS, 486.3599), (('skin',), 486.6128),
                       (('base',), 487.2331), (('flange',), 487.3477)):
        assy, w = _flutter_omega1(run_static_case=False,
                                  **{'Nxx_%s' % p: -100. for p in parts})
        assert np.isclose(w, ref, rtol=1e-6)

    for comp in ('Nxx', 'Nyy'):
        wc = _flutter_omega1(run_static_case=False,
                             **{'%s_%s' % (comp, p): -100. for p in PARTS})[1]
        wt = _flutter_omega1(run_static_case=False,
                             **{'%s_%s' % (comp, p): 100. for p in PARTS})[1]
        assert wc < w0 < wt
    ws = _flutter_omega1(run_static_case=False,
                         **{'Nxy_%s' % p: -100. for p in PARTS})[1]
    assert not np.isclose(ws, w0, rtol=1e-6)


def test_tstiff2d_1stiff_flutter_static_edge_loads_fig8():
    r"""In-plane loads of :func:`.tstiff2d_1stiff_flutter` as edge loads
    of a static case, ``run_static_case=True``

    The panel of the function's Examples, with the boundary conditions of
    Castro and Donadon (2017), Fig. 8: `u = v = w = 0` at the edges of the
    skin and at the ends of the flange. The test verifies that without
    loads `\sqrt{-\lambda_1}` is that of the Examples; that the edge loads
    ``Nxx`` and ``Nxy`` of the skin and of the flange, which act on these
    restrained displacements, leave it unchanged (``rtol=1e-10``); that
    ``Nyy`` is not used; and that the edge loads of the base, whose edges
    are free, reach the restrained skin through the connection between
    base and skin and change it by less than 1e-4, relatively, for 100 N/m
    (regression value of ``Nxx_base`` stored in the test, 487.4215 rad/s;
    with null-space connections the shear load of the base leaves it
    unchanged to the digits printed).

    """
    assy, w0 = _flutter_omega1(run_static_case=True)
    assert np.isclose(w0, W0, rtol=1e-6)
    restrained = dict(Nxx_skin=-100., Nxx_flange=-100.,
                      Nxy_skin=-100., Nxy_flange=-100.)
    assert np.isclose(_flutter_omega1(run_static_case=True, **restrained)[1],
                      w0, rtol=1e-10)
    nyy = {'Nyy_%s' % p: -100. for p in PARTS}
    assert np.isclose(_flutter_omega1(run_static_case=True, **nyy)[1], w0,
                      rtol=1e-10)
    w = _flutter_omega1(run_static_case=True, Nxx_base=-100.)[1]
    assert np.isclose(w, 487.4215, rtol=1e-6)
    w = _flutter_omega1(run_static_case=True, Nxy_base=-100.)[1]
    assert abs(w - w0)/w0 < 1e-4


if __name__ == '__main__':
    test_tstiff2d_1stiff_freq()
    test_tstiff2d_1stiff_compression()
    test_tstiff2d_1stiff_flutter()
