import numpy as np

from panels.panel import cylinder_blade_stiffened_compression_lb_Nxx_cte, cylinder_blade_stiffened_compression_lb_Nxx_from_static


def test_cylinder_blade_stiffened_compression_lb_Nxx_cte():
    print('Testing assembly function: cylinder_blade_stiffened_compression_lb_Nxx_cte')
    height = 0.500
    r = 0.250
    nshells = 5
    Nxxs = [-100.]*nshells
    assy, eigvals, eigvecs = cylinder_blade_stiffened_compression_lb_Nxx_cte(
        height=height,
        r=r,
        plyt=0.125e-3,
        stack=[0, 45, -45, 90, -45, 45],
        stack_blades=[[0, 90, 0]*4]*nshells,
        width_blades=[0.02]*nshells,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        nshells=nshells,
        Nxxs_skin=Nxxs,
        Nxxs_blade=Nxxs,
        m=8, n=12)

    assy.plot(eigvecs[:, 0], 'skin', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_cte.png')

    assert np.isclose(Nxxs[0]*eigvals[0], -56569.62172, atol=0.01, rtol=0.001)


def test_cylinder_blade_stiffened_compression_lb_Nxx_from_static():
    print('Testing assembly function: cylinder_blade_stiffened_compression_lb_Nxx_from_static')
    height = 0.500
    r = 0.250
    nshells = 5
    Nxxs = [-100.]*nshells
    assy, c, eigvals, eigvecs = cylinder_blade_stiffened_compression_lb_Nxx_from_static(
        height=height,
        r=r,
        plyt=0.125e-3,
        stack=[0, 45, -45, 90, -45, 45],
        stack_blades=[[0, 90, 0]*4]*nshells,
        width_blades=[0.02]*nshells,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        nshells=nshells,
        Nxxs_skin=Nxxs,
        Nxxs_blade=Nxxs,
        m=8, n=12)

    assy.plot(c, 'skin', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_from_static_c.png')
    assy.plot(eigvecs[:, 0], 'skin', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_from_static_eigvec.png')

    assert np.isclose(Nxxs[0]*eigvals[0], -40835.14008, atol=0.01, rtol=0.001)


if __name__ == '__main__':
    test_cylinder_blade_stiffened_compression_lb_Nxx_from_static()
