import numpy as np

from panels.multidomain import (cylinder_blade_stiffened_compression_lb_Nxx_cte,
        cylinder_blade_stiffened_compression_lb_Nxx_from_static)


def test_cylinder_blade_stiffened_compression_lb_Nxx_cte():
    print('Testing assembly function: cylinder_blade_stiffened_compression_lb_Nxx_cte')
    height = 0.500
    r = 0.250
    npanels = 5
    Nxxs = [-100.]*npanels
    assy, eigvals, eigvecs = cylinder_blade_stiffened_compression_lb_Nxx_cte(
        height=height,
        r=r,
        plyt=0.125e-3,
        stack=[0, 45, -45, 90, -45, 45],
        stack_blades=[[0, 90, 0]*4]*npanels,
        width_blades=[0.02]*npanels,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        npanels=npanels,
        Nxxs_skin=Nxxs,
        Nxxs_blade=Nxxs,
        m=8, n=12)

    assy.plot(eigvecs[:, 0], 'skin', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_cte.png')

    assert np.isclose(Nxxs[0]*eigvals[0], -51905.843, atol=0.01, rtol=0.001)


def test_cylinder_blade_stiffened_compression_lb_Nxx_from_static():
    print('Testing assembly function: cylinder_blade_stiffened_compression_lb_Nxx_from_static')
    height = 0.500
    r = 0.250
    npanels = 5
    Nxxs = [-100.]*npanels
    assy, c, eigvals, eigvecs = cylinder_blade_stiffened_compression_lb_Nxx_from_static(
        height=height,
        r=r,
        plyt=0.125e-3,
        stack=[0, 45, -45, 90, -45, 45],
        stack_blades=[[0, 90, 0]*4]*npanels,
        width_blades=[0.02]*npanels,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        npanels=npanels,
        Nxxs_skin=Nxxs,
        Nxxs_blade=Nxxs,
        m=14, n=14)

    assy.plot(c, 'skin', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_from_static_c.png')
    assy.plot(eigvecs[:, 0], 'skin', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_from_static_eigvec.png')

    assert np.isclose(Nxxs[0]*eigvals[0], -51905.843, atol=0.01, rtol=0.001)
