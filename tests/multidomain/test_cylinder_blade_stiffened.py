import sys
sys.path.append('../..')

import numpy as np

from panels.multidomain import (cylinder_blade_stiffened_compression_lb_Nxx_cte,
        cylinder_blade_stiffened_compression_lb_Nxx_from_static,
        cylinder_blade_stiffened_compression_lb_pd_from_static)


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

    assy.plot(eigvecs[:, 0], 'skin', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_cte_eigvec.png')

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
        m=8, n=12)

    assy.plot(c, 'skin', vec='u', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_from_static_u.png', colorbar=True)
    assy.plot(c, 'skin', vec='w', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_from_static_w.png', colorbar=True)
    assy.plot(c, 'skin', vec='Nxx', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_from_static_Nxx.png', colorbar=True)
    assy.plot(c, 'skin', vec='Nyy', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_from_static_Nyy.png', colorbar=True)
    assy.plot(eigvecs[:, 0], 'skin', filename='tmp_cylinder_blade_stiffened_compression_lb_Nxx_from_static_eigvec.png')

    assert np.isclose(Nxxs[0]*eigvals[0], -40003, atol=0.01, rtol=0.001)


def test_cylinder_blade_stiffened_compression_lb_pd_from_static():
    print('Testing assembly function: cylinder_blade_stiffened_compression_lb_pd_from_static')
    height = 0.500
    r = 0.250
    npanels = 5
    pds = [-1.e-3]*npanels
    assy, c, eigvals, eigvecs = cylinder_blade_stiffened_compression_lb_pd_from_static(
        height=height,
        r=r,
        plyt=0.125e-3,
        stack=[0, 45, -45, 90, -45, 45],
        stack_blades=[[0, 90, 0]*4]*npanels,
        width_blades=[0.02]*npanels,
        laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
        npanels=npanels,
        pds_skin=pds,
        pds_blade=pds,
        m=8, n=12)

    assy.plot(c, 'skin', vec='u', filename='tmp_cylinder_blade_stiffened_compression_lb_pd_from_static_u.png', colorbar=True)
    assy.plot(c, 'skin', vec='w', filename='tmp_cylinder_blade_stiffened_compression_lb_pd_from_static_w.png', colorbar=True)
    assy.plot(c, 'skin', vec='Nxx', filename='tmp_cylinder_blade_stiffened_compression_lb_pd_from_static_Nxx.png', colorbar=True)
    assy.plot(c, 'skin', vec='Nyy', filename='tmp_cylinder_blade_stiffened_compression_lb_pd_from_static_Nyy.png', colorbar=True)
    assy.plot(eigvecs[:, 0], 'skin', filename='tmp_cylinder_blade_stiffened_compression_lb_pd_from_static_eigvec.png')

    #assert np.isclose(Nxxs[0]*eigvals[0], -51905.843, atol=0.01, rtol=0.001)
