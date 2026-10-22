import sys
sys.path.append('../..')

import numpy as np

from panels.multidomain import (cylinder_compression_lb_Nxx_from_static,
        cylinder_blade_stiffened_compression_lb_Nxx_from_static,
        cylinder_blade_stiffened_compression_lb_pd_from_static)


NPANELS = 3
COMMON = dict(height=0.5, r=0.25, plyt=0.125e-3,
              stack=[0, 45, -45, 90, -45, 45],
              laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
              npanels=NPANELS, m=4, n=4, num_eigvalues=1,
              conn_method='penalty')
BLADES = dict(stack_blades=[[0, 90, 0]*4]*NPANELS,
              width_blades=[0.02]*NPANELS)


def _values_by_group(md, values_skin, values_blade):
    """The value expected for each panel of ``md``, in the order of the
    loop over the panels of the assembly functions"""
    i_skin = i_blade = -1
    out = []
    for p in md.panels:
        if 'skin' in p.group:
            i_skin += 1
            out.append(values_skin[i_skin])
        elif 'blade' in p.group:
            i_blade += 1
            out.append(values_blade[i_blade])
    return out


def test_cylinder_Nxx_from_static_one_load_per_panel():
    """Each panel receives its own Nxx, not the last one of the list"""
    Nxxs = [-100., -200., -300.]
    md, c, eigvals, eigvecs = cylinder_compression_lb_Nxx_from_static(
        Nxxs=Nxxs, **COMMON)
    for p, Nxx in zip(md.panels, Nxxs):
        assert len(p.distr_loads) == 1
        x, y, funcx, funcy, funcz = p.distr_loads[0]
        assert x == p.a
        assert funcx(0.) == Nxx
        assert funcx(p.b) == Nxx


def test_cylinder_blade_stiffened_Nxx_from_static_one_load_per_panel():
    Nxxs_skin = [-100., -200., -300.]
    Nxxs_blade = [-10., -20., -30.]
    md, c, eigvals, eigvecs = \
        cylinder_blade_stiffened_compression_lb_Nxx_from_static(
            Nxxs_skin=Nxxs_skin, Nxxs_blade=Nxxs_blade, **BLADES, **COMMON)
    expected = _values_by_group(md, Nxxs_skin, Nxxs_blade)
    assert len(expected) == 2*NPANELS
    for p, Nxx in zip(md.panels, expected):
        assert len(p.distr_loads) == 1
        assert p.distr_loads[0][2](0.) == Nxx


def test_cylinder_blade_stiffened_pd_from_static_one_pd_per_panel():
    pds_skin = [-1.e-4, -2.e-4, -3.e-4]
    pds_blade = [-1.e-5, -2.e-5, -3.e-5]
    ku = 1.e6
    md, c, eigvals, eigvecs = \
        cylinder_blade_stiffened_compression_lb_pd_from_static(
            pds_skin=pds_skin, pds_blade=pds_blade, ku=ku, **BLADES,
            **COMMON)
    expected = _values_by_group(md, pds_skin, pds_blade)
    assert len(expected) == 2*NPANELS
    for p, pd in zip(md.panels, expected):
        assert len(p.distr_pds) == 1
        # the prescribed displacement and the force k_u u_p of the penalty
        assert p.distr_pds_values[0][2](0.) == pd
        assert np.isclose(p.distr_pds[0][2](0.), ku*pd, rtol=1e-15)
