r"""Linear buckling of a blade-stiffened cylinder made of multiple domains

The cylinder is cylinder 4 of Card and Jones (1966), Table I: an aluminium
cylinder with 60 integral blade stringers inside the skin, of the geometry
given there, built with :func:`.create_cylinder_blade_stiffened` (skin
panels of the default Sanders kinematics, blades of the model
``'plate_clpt_gl'``). The tests verify:

- the smeared-stringer stability equation of Block, Card and Mikulas
  (1965), Eq. (15), implemented in the model module
  ``smeared_stringer_cylinder.py``, against the 16 buckling loads of their
  Table I(c) and against the four simply supported theoretical loads
  `\bar{N}_s` of the integrally stiffened cylinders of Card and Jones
  (1966), Table I;
- the discrete model, with the simply supported ends of the smeared theory
  (`w = v = N_x = M_x = 0`) and the membrane pre-buckling state of equal
  axial strains in the skin and in the blades, against the smeared theory
  for the same stringer section, within the discrete-versus-smeared
  difference explained in the Notes, against `\bar{N}_s` of Card and Jones
  and against a regression value; with blades of ``'plate_clpt_donnell'``
  the load is about twice the smeared one;
- the analysis functions :func:`.cylinder_blade_stiffened_compression_lb_Nxx_cte`,
  :func:`.cylinder_blade_stiffened_compression_lb_Nxx_from_static` (slow)
  and :func:`.cylinder_blade_stiffened_compression_lb_pd_from_static`
  (slow), whose ends restrain the axial displacement, for the same cylinder,
  against regression values, their ratios to the smeared theory being
  given in the Notes; with prescribed displacements, the end `x` =
  ``height`` of every domain reaches `u` = ``pd`` exactly in the
  pre-buckling state.

Notes
-----
**Smeared theory.** Block, Card and Mikulas [1]_ smear the stringers over
their spacing, treat them as beams that rotate with the skin, and use the
Donnell shell; their Eq. (15) (p. 12) is minimized over the integer numbers
of waves for classical simple supports, `w = v = N_x = M_x = 0` at both ends
(Eq. (14), p. 11). Table I(c) (p. 26) gives `\bar{N}_x/(E \bar{t})` of
longitudinally stiffened cylinders with the Z stringers of Fig. 2(b)
(p. 28), `R = 200` in, `t = 0.10` in, `d = 2.5` in, `\mu = 0.32` (p. 16),
read as outer dimensions (flanges `0.90 \times 0.10` in, overall depth 2.00
in) with `\bar{t} = A_s/d + t` (p. 16): the 16 values and their numbers of
waves are reproduced to the last printed digit. Card and Jones [2]_
computed `\bar{N}_s` with the same equation (p. 9) and the torsion constant
of their p. 3, for the stringer of their Fig. 2(b) (p. 6), 0.097 in wide and
0.33 in deep including the skin: the four values of their Table I (p. 7,
lb/in although headed kips/in.) are reproduced within 0.2 %.

**Discrete model.** The blade is a plate strip from the middle surface of
the skin to the stringer tip, `h_b = 0.33 - t/2` in, 0.097 in thick, and the
smeared reference is Eq. (15) for this rectangle standing on the middle
surface (``blade_stringer``), so that both count the same material and the
same twisting stiffness, that of the Kirchhoff plate strip, `G h_b
t_b^3/3`: 771 lb/in, 6.8 % above `\bar{N}_s` of Card and Jones (742 lb/in
with their torsion constant of a rectangle, 723 lb/in with their section).
The simply supported ends are represented exactly by half of the cylinder, `0
\le x \le L/2`, with `u` free at `x = 0` and the symmetry conditions `u =
v_{,x} = w_{,x} = 0` at `x = L/2`, which contain the critical mode of one
axial half-wave; the axial load is a constant `N_{xx}` in every domain,
`-E t \varepsilon` in the skin and `-E t_b \varepsilon` in the blades, so
that `\bar{N}_x = \varepsilon (E t + E A_s/d)`. With the blades of
``'plate_clpt_gl'`` the discrete load is 6.6 % below the smeared one and
0.25 % below `\bar{N}_s` of Card and Jones. The difference from the smeared
theory has two parts. The skin panels have the Sanders kinematics, the
smeared theory the Donnell one: with ``'cylshell_clpt_donnell'`` panels the
discrete load is 2.5 % higher, 4.3 % below the smeared load; for the
cylinders 1 to 3 of the same table, with the same model, the differences
are -3.8, -7.8 and -2.7 % with the Sanders skin and -0.7, -4.4 and -1.1 %
with the Donnell skin (cylinders 1 and 3 have external stringers; runs
made when this test was written). The remainder comes from the
discreteness of the blades, plate strips that deform in their
cross-section, whereas the smeared theory treats the stringer as a beam
that rotates with the skin and averages it over the circumference. The
test accepts a discrete load between 10 % below the smeared value, beyond
the largest difference of the four cylinders, and the smeared value, and
within 5 % of
`\bar{N}_s` of Card and Jones, whose section differs from the blade by the
overlap with the skin and the torsion constant. With ``'plate_clpt_donnell'`` blades the geometric
stiffness lacks the work of the blade load on the in-plane (radial)
bending of the blade, `N_{xx} v_{,x}^2`, see the Notes of
:func:`.create_cylinder_blade_stiffened`, and the load is about twice the
smeared one.

**Analysis functions.** They keep the boundary conditions of
:func:`.create_cylinder_blade_stiffened`, `u = v = w = 0` at `x = 0` and
`v = w = 0` at `x = L`: `u = 0` at `x = 0` restrains the axial
displacement of the buckling mode, which the smeared solution leaves free
(Eq. (13c), p. 11, gives `N_x = 0` or `u = 0`), and the prescribed
displacements also impose a uniform `u` at `x = L`. Their loads are above
the smeared one: 1.153 times with a constant `N_{xx}`, 1.163 times with the
pre-buckling state of the static analysis and 1.546 times with the
prescribed end-shortening.

References
----------
.. [1] Block, D. L., Card, M. F., and Mikulas, M. M., Jr., "Buckling of
   Eccentrically Stiffened Orthotropic Cylinders", NASA TN D-2960, 1965.
   https://ntrs.nasa.gov/citations/19650021133
.. [2] Card, M. F., and Jones, R. M., "Experimental and Theoretical Results
   for Buckling of Eccentrically Stiffened Cylinders", NASA TN D-3639,
   1966.

"""
import sys
sys.path.append('../..')

import pytest
import numpy as np
from structsolve import lb

from panels.multidomain import (create_cylinder_blade_stiffened,
        cylinder_blade_stiffened_compression_lb_Nxx_cte,
        cylinder_blade_stiffened_compression_lb_Nxx_from_static,
        cylinder_blade_stiffened_compression_lb_pd_from_static)

from smeared_stringer_cylinder import (blade_stringer, isotropic_skin,
        nx_cr, nx_cr_isotropic_eq18, rectangle_stringer)


IN = 0.0254
PSI = 6894.757293168
LBF = 4.4482216152605
# Card and Jones (1966), Table I, cylinder 4, and Fig. 2(b), in SI units
E = 10.5e6*PSI
NU = 0.32
R = 9.55*IN
L = 23.75*IN
T = 0.0280*IN
NSTIFF = 60
TB = 0.097*IN
HB = 0.33*IN - T/2
PLYT = 0.001*IN
NS_CARD = 722.  # lb/in
# regression values, N/m
REF = dict(half_gl=126130.9, Nxx_cte=155764.9, Nxx_from_static=157020.9,
           pd_from_static=208841.1)


def smeared_reference():
    skin = isotropic_skin(E, NU, T)
    st = blade_stringer(E, NU, HB, TB, internal=True)
    return nx_cr(skin, st, 2*np.pi*R/NSTIFF, R, L)


def nbar(eps):
    return eps*(E*T + E*TB*HB/(2*np.pi*R/NSTIFF))


def test_smeared_block1965_table1c():
    E_, nu, t, R_, d = 10.e6, 0.32, 0.10, 200., 2.5
    G = E_/(2*(1 + nu))
    tf = 0.10
    A = 2*0.90*tf + (2.00 - 2*tf)*tf
    I = 2*(0.90*tf**3/12 + 0.90*tf*(1.00 - tf/2)**2) + tf*(2.00 - 2*tf)**3/12
    J = (2*0.90 + 2.00 - 2*tf)*tf**3/3
    zc = t/2 + 1.00
    tbar = A/d + t
    skin = isotropic_skin(E_, nu, t)
    table = {
        'external': [(0.15, 0.007043, 1, 0), (0.20, 0.004739, 1, 8),
                     (0.25, 0.003614, 1, 18), (0.30, 0.002814, 1, 25),
                     (0.35, 0.002243, 1, 27), (0.40, 0.001843, 1, 27),
                     (0.45, 0.001554, 1, 26), (0.50, 0.001340, 1, 25)],
        'internal': [(0.15, 0.006747, 1, 21), (0.20, 0.003804, 1, 20),
                     (0.25, 0.002443, 1, 19), (0.30, 0.001706, 1, 18),
                     (0.35, 0.001262, 1, 18), (0.40, 0.000975, 1, 18),
                     (0.45, 0.000780, 1, 18), (0.50, 0.000641, 1, 17)],
        }
    for side, rows in table.items():
        z = zc if side == 'external' else -zc
        st = dict(EA=E_*A, EI0=E_*(I + A*zc**2), GJ=G*J, z=z)
        for l_R, ref, m_ref, n_ref in rows:
            Nx, m, n = nx_cr(skin, st, d, R_, l_R*R_)
            assert abs(Nx/(E_*tbar) - ref) <= 1.e-6, (side, l_R)
            assert (m, n) == (m_ref, n_ref)
            assert np.isclose(nx_cr_isotropic_eq18(E_, nu, t, st, d, R_,
                l_R*R_, m, n), Nx, rtol=1e-12)


def test_smeared_card1966_ns():
    R_ = 9.55
    d = 2*np.pi*R_/NSTIFF
    cylinders = [('external', 0.0283, 38.00, 1145.),
                 ('internal', 0.0277, 38.00, 687.),
                 ('external', 0.0275, 23.75, 1326.),
                 ('internal', 0.0280, 23.75, 722.)]
    for side, t, L_, Ns in cylinders:
        st = rectangle_stringer(10.5e6, NU, 0.33 - t, 0.097, t,
                                internal=(side == 'internal'))
        Nx, m, n = nx_cr(isotropic_skin(10.5e6, NU, t), st, d, R_, L_)
        assert abs(Nx/Ns - 1) < 0.002, (side, t, L_)


def _half_cylinder(blade_model):
    md, conns = create_cylinder_blade_stiffened(height=L/2, r=R,
            stack=[0]*28, stack_blades=[[0]*97]*NSTIFF,
            width_blades=[HB]*NSTIFF, plyt=PLYT, laminaprop=(E, NU),
            npanels=NSTIFF, m=8, n=6)
    for p in md.panels:
        # x = 0: simply supported, u free; x = L/2: symmetry
        p.x1u = 1
        p.x2u = 0
        p.x2v = 1
        p.x2vr = 0
        p.x2w = 1
        p.x2wr = 0
        if 'blade' in p.group:
            p.model = blade_model
            p.Nxx = -E*TB
        else:
            p.Nxx = -E*T
    k0 = md.reduce(md.calc_kC(conns, silent=True))
    kG = md.reduce(md.calc_kG(silent=True))
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
                          num_eigvalues=5, num_eigvalues_print=5)
    return nbar(eigvals[0])


def test_discrete_vs_smeared():
    Ns, m, n = smeared_reference()
    assert (m, n) == (1, 6)
    N = _half_cylinder('plate_clpt_gl')
    print('discrete %.1f N/m, smeared %.1f N/m, %+.2f%%; %.1f lb/in, Card '
          'N_s %+.2f%%' % (N, Ns, 100*(N/Ns - 1), N/(LBF/IN),
                           100*(N/(LBF/IN)/NS_CARD - 1)))
    assert -0.10 < N/Ns - 1 < 0.
    assert abs(N/(LBF/IN)/NS_CARD - 1) < 0.05
    assert np.isclose(N, REF['half_gl'], rtol=1e-4)
    N_donnell = _half_cylinder('plate_clpt_donnell')
    print('discrete, plate_clpt_donnell blades %.1f N/m, %.3f x smeared'
          % (N_donnell, N_donnell/Ns))
    assert 1.8 < N_donnell/Ns < 2.2


def _common():
    return dict(height=L, r=R, plyt=PLYT, stack=[0]*28,
                stack_blades=[[0]*97]*NSTIFF, width_blades=[HB]*NSTIFF,
                laminaprop=(E, NU), npanels=NSTIFF, m=8, n=6,
                num_eigvalues=5)


def test_cylinder_blade_stiffened_compression_lb_Nxx_cte():
    Ns = smeared_reference()[0]
    assy, eigvals, eigvecs = cylinder_blade_stiffened_compression_lb_Nxx_cte(
        Nxxs_skin=[-E*T]*NSTIFF, Nxxs_blade=[-E*TB]*NSTIFF, **_common())
    N = nbar(eigvals[0])
    print('Nxx_cte: %.1f N/m, %.4f x smeared' % (N, N/Ns))
    assert np.isclose(N, REF['Nxx_cte'], rtol=1e-4)


@pytest.mark.slow
def test_cylinder_blade_stiffened_compression_lb_Nxx_from_static():
    Ns = smeared_reference()[0]
    assy, c, eigvals, eigvecs = \
        cylinder_blade_stiffened_compression_lb_Nxx_from_static(
            Nxxs_skin=[-E*T]*NSTIFF, Nxxs_blade=[-E*TB]*NSTIFF, **_common())
    N = nbar(eigvals[0])
    print('Nxx_from_static: %.1f N/m, %.4f x smeared' % (N, N/Ns))
    assert np.isclose(N, REF['Nxx_from_static'], rtol=1e-4)


@pytest.mark.slow
def test_cylinder_blade_stiffened_compression_lb_pd_from_static():
    Ns = smeared_reference()[0]
    pds = [-1.e-3*L]*NSTIFF
    assy, c, eigvals, eigvecs = \
        cylinder_blade_stiffened_compression_lb_pd_from_static(
            pds_skin=pds, pds_blade=pds, **_common())
    # the prescribed displacements are imposed exactly
    for p in assy.panels:
        mesh, f = p.uvw(c[p.col_start:p.col_end],
                        xs=np.full(5, p.a), ys=np.linspace(0, p.b, 5))
        assert np.allclose(f['u'], pds[0], rtol=1e-8, atol=0)
    # eigvals multiply the prescribed end-shortening, a strain of 1e-3
    N = nbar(1.e-3*eigvals[0])
    print('pd_from_static: %.1f N/m, %.4f x smeared' % (N, N/Ns))
    assert np.isclose(N, REF['pd_from_static'], rtol=1e-4)
