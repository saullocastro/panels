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
  for the same stringer section (Eq. (15) of Block, Card and Mikulas for
  the blade as modelled, 771.2 lb/in), within the discrete-versus-smeared
  difference explained in the Notes, against `\bar{N}_s` of Card and Jones
  (722 lb/in; Eq. (15) for the stringer section drawn by Card and Jones
  gives 722.8 lb/in) and against a regression value (720.2 lb/in, 6.6 %
  below 771.2 lb/in and 0.4 % below 722.8 lb/in); with blades of
  ``'plate_clpt_donnell'``
  the load is about twice the smeared one;
- the analysis functions :func:`.cylinder_blade_stiffened_compression_lb_Nxx_cte`,
  :func:`.cylinder_blade_stiffened_compression_lb_Nxx_from_static` (slow)
  and :func:`.cylinder_blade_stiffened_compression_lb_pd_from_static`
  (slow), whose ends restrain the axial displacement, for the same cylinder,
  against regression values, their ratios to the smeared theory being
  given in the Notes; with prescribed displacements, the end `x` =
  ``height`` of every domain reaches `u` = ``pd`` exactly in the
  pre-buckling state;
- the skin ``'cylshell_clpt_tl'`` and the blades ``'plate_clpt_tl'``
  (argument ``models``) against the GL models: equal loads with the
  constant membrane state, within ``1e-6`` with the static pre-buckling
  state (slow); the models of the groups and the errors of ``models``.

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
made when this test was written). The remainder comes from the blades.
Their axial load acts, through the geometric stiffness of
``'plate_clpt_gl'``, also on their out-of-plane displacement, the
circumferential displacement of the stringer section, whereas the load
term of the smeared theory, Eq. (9) of [1]_ (p. 8), acts on the radial
displacement only; and they twist as plate strips tied to the skin, whereas
Eq. (6) of [1]_ (p. 7) adds the approximate twisting energy `G_s J_s
w_{,xy}^2/d` of a stringer that twists with the shell. For the whole length
of this cylinder with the Donnell skin (simply supported ends of the blades
holding only their radial displacement), removing the first term from the
geometric stiffness of the blades raises the discrete load by 3.7 %, to
1.8 % below the smeared one, and removing also the twisting stiffness of
the blades gives Eq. (15) with `J = 0` within 0.1 % (runs made when these
Notes were written).
Almroth [3]_ shows for unstiffened cylinders (Donnell equations) that, with
the membrane pre-buckling state, simple supports with `v = 0` and either
`u = 0` or `N_x = 0` give a load equal to or insignificantly higher than the
classical one (p. 138), and only edges free circumferentially, `N_{xy} =
0`, lower it, to about one half (Table 2, p. 137); the discrete model and
the smeared theory share `w = v = N_x = M_x = 0` and the membrane state, so
the edge conditions of the skin do not contribute to their difference. The
paper treats no stiffeners and gives no value for the part due to the
discreteness. The test accepts a discrete load between 10 % below the smeared value, beyond
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
.. [3] Almroth, B. O., "Influence of Edge Conditions on the Stability of
   Axially Compressed Cylindrical Shells", AIAA Journal, Vol. 4, No. 1,
   1966, pp. 134-140. https://doi.org/10.2514/3.3396

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
# cylinder 4 of Table I and the stringer of Fig. 2(b) of [2]_, in SI units
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
REF = dict(half_gl=126130.9, half_gl_cyl3_external=235662.7,
           Nxx_cte=155764.9, Nxx_from_static=157020.9,
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


def _half_cylinder(blade_model, model='cylshell_clpt_sanders', m=8, n=6,
                   side='internal', t=T, nply=28, nply_blade=97, plyt=PLYT):
    hb = 0.33*IN - t/2
    md, conns, info = create_cylinder_blade_stiffened(height=L/2, r=R,
            stack=[0]*nply, stack_blades=[[0]*nply_blade]*NSTIFF,
            width_blades=[hb]*NSTIFF, plyt=plyt, laminaprop=(E, NU),
            npanels=NSTIFF, m=m, n=n,
            models=dict(skin=model, blade=blade_model), side=side)
    for p in md.panels:
        # x = 0: simply supported, u free; x = L/2: symmetry
        p.x1u = 1
        p.x2u = 0
        p.x2v = 1
        p.x2vr = 0
        p.x2w = 1
        p.x2wr = 0
        if 'blade' in p.group:
            p.Nxx = -E*TB
        else:
            p.Nxx = -E*t
    k0 = md.reduce(md.calc_kC(conns, silent=True))
    kG = md.reduce(md.calc_kG(silent=True))
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
                          num_eigvalues=5, num_eigvalues_print=5)
    return eigvals[0]*(E*t + E*TB*hb/(2*np.pi*R/NSTIFF))


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


def test_discrete_gl_skin():
    r"""Cylinder 4 of Card and Jones (1966) with skin panels of
    ``'cylshell_clpt_gl'``, otherwise the set-up of
    ``test_discrete_vs_smeared``

    The load is compared with `\bar{N}_s` of Card and Jones (1966), Table I,
    and with the discrete load of the Sanders skin, ``REF['half_gl']``. It
    must be below the latter, and by at most 2.5 %.

    Notes
    -----
    Both skin models have the same linear stiffness. The geometric
    stiffness of ``'cylshell_clpt_gl'`` adds to the Sanders one the work
    `\frac{1}{2} N_{xx} (u_{,x}^2 + v_{,x}^2)` of the skin load, which is
    destabilizing under compression, so that the load cannot be higher
    than with the Sanders skin. Its size is estimated from the critical mode,
    one axial half-wave and `n = 6` circumferential waves: an inextensional
    mode, `v_{,y} + w/r = 0`, has `v_{,x}^2 / w_{,x}^2 \approx 1/n^2 = 2.8`
    %, and the skin carries `t/(t + A_s/d) = 48` % of the load, so the load
    should drop by about `0.48 \times 2.8 = 1.3` %. The computed drop is
    1.33 % (124455.2 against 126130.9 N/m, the same for 6 x 6 and 10 x 8
    terms), and the GL load is 1.6 % below `\bar{N}_s`. The bounds, 0 and
    -2.5 %, are the sign of the term and about twice the estimate; the
    agreement with `\bar{N}_s` is checked within the 5 % of
    ``test_discrete_vs_smeared``, and the load against a regression value
    with ``rtol=1e-4``.

    """
    N = _half_cylinder('plate_clpt_gl', model='cylshell_clpt_gl')
    print('discrete, GL skin %.1f N/m, %+.2f%% vs Sanders skin; %.1f lb/in, '
          'Card N_s %+.2f%%' % (N, 100*(N/REF['half_gl'] - 1), N/(LBF/IN),
                                100*(N/(LBF/IN)/NS_CARD - 1)))
    assert -0.025 < N/REF['half_gl'] - 1 < 0.
    assert abs(N/(LBF/IN)/NS_CARD - 1) < 0.05
    assert np.isclose(N, 124455.2, rtol=1e-4)


def test_discrete_tl_skin():
    r"""Cylinder 4 of Card and Jones (1966) with skin panels of
    ``'cylshell_clpt_tl'``, with blades of ``'plate_clpt_gl'`` and of
    ``'plate_clpt_tl'``, otherwise the set-up of ``test_discrete_vs_smeared``

    Both loads must equal the load of ``test_discrete_gl_skin``, skin
    ``'cylshell_clpt_gl'`` and blades ``'plate_clpt_gl'``, to rounding
    (``rtol=1e-12``), and so lie within 5 % of `\bar{N}_s` of Card and
    Jones (1966), Table I, and at the regression value ``124455.2`` N/m.

    Notes
    -----
    The TL models have the membrane strains of the GL models and changes
    of curvature measured with the deformed normal, whose linear part is
    that of Sanders for the cylinder (van der Heijden [1]_, Eq. (3.16.8),
    p. 202) and that of von Kármán for the plate. The geometric stiffness of
    the constant membrane state `N_{xx}` used here is the second variation
    of the membrane strains times `N_{xx}`, the same for both models: the
    curvatures of the TL models enter it only through the bending moments,
    zero in this state. The linear stiffness is the same too, so the
    buckling loads must agree to rounding.

    References
    ----------
    .. [1] van der Heijden, A. M. A. (ed.), "W. T. Koiter's Elastic
       Stability of Solids and Structures", Cambridge University Press, 2008.
       https://doi.org/10.1017/CBO9780511546174

    """
    N_gl = _half_cylinder('plate_clpt_gl', model='cylshell_clpt_gl')
    for blade_model in ('plate_clpt_gl', 'plate_clpt_tl'):
        N = _half_cylinder(blade_model, model='cylshell_clpt_tl')
        print('discrete, TL skin, %s blades %.6f N/m, %+.1e vs GL'
              % (blade_model, N, N/N_gl - 1))
        assert np.isclose(N, N_gl, rtol=1e-12)
        assert abs(N/(LBF/IN)/NS_CARD - 1) < 0.05
        assert np.isclose(N, 124455.2, rtol=1e-4)


@pytest.mark.slow
def test_tl_skin_Nxx_from_static():
    r"""Cylinder 4 of Card and Jones (1966) with the pre-buckling state of
    a linear static analysis,
    :func:`.cylinder_blade_stiffened_compression_lb_Nxx_from_static`, with
    the skin ``'cylshell_clpt_tl'`` against ``'cylshell_clpt_gl'``, and
    with the blades ``'plate_clpt_tl'`` too: the loads must agree within
    ``1e-6`` (relative), and the GL skin is checked against the regression
    value ``155419.52`` N/m.

    Notes
    -----
    The static pre-buckling state bends the skin and the blades near the
    ends, where `u = 0` at `x = 0` and the load is applied at `x = L`.
    The numerical geometric stiffness of the TL models contains, besides
    the terms of the GL models, the bending moments of this state times the
    second variation of the quadratic part of the changes of curvature. The
    moments are confined to the edge zones, and the critical mode has one
    axial half-wave, so the difference is expected far below the
    differences between the kinematic models (1.3 % between the GL and
    Sanders skins); it is -6.6e-8 for both blade models, and the bound is
    ``1e-6``.

    """
    loads = {}
    for models in (dict(skin='cylshell_clpt_gl'),
                   dict(skin='cylshell_clpt_tl'),
                   dict(skin='cylshell_clpt_tl', blade='plate_clpt_tl')):
        assy, c, eigvals, eigvecs = \
            cylinder_blade_stiffened_compression_lb_Nxx_from_static(
                Nxxs_skin=[-E*T]*NSTIFF, Nxxs_blade=[-E*TB]*NSTIFF,
                models=models, **_common())
        loads[tuple(sorted(models.values()))] = nbar(eigvals[0])
    N_gl = loads[('cylshell_clpt_gl', )]
    assert np.isclose(N_gl, 155419.52, rtol=1e-6)
    for key, N in loads.items():
        print('Nxx_from_static %s: %.6f N/m, %+.1e vs GL skin'
              % (key, N, N/N_gl - 1))
        assert abs(N/N_gl - 1) < 1e-6


def test_models_argument():
    # the models of the groups, by default and from models, and the errors
    md, conns, info = create_cylinder_blade_stiffened(height=L/2, r=R,
            stack=[0]*28, stack_blades=[[0]*97]*4, width_blades=[HB]*4,
            plyt=PLYT, laminaprop=(E, NU), npanels=4, m=4, n=4)
    assert {(p.group[:4], p.model) for p in md.panels} == {
        ('skin', 'cylshell_clpt_sanders'), ('blad', 'plate_clpt_gl')}
    assert info['models'] == dict(skin='cylshell_clpt_sanders',
                                  blade='plate_clpt_gl')
    md, conns, info = create_cylinder_blade_stiffened(height=L/2, r=R,
            stack=[0]*28, stack_blades=[[0]*97]*4, width_blades=[HB]*4,
            plyt=PLYT, laminaprop=(E, NU), npanels=4, m=4, n=4,
            models=dict(skin='cylshell_clpt_tl'))
    assert {(p.group[:4], p.model) for p in md.panels} == {
        ('skin', 'cylshell_clpt_tl'), ('blad', 'plate_clpt_gl')}
    assert info['models'] == dict(skin='cylshell_clpt_tl',
                                  blade='plate_clpt_gl')
    for models in (dict(stringer='plate_clpt_gl'),
                   dict(skin='cylshell_fsdt_sanders'),
                   dict(blade='plate_fsdt_donnell'),
                   dict(skin='plate_clpt_gl'),
                   dict(blade='cylshell_clpt_sanders')):
        with pytest.raises(ValueError):
            create_cylinder_blade_stiffened(height=L/2, r=R, stack=[0]*28,
                    stack_blades=[[0]*97]*4, width_blades=[HB]*4,
                    plyt=PLYT, laminaprop=(E, NU), npanels=4,
                    models=models)


def test_return_info():
    # the assembly, the connections and the dictionary info; the unpacking
    # into two names fails, as does an unknown side
    kw = dict(height=L/2, r=R, stack=[0]*28, stack_blades=[[0]*97]*4,
              width_blades=[HB]*4, plyt=PLYT, laminaprop=(E, NU), npanels=4,
              m=4, n=4)
    md, conns, info = create_cylinder_blade_stiffened(**kw)
    assert info['skin'] == md.panels[:4]
    assert info['blades'] == md.panels[4:]
    assert np.isclose(info['ws'], 2*np.pi*R/4)
    assert len(conns) == 8
    assert md.conn is None
    with pytest.raises(ValueError):
        md, conns = create_cylinder_blade_stiffened(**kw)
    with pytest.raises(ValueError):
        create_cylinder_blade_stiffened(side='outside', **kw)


def test_side():
    r"""Blades inside and outside the skin, argument ``side``: the
    connections, and the buckling load of cylinder 3 of Card and Jones
    (1966), Table I, whose blades are outside, against the smeared theory,
    against `\bar{N}_s` of Card and Jones and against the same cylinder
    with the blades inside

    The set-up is that of ``test_discrete_vs_smeared`` (half of the
    cylinder, simply supported, constant membrane pre-buckling state). With
    ``side='external'`` every ``'BFycte'`` connection joins the edge `y = 0`
    of the skin panel to the edge `y` = ``width_blades[i]`` of the blade,
    with ``side='internal'`` to its edge `y = 0`. With the blades outside
    the load must lie between 10 % below the smeared value of Eq. (15) of
    Block, Card and Mikulas (1965) for the blade as modelled and that
    value, within 5 % of `\bar{N}_s` of Card and Jones (1966), 1326 lb/in,
    and at a regression value (``rtol=1e-4``); with the blades inside, the
    same cylinder must give a lower load, within the same bounds of its own
    smeared value.

    Notes
    -----
    Cylinder 3 of Card and Jones [1]_ (Table I, p. 7): `t = 0.0275` in, `L
    = 23.75` in, `R = 9.55` in, 60 integral blade stringers outside, 0.097
    in wide and 0.33 in deep including the skin; the blade is modelled from
    the middle surface of the skin, `h_b = 0.33 - t/2` in. In the smeared
    theory of Block, Card and Mikulas [2]_ outside stringers raise the axial
    buckling load of such cylinders: their Table I(c) (p. 26) gives higher
    loads with the stringers outside than inside for every length, and
    Card and Jones give higher `\bar{N}_s` for their external cylinders 1
    and 3 than for the internal cylinders 2 and 4. Eq. (15) for the blade
    as modelled gives 1382.8 lb/in outside and 754.4 lb/in inside; the
    discrete loads are 1345.7 lb/in (-2.7 %; +1.5 % from `\bar{N}_s`) and
    704.2 lb/in (-6.7 %) (runs made when this test was written).

    References
    ----------
    .. [1] Card, M. F., and Jones, R. M., "Experimental and Theoretical
       Results for Buckling of Eccentrically Stiffened Cylinders", NASA TN
       D-3639, 1966.
    .. [2] Block, D. L., Card, M. F., and Mikulas, M. M., Jr., "Buckling of
       Eccentrically Stiffened Orthotropic Cylinders", NASA TN D-2960, 1965.
       https://ntrs.nasa.gov/citations/19650021133

    """
    for side, ycte2 in (('internal', 0.), ('external', HB)):
        md, conns, info = create_cylinder_blade_stiffened(height=L/2, r=R,
                stack=[0]*28, stack_blades=[[0]*97]*4, width_blades=[HB]*4,
                plyt=PLYT, laminaprop=(E, NU), npanels=4, m=4, n=4,
                side=side)
        bf = [c for c in conns if c['func'] == 'BFycte']
        assert len(bf) == 4
        assert all(c['ycte1'] == 0 and c['ycte2'] == ycte2 for c in bf)
    t3 = 0.0275*IN
    hb3 = 0.33*IN - t3/2
    skin = isotropic_skin(E, NU, t3)
    loads = {}
    for side in ('internal', 'external'):
        st = blade_stringer(E, NU, hb3, TB, internal=(side == 'internal'))
        Ns = nx_cr(skin, st, 2*np.pi*R/NSTIFF, R, L)[0]
        # plies of 0.0005 in: 55 in the skin, 194 in a blade
        N = _half_cylinder('plate_clpt_gl', side=side, t=t3, nply=55,
                           nply_blade=194, plyt=0.0005*IN)
        loads[side] = N
        print('cylinder 3, blades %s: discrete %.1f N/m (%.1f lb/in), '
              'smeared %.1f lb/in, %+.2f%%'
              % (side, N, N/(LBF/IN), Ns/(LBF/IN), 100*(N/Ns - 1)))
        assert -0.10 < N/Ns - 1 < 0.
    assert abs(loads['external']/(LBF/IN)/1326. - 1) < 0.05
    assert loads['internal'] < loads['external']
    assert np.isclose(loads['external'], REF['half_gl_cyl3_external'],
                      rtol=1e-4)
