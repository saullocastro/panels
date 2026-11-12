r"""Linear buckling of the clamped stringer-stiffened shell AS-2 of Singer,
Arbocz and Babcock (1971)

The shell AS-2 of Singer, Arbocz and Babcock (1971), Table 1: an aluminium
cylinder with 80 flat integral stringers outside the skin, built with
:func:`.create_cylinder_blade_stiffened` with ``side='external'`` (skin
panels of the default Sanders kinematics, one blade of the model
``'plate_clpt_gl'`` per stringer), with
clamped ends, `u = v = w = w_{,x} = 0` (their C-4), and the membrane
pre-buckling state of equal axial strains in the skin and in the blades. The
test verifies the critical load against the C-4 buckling load of their Table
2 for the membrane pre-buckling state, 183.8 lb/in, within 3 %, and against
a regression value. The twisting stiffness of the blades is that of the
rectangular stringer section (see the Notes).

Notes
-----
Singer, Arbocz and Babcock [1]_ give the stringers of AS-2 as `A/(d t) =
0.506`, `-e/t = 1.72` (outside) and `I/(d t^3) = 0.2466` (Table 1, p. 71),
with `t = 7.74 \times 10^{-3}` in, `R/t = 517`, `L = 5.50` in, 80 stringers,
`E = 10 \times 10^6` psi and `\nu = 0.3`. A rectangle of height `h_s =
\sqrt{12 I/A}` and width `t_s = A/h_s` (with `d = 2 \pi R/80`) reproduces
`e = (t + h_s)/2` within 0.6 %; the blade is this rectangle, standing on
the outer surface of the skin, to which
:func:`.create_cylinder_blade_stiffened` ties its root, of thickness `t_s`
and width `H_s = h_s`. The torsion constant of the stringers is not
printed. The Kirchhoff plate strip has the twisting stiffness `4 D_{66} H_s
= G H_s t_s^3/3`, 15 times
`G J_r`, with `J_r` the Saint-Venant constant of the rectangle `t_s \times
h_s`; `D_{66}` of the blades is scaled to `G J_r` through the attribute
``ABDnxny``, which also gives the blades their thickness `t_s` (the plies of
the assembly have the skin thickness). With ``side='external'`` the
assembly connects each blade along its edge `y = H_s`, which stands it
outside, as in the shell.

The C-4 values of [1]_ come from a smeared-stiffener analysis (Table 2;
the program is not named). The smeared theory of Block, Card and Mikulas
[2]_ with the clamped Galerkin functions of Card and Jones [3]_ (equation
(A5)) and `J = J_r` gives 184.3 lb/in (+0.3 %), with `J = 0` 164.4 lb/in
(-10.6 %); the discrete model, about 2.2 % below the smeared theory for the
same blade, gives with the 10 by 4 terms per domain of this test 180.3 lb/in
(-1.9 %) with `J = J_r` and 161.9 lb/in with `J = 0`, and with 14 by 6
terms 180.0 lb/in (-2.1 %) (runs made when this test was written). The 3 %
tolerance thus accepts the discrete-versus-smeared
difference and rejects a blade without twisting stiffness. Under the simple support SS-3 of [1]_ the discrete load with `J =
J_r` was 7 % above its SS-3 value (123.8 lb/in), which was matched with `J
= 0`, with the blade extended to the middle surface of the skin, `H_s =
h_s + t/2` (runs made when this test was written): no single `J` reproduces both columns of [1]_, so the clamped case,
where the authors judged the test shells to be (p. 72), is the one tested.
The full cylinder and the whole length are modelled: the critical mode has
two axial half-waves and 15 circumferential waves.

References
----------
.. [1] Singer, J., Arbocz, J., and Babcock, C. D., Jr., "Buckling of
   Imperfect Stiffened Cylindrical Shells under Axial Compression", AIAA
   Journal, Vol. 9, No. 1, 1971, pp. 68-75. https://doi.org/10.2514/3.6125
.. [2] Block, D. L., Card, M. F., and Mikulas, M. M., Jr., "Buckling of
   Eccentrically Stiffened Orthotropic Cylinders", NASA TN D-2960, 1965.
   https://ntrs.nasa.gov/citations/19650021133
.. [3] Card, M. F., and Jones, R. M., "Experimental and Theoretical Results
   for Buckling of Eccentrically Stiffened Cylinders", NASA TN D-3639,
   1966. https://ntrs.nasa.gov/citations/19660027866

"""
import sys
sys.path.append('../..')

import numpy as np
from structsolve import lb
from composites import isotropic_plate

from panels.multidomain import create_cylinder_blade_stiffened


# shell AS-2, Table 1 of [1]_, in, lb, psi
E = 10.e6
NU = 0.3
T = 7.74e-3
R = 517*T
L = 5.50
NSTIFF = 80
C4_MEMBRANE = 183.8  # lb/in, Table 2 of [1]_
REF = 180.3085  # regression value, lb/in (m = 10, n = 4)


def section():
    d = 2*np.pi*R/NSTIFF
    A = 0.506*d*T
    I = 0.2466*d*T**3
    hs = np.sqrt(12*I/A)
    ts = A/hs
    a, c = ts, hs
    Jr = c**3*a*(1/3 - 64/np.pi**5*c/a*np.tanh(np.pi*a/(2*c)))
    return d, ts, hs, Jr


def clamped_membrane(m, n, J=True):
    d, ts, hs, Jr = section()
    Hs = hs
    md, conns, info = create_cylinder_blade_stiffened(height=L, r=R,
            stack=[0], stack_blades=[[0]]*NSTIFF, width_blades=[Hs]*NSTIFF,
            plyt=T, laminaprop=(E, NU), npanels=NSTIFF, m=m, n=n,
            side='external')
    ABD = np.array(isotropic_plate(thickness=ts, E=E, nu=NU).ABD, dtype=float)
    ABD[5, 5] *= Jr/(Hs*ts**3/3) if J else 0.
    for p in md.panels:
        # u = v = w = 0 at both ends (create_cylinder_blade_stiffened) and
        # w,x = 0 for the skin; for a blade, v (radial) and w
        # (circumferential) with their derivatives along x
        if p.group == 'skin':
            p.x1wr = p.x2wr = 0
            p.Nxx = -E*T
        else:
            p.x1vr = p.x2vr = 0
            p.x1wr = p.x2wr = 0
            p.ABDnxny = ABD
            p.Nxx = -E*ts
    k0 = md.reduce(md.calc_kC(conns, silent=True))
    kG = md.reduce(md.calc_kG(silent=True))
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
                          num_eigvalues=5, num_eigvalues_print=5)
    return eigvals[0]*(E*T + E*ts*Hs/d)


def test_singer1971_as2_clamped():
    N = clamped_membrane(m=10, n=4)
    print('AS-2, C-4: %.2f lb/in, %+.2f %% from Singer et al.'
          % (N, 100*(N/C4_MEMBRANE - 1)))
    assert abs(N/C4_MEMBRANE - 1) < 0.03
    assert np.isclose(N, REF, rtol=1e-4)
