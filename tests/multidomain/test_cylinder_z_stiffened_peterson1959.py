r"""Axial buckling of the Z-stiffened cylinders 1 and 2 of Peterson and Dow (1959)

Cylinders 1 and 2 of Peterson and Dow (1959), Table I (p. 11), 7075-T6
aluminium alloy, 23.74 in radius, with internal Z-section stringers riveted
to the skin 1.25 in apart (Fig. 2), loaded in compression between the
platens of the testing machine, with the ends ground flat. The whole
cylinder is built with the model module ``z_stringer_cylinder.py`` of this
folder: 120 stringers, each assembled from three plates (the attached
flange in the laminate of the strip of skin under it, the web and the free
flange, ``plate_clpt_gl``), skin panels ``cylshell_clpt_sanders``, clamped
ends and the membrane pre-buckling state of a uniform axial strain, with
few terms. The tests verify, for each cylinder:

- the measured buckling load `P_{cr}` of Table I lies between 70 % and
  100 % of the computed one, the range of the tests of the same Z
  stringers on cylinders of 15.92 in radius against the clamped
  smeared-stiffener theory, Card and Jones (1966), Table I (71 to 96 %);
- the critical number of circumferential waves `n` and the load against
  regression values (``rtol=1e-4``).

:func:`test_z_section` checks the dimensions of the plates and the section
of the module against Fig. 2(b), and the assembly (domains, groups).

Notes
-----
**Data** (Peterson and Dow [1]_). Table I, p. 11: cylinder 1, `t_s =
0.0395` in, `l = 25.40` in, `R = 23.74` in, `A = 10.80` in^2, `P_{cr} =
270` kips; cylinder 2, `l = 19.28` in, `P_{cr} = 413` kips (the other
values equal); `P_{cr}` was determined by strain reversal (p. 7). `E =
10\,900` ksi, `\mu = 0.32` (p. 4). Fig. 2(b), p. 12: sheet 0.040 in,
attached flange 0.38 in and free flange 0.22 in from the outer face of the
web, depth 0.54 in from the skin to the outer face of the free flange.
Fig. 2(a): spacing 1.25 in, stringers inside. The number of stringers is
not printed; `2 \pi R/1.25 = 119.3`, and 120 are modelled (spacing 1.243
in). The middle lines of the plates have square corners (the bends, of
0.04 in inner radius, are not represented). Card and Jones [2]_ give no
theoretical value for these two cylinders, so the tests are the reference.

**Model.** The ends ground flat and pressed by the platens are modelled as
clamped, `u = v = w = w_{,x} = 0` on the skin and the plates; the rivets
(3/32 in at 5/16 in, Card and Jones [2]_, p. 6) as a continuous bond of
the attached flange; the pre-buckling state as the membrane state of a
uniform axial strain without Poisson expansion, the classical assumption of
the smeared theory (Block, Card and Mikulas [3]_, p. 8). The free flanges
have 4 terms across, the fewest that do not lock their connection to the
webs (:class:`panels.multidomain.ConnectionTermsWarning`).

**Bounds and wave numbers.** Peterson and Dow (p. 10) found that the
small-deflection orthotropic theories predict the buckling loads of their
six cylinders with errors of about 15 %; for the eight cylinders with these
Z stringers of Card and Jones [2]_ (Table I, p. 7, cylinders 5 to 12, four
of them the specimens 3 to 6 of Peterson and Dow) the tests are at 71 to 96
% of their clamped smeared theory `\bar{N}_c`. With this model the computed
loads are above the tests by 15 to 18 % (runs made when this test was
written; more terms change them by less than 0.4 %), so the ratio
test/model of 0.70 to 1.00 is checked. The model buckles with `n = 12`
(cylinder 1) and `n = 13` (cylinder 2) circumferential waves against 9,
then 10, and 11 in the tests (p. 9). The loads of the wave numbers are
close near the minimum: for cylinder 1, `n = 11, 13` are 1.6 % above the
critical load, `n = 10` 7.2 % and `n = 9` 18.5 %; for cylinder 2, `n = 12,
14` 1.0 and 0.6 %, `n = 11` 4.1 %. The measured wave numbers are not
checked: they were observed on the buckled cylinders, and that of cylinder
1 changed from 9 to 10 as the shortening increased (p. 9). The bfsccylinder
implementation of these two cylinders (one stringer pitch with cyclic
symmetry, the same three plates, a coarse mesh) gives the same critical `n`
and 2128.4 and 3196.1 lb/in, against 2127.6 and 3194.0 lb/in here (runs
made when this test was written).

References
----------
.. [1] Peterson, J. P., and Dow, M. B., "Compression Tests on Circular
   Cylinders Stiffened Longitudinally by Closely Spaced Z-Section
   Stringers", NASA Memo 2-12-59L, 1959.
.. [2] Card, M. F., and Jones, R. M., "Experimental and Theoretical Results
   for Buckling of Eccentrically Stiffened Cylinders", NASA TN D-3639,
   1966.
.. [3] Block, D. L., Card, M. F., and Mikulas, M. M., Jr., "Buckling of
   Eccentrically Stiffened Orthotropic Cylinders", NASA TN D-2960, 1965.
   https://ntrs.nasa.gov/citations/19650021133

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest

from z_stringer_cylinder import (IN, LBF, PSI, build_cylinder_z,
        buckling_membrane, section_properties, z_midlines)


E = 10.9e6*PSI
MU = 0.32
R = 23.74*IN
TS = 0.0395*IN
NS = 120
# cylinder: length (in), P_cr (kips)
PETERSON = {1: (25.40, 270.), 2: (19.28, 413.)}
# regression values: critical n and N (N/m)
REF = {1: (12, 372604.2), 2: (13, 559351.0)}


def _sec():
    return {k: v*IN for k, v in z_midlines(0.040, 0.38, 0.54, 0.22,
                                           h=0.0395).items()}


def test_z_section():
    sec = z_midlines(0.040, 0.38, 0.54, 0.22, h=0.0395)
    assert np.isclose(sec['bf'], 0.36)
    assert np.isclose(sec['Hw'], 0.0395/2 + 0.52)
    assert np.isclose(sec['bl'], 0.20)
    p = section_properties(sec, 0.0395, 10.9e6, MU)
    assert np.isclose(p['A'], 0.040*(0.36 + 0.0395/2 + 0.52 + 0.20))
    sec = z_midlines(0.040, 0.38, 0.54, 0.22, h=0.0395, bend_radius=0.04)
    assert np.isclose(sec['bf'], 0.36 - (2 - np.pi/2)*0.06)
    md, groups = build_cylinder_z(R, TS, 19.28*IN, E, MU, 4, _sec(), mx=4,
                                  ny=4, nf=4, nw=4, nl=4)
    assert [len(groups[k]) for k in ('skin', 'strip', 'web', 'lip')] == [
        8, 4, 4, 4]
    assert np.isclose(sum(p.b for p in groups['skin']), 2*np.pi*R)
    assert {p.model for p in md.panels} == {'cylshell_clpt_sanders',
                                            'plate_clpt_gl'}


@pytest.mark.parametrize('cyl', [1, 2])
def test_peterson1959_cylinder(cyl):
    L, Pcr = PETERSON[cyl]
    N, modes, md, groups, ndof, dt = buckling_membrane(R, TS, L*IN, E, MU,
            NS, _sec(), side='internal', bc='CC', mx=12, ny=6, nf=4, nw=4,
            nl=4, num_eigvalues=24)
    n = modes[0][2]
    Ntest = Pcr*1e3*LBF/(2*np.pi*R)
    print('cylinder %d: n = %d, N = %.1f N/m (%.1f lb/in), P = %.1f kips; '
          'test %.3f of the model; %.0f s'
          % (cyl, n, N, N/(LBF/IN), N*2*np.pi*R/LBF/1e3, Ntest/N, dt))
    assert 0.70 < Ntest/N < 1.
    assert n == REF[cyl][0]
    assert np.isclose(N, REF[cyl][1], rtol=1e-4)
