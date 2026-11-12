r"""Z-stiffened cylinders of Peterson and Dow (1959)

The Z-section stringers of Peterson and Dow (1959), riveted inside 7075-T6
aluminium alloy cylinders of 23.74 in radius 1.25 in apart (Fig. 2), and
the cylinders built with :func:`.create_cylinder_z_stiffened`, each
stringer assembled from three plates (the attached flange in the laminate
of the strip of skin under it, the web and the free flange,
``plate_clpt_gl``), skin panels ``cylshell_clpt_sanders``.

:func:`test_z_section` checks the middle-line dimensions of the plates and
the area of the section against Fig. 2(b), and
:func:`test_create_cylinder_z_stiffened` the assembly (domains, groups,
models, connections, also stored in ``md.conn``, the side and the
orientation of the stringers, the laminate of the strips, the errors).
:func:`test_stack_z_bent_sheet` and :func:`test_stack_z_mirror` check, on a
small laminated cylinder, the laminate of each plate of the profile given
as one bent sheet or as three stacks: against the matrices of the sheet
seen from either face, and by the mirror symmetry of the buckling loads.

Notes
-----
**Data** (Peterson and Dow [1]_). `E = 10\,900` ksi, `\mu = 0.32` (p. 4).
Fig. 2(b), p. 12: sheet 0.040 in, attached flange 0.38 in and free flange
0.22 in from the outer face of the web, depth 0.54 in from the skin to the
outer face of the free flange. Fig. 2(a): spacing 1.25 in, stringers
inside. The plates have square corners (the bends, of 0.04 in inner radius,
are not represented): the attached flange `b_f = 0.38 - 0.02 = 0.36` in and
the free flange `b_l = 0.22 - 0.02 = 0.20` in from the middle surface of
the web, and the web, whose root
:func:`.create_cylinder_z_stiffened` ties to the face of the attached
flange, from that face to the middle surface of the free flange, `H_w =
0.54 - 0.040 - 0.020 = 0.48` in; the section as modelled has the area
`1.04 t_z` per unit thickness, against `1.06 t_z` of the sheet with square
corners, the two corner squares of half the web thickness not being
counted.

**Model.** The free flanges have 4 terms across, the fewest that do not
lock their connection to the webs
(:class:`panels.multidomain.ConnectionTermsWarning`).

References
----------
.. [1] Peterson, J. P., and Dow, M. B., "Compression Tests on Circular
   Cylinders Stiffened Longitudinally by Closely Spaced Z-Section
   Stringers", NASA Memo 2-12-59L, 1959.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from numpy import pi
from structsolve import lb

from panels.multidomain import create_cylinder_z_stiffened


IN = 0.0254
LBF = 4.4482216152605
PSI = LBF/IN**2
E = 10.9e6*PSI
MU = 0.32
R = 23.74*IN
TS = 0.0395*IN
# cylinder: length (in), P_cr (kips)
# regression values: critical n and N (N/m)


def z_midlines(tz, flange, depth, free_flange):
    # widths of the plates, square corners, from the outer dimensions of the
    # sheet: attached flange from the middle surface of the web, web from
    # the face of the attached flange to the middle surface of the free
    # flange, free flange from the middle surface of the web
    return dict(tz=tz, bf=flange - tz/2, Hw=depth - 3*tz/2,
                bl=free_flange - tz/2)


def _sec():
    return {k: v*IN for k, v in z_midlines(0.040, 0.38, 0.54,
                                           0.22).items()}


def _cylinder(L, nstiff, sec, m, n, n_flange, **kwargs):
    return create_cylinder_z_stiffened(height=L, r=R, stack=[0.], plyt=TS,
            laminaprop=(E, MU), nstiff=nstiff, stack_z=[0.],
            plyt_z=sec['tz'], width_flange=sec['bf'], width_web=sec['Hw'],
            width_free_flange=sec['bl'], m=m, n=n, n_flange=n_flange,
            n_web=n_flange, n_free_flange=n_flange, **kwargs)


def test_z_section():
    sec = z_midlines(0.040, 0.38, 0.54, 0.22)
    assert np.isclose(sec['bf'], 0.36)
    assert np.isclose(sec['Hw'], 0.48)
    assert np.isclose(sec['bl'], 0.20)
    A = sec['tz']*(sec['bf'] + sec['Hw'] + sec['bl'])
    assert np.isclose(A, 0.040*(0.38 + 0.54 - 2*0.040 + 0.22 - 0.040/2))


def test_create_cylinder_z_stiffened():
    sec = _sec()
    md, conns, info = _cylinder(19.28*IN, 4, sec, 4, 4, 4)
    assert [len(info[k]) for k in ('skin', 'strips', 'webs',
                                   'free_flanges')] == [8, 4, 4, 4]
    assert md.panels == info['skin'] + info['webs'] + info['free_flanges']
    assert md.conn is conns
    assert np.isclose(sum(p.b for p in info['skin']), 2*pi*R)
    assert np.isclose(info['pitch'], 2*pi*R/4)
    assert info['models'] == dict(skin='cylshell_clpt_sanders',
                                  stringer='plate_clpt_gl')
    assert {p.model for p in md.panels} == {'cylshell_clpt_sanders',
                                            'plate_clpt_gl'}
    assert {p.group for p in md.panels} == {'skin', 'strip', 'web',
                                            'free_flange'}
    assert [c['func'] for c in conns] == ['SSycte']*8 + ['BFycte']*8
    with pytest.raises(ValueError):
        md, conns = _cylinder(19.28*IN, 4, sec, 4, 4, 4)
    # the strip: laminate of the skin and of the flange, offset towards the
    # stringer; the web connected at its edge y = 0 inside, y = H_w outside,
    # at the face of the strip;
    # the free flange towards +y (flange_side=-1) or -y (flange_side=1)
    for side, sign in (('internal', -1), ('external', 1)):
        for flange_side in (-1, 1):
            md, conns, info = _cylinder(19.28*IN, 4, sec, 4, 4, 4, side=side,
                                        flange_side=flange_side)
            strip = info['strips'][0]
            assert np.isclose(strip.lam.h, TS + sec['tz'])
            assert np.isclose(strip.offset, sign*sec['tz']/2)
            web_conn, lip_conn = conns[8:10]
            assert web_conn['p1'] is strip
            assert np.isclose(web_conn['ycte1'],
                              strip.b if flange_side == -1 else 0.)
            assert np.isclose(web_conn['ycte2'],
                              0. if side == 'internal' else sec['Hw'])
            assert np.isclose(lip_conn['ycte1'],
                              sec['Hw'] if side == 'internal' else 0.)
            assert np.isclose(lip_conn['ycte2'],
                              sec['bl'] if flange_side == -1 else 0.)
            # the root of the web at the face of the strip, the free
            # flange at the middle surface of the web
            assert web_conn['face'] == ('lower' if side == 'internal'
                                        else 'upper')
            assert lip_conn.get('face') is None
    for kwargs in (dict(side='inside'), dict(flange_side=0), dict(bc='CF'),
                   dict(models=dict(blade='plate_clpt_gl'))):
        with pytest.raises(ValueError):
            _cylinder(19.28*IN, 4, sec, 4, 4, 4, **kwargs)


# laminated cylinder of the tests of the bent sheet: skin cross-ply, the
# lamina of the Examples of the package
LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
SKIN = [0., 90., 0.]
SHEET = [0., 45., 90.]
PLYT = 0.25e-3


def _laminated(stack_z, side='internal', flange_side=-1, m=6, nstiff=3):
    return create_cylinder_z_stiffened(height=0.3, r=0.25, stack=SKIN,
            plyt=PLYT, laminaprop=LAMINAPROP, nstiff=nstiff, stack_z=stack_z,
            plyt_z=PLYT, width_flange=0.012, width_web=0.02,
            width_free_flange=0.01, side=side, flange_side=flange_side, m=m,
            n=4, n_flange=4, n_web=4, n_free_flange=4)


def _abd(p):
    return np.asarray(p.lam.ABD)


def test_stack_z_bent_sheet():
    r"""A single ``stack_z`` (bent sheet) against three stacks, and the
    laminate of each plate against the sheet seen from its face

    The unsymmetric sheet ``[0, 45, 90]`` gives in each plate the stack of
    ``info['stacks_z']``; the same three stacks given explicitly build the
    same domains (equal matrices ``ABD`` and, for one orientation, equal
    ``calc_kC``). Independently of the reversal of the plies, the matrix
    ``ABD`` of a plate whose `z` axis is opposite to the normal of the
    sheet must be `P\,ABD\,P` of the sheet, `P = \mathrm{diag}(1, 1, -1,
    -1, -1, 1)`, the change of sign of `\gamma_{xy}, \kappa_{xx},
    \kappa_{yy}` and of `N_{xy}, M_{xx}, M_{yy}` when the axes `y` and `z`
    are reversed (`x` common), and equal to that of the sheet otherwise; the
    sheet is a flat laminate of :mod:`composites`. The free flange is
    reversed in every orientation, the web for internal stringers with
    ``flange_side=1`` and external stringers with ``flange_side=-1`` (Notes
    of :func:`.create_cylinder_z_stiffened`). A symmetric cross-ply sheet
    gives in every plate the laminate of the sheet in that plate's own `z`
    order, and ``stack_z`` that is neither one stack nor three non-empty
    stacks raises ``ValueError``.

    """
    from composites import laminated_plate
    abd_sheet = np.asarray(laminated_plate(stack=SHEET, plyt=PLYT,
                                           laminaprop=LAMINAPROP).ABD)
    atol = 1e-12*np.abs(abd_sheet).max()
    P = np.diag([1., 1., -1., -1., -1., 1.])
    flipped = [-a for a in SHEET[::-1]]
    for side in ('internal', 'external'):
        for flange_side in (-1, 1):
            md, conns, info = _laminated(SHEET, side, flange_side, m=4)
            web_flipped = (side, flange_side) in (('internal', 1),
                                                  ('external', -1))
            assert info['stacks_z'] == dict(
                flange=SHEET, web=flipped if web_flipped else SHEET,
                free_flange=flipped)
            web, lip = info['webs'][0], info['free_flanges'][0]
            ref_web = P @ abd_sheet @ P if web_flipped else abd_sheet
            assert np.allclose(_abd(web), ref_web, rtol=1e-12, atol=atol)
            assert np.allclose(_abd(lip), P @ abd_sheet @ P, rtol=1e-12,
                               atol=atol)
            # the same three stacks given explicitly
            st = info['stacks_z']
            md2, conns2, info2 = _laminated(
                [st['flange'], st['web'], st['free_flange']], side,
                flange_side, m=4)
            assert info2['stacks_z'] == st
            for p, p2 in zip(md.panels, md2.panels):
                assert np.array_equal(_abd(p), _abd(p2))
            if side == 'internal' and flange_side == -1:
                k1 = md.calc_kC(silent=True)
                k2 = md2.calc_kC(silent=True)
                assert abs(k1 - k2).max() == 0
    md, conns, info = _laminated([0., 90., 0.], m=4)
    md2, conns2, info2 = _laminated([[0., 90., 0.]]*3, m=4)
    for p, p2 in zip(md.panels, md2.panels):
        assert np.allclose(_abd(p), _abd(p2), rtol=1e-12,
                           atol=1e-12*np.abs(_abd(p2)).max())
    for stack_z in ([], [[0.], [0.]], [[0.], [], [0.]], [[0.], 0., [0.]]):
        with pytest.raises(ValueError):
            _laminated(stack_z, m=4)


def _first_loads(md, num=6):
    # membrane pre-buckling state of a uniform axial strain, the first
    # positive load multipliers
    for p in md.panels:
        p.Nxx = -p.lam.A[0, 0]*1.e-4
    K = md.reduce(md.calc_kC(silent=True))
    KG = md.reduce(md.calc_kG(silent=True))
    eigvals = lb(K, KG, tol=0, sparse_solver=True, silent=True,
                 num_eigvalues=20)[0]
    eigvals = np.sort(eigvals.real[eigvals.real > 0])
    return eigvals[:num]


@pytest.mark.parametrize('side', ['internal', 'external'])
def test_stack_z_mirror(side):
    r"""Mirror symmetry of the bent sheet

    The cylinder with ``flange_side=-1`` and the sheet `[\theta_k]` is the
    mirror image, `y \to -y`, of the cylinder with ``flange_side=1`` and the
    sheet `[-\theta_k]` (the skin, a cross-ply, is its own mirror image), so
    both have the same buckling loads: the bent sheet ``[0, 45, 90]``
    gives them to ``rtol=1e-8`` (first 6 loads). The same sheet given in
    the order of the `z` axis of each plate, ``[stack]*3``, which is not a
    bent sheet, breaks the symmetry: the first loads of the two cylinders
    differ by 0.50 % (internal) and 1.21 % (external) (runs made when this
    test was written); more than 0.3 % is asserted, so that the test
    detects a wrong order of the plies.

    """
    mirrored = [-a for a in SHEET]
    loads = _first_loads(_laminated(SHEET, side, -1)[0])
    loads_m = _first_loads(_laminated(mirrored, side, 1)[0])
    print(side, 'bent sheet', loads, loads_m)
    assert np.allclose(loads, loads_m, rtol=1e-8)
    own = _first_loads(_laminated([SHEET]*3, side, -1)[0])
    own_m = _first_loads(_laminated([mirrored]*3, side, 1)[0])
    print(side, 'own z order', own, own_m, own[0]/own_m[0] - 1)
    assert abs(own[0]/own_m[0] - 1) > 3e-3


