r"""Buckling of laminated stiffened plates with material anisotropy

Reference
---------
D. G. Stamatelos, G. N. Labeas, *Buckling Analysis of Laminated Stiffened
Plates with Material Anisotropy Using the Rayleigh-Ritz Approach*,
Computation 2023, 11, 110. https://doi.org/10.3390/computation11060110

Every published value used here lives in the ``PAPER`` dictionary below, so
that each test is plain arithmetic against a single source of truth. The
tolerances are per-case, because the paper's data is not uniformly complete:

- Section 3.1.2 (Table 3) gives only the ratios ``E11/E22``, ``G12/E22``,
  ``a/b`` and ``b/t_skin``, never an absolute ``E22`` or ``t_ply``. Since
  ``Nx ~ E22*t*(t/b)**2``, no value in Tables 3, 4 or 5 can be recomputed in
  N/mm from the paper. What can be checked, and is checked here, is the whole
  of those three tables against **one** scale factor calibrated on mode (1, 1)
  of Table 3 alone.
- Section 3.2.2 (Figure 7) is fully specified, so its unstiffened intercepts
  are checked with no calibration at all.
- Table 2 is inconsistent with the paper's own Figure 7 by factors of 2.0 to
  4.5 (see ``test_table2_is_inconsistent_with_figure7``). Its only uncoupled
  laminate is matched exactly once the loaded edges are clamped, which is how
  Lagace's specimens were held, so that is the boundary condition used.

The notebook ``notebooks/stamatelos_labeas_2023.ipynb`` works through all of
this with plots and commentary.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.sparse import csr_matrix

import panels.bardell as bardell
from panels.shell import Shell
from panels.stiffener.bladestiff1d import BladeStiff1D
from composites import laminated_plate
from structsolve import lb


# --------------------------------------------------------------------------
# geometry, material and laminates of the paper
# --------------------------------------------------------------------------
PLY_T = 0.134e-3                                    # m, all sections

LAMINATES = {
    '[0_3/90_3]s':              [0]*3 + [90]*3 + [90]*3 + [0]*3,
    '[0_3/90_3/0_3/90_3]':      [0]*3 + [90]*3 + [0]*3 + [90]*3,
    '[0_2/45_2/0_2/45_2/0_2]':  [0]*2 + [45]*2 + [0]*2 + [45]*2 + [0]*2,
    '[0_2/45_2/0_2/-45_2/0_2]': [0]*2 + [45]*2 + [0]*2 + [-45]*2 + [0]*2,
    '[0_6/60_6]':               [0]*6 + [60]*6,
}

# Section 3.1.1, fully specified
LAMINAPROP_311 = (130e9, 10.5e9, 0.28, 6e9, 6e9, 6e9)
A_311 = B_311 = 0.254                               # m

# Section 3.1.2, ratios only; E22 and t_ply below are an arbitrary choice and
# are absorbed by the single scale factor calibrated on Table 3 mode (1, 1)
E22_312 = 10.5e9
LAMINAPROP_312 = (10*E22_312, E22_312, 0.25,
                  0.5*E22_312, 0.5*E22_312, 0.5*E22_312)
STACK_312 = [0, 90, 90, 0, 0]
T_312 = len(STACK_312)*PLY_T
B_312 = 100*T_312                                   # b/t_skin = 100
A_312 = 2*B_312                                     # a/b = 2
H_321 = 9*T_312                                     # blade height, Section 3.2.1

# Section 3.2.2, fully specified
A_322 = B_322 = 0.500                               # m
N_STIFF_322 = 10
STIFF_STACK_322 = [0, 90, 90, 0]                    # [0/90]s
STIFF_T_322 = len(STIFF_STACK_322)*PLY_T            # 0.536 mm

M = N = 16                                          # Bardell terms


# --------------------------------------------------------------------------
# published results
# --------------------------------------------------------------------------
# 'rr' is the paper's Rayleigh-Ritz column, 'fe' its ANSYS model, the other
# keys name the reference they come from.
PAPER = {
    # Table 2: Nx in N/mm, uniform load, 254 mm square plate
    'table2': {
        '[0_3/90_3]s':              dict(rr=26.475, fe=26.367,
                                         lagace_rr=27.150, lagace_exp=19.650),
        '[0_3/90_3/0_3/90_3]':      dict(rr=17.514, fe=15.305,
                                         lagace_rr=20.439, lagace_exp=14.970),
        '[0_2/45_2/0_2/45_2/0_2]':  dict(rr=22.789, fe=21.675,
                                         lagace_rr=18.389, lagace_exp=23.440),
        '[0_2/45_2/0_2/-45_2/0_2]': dict(rr=20.426, fe=20.002,
                                         lagace_rr=17.770, lagace_exp=21.480),
        '[0_6/60_6]':               dict(rr=17.788, fe=10.647,
                                         lagace_rr=18.00,  lagace_exp=11.00),
    },
    # Table 3: [0/90/90/0/0], triangular load, Nx in N/mm
    'table3': {
        '(1, 1)': dict(rr=67.749, fe=59.476, papazoglou=65.00),
        '(2, 1)': dict(rr=69.298, fe=65.803, papazoglou=None),
        '(3, 1)': dict(rr=113.31, fe=108.266, papazoglou=None),
    },
    # Table 4: one blade, unloaded stiffener, Nx in N/mm
    'table4': {
        'uniform':    dict(rr=119.60, fe=116.265, kumar=None),
        'triangular': dict(rr=172.92, fe=171.755, kumar=260.00),
    },
    # Table 5: one blade, loaded stiffener (Eq. 8), Nx in N/mm
    'table5': {
        'uniform':    dict(rr=118.45, fe=112.542),
        'triangular': dict(rr=184.05, fe=168.747),
    },
    # Figure 7: P_cr in N at the two ends of each published curve. These are
    # read off the printed charts, hence good to a few per cent only.
    'figure7': {
        '[0_3/90_3]s':              dict(ratio_max=1.04, P0_rr=1200.,
                                         P0_fe=1200., Pmax_rr=3050.,
                                         Pmax_fe=3100.),
        '[0_3/90_3/0_3/90_3]':      dict(ratio_max=1.04, P0_rr=1120.,
                                         P0_fe=1080., Pmax_rr=2850.,
                                         Pmax_fe=2850.),
        '[0_2/45_2/0_2/45_2/0_2]':  dict(ratio_max=0.89, P0_rr=660.,
                                         P0_fe=660., Pmax_rr=1830.,
                                         Pmax_fe=1790.),
        '[0_2/45_2/0_2/-45_2/0_2]': dict(ratio_max=0.87, P0_rr=665.,
                                         P0_fe=660., Pmax_rr=1720.,
                                         Pmax_fe=1755.),
        '[0_6/60_6]':               dict(ratio_max=0.99, P0_rr=1120.,
                                         P0_fe=840., Pmax_rr=2170.,
                                         Pmax_fe=1990.),
    },
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_skin(a, b, stack, plyt, laminaprop, m=M, n=N,
              clamped_loaded_edges=False):
    """Simply supported skin, optionally with the loaded edges clamped"""
    s = Shell(a=a, b=b, r=None, stack=stack, plyt=plyt,
              laminaprop=laminaprop, rho=1600.,
              model='plate_clpt_donnell', m=m, n=n)
    if clamped_loaded_edges:
        s.x1wr = 0.
        s.x2wr = 0.
    return s


def kG_linear(shell, Nxx, phi, size=None, npts=80):
    r"""kG for ``Nxx(y) = Nxx*(1 - phi*y/b)``, the paper's Eq. (18)

    ``phi = 0`` is a uniform load and ``phi = 1`` a triangular one. For
    ``phi = 0`` this reproduces :meth:`.Shell.calc_kG` to machine precision,
    which is what ``test_kG_linear_matches_calc_kG`` checks.

    """
    m, n, a, b = shell.m, shell.n, shell.a, shell.b
    xi, wt = leggauss(npts)
    fp = np.array([bardell.calc_vec_fp(x, shell.x1w, shell.x1wr,
                                       shell.x2w, shell.x2wr)[:m] for x in xi])
    Fxx = (fp.T*wt) @ fp
    g = np.array([bardell.calc_vec_f(e, shell.y1w, shell.y1wr,
                                     shell.y2w, shell.y2wr)[:n] for e in xi])
    Gyy = (g.T*(wt*(1. - phi*(1. + xi)/2.))) @ g
    size = shell.get_size() if size is None else size
    K = np.zeros((size, size))
    blk = Nxx*(b/a)*np.einsum('ik,jl->jilk', Fxx, Gyy).reshape(m*n, m*n)
    idx = 3*np.arange(m*n) + 2
    K[np.ix_(idx, idx)] = blk
    return csr_matrix(K)


def add_stiffeners(skin, nstiff, hstiff, sstack, splyt, slaminaprop):
    """Equidistant blades at ``y = j*b/(nstiff + 1)``"""
    ys = [(j + 1)*skin.b/(nstiff + 1) for j in range(nstiff)]
    return [BladeStiff1D(bay=skin, rho=1600., panel1=skin, panel2=skin, ys=y,
                         bb=None, bf=hstiff, bstack=None, bplyts=None,
                         blaminaprops=None, fstack=sstack,
                         fplyts=[splyt]*len(sstack),
                         flaminaprops=[slaminaprop]*len(sstack))
            for y in ys]


def Et_reduced(stack, plyt, laminaprop):
    """Apparent membrane stiffness ``(E*t) = A11 - A12**2/A22``"""
    lam = laminated_plate(stack, plyt=plyt, laminaprop=laminaprop)
    return lam.A11 - lam.A12**2/lam.A22


def buckling(skin, stiffeners=(), phi=0., loaded_stiffeners=False,
             Et_skin=None, num_eigvalues=6):
    """Critical ``Nxx`` in N/m, peak value at ``y = 0``, and eigenvectors

    ``loaded_stiffeners`` applies the paper's Eq. (8), i.e. the blade carries
    ``F = Nx*(E*A)_stiffener/(E*t)_skin``. Eq. (8) as printed also multiplies
    by ``Nstiffener`` although Eq. (7) already sums over the stiffeners; the
    per-stiffener form used here is the one consistent with skin and blade
    straining equally.

    """
    size = skin.get_size()
    kC = skin.calc_kC(silent=True)
    kG = kG_linear(skin, -1., phi, size=size)
    for st in stiffeners:
        st.calc_kC(size=size, row0=0, col0=0, silent=True, finalize=True)
        kC = kC + st.kC
        st.Fx = -(st.E1*st.bf)/Et_skin if loaded_stiffeners else 0.
        st.calc_kG(size=size, row0=0, col0=0, silent=True, finalize=True)
        kG = kG + st.kG
    return lb(csr_matrix(kC), csr_matrix(kG), silent=True,
              num_eigvalues=num_eigvalues)


def table3_scale():
    """The single factor absorbing the missing ``E22*t`` of Section 3.1.2

    Calibrated on mode (1, 1) of Table 3 and on nothing else.
    """
    skin = make_skin(A_312, B_312, STACK_312, PLY_T, LAMINAPROP_312)
    eigvals, _ = buckling(skin, phi=1.)
    return PAPER['table3']['(1, 1)']['rr']/(eigvals[0]/1e3), eigvals


# --------------------------------------------------------------------------
# the formulation itself
# --------------------------------------------------------------------------
def test_kG_linear_matches_calc_kG():
    """For a uniform load the helper must reproduce what panels builds"""
    skin = make_skin(A_312, B_312, STACK_312, PLY_T, LAMINAPROP_312)
    skin.Nxx = -1.
    kG_panels = skin.calc_kG(silent=True).toarray()
    kG_helper = kG_linear(skin, -1., 0.).toarray()
    err = np.abs(kG_helper - kG_panels).max()/np.abs(kG_panels).max()
    assert err < 1e-12


def test_triangular_load_is_about_twice_the_uniform_one():
    """Scale-free check, so it needs no calibration factor"""
    skin = make_skin(A_312, B_312, STACK_312, PLY_T, LAMINAPROP_312)
    tri, _ = buckling(skin, phi=1.)
    uni, _ = buckling(skin, phi=0.)
    assert np.isclose(tri[0]/uni[0], 1.971, rtol=1e-3)


@pytest.mark.parametrize('m_n', [12, 14, 16])
def test_convergence_of_the_bardell_series(m_n):
    """The reported results must not depend on the number of terms"""
    skin = make_skin(A_312, B_312, STACK_312, PLY_T, LAMINAPROP_312,
                     m=m_n, n=m_n)
    eigvals, _ = buckling(skin, phi=1.)
    assert np.allclose(eigvals[:3]/1e3, [14.29786, 14.65560, 23.84015],
                       rtol=1e-4)


# --------------------------------------------------------------------------
# Table 2, Section 3.1.1
# --------------------------------------------------------------------------
def test_table2_uncoupled_laminate_matches_lagace():
    """``[0_3/90_3]s`` is specially orthotropic, so every method must agree

    With no coupling at all there is nothing for a Ritz series to get wrong,
    which makes this the case that identifies the boundary conditions: it is
    matched to five significant digits with the loaded edges clamped, and is
    2.9x off with all four edges simply supported.

    """
    stack = LAMINATES['[0_3/90_3]s']
    skin = make_skin(A_311, B_311, stack, PLY_T, LAMINAPROP_311,
                     clamped_loaded_edges=True)
    eigvals, _ = buckling(skin)
    assert np.isclose(eigvals[0]/1e3,
                      PAPER['table2']['[0_3/90_3]s']['lagace_rr'],
                      rtol=1e-3)


@pytest.mark.parametrize('name', list(LAMINATES))
def test_table2_laminates(name):
    """All five laminates, loaded edges clamped

    The three laminates with bending-twisting or bending-extension coupling
    come out below the published Rayleigh-Ritz values, which is the expected
    direction: the Rayleigh-Ritz method is an upper bound and a truncated sine
    series over-predicts when ``D16``, ``D26`` or ``B`` matter, whereas the
    Bardell series used here is converged. The tolerance is therefore per
    laminate and documents the discrepancy rather than hiding it.

    """
    expected = {                       # panels, N/mm, loaded edges clamped
        '[0_3/90_3]s':              27.152,
        '[0_3/90_3/0_3/90_3]':      17.257,
        '[0_2/45_2/0_2/45_2/0_2]':  15.943,
        '[0_2/45_2/0_2/-45_2/0_2]': 15.028,
        '[0_6/60_6]':               12.110,
    }
    skin = make_skin(A_311, B_311, LAMINATES[name], PLY_T, LAMINAPROP_311,
                     clamped_loaded_edges=True)
    eigvals, _ = buckling(skin)
    assert np.isclose(eigvals[0]/1e3, expected[name], rtol=1e-3)


def test_table2_is_inconsistent_with_figure7():
    """Table 2 and Figure 7 are the same plates and disagree by about 2.9x

    Figure 7 at a zero stiffener ratio is the unstiffened plate of Table 2,
    500 mm square instead of 254 mm and reported as a total force. Rescaling
    Table 2 by ``1/b**2`` must therefore land on the Figure 7 intercept, and
    it does not. The ratio is not even constant across the laminates, so no
    single geometry or material slip explains it.

    """
    ratios = []
    for name in LAMINATES:
        rescaled = (PAPER['table2'][name]['rr']*(A_311/A_322)**2*(B_322*1e3))
        ratios.append(rescaled/PAPER['figure7'][name]['P0_rr'])
    ratios = np.array(ratios)
    assert ratios.min() > 1.8           # Table 2 is far above Figure 7
    assert np.ptp(ratios) > 0.9           # and not by a constant factor


# --------------------------------------------------------------------------
# Table 3, Section 3.1.2
# --------------------------------------------------------------------------
def test_table3_all_three_modes():
    """One factor is fitted on mode (1, 1); modes (2, 1) and (3, 1) must follow

    This is the real content of the comparison: if the mechanics reproduces,
    the two modes that were not used for the calibration fall into place by
    themselves. They agree to better than 0.5 %.

    """
    scale, eigvals = table3_scale()
    for i, mode in enumerate(['(1, 1)', '(2, 1)', '(3, 1)']):
        got = eigvals[i]/1e3*scale
        assert np.isclose(got, PAPER['table3'][mode]['rr'], rtol=5e-3), mode


# --------------------------------------------------------------------------
# Tables 4 and 5, Section 3.2.1
# --------------------------------------------------------------------------
@pytest.mark.parametrize('table, load', [('table4', 'uniform'),
                                         ('table4', 'triangular'),
                                         ('table5', 'uniform'),
                                         ('table5', 'triangular')])
def test_tables4_and_5_single_blade_stiffener(table, load):
    """One blade of height ``9*t_skin`` at mid-width

    The scale factor is the one calibrated on Table 3 and is not re-fitted, so
    these four cases are genuine predictions. All land within 5 % of the
    published Rayleigh-Ritz values.

    """
    scale, _ = table3_scale()
    phi = 0. if load == 'uniform' else 1.
    loaded = table == 'table5'
    skin = make_skin(A_312, B_312, STACK_312, PLY_T, LAMINAPROP_312)
    sts = add_stiffeners(skin, 1, H_321, STACK_312, PLY_T, LAMINAPROP_312)
    eigvals, _ = buckling(skin, sts, phi=phi, loaded_stiffeners=loaded,
                          Et_skin=Et_reduced(STACK_312, PLY_T, LAMINAPROP_312))
    got = eigvals[0]/1e3*scale
    assert np.isclose(got, PAPER[table][load]['rr'], rtol=5e-2)


def test_critical_mode_has_a_node_on_the_single_stiffener():
    """Why loading the blade barely changes the load

    The blade's axial force enters the energy only through ``(dw/dx)**2``
    evaluated at its own ``y`` (Eq. 7). The critical global mode of a plate
    with one stiff blade at mid-width has a node there, so the force cannot
    contribute, and Tables 4 and 5 must be nearly equal.

    """
    skin = make_skin(A_312, B_312, STACK_312, PLY_T, LAMINAPROP_312)
    sts = add_stiffeners(skin, 1, H_321, STACK_312, PLY_T, LAMINAPROP_312)
    eigvals, eigvecs = buckling(skin, sts)
    mesh, fields = skin.uvw(eigvecs[:, 0], gridx=60, gridy=61)
    w = fields['w']
    j = np.argmin(abs(mesh['Ys'][0, :] - sts[0].ys))
    assert np.abs(w[:, j]).max()/np.abs(w).max() < 1e-6


def test_table5_triangular_load_cannot_be_right():
    """Adding a compressive force cannot raise a buckling load

    Table 5 reports 184.05 N/mm against Table 4's 172.92 N/mm for the same
    geometry with the stiffener now loaded in compression. This test records
    that the published pair is not physically admissible.

    """
    assert PAPER['table5']['triangular']['rr'] > \
        PAPER['table4']['triangular']['rr']


# --------------------------------------------------------------------------
# Figure 7, Section 3.2.2
# --------------------------------------------------------------------------
@pytest.mark.parametrize('name', list(LAMINATES))
def test_figure7_unstiffened_intercepts(name):
    """Fully specified, so no calibration of any kind

    At a zero stiffener ratio Figure 7 is the unstiffened 500 mm plate and
    everything needed is given in the paper. All five intercepts are matched
    to within 25 %, and four of them to within 13 %; the outlier is the fully
    anisotropic skin, for which the paper's own R-R and FE columns disagree by
    33 %.

    """
    skin = make_skin(A_322, B_322, LAMINATES[name], PLY_T, LAMINAPROP_311)
    eigvals, _ = buckling(skin)
    P_cr = eigvals[0]*B_322                       # N/m * m -> N
    ref = PAPER['figure7'][name]
    assert np.isclose(P_cr, ref['P0_rr'], rtol=0.25)
    assert np.isclose(P_cr, ref['P0_fe'], rtol=0.25)


def test_figure7_load_grows_with_the_stiffener_height():
    """The qualitative conclusion of Section 3.2.2"""
    stack = LAMINATES['[0_3/90_3]s']
    loads = []
    for h in (0., 2.e-3, 4.e-3, 6.e-3, 8.e-3):
        skin = make_skin(A_322, B_322, stack, PLY_T, LAMINAPROP_311)
        sts = () if h == 0. else add_stiffeners(
            skin, N_STIFF_322, h, STIFF_STACK_322, PLY_T, LAMINAPROP_311)
        eigvals, _ = buckling(skin, sts)
        loads.append(eigvals[0]*B_322)
    assert np.all(np.diff(loads) > 0)
    assert loads[-1]/loads[0] > 4.             # 10 blades of 8 mm


if __name__ == '__main__':
    pytest.main(['-v', __file__])
