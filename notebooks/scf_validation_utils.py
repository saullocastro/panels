r"""Utilities of the shear correction validation notebooks

The models, settings and reference data are those of the use-case studies of
the paper on the shear correction factors (SCF) of the first-order shear
deformation theory (FSDT), adapted to use :class:`panels.shell.Shell`
directly. The notebooks are organized per reference:

- ``pagano1970_plates.ipynb``: cross-ply and sandwich plates under a
  sinusoidal pressure, 3D elasticity of Pagano (1970), see ``pagano1970.py``;
- ``noor1975_buckling.ipynb``: 3D buckling of cross-ply plates of Noor (1975);
- ``sze2004_hinged_roofs.ipynb``: limit loads of the hinged cylindrical roofs
  of Sze et al. (2004), Riks;
- ``varadan1991_cylinders.ipynb``: cross-ply cylinders of Varadan and Bhaskar
  (1991) under a sinusoidal load;
- ``kardomateas1993b_1995_thick_cylinders.ipynb``: thick cylinders in axial
  compression, Kardomateas (1993b), and under combined external pressure and
  axial compression, Kardomateas and Philobos (1995).

The shear corrections of the FSDT are selected with
``Shell.fsdt_shear_correction``, see :data:`SCF`. The results of panels are
saved in ``results/<name>.json`` and loaded unless ``rerun=True``, see
:func:`run_or_load`.

On machines with a threaded MKL, ``scipy.linalg.eig``, used by
:func:`structsolve.lb` for the unsymmetric problems of the follower pressure,
may crash; set ``MKL_NUM_THREADS=1`` and ``OMP_NUM_THREADS=1`` before
starting Python in that case.
"""
import numpy as np
from composites import laminated_plate
from structsolve import Analysis, lb, solve

from panels.shell import Shell
from follower_utils import fmt, markdown_table, pct, run_or_load  # noqa: F401

#: value of ``Shell.fsdt_shear_correction`` of each shear correction; a float
#: multiplies the uncorrected transverse shear stiffness ``Abar_ts``, a
#: string selects the ``shear_correction`` of ``composites``
SCF = {
    'K1': None,
    'K56': 5/6,
    'KPI': np.pi**2/12,
    'WHI': 'whitney',
    'CHO': 'chow',
    'VLA': 'vlachoutsis',
    'BB': 'birman_bert',
    'ROH': 'rohwer',
    'TSF': 'thickness_shear',
}

#: descriptions of the shear corrections, for the tables
SCF_NAMES = {
    'K1': r'$\kappa = 1$',
    'K56': r'$\kappa = 5/6$',
    'KPI': r'$\kappa = \pi^2/12$',
    'WHI': 'Whitney (1973)',
    'CHO': 'Chow (1971)',
    'VLA': 'Vlachoutsis (1992)',
    'BB': 'Birman and Bert (2002)',
    'ROH': 'Rohwer (1988)',
    'TSF': 'thickness-shear frequency',
}

THEORIES = {
    'plate': dict(clpt='plate_clpt_donnell', fsdt='plate_fsdt_donnell',
                  tsdt='plate_tsdt_donnell'),
    'cylinder': dict(clpt='cylshell_clpt_sanders',
                     fsdt='cylshell_fsdt_sanders',
                     tsdt='cylshell_tsdt_sanders'),
}


# Shear corrections
# -----------------

def scf_applicable(scf_id, stack, plyts, laminaprops, rhos=None):
    r"""``True`` if the shear correction is defined for the laminate

    ``'chow'`` needs a symmetric laminate and ``'thickness_shear'`` positive
    densities.
    """
    value = SCF[scf_id]
    try:
        laminated_plate(stack, plyts=plyts, laminaprops=laminaprops,
                        rhos=rhos, shear_correction=(value if
                        isinstance(value, str) else None))
    except ValueError:
        return False
    return True


def kappas(scf_id, stack, plyts, laminaprops, rhos=None):
    r"""``(kappa_13, kappa_23)`` of a shear correction, as reported by
    composites, `\kappa_{13} = A_{55}/\bar{A}_{55}` and `\kappa_{23} =
    A_{44}/\bar{A}_{44}`"""
    value = SCF[scf_id]
    if isinstance(value, str) or value is None:
        lam = laminated_plate(stack, plyts=plyts, laminaprops=laminaprops,
                              rhos=rhos, shear_correction=value)
        return float(lam.scf_k13), float(lam.scf_k23)
    return value, value


def models(scf_ids, stack=None, plyts=None, laminaprops=None, rhos=None):
    r"""The models ``(label, theory, scf_id)`` of a comparison: ``CLPT``,
    ``FSDT-<ID>`` for the shear corrections ``scf_ids`` that are applicable
    to the laminate, and ``TSDT``"""
    out = [('CLPT', 'clpt', 'ROH')]
    for scf_id in scf_ids:
        if stack is None or scf_applicable(scf_id, stack, plyts, laminaprops,
                                           rhos):
            out.append(('FSDT-' + scf_id, 'fsdt', scf_id))
    out.append(('TSDT', 'tsdt', 'ROH'))
    return out


# Boundary conditions
# -------------------

def set_edge(s, edge, kind):
    r"""Boundary condition of one edge of a plate or cylindrical panel

    ``kind``: ``'S'`` hard simply supported (SS-1: tangential displacement,
    `w` and tangential rotation zero, normal displacement free), ``'s'`` soft
    simply supported (tangential rotation free), ``'D'`` shear diaphragm, the
    same as ``'S'`` (named for the cylinders), ``'C'`` clamped, ``'F'`` free.
    For the CLPT and the TSDT the clamped edge also clamps the normal slope of
    `w`. For the CLPT the soft and hard simple supports coincide.
    """
    x_edge = edge.startswith('x')
    tangential = 'v' if x_edge else 'u'
    rot_tangential = 'phiy' if x_edge else 'phix'
    for field in ('u', 'v', 'w', 'phix', 'phiy'):
        setattr(s, edge + field, 1.)
        setattr(s, edge + field + 'r', 1.)
    if kind in ('S', 'D'):
        setattr(s, edge + tangential, 0.)
        setattr(s, edge + 'w', 0.)
        setattr(s, edge + rot_tangential, 0.)
    elif kind == 's':
        setattr(s, edge + tangential, 0.)
        setattr(s, edge + 'w', 0.)
    elif kind == 'C':
        for field in ('u', 'v', 'w', 'phix', 'phiy'):
            setattr(s, edge + field, 0.)
        # the normal slope of w is an essential variable of the CLPT and of
        # the TSDT, where it enters the in-plane displacements through
        # -4/(3h^2) z^3 (phi + w_,n)
        if 'clpt' in s.model or 'tsdt' in s.model:
            setattr(s, edge + 'wr', 0.)
    elif kind != 'F':
        raise ValueError(kind)


def set_bcs(s, bcs):
    r"""``bcs`` gives the edges ``x1, y1, x2, y2``, e.g. ``'SSSS'``"""
    for edge, kind in zip(('x1', 'y1', 'x2', 'y2'), bcs):
        set_edge(s, edge, kind)


# Shells
# ------

def make_shell(theory, a, b, stack, plyts, laminaprops, rhos=None, r=None,
               scf='ROH', bcs='SSSS', m=12, n=12):
    r"""Plate (``r=None``) or cylindrical panel of radius ``r``

    ``theory`` is ``'clpt'``, ``'fsdt'`` or ``'tsdt'``, with the Donnell
    kinematics for the plates and those of Sanders for the cylinders. ``scf``
    is a key of :data:`SCF`, used only by the FSDT; ``bcs`` gives the edges
    ``x1, y1, x2, y2``, see :func:`set_edge`. The stack, the ply thicknesses
    and the material properties go from the bottom (inner) to the top (outer)
    surface.
    """
    geometry = 'plate' if r is None else 'cylinder'
    s = Shell(a=a, b=b, r=r, m=m, n=n, model=THEORIES[geometry][theory],
              stack=list(stack), plyts=list(plyts),
              laminaprops=list(laminaprops),
              rhos=None if rhos is None else list(rhos))
    s.plyt = plyts[0]
    s.laminaprop = laminaprops[0]
    s.fsdt_shear_correction = SCF[scf]
    set_bcs(s, bcs)
    s._rebuild()
    return s


# Solvers
# -------

def center(s, c, x=None, y=None, field='w'):
    r"""Value of ``field`` at ``(x, y)``, by default the center"""
    x = s.a/2 if x is None else x
    y = s.b/2 if y is None else y
    return float(s.uvw(c, xs=np.array([x]), ys=np.array([y]))[1][field][0])


def static_linear(s):
    r"""Linear static solution of the loads of ``s``"""
    return solve(s.calc_kC(), s.calc_fext(), silent=True)


def linear_buckling(s, num_eigvalues=5):
    r"""Lowest eigenvalue of `(K_C + \lambda K_G) c = 0` for the membrane
    prestate ``Nxx``, ``Nyy``, ``Nxy`` of the shell, and its mode"""
    eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True,
                          num_eigvalues=num_eigvalues)
    return float(eigvals[0]), eigvecs[:, 0]


class _LimitPassed(Exception):
    pass


def nonlinear_path(s, method='riks', initial_inc=0.1, max_arc_length=None,
                   max_inc=None, stop_after_limit=None):
    r"""Load-displacement path with the loads of ``cte=False`` incremented

    ``method`` is ``'NR'`` (Newton-Raphson, load control) or ``'riks'``.
    With ``'riks'``, ``max_arc_length`` is the cumulative arc length at which
    the analysis stops and ``max_inc`` the largest arc length of one step,
    both relative to the linear solution at load factor 1. Since structsolve
    caps the arc length of a step at ``max(max_inc, initial arc length)``,
    ``initial_inc`` should not be larger than ``max_inc``. With
    ``stop_after_limit``, e.g. ``0.9``, the analysis stops when the load
    factor drops below this fraction of its first maximum, i.e. once the limit
    point has been passed: the check is done when structsolve computes `K_C`
    at the start of each step. Returns the load factors and the solutions.
    """
    calc_kC = s.calc_kC
    an = None

    def calc_kC_check(*args, **kwargs):
        if stop_after_limit is not None and an.increments:
            lbds = np.asarray(an.increments)
            drops = np.where(np.diff(lbds) < 0)[0]
            if len(drops) and lbds[-1] < stop_after_limit*lbds[drops[0]]:
                raise _LimitPassed()
        return calc_kC(*args, **kwargs)

    an = Analysis(s.calc_fext, s.calc_fint, calc_kC_check, s.calc_kG)
    if method == 'riks':
        an.NL_method = 'arc_length_riks'
        if max_arc_length is not None:
            an.maxArcLength = max_arc_length
    else:
        an.NL_method = 'NR'
    an.initialInc = initial_inc
    if max_inc is not None:
        an.maxInc = max_inc
    try:
        increments, cs = an.static(NLgeom=True, silent=True)
    except _LimitPassed:
        increments, cs = an.increments, an.cs
    return np.asarray(increments), cs


def first_maximum(P, tol=1e-3):
    r"""Index of the first local maximum of the load followed by a drop of
    more than ``tol``, relative, before the load recovers, or ``None``"""
    P = np.asarray(P, dtype=float)
    i = 0
    while i < len(P) - 1:
        if P[i+1] < P[i]:
            j = i + 1
            while j < len(P) and P[j] <= P[i]:
                if P[j] < (1 - tol)*P[i]:
                    return i
                j += 1
            i = j
        else:
            i += 1
    return None


def refine_peak(P, wc, h, Pref, i):
    r"""Parabola through the three points around the maximum ``i``, in terms
    of the length along the path in the plane of `P/P_{ref}` and `w_C/h`"""
    if i == 0 or i == len(P) - 1:
        return float(P[i]), float(wc[i])
    x = np.asarray(wc, dtype=float)/h
    y = np.asarray(P, dtype=float)/Pref
    t = np.concatenate(([0.], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))
    ts, ys, xs = t[i-1:i+2], y[i-1:i+2], x[i-1:i+2]
    c = np.polyfit(ts, ys, 2)
    if c[0] >= 0:
        return float(P[i]), float(wc[i])
    tp = min(max(-c[1]/(2*c[0]), ts[0]), ts[2])
    yp = np.polyval(c, tp)
    xp = np.polyval(np.polyfit(ts, xs, 2), tp)
    return float(yp*Pref), float(xp*h)


def limit_point(P, wc, h, Pref):
    r"""First maximum of the load along the path, refined with a parabola:
    ``Plim``, ``w_lim``, ``has_limit`` and ``dw_max``, the largest increment
    of `w_C/h` of one step up to the limit point"""
    P = np.asarray(P, dtype=float)
    wc = np.asarray(wc, dtype=float)
    ilim = first_maximum(P)
    has_limit = ilim is not None
    if has_limit:
        Plim, w_lim = refine_peak(P, wc, h, Pref, ilim)
    else:
        ilim = int(np.argmax(P))
        Plim, w_lim = float(P[ilim]), float(wc[ilim])
    dw = np.abs(np.diff(np.concatenate(([0.], wc[:ilim+1]))))
    return dict(Plim=Plim, w_lim=w_lim, P_step=float(P[ilim]),
                has_limit=has_limit, dw_max=float(dw.max()/h))


def finite_cylinder(theory, scf, R, L, stack, plyts, laminaprops, rhos=None,
                    mx=20, my=10):
    r"""Closed cylinder of radius ``R`` and length ``L`` with shear-diaphragm
    ends, represented by a shear-diaphragm panel of angle `\pi`, which
    contains every mode with `n \ge 1` circumferential waves"""
    return make_shell(theory, L, np.pi*R, stack, plyts, laminaprops, rhos,
                      r=R, scf=scf, bcs='DDDD', m=mx, n=my)


def cylinder_pressure(s, zp, Nxx_per_p, mode=False):
    r"""Critical external pressure `p` of the cylinder ``s``: follower
    pressure on the surface ``zp``, hoop prestate `N_{yy} = -p (R + z_p)`
    and axial prestate `N_{xx} = ` ``Nxx_per_p`` `\times p`; the follower
    load stiffness makes the eigenvalue problem unsymmetric. With
    ``mode=True`` the half-waves of the mode, :func:`halfwaves`, are also
    returned"""
    s.clear_loads()
    s.Nxx = Nxx_per_p
    s.Nyy = -(s.r + zp)
    s.Nxy = 0.
    s.add_pressure_load(-1., cte=False, follower=True, zp=zp)
    K = s.calc_kG() + s.calc_kCfollower()
    eigvals, eigvecs = lb(s.calc_kC(), K, silent=True, num_eigvalues=4)
    p = float(np.real(eigvals[0]))
    if mode:
        return p, halfwaves(s, np.real(eigvecs[:, 0]))
    return p


def halfwaves(s, c, n=61):
    r"""Numbers of half-waves of `w` of the mode ``c`` along `x` and `y`,
    counted through the point of maximum `|w|`; for the cylinder of
    :func:`finite_cylinder`, the panel of angle `\pi`, they are the axial
    half-waves `m` and the circumferential full waves `n` of the closed
    cylinder"""
    xs, ys = np.linspace(0, s.a, n), np.linspace(0, s.b, n)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    w = s.uvw(c, xs=X.ravel(), ys=Y.ravel())[1]['w'].copy().reshape(n, n)
    i, j = np.unravel_index(np.argmax(np.abs(w)), w.shape)

    def count(v):
        v = v[np.abs(v) > 1e-3*np.abs(w).max()]
        return int(np.sum(np.diff(np.sign(v)) != 0)) + 1
    return count(w[:, j]), count(w[i, :])


# Hinged roof of Sabir and Lock (1972), Sze et al. (2004)
# -------------------------------------------------------

ROOF_R = 2540.
ROOF_A = 508.
ROOF_B = 0.2*ROOF_R
ROOF_E = 3102.75
ROOF_NU = 0.3
#: G_TT, not given by Sze et al. (2004), is taken as G_LT
ROOF_LAMINA = (3300., 1100., 0.25, 660., 660., 660.)
ROOF_M = 12
ROOF_MAX_INC = 0.01
ROOF_MAX_ARC_LENGTH = 8.


def roof_section(kind, h):
    r"""``(stack, plyts, laminaprops)`` of the roofs ``'isotropic'``,
    ``'[0/90/0]'``, ``'[90/0/90]'`` and ``'sandwich'``"""
    if kind == 'isotropic':
        return [0], [h], [(ROOF_E, ROOF_NU)]
    if kind == 'sandwich':
        core = (ROOF_E/100, ROOF_NU)
        return ([0, 0, 0], [h/10, 0.8*h, h/10],
                [(ROOF_E, ROOF_NU), core, (ROOF_E, ROOF_NU)])
    stack = {'[90/0/90]': [90, 0, 90], '[0/90/0]': [0, 90, 0]}[kind]
    return stack, [h/3]*3, [ROOF_LAMINA]*3


def roof_pref(h):
    r"""Reference load of the Riks analysis, above the limit loads"""
    return 4000.*(h/12.7)**2.5


def hinged_roof(theory, scf, kind, h, m=ROOF_M):
    r"""Roof with hinged straight edges, ``u = v = w = 0`` and the rotations
    free at ``y = 0, b``, and free curved edges"""
    stack, plyts, lps = roof_section(kind, h)
    s = make_shell(theory, ROOF_A, ROOF_B, stack, plyts, lps,
                   [1.]*len(stack), r=ROOF_R, scf=scf, bcs='FFFF', m=m, n=m)
    for edge in ('y1', 'y2'):
        for field in 'uvw':
            setattr(s, edge + field, 0.)
    s._rebuild()
    return s


def roof_limit_load(theory, scf, kind, h, m=ROOF_M, max_inc=ROOF_MAX_INC,
                    initial_inc=None):
    r"""Riks path of the roof under the central point load, stopped after a
    drop of 10% below the first maximum, and its limit point; by default
    ``initial_inc = max_inc``"""
    Pref = roof_pref(h)
    s = hinged_roof(theory, scf, kind, h, m=m)
    s.add_point_load(ROOF_A/2, ROOF_B/2, 0., 0., -Pref, cte=False)
    # structsolve caps the arc length of a step at max(maxInc, initial arc
    # length), hence initial_inc = max_inc
    initial_inc = max_inc if initial_inc is None else initial_inc
    lbds, cs = nonlinear_path(s, method='riks', initial_inc=initial_inc,
                              max_arc_length=ROOF_MAX_ARC_LENGTH,
                              max_inc=max_inc, stop_after_limit=0.9)
    wc = np.array([center(s, c) for c in cs])
    P = Pref*np.asarray(lbds)
    out = dict(P=P.tolist(), wc=wc.tolist())
    out.update(limit_point(P, wc, h, Pref))
    return out


# Summaries
# ---------

def error_stats(errors):
    r"""``(max |e|, mean |e|, min e, max e)`` in percent of a list of
    relative differences in percent, ignoring ``None``"""
    e = np.array([v for v in errors if v is not None], dtype=float)
    if not len(e):
        return None
    return float(np.abs(e).max()), float(np.abs(e).mean()), float(e.min()), \
        float(e.max())


def stats_table(columns, label='model'):
    r"""Markdown table of :func:`error_stats` of ``{name: errors}``"""
    rows = []
    for name, errors in columns.items():
        st = error_stats(errors)
        if st is None:
            continue
        rows.append([name, '%.1f' % st[0], '%.1f' % st[1],
                     '%+.1f to %+.1f' % (st[2], st[3])])
    return markdown_table([label, 'max abs. diff. (%)', 'mean abs. diff. (%)',
                           'range (%)'], rows)


# Reference data
# --------------
# Copied from the scripts of the studies of the paper and from
# REFERENCE_DATA.md, which were checked against the papers

#: Pagano (1970), Table 1, square [0/90/0], as printed: S: (sigma_x(a/2, b/2,
#: 1/2), tau_xz(0, b/2, 0), tau_yz(a/2, 0, 0))
PAGANO_TABLE_1 = {4: ('.801', '.256', '.2172'), 10: ('.590', '.357', '.1228'),
                  20: ('.552', '.385', '.0938'), 50: ('.541', '.393', '.0842'),
                  100: ('.539', '.395', '.0828')}
#: Pagano (1970), Table 2, [0/90/0] with b = 3a: S: (sigma_x(a/2, b/2, 1/2),
#: tau_xz(0, b/2, 0), tau_yz(a/2, 0, 0), wbar)
PAGANO_TABLE_2 = {2: ('2.13', '.257', '.0668', '8.17'),
                  4: ('1.14', '.351', '.0334', '2.82'),
                  10: ('.726', '.420', '.0152', '.919'),
                  20: ('.650', '.434', '.0119', '.610'),
                  50: ('.628', '.439', '.0110', '.520'),
                  100: ('.624', '.439', '.0108', '.508')}
#: Pagano (1970), Table 3, square sandwich: S: (sigma_x(a/2, b/2, 1/2),
#: |sigma_y(a/2, b/2, 1/2)|, tau_xz(0, b/2, 0), tau_yz(a/2, 0, 0),
#: |tau_xy(0, 0, 1/2)|)
PAGANO_TABLE_3 = {2: ('3.278', '.4517', '.185', '.1399', '.2403'),
                  4: ('1.556', '.2595', '.239', '.1072', '.1437'),
                  10: ('1.153', '.1104', '.300', '.0527', '.0707'),
                  20: ('1.110', '.0700', '.317', '.0361', '.0511'),
                  50: ('1.099', '.0569', '.323', '.0306', '.0446'),
                  100: ('1.098', '.0550', '.324', '.0297', '.0437')}

#: Noor (1975), Table 2, 3D, a/h = 10: {NL: [E_L/E_T = 3, 10, 20, 30, 40]}
NOOR_1975_TABLE_2 = {
    2: [4.6948, 6.1181, 7.8196, 9.3746, 10.8167],
    4: [5.1738, 9.0164, 13.7429, 17.7829, 21.2796],
    6: [5.2673, 9.6051, 15.0014, 19.6394, 23.6689],
    10: [5.3159, 9.9134, 15.6685, 20.6347, 24.9636],
    3: [5.3044, 9.7621, 15.0191, 19.3040, 22.8807],
    5: [5.3255, 9.9603, 15.6527, 20.4663, 24.5929],
    9: [5.3352, 10.0417, 15.9153, 20.9614, 25.3436],
}
NOOR_E = [3, 10, 20, 30, 40]
#: Noor (1975), Table 3, E_L/E_T = 30: {(a/h, NL): (3D, N_SDT/N_3D with
#: kappa = 1, 5/6 and the composite factors of his Table 1)}
NOOR_1975_TABLE_3 = {
    (10, 2): (9.375, 1.057, 1.038, 1.006),
    (10, 3): (19.304, 1.074, 1.023, 1.010),
    (10, 9): (20.961, 1.056, 1.014, 1.008),
    (10, 10): (20.635, 1.058, 1.018, 1.007),
    (5, 2): (6.664, 1.171, 1.108, 1.011),
    (5, 3): (10.383, 1.172, 1.062, 1.015),
    (5, 9): (12.138, 1.130, 1.027, 1.008),
    (5, 10): (12.070, 1.134, 1.031, 1.005),
}
#: Noor (1975), Table 1, E_L/E_T = 30: composite shear correction factors,
#: NL: (k1, k2)
NOOR_1975_TABLE_1 = {2: (0.6421, 0.6421), 4: (0.6523, 0.6523),
                     6: (0.7422, 0.7422), 10: (0.7947, 0.7947),
                     3: (0.8274, 0.5412), 5: (0.8732, 0.5914),
                     7: (0.8769, 0.6749), 9: (0.8736, 0.7170)}

#: Sze et al. (2004), Tables 9a-f: first maximum of P/Pmax along the path,
#: Pmax = 3000, and the central deflection -W_C at that point
SZE_2004 = {
    ('isotropic', 12.7): (0.7421, 11.293),
    ('[0/90/0]', 12.7): (0.3618, 9.884),
    ('[90/0/90]', 12.7): (0.5970, 14.192),
    ('isotropic', 6.35): (0.1953, 12.892),
    ('[0/90/0]', 6.35): (0.0782, 12.280),
    ('[90/0/90]', 6.35): (0.1585, 15.905),
}
PMAX_SZE = 3000.

#: Varadan and Bhaskar (1991), Tables 1-4, Ubar_r = 10 E_L u_r/(Q h S^4) at
#: the mid-surface, for R0/h = 2, 4, 10, 50, 100, 500
VARADAN_1991 = {
    '90': [7.503, 2.783, 0.9189, 0.5385, 0.5170, 0.3060],
    '90/0': [14.034, 6.100, 3.330, 2.242, 1.367, 0.1005],
    '90/0/90': [10.11, 4.009, 1.223, 0.5495, 0.4715, 0.1027],
    '(90/0/90/0/90)s': [11.44, 4.206, 1.380, 0.7622, 0.6261, 0.1006],
}
VARADAN_R_H = [2, 4, 10, 50, 100, 500]

#: Kardomateas (1993b), sigma0 R2/(E3 h), 3D elasticity, {R2/R1: value}
KARDOMATEAS_1993B = {
    'isotropic': {1.15: 0.454, 1.20: 0.437, 1.25: 0.443, 1.30: 0.449},
    'transversely isotropic': {1.10: 0.167, 1.15: 0.162, 1.20: 0.164,
                               1.25: 0.163, 1.30: 0.167},
}
#: Kardomateas (1993b), modes (n, m) of the 3D solution
KARDOMATEAS_1993B_MODES = {
    'isotropic': {1.15: (2, 1), 1.20: (2, 2), 1.25: (2, 2), 1.30: (1, 1)},
    'transversely isotropic': {1.10: (2, 1), 1.15: (2, 1), 1.20: (2, 2),
                               1.25: (2, 2), 1.30: (2, 2)},
}
#: Kardomateas (1993b), Table 2, Flugge shell theory, isotropic
FLUGGE_1993B = {1.15: 0.471, 1.20: 0.462, 1.25: 0.473, 1.30: 0.492}

#: Kardomateas and Philobos (1995), Tables 1 and 2, 3D elasticity:
#: {(material, S): [(b/a, pbar, Pbar, (n, m))]}, pbar = p b^3/(E2 h^3)
KARDOMATEAS_1995 = {
    ('glass', 5): [(1.03, 0.5561, 0.3346, (2, 1)),
                   (1.05, 0.3014, 0.2993, (2, 1)),
                   (1.10, 0.1971, 0.3822, (2, 1)),
                   (1.15, 0.1665, 0.4730, (2, 2)),
                   (1.20, 0.1335, 0.4940, (2, 2)),
                   (1.25, 0.1167, 0.5278, (1, 1))],
    ('glass', 1): [(1.03, 0.7311, 0.0880, (3, 1)),
                   (1.05, 0.4666, 0.0927, (2, 1)),
                   (1.10, 0.3038, 0.1178, (2, 1)),
                   (1.15, 0.2758, 0.1567, (2, 1)),
                   (1.20, 0.2659, 0.1968, (2, 1)),
                   (1.25, 0.2600, 0.2353, (2, 1))],
    ('graphite', 5): [(1.03, 0.2511, 0.5708, (2, 1)),
                      (1.05, 0.1826, 0.6852, (2, 1)),
                      (1.10, 0.1125, 0.8245, (2, 2)),
                      (1.15, 0.0754, 0.8089, (2, 3)),
                      (1.20, 0.0483, 0.6760, (1, 1)),
                      (1.25, 0.0324, 0.5540, (1, 1))],
    ('graphite', 1): [(1.03, 0.3899, 0.1773, (2, 1)),
                      (1.05, 0.2834, 0.2127, (2, 1)),
                      (1.10, 0.2352, 0.3446, (2, 1)),
                      (1.15, 0.2140, 0.4593, (2, 2)),
                      (1.20, 0.1810, 0.5063, (2, 2)),
                      (1.25, 0.1597, 0.5461, (2, 2))],
}
#: Kardomateas and Philobos (1995), Tables 1 and 2, critical pbar and (n, m)
#: of the nonshallow Donnell and of the Timoshenko shell theories:
#: {(material, S): {b/a: ((Donnell pbar, (n, m)), (Timoshenko pbar, (n, m)))}}
#: NOTE transcribed from the tables of the paper for these notebooks, they
#:      are not in REFERENCE_DATA.md of the paper study; each value is
#:      consistent with the "% increase" over the elasticity value printed
#:      below it in the paper
KP1995_SHELLS = {
    ('glass', 5): {1.03: ((0.6209, (2, 1)), (0.5653, (2, 1))),
                   1.05: ((0.3435, (2, 1)), (0.3130, (2, 1))),
                   1.10: ((0.2371, (2, 1)), (0.2165, (2, 1))),
                   1.15: ((0.2218, (2, 2)), (0.1886, (2, 2))),
                   1.20: ((0.1909, (2, 2)), (0.1624, (2, 2))),
                   1.25: ((0.1753, (2, 3)), (0.1241, (1, 1)))},
    ('glass', 1): {1.03: ((0.7518, (3, 1)), (0.7480, (3, 1))),
                   1.05: ((0.4965, (2, 1)), (0.4829, (2, 1))),
                   1.10: ((0.3386, (2, 1)), (0.3297, (2, 1))),
                   1.15: ((0.3235, (2, 1)), (0.3152, (2, 1))),
                   1.20: ((0.3297, (2, 1)), (0.3214, (2, 1))),
                   1.25: ((0.3418, (2, 1)), (0.3334, (2, 1)))},
    ('graphite', 5): {1.03: ((0.2845, (2, 1)), (0.2591, (2, 1))),
                      1.05: ((0.2137, (2, 1)), (0.1949, (2, 1))),
                      1.10: ((0.1519, (2, 2)), (0.1290, (2, 2))),
                      1.15: ((0.1092, (2, 4)), (0.0819, (1, 1))),
                      1.20: ((0.0867, (2, 5)), (0.0501, (1, 1))),
                      1.25: ((0.0696, (1, 1)), (0.0348, (1, 1)))},
    ('graphite', 1): {1.03: ((0.4134, (2, 1)), (0.4019, (2, 1))),
                      1.05: ((0.3090, (2, 1)), (0.3005, (2, 1))),
                      1.10: ((0.2793, (2, 1)), (0.2719, (2, 1))),
                      1.15: ((0.2880, (2, 1)), (0.2704, (2, 2))),
                      1.20: ((0.2815, (2, 2)), (0.2505, (1, 1))),
                      1.25: ((0.2743, (2, 3)), (0.1737, (1, 1)))},
}
