r"""Double cantilever beam (DCB) with the multidomain cohesive zone model

Model used by the notebooks that compare ``panels`` with DCB results from the
literature. Along `x`, measured from the far (clamped) end of the specimen:

- ``top0``/``bot0``: bonded region that the crack does not reach, perfect
  bond (``'SB'`` connection), omitted when ``L_bond = 0``
- ``top1``/``bot1``: cohesive domain of length ``L_tsl`` (``'SB_TSL'``
  connection with a bilinear traction-separation law), whose edge at
  ``x = L - a0`` is the initial crack front
- ``top2``/``bot2``: the two arms of length ``a0``

The far end is clamped and the tips of the arms are opened symmetrically,
`w = \pm \delta/2`, by prescribed displacements, rotations and in-plane
displacements free. The load `P` is the reaction of the prescribed
displacement of the top arm.

The nonlinear solution follows ``tests/multidomain/test_dcb_damage.py``:
modified Newton-Raphson with the consistent tangent of the cohesive zone, a
linear predictor, backtracking line search and bisection of increments.
See doc/source/cohesive_zone.rst

Units: N, mm, MPa.

"""
import os
import time

import numpy as np
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import calc_kt_kr, fkCld_xcte


def _stack(case):
    """Stacking sequence and ply thickness of each arm"""
    if 'stack' in case:
        return list(case['stack']), case['plyt']
    return [0], case['h']


def arm_thickness(case):
    stack, plyt = _stack(case)
    return plyt*len(stack)


def _free(p):
    for e in ('x1', 'x2', 'y1', 'y2'):
        for d in ('u', 'v', 'w'):
            setattr(p, e + d, 1.)
            setattr(p, e + d + 'r', 1.)


def _scaling(vec, D):
    non_nulls = ~np.isclose(D, 0)
    vec = vec[non_nulls]
    D = D[non_nulls]
    return np.sqrt((vec*np.abs(1/D)) @ vec)


def build_dcb(case):
    r"""Build the multidomain model of a DCB

    Parameters
    ----------
    case : dict
        Keys: ``E1, E2, nu12, G12`` (MPa), ``h`` (arm thickness), ``b``
        (width), ``L`` (length), ``a0`` (initial crack length), ``L_tsl``
        (length of the cohesive domain, ``<= L - a0``), ``G1c`` (N/mm),
        ``tau_o`` (MPa), ``k_o`` (N/mm3). Optional: ``m_tsl, n_tsl`` (terms
        of the cohesive domain, default 15, 10), ``m, n`` (terms of the other
        domains, default 8, 8), ``nx, ny`` (Gauss points of the cohesive
        domain, default 120, 30), ``edge_penalty_factor`` (default 100),
        ``stack`` and ``plyt`` (stacking sequence of each arm and ply
        thickness, default a single 0-degree ply of thickness ``h``; when
        given, ``h`` is ``plyt*len(stack)``), ``bond_width`` (width of a
        central bonded strip narrower than the arms, see
        :func:`solve_dcb`).

    Returns
    -------
    model : dict
        With the assembly, the panels and the connections.

    """
    E1, E2, nu12, G12 = case['E1'], case['E2'], case['nu12'], case['G12']
    lam = (E1, E2, nu12, G12, G12, G12)
    b, L, a0 = case['b'], case['L'], case['a0']
    stack, plyt = _stack(case)
    L_tsl = case['L_tsl']
    L_bond = L - a0 - L_tsl
    assert L_bond >= -1e-9, 'L_tsl must be <= L - a0'
    m_tsl, n_tsl = case.get('m_tsl', 15), case.get('n_tsl', 10)
    m, n = case.get('m', 8), case.get('n', 8)
    nx, ny = case.get('nx', 120), case.get('ny', 30)

    def shell(group, x0, a, mm, nn):
        s = Shell(group=group, x0=x0, y0=0, a=a, b=b, m=mm, n=nn, plyt=plyt,
                  stack=stack, laminaprop=lam)
        _free(s)
        return s

    top, bot = [], []
    x0 = 0.
    if L_bond > 1e-9:
        top.append(shell('top', x0, L_bond, m, n))
        bot.append(shell('bot', x0, L_bond, m, n))
        x0 += L_bond
    top_tsl = shell('top', x0, L_tsl, m_tsl, n_tsl)
    bot_tsl = shell('bot', x0, L_tsl, m_tsl, n_tsl)
    top.append(top_tsl)
    bot.append(bot_tsl)
    x0 += L_tsl
    top_arm = shell('top', x0, a0, m, n)
    bot_arm = shell('bot', x0, a0, m, n)
    top.append(top_arm)
    bot.append(bot_arm)

    # far end clamped
    for p in (top[0], bot[0]):
        p.x1u = p.x1v = p.x1w = p.x1wr = 0.

    conn = []
    for pan in (top, bot):
        for p1, p2 in zip(pan[:-1], pan[1:]):
            conn.append(dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0))
    if L_bond > 1e-9:
        conn.append(dict(p1=top[0], p2=bot[0], func='SB'))
    conn.append(dict(p1=top_tsl, p2=bot_tsl, func='SB_TSL', tsl_type='bilinear',
                     nr_x_gauss=nx, nr_y_gauss=ny, k_o=case['k_o'],
                     tau_o=case['tau_o'], G1c=case['G1c']))
    factor = case.get('edge_penalty_factor', 100.)
    for c in conn:
        if c['func'] == 'SSxcte':
            kt, kr = calc_kt_kr(c['p1'], c['p2'], 'xcte')
            c['kt'] = factor*kt
            c['kr'] = factor*kr

    panels = bot + top
    assy = MultiDomain(panels=panels, conn=conn)
    return dict(assy=assy, conn=conn, top_tsl=top_tsl, bot_tsl=bot_tsl,
                top_arm=top_arm, bot_arm=bot_arm, nx=nx, ny=ny,
                size=assy.get_size())


def solve_dcb(case, openings, kw=1.e6, epsilon=1.e-4, NR_kT_update=3,
              kT_pan_reuse_steps=5, line_search_max=6, max_NR_iter=50,
              max_bisections=6, save=None, verbose=False):
    r"""Solve the DCB under a list of prescribed openings

    Parameters
    ----------
    case : dict
        See :func:`build_dcb`.
    openings : array-like
        Increasing opening displacements `\delta` of the arm tips, the tips
        move `\pm \delta/2`.
    kw : float, optional
        Penalty stiffness of the prescribed displacements.
    epsilon : float, optional
        Tolerance of the convergence criterion, the ratio between the scaled
        norm of the residual and the largest of those of the internal force
        and of the reaction of the prescribed displacements.
    NR_kT_update : int, optional
        The tangent stiffness of the panels is refreshed every
        ``NR_kT_update`` iterations, the secant and the damage-rate
        stiffness of the cohesive zone at every iteration.
    kT_pan_reuse_steps : int, optional
        While no point of the cohesive zone is softening (`0 < d < 1`), the
        tangent stiffness of the panels is reused for up to this number of
        openings. The points that are fully damaged from the start, outside
        ``bond_width``, do not count as damage.
    line_search_max : int, optional
        Maximum number of halvings of the step of the backtracking line
        search. A trial whose residual is not finite counts as an infinite
        norm.
    max_NR_iter, max_bisections : int, optional
        An opening that does not converge in ``max_NR_iter`` iterations, or
        whose ratio of the convergence criterion is not finite or exceeds
        `10^3`, is bisected from the last converged state; the analysis is
        aborted only when a sub-increment of ``2**-max_bisections`` of the
        original increment still does not converge.
    save : str, optional
        ``.npz`` file updated after every converged opening.

    Returns
    -------
    res : dict
        ``delta`` (openings reached), ``P`` (reaction), ``P_traction``
        (integral of the cohesive tractions, must match ``P``), ``a``
        (crack length at the mid-width, from the fully damaged points),
        ``dmax`` (maximum damage), ``iters`` (Newton-Raphson iterations per
        opening), ``failed`` (number of bisections), ``aborted``, ``time``.

    """
    t0 = time.time()
    model = build_dcb(case)
    assy, conn = model['assy'], model['conn']
    top_tsl, bot_tsl = model['top_tsl'], model['bot_tsl']
    top_arm, bot_arm = model['top_arm'], model['bot_arm']
    nx, ny, size = model['nx'], model['ny'], model['size']
    k_o, tau_o, G1c = case['k_o'], case['tau_o'], case['G1c']

    def prescribe(delta):
        kCp = 0
        for p, sign in ((top_arm, +1), (bot_arm, -1)):
            p.clear_disps()
            kCp += fkCld_xcte(0., 0., kw, p, p.a, size, p.row_start, p.col_start)
            p.add_distr_pd_fixed_x(p.a, None, None, kw, funcu=None, funcv=None,
                                   funcw=lambda y, s=sign: s*delta/2)
        kCp = finalize_symmetric_matrix(kCp)
        fext = assy.calc_fext()
        return kCp, fext

    conv = [(0., np.zeros(size))]
    history = np.zeros((ny, nx))
    if case.get('bond_width') is not None:
        # points outside the central bonded strip start fully damaged, which
        # removes the connection there, e.g. a weld narrower than the arms
        eta, _ = np.polynomial.legendre.leggauss(ny)
        y = top_tsl.b/2*(eta + 1)
        outside = np.abs(y - top_tsl.b/2) > case['bond_width']/2
        history[outside, :] = 1.
    assy.update_TSL_history(curr_max_dmg_index=history)
    state = dict(kT_pan=None, age=0)

    def newton(delta):
        kCp, fext = prescribe(delta)
        d_a, c_a = conv[-1]
        ci = c_a.copy()
        if len(conv) >= 2:
            d_b, c_b = conv[-2]
            if d_a != d_b:
                ci = c_a + (delta - d_a)/(d_a - d_b)*(c_a - c_b)
        elastic = np.max(assy.dmg_index[assy.dmg_index < 1.], initial=0.) == 0.

        def residual(c):
            kC_conn = assy.get_kC_conn(c=c)
            fint = np.asarray(assy.calc_fint(c=c, kC_conn=kC_conn))
            return fint - fext + kCp*c, fint, kC_conn

        Ri, fint, kC_conn = residual(ci)
        if (state['kT_pan'] is None or not elastic
                or state['age'] >= kT_pan_reuse_steps):
            state['kT_pan'] = assy.calc_kT(c=ci, kC_conn=0.)
            state['age'] = 0
        else:
            state['age'] += 1
        k0 = state['kT_pan'] + kC_conn + assy.calc_kT_TSL(c=ci) + kCp
        D = k0.diagonal()
        count = 0
        while True:
            dc = solve(k0, -Ri, silent=True)
            r0 = _scaling(Ri, D)
            step = 1.
            best = None
            for _ in range(line_search_max + 1):
                c = ci + step*dc
                R, fint_t, kC_t = residual(c)
                r = _scaling(R, D)
                if not np.isfinite(r):
                    r = np.inf
                if best is None or r < best[0]:
                    best = (r, c, R, fint_t, kC_t)
                if r <= (1 - 1e-4*step)*r0:
                    break
                step *= 0.5
            _, c, Ri, fint, kC_conn = best
            crit = _scaling(Ri, D)/max(_scaling(fint, D), _scaling(kCp*c - fext, D))
            count += 1
            if crit < epsilon:
                return True, c, count
            if not np.isfinite(crit) or crit > 1e3 or count >= max_NR_iter:
                return False, c, count
            if count % NR_kT_update == 1 and not (elastic and count == 1):
                state['kT_pan'] = assy.calc_kT(c=c, kC_conn=0.)
                state['age'] = 0
            k0 = state['kT_pan'] + kC_conn + assy.calc_kT_TSL(c=c) + kCp
            ci = c

    def accept(delta, c):
        _, dmg, _, _ = assy.calc_k_dmg(c=c, pA=top_tsl, pB=bot_tsl,
                nr_x_gauss=nx, nr_y_gauss=ny, tsl_type='bilinear',
                prev_max_dmg_index=assy.dmg_index, k_i=k_o, tau_o=tau_o, G1c=G1c)
        assy.update_TSL_history(curr_max_dmg_index=dmg)
        conv.append((delta, c.copy()))
        del conv[:-2]

    xi, _ = np.polynomial.legendre.leggauss(nx)
    x_gauss = top_tsl.a/2*(xi + 1)
    res = dict(delta=[], P=[], P_traction=[], a=[], dmax=[], iters=[],
               failed=0, aborted=False)
    for delta_target in openings:
        pending = [delta_target]
        min_inc = abs(delta_target - conv[-1][0])/2**max_bisections
        iters = 0
        while pending:
            delta = pending[-1]
            ok, c_new, n_it = newton(delta)
            iters += n_it
            if ok:
                accept(delta, c_new)
                pending.pop()
                continue
            d_last = conv[-1][0]
            d_mid = 0.5*(d_last + delta)
            res['failed'] += 1
            if abs(d_mid - d_last) < min_inc*(1 - 1e-9):
                res['aborted'] = True
                break
            pending.append(d_mid)
        if res['aborted']:
            break
        c = conv[-1][1]
        P = assy.reaction_line_pd_xcte(c, top_arm, top_arm.a, kw,
                                       lambda y: delta_target/2)
        Pt = assy.force_out_plane_damage(conn=conn, c=c)
        dmg = assy.dmg_index
        row = dmg[ny//2]
        failed = np.flatnonzero(row >= 1.)
        # crack front: fully damaged points connected to the initial front
        a_crack = case['a0']
        if failed.size and failed[-1] == nx - 1:
            j = nx - 1
            while j - 1 >= 0 and row[j - 1] >= 1.:
                j -= 1
            a_crack = case['a0'] + top_tsl.a - x_gauss[j]
        res['delta'].append(delta_target)
        res['P'].append(P)
        res['P_traction'].append(Pt)
        res['a'].append(a_crack)
        res['dmax'].append(float(np.max(dmg)))
        res['iters'].append(iters)
        if verbose:
            print(f'delta={delta_target:.4f} P={P:.3f} Ptrac={Pt:.3f} '
                  f'a={a_crack:.2f} dmax={np.max(dmg):.3f} it={iters}', flush=True)
        if save is not None:
            _save(save, res, t0)
    res['time'] = time.time() - t0
    if save is not None:
        _save(save, res, t0)
    return {k: np.asarray(v) if isinstance(v, list) else v for k, v in res.items()}


def _save(path, res, t0):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in res.items() if k != 'time'},
             time=time.time() - t0)


def load(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}


def run_or_load(case, openings, path, rerun=False, **kwargs):
    """Loads the results in ``path`` or runs :func:`solve_dcb` and saves them"""
    if os.path.exists(path) and not rerun:
        res = load(path)
        # a run interrupted before the last opening is run again
        complete = (len(res['delta']) and np.isclose(res['delta'][-1], openings[-1]))
        if complete or res['aborted']:
            return res
    return solve_dcb(case, openings, save=path, **kwargs)


# ----------------------------------------------------------- LEFM references
def D11(case):
    """Bending stiffness of one arm per unit width, CLPT"""
    stack, plyt = _stack(case)
    lam = (case['E1'], case['E2'], case['nu12'], case['G12'], case['G12'], case['G12'])
    s = Shell(group='arm', x0=0, y0=0, a=1., b=1., m=4, n=4, plyt=plyt,
              stack=stack, laminaprop=lam)
    return float(s.lam.D[0, 0])


def bending_modulus(case):
    """Effective bending modulus of one arm, ``12 D11/h^3``, or ``E1`` of a
    single unidirectional ply, used in the correction of beam theory"""
    if 'stack' not in case:
        return case['E1']
    return 12*D11(case)/arm_thickness(case)**3


def chi_williams(E11, E22, G13):
    r"""Crack length correction factor of corrected beam theory

    `a_{eff} = a + \chi h`, Williams (1989), with:

    .. math::

        \chi = \sqrt{\frac{E_{11}}{11 G_{13}} \left[3 - 2 \left(
        \frac{\Gamma}{1 + \Gamma} \right)^2 \right]}, \qquad
        \Gamma = 1.18 \frac{\sqrt{E_{11} E_{22}}}{G_{13}}

    """
    Gamma = 1.18*np.sqrt(E11*E22)/G13
    return np.sqrt(E11/(11*G13)*(3 - 2*(Gamma/(1 + Gamma))**2))


def lefm_curve(case, a_max, corrected=False, n=200):
    r"""Load-opening curve of linear elastic fracture mechanics

    Beam theory for each arm with bending stiffness ``D11*b``. With
    ``corrected=True`` the crack length is replaced by `a + \chi h`, see
    :func:`chi_williams`, with ``G13 = G12``. Under displacement control the
    curve is linear up to the critical opening of ``a0`` and then follows
    the propagation branch `G = G_{Ic}`:

    .. math::

        P = \frac{\sqrt{G_{Ic} \, b \, D_{11} b}}{a_e}, \qquad
        \delta = \frac{2 P a_e^3}{3 D_{11} b}

    Returns
    -------
    delta, P : np.ndarray

    """
    EI = D11(case)*case['b']
    dh = chi_williams(bending_modulus(case), case['E2'], case['G12'])*arm_thickness(case) if corrected else 0.
    a = np.linspace(case['a0'], a_max, n) + dh
    P = np.sqrt(case['G1c']*_bonded_width(case)*EI)/a
    delta = 2*P*a**3/(3*EI)
    return np.concatenate(([0.], delta)), np.concatenate(([0.], P))


def _bonded_width(case):
    """Width that dissipates G1c, the bonded strip when narrower than the arms"""
    return case.get('bond_width') or case['b']


def elastic_compliance(case, a0, delta=0.05, **kwargs):
    """Compliance ``delta/P`` of the DCB with the initial crack length ``a0``"""
    c = dict(case)
    c['a0'] = a0
    res = solve_dcb(c, [delta], **kwargs)
    return delta/res['P'][-1]


def compliance_cbt(case, a, corrected=True):
    """Compliance of beam theory, with the crack length ``a + chi h``"""
    EI = D11(case)*case['b']
    dh = chi_williams(bending_modulus(case), case['E2'], case['G12'])*arm_thickness(case) if corrected else 0.
    return 2*(np.asarray(a) + dh)**3/(3*EI)


def a_at_opening(case, delta, corrected=False):
    """Crack length reached at the opening ``delta`` according to LEFM"""
    EI = D11(case)*case['b']
    dh = chi_williams(bending_modulus(case), case['E2'], case['G12'])*arm_thickness(case) if corrected else 0.
    ae = np.sqrt(1.5*delta*np.sqrt(EI/(case['G1c']*_bonded_width(case))))
    return max(ae - dh, case['a0'])


def summary(delta, P, delta_ref=None, first_peak=False):
    r"""Metrics of a load-opening curve

    Returns the initial stiffness ``K0`` (least squares through the origin
    of the points below 60% of the opening at the peak, or the secant to the
    peak when there are none), the peak load ``Pmax``, the opening at the
    peak ``delta_Pmax`` and, when ``delta_ref`` is given, the load ``P_ref``
    at the opening ``delta_ref`` (linear interpolation, ``nan`` outside the
    curve). With ``first_peak=True`` the first local maximum is used instead
    of the global one, for curves that stiffen again after the crack passes
    a load point.

    """
    delta = np.asarray(delta, dtype=float)
    P = np.asarray(P, dtype=float)
    i = int(np.argmax(P))
    if first_peak:
        drops = np.flatnonzero(np.diff(P) < 0)
        if drops.size:
            i = int(drops[0])
    pre = (delta > 0) & (delta <= 0.6*delta[i])
    if not np.any(pre):
        pre = (delta > 0) & (delta <= delta[i])
    d, p = delta[pre], P[pre]
    K = float(np.sum(d*p)/np.sum(d*d)) if len(d) else np.nan
    out = dict(K0=K, Pmax=float(P[i]), delta_Pmax=float(delta[i]))
    if delta_ref is not None:
        if delta.min() <= delta_ref <= delta.max():
            out['P_ref'] = float(np.interp(delta_ref, delta, P))
        else:
            out['P_ref'] = np.nan
    return out


def markdown_table(rows, ref=None, delta_ref=None, first_peak=False):
    r"""Markdown table comparing curves

    Parameters
    ----------
    rows : dict
        ``{label: (delta, P)}``.
    ref : str or list of str, optional
        Labels of the references, the relative differences of every other
        row to the first reference are added.
    delta_ref : float, optional
        Opening where the load after the peak is compared.

    """
    refs = [ref] if isinstance(ref, str) else (ref or [])
    s = {k: summary(d, p, delta_ref, first_peak) for k, (d, p) in rows.items()}
    head = '| curve | K0 (N/mm) | Pmax (N) | delta at Pmax (mm) |'
    sep = '|---|---|---|---|'
    if delta_ref is not None:
        head += f' P at delta = {delta_ref:.2f} mm (N) |'
        sep += '---|'
    for r in refs:
        head += f' dK0, dPmax, d(delta) vs {r} |'
        sep += '---|'
    lines = [head, sep]
    for k, v in s.items():
        line = f'| {k} | {v["K0"]:.2f} | {v["Pmax"]:.1f} | {v["delta_Pmax"]:.2f} |'
        if delta_ref is not None:
            line += f' {v["P_ref"]:.1f} |'
        for r in refs:
            if k == r:
                line += ' - |'
                continue
            w = s[r]
            line += (f' {100*(v["K0"]/w["K0"] - 1):+.1f}%, '
                     f'{100*(v["Pmax"]/w["Pmax"] - 1):+.1f}%, '
                     f'{100*(v["delta_Pmax"]/w["delta_Pmax"] - 1):+.1f}% |')
        lines.append(line)
    return '\n'.join(lines)


def resolution(case, terms_per_decay=1.5, points_per_lcz=10., m_max=25):
    r"""Discretisation of the cohesive domain from the convergence study

    The Ritz basis must resolve the decay of the opening ahead of the
    process zone, beam on elastic foundation with `\lambda = (k_o/(4
    D_{11}))^{1/4}`, and the Gauss points must sample the cohesive zone,
    of length `l_{cz} = 0.88 E_2 G_{Ic}/\tau_o^2` (Rice's estimate, Turon
    et al. 2007). The converged discretisation of the reference DCB (15
    terms and 120 points over 17 mm) corresponds to ``m_tsl = 1.5 \lambda
    L_tsl`` and a spacing of the Gauss points at the middle of the domain,
    `\pi L_{tsl}/(2 n_x)`, of about `l_{cz}/6`, i.e. ``nx = 10
    L_tsl/l_cz``.

    Returns
    -------
    lam, lcz, L_tsl_max : float
        ``L_tsl_max`` is the longest cohesive domain resolved with
        ``m_max`` terms.

    """
    lam = (case['k_o']/(4*D11(case)))**0.25
    lcz = 0.88*case['E2']*case['G1c']/case['tau_o']**2
    return lam, lcz, m_max/(terms_per_decay*lam)


def discretize(case, L_tsl, terms_per_decay=1.5, points_per_lcz=10., m_max=25):
    r"""Sets ``L_tsl``, ``m_tsl`` and ``nx`` of ``case`` from :func:`resolution`

    With `\lambda` and `l_{cz}` of :func:`resolution`:

    - `m_{tsl} = \lceil 1.5\,\lambda L_{tsl} \rceil`, at most ``m_max`` = 25
      terms;
    - `n_x = 10\,L_{tsl}/l_{cz}`, but not less than `4\,m_{tsl}` to
      integrate the products of the approximation functions, rounded up to a
      multiple of 10 and with at most 300 points, the Gauss-Legendre rules of
      ``panels`` going up to 304 points.

    Returns a copy of ``case``.

    """
    lam, lcz, _ = resolution(case, terms_per_decay, points_per_lcz, m_max)
    case = dict(case)
    case['L_tsl'] = L_tsl
    case['m_tsl'] = int(min(m_max, np.ceil(terms_per_decay*lam*L_tsl)))
    nx = points_per_lcz*L_tsl/lcz
    # at least 4 points per term, to integrate the products of the functions
    nx = max(nx, 4*case['m_tsl'])
    # the Gauss-Legendre rules of panels go up to 304 points
    case['nx'] = int(min(300, np.ceil(nx/10)*10))
    return case


def plot_curves(ax, curves, style=None):
    """Plots ``{label: (delta, P)}`` with optional ``{label: kwargs}``"""
    style = style or {}
    for k, (d, p) in curves.items():
        ax.plot(d, p, label=k, **style.get(k, {}))
    ax.set_xlabel(r'opening $\delta$ (mm)')
    ax.set_ylabel(r'load $P$ (N)')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
