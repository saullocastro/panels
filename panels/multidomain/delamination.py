r"""
Delamination test coupons: DCB, ENF and MMB with cohesive interfaces

:func:`.create_coupon` builds the double cantilever beam (DCB), end-notched
flexure (ENF) and mixed-mode bending (MMB) specimens as a
:class:`.MultiDomain` of two arms, each one split along `x` into a cracked
domain, ``nd`` cohesive domains and, optionally, a far domain, connected by
``'SSxcte'`` and imposed exactly with the null-space method. The arms are
bonded over the uncracked length by the cohesive connections ``'CZM'`` of
:mod:`panels.multidomain.connections.czm`, one per pair of domains, and are
in contact-only interaction over the pre-crack. :func:`.solve_coupon` traces
the load-displacement curve under displacement control with
``structsolve``, the damage history being updated at the converged
increments only.

Notes
-----
The coordinate `x` starts at the load line of the split end, from which the
crack length `a_0` is measured, ASTM D6671-22 [1]_. The upper arm is ``p1``
of the cohesive connections, on the positive side of the interface.

MMB, [1]_: the lower arm is held by a hinge at `x = 0`, `w = u = 0`, and
rests on a roller at `x = 2L`, `w = 0`; the rigid lever is hinged to the
upper arm at `x = 0` and rests on it at the saddle, `x = L`, and its load
point at the lever length `c` beyond the saddle moves down by `\delta`. With
small rotations of the lever:

.. math::

    \frac{c}{L} \bar{w}^t(0) - \left(1 + \frac{c}{L}\right) \bar{w}^t(L) =
    \delta

with `\bar{w}^t` the mean deflection of the upper arm across the width
along each line, imposed by :meth:`.MultiDomain.add_linear_constraint`
with `\delta = \lambda`, the load factor. The lever load `P` is the
generalized reaction of the incremented prescriptions,
:meth:`.MultiDomain.reaction_pd`, which by virtual work transmits the line
loads `P c/L` (upward, `x = 0`) and `P (c + L)/L` (downward, `x = L`). The
ENF is the MMB with `c = 0`. The lever weight is not modelled.

DCB: the lower arm is held at `x = 0` by `w = u = 0`, the upper arm is
opened by `w^t(0) = \delta` along the load line, and the slope of the upper
arm is zero at the far end, `x = x_{end}`, as in the DCB of D'Souza et al.
[2]_; the reaction of the opening is the load.

References
----------
.. [1] ASTM D6671/D6671M-22, Standard Test Method for Mixed Mode I-Mode II
   Interlaminar Fracture Toughness of Unidirectional Fiber Reinforced
   Polymer Matrix Composites, ASTM International, 2022.
.. [2] D'Souza, N. T. A., Tijs, B. H. A. H., Castro, S. G. P., A multidomain
   semi-analytical cohesive zone model of the double cantilever beam
   (submitted), 2026.

"""
import numpy as np

from panels.shell import Shell
from panels.multidomain import MultiDomain


_SDT = ('fsdt', 'tsdt')


def _is_sdt(model):
    return any(k in model for k in _SDT)


def create_coupon(kind, laminaprop, plyt, stack, b, a0, laws, L=None, c=None,
                  x_end=None, model='plate_clpt_donnell', nd=2, L_coh=None,
                  m_crack=10, m_coh=15, m_far=10, n=6, nr_x_gauss=None,
                  nr_y_gauss=10, k_contact=None, mode_ratio='per-law',
                  lever='mean', u_hinge_z=0., fsdt_shear_correction=5/6.,
                  dcb_bc='hinge', far='czm', conn_method='null-space',
                  penalty_factor=1.):
    r"""DCB, ENF or MMB coupon with cohesive interfaces

    Parameters
    ----------
    kind : str
        ``'DCB'``, ``'ENF'`` or ``'MMB'``.
    laminaprop : tuple
        Ply properties ``(E11, E22, nu12, G12, G13, G23)``.
    plyt : float
        Ply thickness.
    stack : list
        Ply angles of one arm, from its top face to its bottom face.
    b : float
        Width.
    a0 : float
        Crack length, from the load line `x = 0`.
    laws : list of :class:`.czm.BilinearLaw`
        Superposed laws of the cohesive interface.
    L : float, optional
        Half-span (ENF, MMB).
    c : float, optional
        Lever length (MMB), from the saddle.
    x_end : float, optional
        End of the model: ``2*L`` for the ENF and MMB, required for the DCB.
    model : str, optional
        Plate model of the domains, e.g. ``'plate_clpt_donnell'`` or
        ``'plate_fsdt_donnell'``.
    nd : int, optional
        Number of cohesive domains of each arm over ``L_coh``.
    L_coh : float, optional
        Length of the refined cohesive region from the crack tip. ``None``
        for the whole bonded length.
    m_crack, m_coh, m_far, n : int, optional
        Terms along `x` of the cracked, cohesive and far domains, and along
        `y` of all the domains.
    nr_x_gauss : int or None, optional
        Gauss-Legendre points along `x` of each cohesive domain, by default
        ``4*m_coh``; the far domain uses ``4*m_far``.
    nr_y_gauss : int, optional
        Gauss-Legendre points along `y` of the interfaces.
    k_contact : float or None, optional
        Normal stiffness of the contact over the pre-crack, by default the
        sum of the stiffnesses ``KI`` of the laws.
    mode_ratio : str, optional
        See :mod:`panels.multidomain.connections.czm`.
    lever : str, optional
        ``'mean'`` (default), the lever constraint on the mean deflections
        across the width, or ``'rigid'``, with the deflection also uniform
        along each loading line.
    u_hinge_z : float, optional
        Thickness coordinate, from the mid-plane of the lower arm, where
        `u = 0` is imposed at its hinge.
    fsdt_shear_correction : float or str, optional
        Shear correction factor of the shear deformation theories.
    dcb_bc : str, optional
        DCB only: ``'hinge'`` (default), see the module notes, or
        ``'symmetric'``, the reference DCB of D'Souza et al.: the far end
        `x = x_{end}` of both arms clamped and the arms opened to `\pm
        \delta/2` at `x = 0`.
    far : str, optional
        Bond of the far domain, beyond ``L_coh``: ``'czm'`` (default), the
        cohesive law with ``4*m_far`` points along `x`, or ``'sb'``, the
        perfect bond ``'SB'`` imposed exactly.
    conn_method : str, optional
        ``'null-space'`` (default) or ``'penalty'`` for the connections
        ``'SSxcte'`` (and ``'SB'``), see :class:`.MultiDomain`; the
        prescribed displacements and the lever are always imposed exactly.
    penalty_factor : float, optional
        With ``conn_method='penalty'``, factor on the penalties of
        :func:`.calc_kt_kr` (``'SSxcte'``) and of the default ``'SB'``.

    Returns
    -------
    md : :class:`.MultiDomain`
    info : dict
        ``'top'`` and ``'bot'`` (lists of domains along `x`), ``'x'`` (the
        boundaries of the domains), ``'cohesive'`` (the connections
        ``'CZM'`` of the bonded length), ``'kind'``, ``'a0'``, ``'L'``,
        ``'c'``, ``'h'`` (thickness of one arm).

    """
    kind = kind.upper()
    if kind not in ('DCB', 'ENF', 'MMB'):
        raise ValueError("kind must be 'DCB', 'ENF' or 'MMB'")
    if kind == 'ENF':
        c = 0.
    if kind in ('ENF', 'MMB'):
        if L is None or c is None and kind == 'MMB':
            raise ValueError('L (and c for the MMB) are required')
        x_end = 2*L if x_end is None else x_end
    elif x_end is None:
        raise ValueError('x_end is required for the DCB')
    h = plyt*len(stack)
    bonded = x_end - a0
    L_coh = bonded if L_coh is None else min(L_coh, bonded)
    xs = [0., a0] + list(a0 + L_coh*np.arange(1, nd + 1)/nd)
    if x_end - xs[-1] > 1e-9*x_end:
        xs.append(x_end)
    xs = np.array(xs)
    if nr_x_gauss is None:
        nr_x_gauss = 4*m_coh

    def shell(group, x0, a, m):
        p = Shell(group=group, x0=x0, y0=0., a=a, b=b, m=m, n=n, plyt=plyt,
                  stack=stack, laminaprop=laminaprop, model=model)
        if _is_sdt(model):
            p.fsdt_shear_correction = fsdt_shear_correction
        #NOTE all the edges free: the supports and the connections are
        #     imposed exactly, as constraints
        fields = ('u', 'v', 'w', 'phix', 'phiy') if _is_sdt(model) else \
                 ('u', 'v', 'w')
        for d in 'xy':
            for e in '12':
                for f in fields:
                    for r in ('', 'r'):
                        setattr(p, d + e + f + r, 1.)
        return p

    top, bot = [], []
    for i in range(len(xs) - 1):
        x0, a = xs[i], xs[i + 1] - xs[i]
        if i == 0:
            m = m_crack
        elif i <= nd:
            m = m_coh
        else:
            m = m_far
        top.append(shell('top', x0, a, m))
        bot.append(shell('bot', x0, a, m))

    conn = []
    for arm in (top, bot):
        for p1, p2 in zip(arm[:-1], arm[1:]):
            ci = dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0.)
            if conn_method == 'penalty':
                from panels.multidomain.connections import calc_kt_kr
                kt, kr = calc_kt_kr(p1, p2, 'xcte')
                ci['kt'] = penalty_factor*kt
                ci['kr'] = penalty_factor*kr
            conn.append(ci)
    if k_contact is None:
        k_contact = sum(law.KI for law in laws)
    conn.append(dict(p1=top[0], p2=bot[0], func='CZM', laws=[],
                     k_contact=k_contact, nr_x_gauss=4*m_crack,
                     nr_y_gauss=nr_y_gauss))
    cohesive = []
    for i in range(1, len(top)):
        if i > nd and far == 'sb':
            conn.append(dict(p1=top[i], p2=bot[i], func='SB'))
            continue
        nx = nr_x_gauss if i <= nd else 4*m_far
        ci = dict(p1=top[i], p2=bot[i], func='CZM', laws=laws,
                  nr_x_gauss=nx, nr_y_gauss=nr_y_gauss,
                  mode_ratio=mode_ratio)
        conn.append(ci)
        cohesive.append(ci)

    md = MultiDomain(bot + top, conn, conn_method=conn_method)

    # supports, exact, constant
    symmetric = kind == 'DCB' and dcb_bc == 'symmetric'
    if symmetric:
        for p in (top[-1], bot[-1]):
            p.add_distr_pd_fixed_x(p.a, ku=1., funcu=lambda y: 0., kv=1.,
                                   funcv=lambda y: 0., kw=1.,
                                   funcw=lambda y: 0., kphix=1.,
                                   funcphix=lambda y: 0.)
    else:
        bot[0].add_distr_pd_fixed_x(0., ku=1., funcu=lambda y: 0., kw=1.,
                                    funcw=lambda y: 0., z=u_hinge_z)
        bot[0].add_point_pd(0., b/2, None, None, 1., 0., None, None)
    if kind in ('ENF', 'MMB'):
        bot[-1].add_distr_pd_fixed_x(bot[-1].a, kw=1., funcw=lambda y: 0.)

    def at(arm, x):
        for p in arm:
            if p.x0 - 1e-9 <= x <= p.x0 + p.a + 1e-9:
                return p, min(max(x - p.x0, 0.), p.a)
        raise ValueError('x = {0} outside the coupon'.format(x))

    if symmetric:
        top[0].add_distr_pd_fixed_x(0., kw=1., funcw=lambda y: 0.5,
                                    cte=False)
        bot[0].add_distr_pd_fixed_x(0., kw=1., funcw=lambda y: -0.5,
                                    cte=False)
    elif kind == 'DCB':
        top[0].add_distr_pd_fixed_x(0., kw=1., funcw=lambda y: 1.,
                                    cte=False)
        if _is_sdt(model):
            top[-1].add_distr_pd_fixed_x(top[-1].a, kphix=1.,
                                         funcphix=lambda y: 0.)
        else:
            top[-1].x2wr = 0
    else:
        pe, xe = at(top, 0.)
        pm, xm = at(top, L)
        md.add_linear_constraint([(pe, 'w', xe, None, 0., c/L),
                                  (pm, 'w', xm, None, 0., -(1 + c/L))],
                                 value=1., cte=False, label='lever')
        if lever == 'rigid':
            # w uniform along each loading line: w - mean(w) = 0 at the
            # Gauss-Legendre points along y, as constant constraints
            from numpy.polynomial.legendre import leggauss
            ys = (leggauss(max(n, 4))[0] + 1)*b/2
            for p, x in ((pe, xe), (pm, xm)):
                for y in ys[1:]:
                    md.add_linear_constraint([(p, 'w', x, y, 0., 1.),
                                              (p, 'w', x, ys[0], 0., -1.)],
                                             value=0., cte=True,
                                             label='rigid line')

    info = dict(top=top, bot=bot, x=xs, cohesive=cohesive, kind=kind,
                a0=a0, L=L, c=c, h=h, b=b, x_end=x_end)
    return md, info


def crack_length(md, info, threshold=1.):
    r"""Crack length from the load line: the largest `x` up to which all
    the points of the cohesive interfaces across the width have every damage
    variable at ``threshold`` or above, plus `a_0`"""
    a = info['a0']
    for ci in info['cohesive']:
        iface = md._czm.get(id(ci))
        if iface is None or iface.d.size == 0:
            continue
        nx, ny = iface.nr_x_gauss, iface.nr_y_gauss
        dmin = iface.d.min(axis=0).reshape(ny, nx)
        failed = np.all(dmin >= threshold, axis=0)
        x = iface.p_top.x0 + iface.x[:nx]
        if failed.all():
            a = iface.p_top.x0 + iface.p_top.a
            continue
        idx = np.argmin(failed)
        if idx > 0:
            a = x[idx - 1]
        break
    return a


def load_line_control_vector(md, info):
    r"""Control vector of the separation at the load line, in the reduced
    unknowns

    The controlled quantity is `q = s_w \bar\Delta_w(0) + s_u
    \bar\Delta_u(0)`, the mean opening and sliding across the width at the
    load line `x = 0` (the faces of the arms in contact, see
    :mod:`.czm`), weighted by their values `s_w`, `s_u` in the linear
    solution for a unit load parameter, such that `q` grows from the start.
    Returns ``psi`` with ``psi @ c_r`` the part of `q` that depends on the
    reduced unknowns, see :meth:`.MultiDomain.get_Z`.

    Notes
    -----
    The crack opens and slides at the load line monotonically while it
    grows, also when the load and the load-point displacement decrease, so
    that `q` can control the analysis through the snap-backs of the
    load-displacement curve (indirect displacement control, de Borst
    1987). The part `\lambda\,\partial q/\partial \lambda` of the
    prescribed displacement is left out of ``psi``; it decreases the
    controlled quantity where the load-point displacement decreases, so
    that ``psi @ c_r`` grows at least as fast as `q` there.

    """
    from panels.multidomain.connections import nullspace, kCsdt
    size = md.get_size()
    top, bot = info['top'][0], info['bot'][0]
    rows = {}
    for comp, zt, zb in ((nullspace.W, 0., 0.),
                         (nullspace.U, kCsdt._face(top, -1),
                          kCsdt._face(bot, +1))):
        rows[comp] = (nullspace.mean_row(top, comp, 0., None, size, zt)
                      - nullspace.mean_row(bot, comp, 0., None, size, zb))
    c_lin = md.linear_static(lbd=1.)
    sw = float((rows[nullspace.W] @ c_lin)[0])
    su = float((rows[nullspace.U] @ c_lin)[0])
    row = sw*rows[nullspace.W] + su*rows[nullspace.U]
    psi = np.asarray((md.get_Z().T @ row.T).todense()).ravel()
    return psi


def solve_coupon(md, info, delta_max, initialInc=0.05, maxInc=None,
                 minInc=1e-6, NL_method='NR', NLgeom=True, relTOL=1e-9,
                 maxNumIter=40, record=None, silent=True, control='load',
                 a_stop=None, max_steps=2000, **analysis_kw):
    r"""Load-displacement curve of a coupon of :func:`.create_coupon`

    With ``control='load'`` (default), displacement control, `\delta =
    \lambda`, up to ``delta_max``, with the Newton-Raphson method of
    ``structsolve`` and its automatic cutting of the increments. With
    ``control='load-line'``, indirect displacement control of the separation
    at the load line, :func:`.load_line_control_vector`, which follows the
    snap-backs of unstable crack growth, until `\delta` reaches
    ``delta_max`` or the crack length reaches ``a_stop``; the step lengths
    are in the units of `\delta` in the linear regime. The cohesive history
    is updated at the converged increments only, see
    :meth:`.MultiDomain.get_on_converged`.

    Parameters
    ----------
    md, info
        From :func:`.create_coupon`.
    delta_max : float
        Final displacement of the load point (MMB, ENF: lever load point;
        DCB: opening of the upper arm).
    initialInc, maxInc, minInc : float, optional
        Increments of `\delta`.
    NL_method : str, optional
        ``'NR'`` or an arc-length method of ``structsolve``.
    record : callable, optional
        ``record(an, c, row)`` called after each converged increment with
        the row of results.

    Returns
    -------
    res : dict of np.ndarray
        ``'delta'``, ``'P'`` (load: reaction of the incremented
        prescriptions), ``'a'`` (crack length, :func:`.crack_length`),
        ``'dissipated'``, ``'max_d'``.
    an : ``structsolve.Analysis``

    """
    from structsolve import Analysis
    md.reset_cohesive()
    rows = []

    def on_converged(an, c):
        lbd = an.increments[-1]
        P = md.reaction_pd(c, lbd)['inc']
        diss = sum(iface.dissipated for iface in md.cohesive_interfaces())
        max_d = max((iface.d.max() for iface in md.cohesive_interfaces()
                     if iface.d.size), default=0.)
        row = dict(delta=lbd, P=P, a=crack_length(md, info),
                   dissipated=diss, max_d=max_d)
        rows.append(row)
        if record is not None:
            record(an, c, row)

    an = Analysis(*md.get_reduced_functions())
    an.NL_method = NL_method
    an.initialInc = initialInc
    an.maxInc = maxInc if maxInc is not None else initialInc
    an.minInc = minInc
    an.final_load_factor = delta_max
    an.relTOL = relTOL
    an.maxNumIter = maxNumIter
    an.on_converged = md.get_on_converged(callback=on_converged)
    if control == 'load-line':
        an.NL_method = 'displacement_control'
        an.control_vector = load_line_control_vector(md, info)
        an.final_load_factor = None
        an.maxArcLength = 1e9

        def stop(an):
            if len(rows) >= max_steps:
                return True
            r = rows[-1]
            if r['delta'] >= delta_max:
                return True
            return a_stop is not None and r['a'] >= a_stop
        an.stop_criterion = stop
    elif control != 'load':
        raise ValueError("control must be 'load' or 'load-line'")
    for k, v in analysis_kw.items():
        setattr(an, k, v)
    an.static(NLgeom=NLgeom, silent=silent)
    res = {k: np.array([r[k] for r in rows]) for k in
           ('delta', 'P', 'a', 'dissipated', 'max_d')}
    return res, an
