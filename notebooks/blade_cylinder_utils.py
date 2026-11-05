r"""Utilities of the notebooks sewall1968_blade_cylinder_vibration.ipynb and
card1966_blade_cylinder_buckling.ipynb

Cylinders with integral blade stringers of the NASA tests of Sewall and
Naumann (1968, natural frequencies) and Card and Jones (1966, axial
buckling), modelled as :class:`panels.multidomain.MultiDomain` assemblies:
curved skin panels between the stringers (``cylshell_clpt_sanders``) and one
flat plate strip (``plate_clpt_donnell``) per stringer, connected with
``'SSycte'`` (skin to skin) and ``'BFycte'`` (skin to blade), see
:func:`build_cylinder`.

All quantities are in SI units; :data:`IN`, :data:`LBF`, :data:`PSI` and
:data:`LBS2IN4` convert the U.S. Customary values of the reports.

"""
import time

import numpy as np
from structsolve import freq

from panels.shell import Shell
from panels.multidomain import MultiDomain

_trapz = getattr(np, 'trapezoid', None) or np.trapz

#: one inch, m
IN = 0.0254
#: one pound-force, N
LBF = 4.4482216152605
#: one psi, Pa
PSI = LBF/IN**2
#: one lb-s^2/in^4 (mass density), kg/m^3
LBS2IN4 = LBF/IN**4


def isotropic(E, nu):
    r"""Lamina properties of an isotropic material, `G = E/(2(1 + \nu))`"""
    G = E/(2*(1 + nu))
    return (E, E, nu, G, G, G)


# Flags of an end of the skin and of the blades (0 removes the value, or the
# derivative normal to the end for the keys ending in r). u, v, w are the
# local displacements of each panel; for a blade, whose local axes are those
# of the skin rotated by 90 deg about x, v is radial and w circumferential.
_FREE = dict(u=1, ur=1, v=1, vr=1, w=1, wr=1)
END_FLAGS = {
    # simple support of Sewall and Naumann: skin v = w = 0 (knife-edge
    # rings), u free; blade: radial v = 0 (ring at the stringer tips)
    'S': (dict(_FREE, v=0, w=0), dict(_FREE, v=0)),
    # clamped (potted): skin u = v = w = w,x = 0; blade u = v = w = 0 and
    # v,x = w,x = 0
    'C': (dict(_FREE, u=0, v=0, w=0, wr=0),
          dict(_FREE, u=0, v=0, vr=0, w=0, wr=0)),
    # free
    'F': (dict(_FREE), dict(_FREE)),
    # clamped, u prescribed (loaded end of a buckling analysis)
    'CL': (dict(_FREE, v=0, w=0, wr=0), dict(_FREE, v=0, vr=0, w=0, wr=0)),
    # simple support of Card and Jones, N_x = v = w = M_x = 0 (loaded end)
    'SL': (dict(_FREE, v=0, w=0), dict(_FREE, v=0)),
}


def _set_end(p, end, flags):
    for k, v in flags.items():
        setattr(p, '%s%s' % (end, k), v)


def build_cylinder(R, h, L, E, nu, rho=0., nstiff=60, tb=None, hb=None,
        side='external', bc='SS', mx=12, ny=6, nyb=4, full=False,
        pd_method='penalty', blade_model='plate_clpt_gl', Dy_factor=1.,
        D12_factor=1.):
    r"""Cylinder with ``nstiff`` equally spaced blade stringers

    The skin is divided at the stringers into curved panels. With
    ``full=False`` only the half cylinder `0 \le \theta \le \pi` is
    modelled, bounded by two symmetry planes through the middle of a bay,
    `\theta = 0` and `\theta = \pi`, where `v = w_{,y} = 0`: its
    ``nstiff/2`` stringers are at `\theta_k = (k + 1/2) 2\pi/n_s`, between
    ``nstiff/2 - 1`` full bays and two half bays. With ``full=True`` the
    whole cylinder is modelled, ``nstiff`` bays with a stringer at the edge
    `y = 0` of each, the last bay connected to the first one.

    The blade of each stringer is a flat plate strip of thickness ``tb`` and
    width ``hb``, from the middle surface of the skin to the tip of the
    stringer, connected with ``'BFycte'`` to the edge `y = 0` of the skin
    panel that starts at the stringer. The local axes of the blade are those
    of the skin rotated by 90 deg about `x`, its `y` axis pointing inwards
    (see :mod:`panels.multidomain.connections.kCBFycte`): with ``ycte2 =
    0`` the blade stands on the inner side of the skin (``side='internal'``)
    and with ``ycte2 = hb`` on the outer side (``side='external'``). Without
    ``tb`` the cylinder is unstiffened, divided in the same panels.

    ``Dy_factor`` multiplies the circumferential bending stiffness `D_{22}`
    of every skin panel, `D_y/D` of Sewall and Naumann (1968), Appendix A,
    and ``D12_factor`` its coupling term `D_{12}`: the modified constant
    ``ABD`` matrix of the laminate is given to the panel as its attribute
    ``ABDnxny`` (see :class:`panels.shell.Shell`), with which the stiffness
    matrix is integrated numerically; the mass still comes from the
    laminate. With both factors equal to 1 (default) the attribute is not
    set.

    Parameters
    ----------
    R, h, L : float
        Radius of the middle surface, skin thickness and length, m.
    E, nu, rho : float
        Young's modulus, Pa, Poisson's ratio and density, kg/m^3.
    nstiff : int
        Number of stringers of the full cylinder (even).
    tb, hb : float or None
        Thickness and width of the blades, m.
    side : str
        ``'external'`` or ``'internal'``.
    bc : str or tuple
        The ends `x = 0` and `x = L`, two keys of :data:`END_FLAGS`, e.g.
        ``'SS'``, ``'CC'``, ``'CF'``, ``'FF'`` or ``('C', 'CL')``.
    mx, ny, nyb : int
        Terms along `x` (skin and blades), along `y` in a skin bay (half of
        them, at least 4, in a half bay) and across a blade.
    full : bool
        Whole cylinder instead of the half.
    pd_method : str
        See :class:`panels.multidomain.MultiDomain`.
    blade_model : str
        Model of the blades.
    Dy_factor, D12_factor : float
        Factors of `D_{22}` and `D_{12}` of the skin.

    Returns
    -------
    md : :class:`panels.multidomain.MultiDomain`
        The assembly, with the connections stored, ``conn_method='null-space'``.
    skin, blades : list
        The skin panels, ordered in `\theta`, and the blades.

    """
    lam = isotropic(E, nu)
    dth = 2*np.pi/nstiff
    if full:
        ths = np.arange(nstiff + 1)*dth
        ks = range(nstiff)
    else:
        ths = np.concatenate(([0.], (np.arange(nstiff//2) + 0.5)*dth, [np.pi]))
        ks = range(1, nstiff//2 + 1)
    if isinstance(bc, str):
        bc = (bc[0], bc[1])
    (sk1, bl1), (sk2, bl2) = END_FLAGS[bc[0]], END_FLAGS[bc[1]]
    skin = []
    for i in range(len(ths) - 1):
        half_bay = not full and i in (0, len(ths) - 2)
        p = Shell(group='skin', x0=0., y0=R*ths[i], a=L, b=R*(ths[i+1] - ths[i]),
                  r=R, m=mx, n=max(4, ny//2) if half_bay else ny, plyt=h,
                  stack=[0.], laminaprop=lam, rho=rho,
                  model='cylshell_clpt_sanders')
        _set_end(p, 'x1', sk1)
        _set_end(p, 'x2', sk2)
        _set_end(p, 'y1', _FREE)
        _set_end(p, 'y2', _FREE)
        if Dy_factor != 1. or D12_factor != 1.:
            # ABD order: (A, B, D) of (xx, yy, xy); D22 = [4, 4], D12 = [3, 4]
            ABD = np.array(p.ABD, dtype=float)
            ABD[4, 4] *= Dy_factor
            ABD[3, 4] *= D12_factor
            ABD[4, 3] *= D12_factor
            p.ABDnxny = ABD
        skin.append(p)
    if not full:
        # symmetry planes
        skin[0].y1v = 0
        skin[0].y1wr = 0
        skin[-1].y2v = 0
        skin[-1].y2wr = 0
    conns = []
    for p1, p2 in zip(skin[:-1], skin[1:]):
        conns.append(dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0.))
    if full:
        conns.append(dict(p1=skin[-1], p2=skin[0], func='SSycte',
                          ycte1=skin[-1].b, ycte2=0.))
    blades = []
    if tb is not None:
        for k in ks:
            pskin = skin[k]
            bl = Shell(group='blade', x0=0., y0=pskin.y0, a=L, b=hb, m=mx,
                       n=nyb, plyt=tb, stack=[0.], laminaprop=lam, rho=rho,
                       model=blade_model)
            _set_end(bl, 'x1', bl1)
            _set_end(bl, 'x2', bl2)
            _set_end(bl, 'y1', _FREE)
            _set_end(bl, 'y2', _FREE)
            blades.append(bl)
            ycte2 = 0. if side == 'internal' else hb
            conns.append(dict(p1=pskin, p2=bl, func='BFycte', ycte1=0.,
                              ycte2=ycte2))
    md = MultiDomain(skin + blades, conns, conn_method='null-space',
                     pd_method=pd_method)
    return md, skin, blades


def skin_w(md, c, skin, R, nx=61, npt=13):
    r"""Normal displacement `w(x, \theta)` of the skin on a grid

    Returns ``x`` (``nx`` values, m), ``theta`` (increasing, rad) and ``w``
    with shape ``(ntheta, nx)``.
    """
    res = md.uvw(c, group='skin', gridx=nx, gridy=npt)
    ths, ws = [], []
    for i, p in enumerate(skin):
        th = (p.y0 + res['y'][i][:, 0])/R
        w = res['w'][i]
        if i > 0:
            th, w = th[1:], w[1:]
        ths.append(th)
        ws.append(w)
    x = res['x'][0][0, :]
    return x, np.concatenate(ths), np.vstack(ws)


def classify(x, th, w, bc, nmax=40):
    r"""Wave numbers ``(m, n)`` of a mode

    ``n`` is the harmonic of `\theta` with the largest projection of `w` on
    the section of maximum `|w|` (`\cos n\theta` for the half cylinder, `\cos
    n\theta` and `\sin n\theta` for the full one); ``m`` counts the sign
    changes of `w` along `x` at the generator of maximum `|w|` (values below
    5 % of the maximum ignored), with the numbering of Sewall and Naumann
    (1968), Fig. 4: the number of half-waves for ``'SS'`` and ``'CC'`` (and
    for buckling), the number of nodes plus one for ``'CF'`` and the number
    of nodes for ``'FF'``.
    """
    i, j = np.unravel_index(np.argmax(np.abs(w)), w.shape)
    wt = w[:, j]
    ns = np.arange(nmax + 1)
    pc = _trapz(wt[None, :]*np.cos(ns[:, None]*th[None, :]), th, axis=1)
    if th[-1] > 1.5*np.pi:
        ps = _trapz(wt[None, :]*np.sin(ns[:, None]*th[None, :]), th, axis=1)
    else:
        ps = 0.
    proj = np.sqrt(pc**2 + ps**2)
    proj[0] /= 2
    n = int(np.argmax(proj))
    wx = w[i, :]
    wx = wx[np.abs(wx) > 0.05*np.abs(wx).max()]
    changes = int(np.sum(np.diff(np.sign(wx)) != 0))
    m = changes if bc == 'FF' else changes + 1
    return m, n


def frequencies(md, skin, R, bc, num_eigvalues=80, fmin=1., check_rtol=1.e-2):
    r"""Natural frequencies of the assembly with their ``(m, n)``

    Returns a list of tuples ``(f, m, n, c)``, the frequency in Hz, the wave
    numbers of :func:`classify` and the mode, in increasing frequency, the
    rigid-body modes (``f < fmin``) excluded, the run time, s, and the
    largest relative residual `\|(\boldsymbol{K} - \omega^2
    \boldsymbol{M}) \boldsymbol{\phi}\| / \|\boldsymbol{K}
    \boldsymbol{\phi}\|` of the returned modes. ``check_rtol`` is the
    tolerance of the relative residual in ``structsolve.freq``: the residual
    of the lowest modes of the free-free shells, whose spectrum starts at
    the rigid-body modes, exceeds ``1e-6`` with backward errors of the order
    of ``1e-14``, so the residual is checked here instead and returned.
    """
    t0 = time.time()
    K = md.reduce(md.calc_kC(silent=True))
    M = md.reduce(md.calc_kM(silent=True))
    eigvals, eigvecs = freq(K, M, tol=0, sparse_solver=True, silent=True,
                            num_eigvalues=num_eigvalues, symmetric=True,
                            check_rtol=check_rtol)
    omegas = np.sqrt(np.abs(eigvals.real))
    out = []
    rres = 0.
    for k in np.argsort(omegas):
        f = omegas[k]/(2*np.pi)
        if f < fmin:
            continue
        phi = eigvecs[:, k].real
        Kphi = K @ phi
        rres = max(rres, np.linalg.norm(Kphi - omegas[k]**2*(M @ phi))
                   /np.linalg.norm(Kphi))
        c = md.expand(phi)
        x, th, w = skin_w(md, c, skin, R)
        m, n = classify(x, th, w, bc)
        out.append((f, m, n, c))
    return out, time.time() - t0, rres


def lowest_by_mn(modes):
    r"""Lowest frequency of each ``(m, n)`` in the output of
    :func:`frequencies`, a dict"""
    out = {}
    for f, m, n, c in modes:
        if (m, n) not in out:
            out[(m, n)] = f
    return out


def buckling_clamped(R, h, L, E, nu, tb, hb, side, mx, ny, nyb,
        num_eigvalues=10, delta=1.e-4):
    r"""Axial buckling of the half cylinder with clamped ends

    The end `x = 0` is clamped, `u = v = w = w_{,x} = 0`, and the end `x =
    L` is clamped with the uniform shortening `u = -\lambda` ``delta``
    prescribed exactly (``pd_method='null-space'``) on the skin and on the
    blades, as between the platens of a testing machine. The pre-buckling
    state is the linear static solution, see
    :meth:`panels.multidomain.MultiDomain.linear_buckling`.

    Returns the critical axial load of the whole cylinder ``P``, N, the
    modes ``[(lambda, m, n, c), ...]`` with `\lambda > 0` in increasing
    order, the assembly, the skin panels and the run time, s.
    """
    t0 = time.time()
    md, skin, blades = build_cylinder(R, h, L, E, nu, tb=tb, hb=hb,
            side=side, bc=('C', 'CL'), mx=mx, ny=ny, nyb=nyb,
            pd_method='null-space')
    for p in md.panels:
        p.add_distr_pd_fixed_x(L, ku=1., funcu=lambda y: -delta, cte=False)
    eigvals, eigvecs = md.linear_buckling(num_eigvalues=num_eigvalues,
            silent=True, tol=0, sparse_solver=True)
    c1 = md.linear_static(lbd=1., silent=True)
    F = md.reaction_resultant(c1, [(p, 'x', L) for p in md.panels], lbd=1.,
                              NLgeom=False)
    P1 = 2*abs(F[0])
    modes = []
    for k in range(len(eigvals)):
        c = eigvecs[:, k].real
        x, th, w = skin_w(md, c, skin, R)
        m, n = classify(x, th, w, 'CC')
        modes.append((eigvals[k].real, m, n, c))
    modes = sorted((mode for mode in modes if mode[0] > 0), key=lambda mode: mode[0])
    return modes[0][0]*P1, modes, md, skin, time.time() - t0


def buckling_ss(R, h, L, E, nu, tb, hb, side, mx, ny, nyb, num_eigvalues=10,
        eps=1.e-4):
    r"""Axial buckling of the half cylinder with simply supported ends

    Both ends have `N_x = v = w = M_x = 0` in the buckling mode (`u` and
    `w_{,x}` free), the simple support of Card and Jones (1966). The uniform
    strain ``-eps`` is applied at both ends as the forces per unit length
    `E h` ``eps`` on the skin and `E t_b` ``eps`` on the blades, compressive
    and in equilibrium, and the rigid axial translation is removed by `u =
    0` prescribed at the single point `x = L/2`, `\theta = 0`, which carries
    no reaction in the pre-buckling state. The pre-buckling state is the
    linear static solution, see
    :meth:`panels.multidomain.MultiDomain.linear_buckling`.

    Returns the critical axial load of the whole cylinder ``P``, N, the
    modes ``[(lambda, m, n, c), ...]`` with `\lambda > 0` in increasing
    order, the assembly, the skin panels and the run time, s.
    """
    t0 = time.time()
    md, skin, blades = build_cylinder(R, h, L, E, nu, tb=tb, hb=hb,
            side=side, bc=('SL', 'SL'), mx=mx, ny=ny, nyb=nyb,
            pd_method='null-space')
    P1 = 0.
    for p in skin:
        p.add_distr_load_fixed_x(0., funcx=lambda y: E*h*eps, cte=False)
        p.add_distr_load_fixed_x(L, funcx=lambda y: -E*h*eps, cte=False)
        P1 += E*h*eps*p.b
    for p in blades:
        p.add_distr_load_fixed_x(0., funcx=lambda y: E*tb*eps, cte=False)
        p.add_distr_load_fixed_x(L, funcx=lambda y: -E*tb*eps, cte=False)
        P1 += E*tb*eps*p.b
    P1 *= 2
    skin[0].add_point_pd(L/2, 0., 1., 0., None, None, None, None)
    eigvals, eigvecs = md.linear_buckling(num_eigvalues=num_eigvalues,
            silent=True, tol=0, sparse_solver=True)
    modes = []
    for k in range(len(eigvals)):
        c = eigvecs[:, k].real
        x, th, w = skin_w(md, c, skin, R)
        m, n = classify(x, th, w, 'SS')
        modes.append((eigvals[k].real, m, n, c))
    modes = sorted((mode for mode in modes if mode[0] > 0), key=lambda mode: mode[0])
    return modes[0][0]*P1, modes, md, skin, time.time() - t0


def buckling_membrane(R, h, L, E, nu, tb, hb, side, bc, mx, ny, nyb,
        num_eigvalues=10, eps=1.e-4):
    r"""Axial buckling of the half cylinder about a membrane pre-buckling state

    The pre-buckling state is the uniform axial strain ``-eps`` without
    Poisson expansion, the constant stress resultants `N_{xx} = -E h`
    ``eps`` of the skin panels and `-E t_b` ``eps`` of the blades (the
    attribute ``Nxx`` of each panel), as in the theory of Card and Jones
    (1966). ``bc='CC'``: both ends clamped, `u = v = w = w_{,x} = 0`;
    ``bc='SS'``: both ends with `N_x = v = w = M_x = 0`, the rigid axial
    translation removed by `u = 0` at the point `x = L/2`, `\theta = 0`.

    Returns the critical axial load of the whole cylinder ``P``, N, the
    modes ``[(lambda, m, n, c), ...]`` with `\lambda > 0` in increasing
    order, the assembly, the skin panels and the run time, s.
    """
    from structsolve import lb
    t0 = time.time()
    if bc == 'CC':
        md, skin, blades = build_cylinder(R, h, L, E, nu, tb=tb, hb=hb,
                side=side, bc='CC', mx=mx, ny=ny, nyb=nyb)
    else:
        md, skin, blades = build_cylinder(R, h, L, E, nu, tb=tb, hb=hb,
                side=side, bc=('SL', 'SL'), mx=mx, ny=ny, nyb=nyb,
                pd_method='null-space')
        skin[0].add_point_pd(L/2, 0., 1., 0., None, None, None, None)
    P1 = 0.
    for p in skin:
        p.Nxx = -E*h*eps
        P1 += E*h*eps*p.b
    for p in blades:
        p.Nxx = -E*tb*eps
        P1 += E*tb*eps*p.b
    P1 *= 2
    K = md.reduce(md.calc_kC(silent=True))
    KG = md.reduce(md.calc_kG(silent=True))
    eigvals, eigvecs = lb(K, KG, tol=0, sparse_solver=True, silent=True,
                          num_eigvalues=num_eigvalues)
    eigvecs = md.expand(eigvecs)
    modes = []
    for k in range(len(eigvals)):
        c = eigvecs[:, k].real
        x, th, w = skin_w(md, c, skin, R)
        m, n = classify(x, th, w, 'SS')
        modes.append((eigvals[k].real, m, n, c))
    modes = sorted((mode for mode in modes if mode[0] > 0), key=lambda mode: mode[0])
    return modes[0][0]*P1, modes, md, skin, time.time() - t0
