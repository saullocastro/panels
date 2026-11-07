r"""Cylinders with Z-section stringers made of three plates

Model module of ``test_cylinder_z_stiffened_peterson1959.py``: a circular
cylinder with ``nstiff`` identical Z-section stringers, equally spaced,
built as a :class:`panels.multidomain.MultiDomain` assembly of the whole
cylinder (:func:`build_cylinder_z`), and its linear buckling under axial
compression about a membrane pre-buckling state (:func:`buckling_membrane`).
Each stringer is assembled from three plates of the same thickness `t_z`,
with the dimensions of their middle lines (:func:`z_midlines`):

- the attached flange, riveted to the skin, of width `b_f` from the web to
  its free edge: the strip of skin under it is a curved panel
  ``cylshell_clpt_sanders`` with the thickness `h + t_z` and the offset
  `\mp t_z/2`, its reference surface remaining the middle surface of the
  skin (a continuous bond);
- the web, a flat plate ``plate_clpt_gl`` normal to the skin, connected with
  ``'BFycte'`` to the edge of the strip, from the middle surface of the skin
  to the middle surface of the free flange, of width `H_w`;
- the free flange, a flat plate ``plate_clpt_gl`` normal to the web,
  connected with ``'BFycte'`` at the free edge of the web, of width `b_l`
  from the middle surface of the web, on the side of the web opposite to
  the attached flange.

The bays of skin between the strips are curved panels
``cylshell_clpt_sanders``, connected with ``'SSycte'``. The Z section has no
plane of symmetry through the axis of the cylinder, so the whole
circumference is modelled.

All quantities are in SI units or in any other consistent set;
:data:`IN`, :data:`LBF` and :data:`PSI` convert U.S. Customary values.

Notes
-----
The local axes of a plate connected with ``'BFycte'`` are those of the
domain it is connected to rotated by 90 degrees about `x`
(:mod:`panels.multidomain.connections.kCBFycte`): the `y` axis of a web
points inwards, so a web stands inside the skin when its edge `y = 0` is
connected (``ycte2 = 0``) and outside when its edge `y = H_w` is; the `y`
axis of a free flange points along `-\theta`, so a free flange extends
towards `+\theta` from its edge `y = b_l` and towards `-\theta` from its
edge `y = 0`.

The web starts at the middle surface of the skin, so that it overlaps the
skin and the attached flange over `h/2 + t_z/2`; the section as modelled
is given by :func:`section_properties`.

"""
import time

import numpy as np
from numpy import pi
from structsolve import lb

from panels.shell import Shell
from panels.multidomain import MultiDomain

_trapz = getattr(np, 'trapezoid', None) or np.trapz

#: one inch, m
IN = 0.0254
#: one pound-force, N
LBF = 4.4482216152605
#: one psi, Pa
PSI = LBF/IN**2

_FREE = dict(u=1, ur=1, v=1, vr=1, w=1, wr=1)
#: flags of an end, ``(skin, plates)``: ``'C'`` clamped, `u = v = w =
#: w_{,x} = 0` on the skin and the plates; ``'S'`` simply supported, `v = w
#: = 0` on the skin and the plates (both in-section displacements of each
#: plate), `u` free
END_FLAGS = {
    'C': (dict(_FREE, u=0, v=0, w=0, wr=0),
          dict(_FREE, u=0, v=0, vr=0, w=0, wr=0)),
    'S': (dict(_FREE, v=0, w=0), dict(_FREE, v=0, w=0)),
}


def z_midlines(tz, flange, depth, free_flange, h=0., bend_radius=0.):
    r"""Middle-line dimensions of the three plates of a Z section

    Parameters
    ----------
    tz : float
        Thickness of the sheet.
    flange : float
        Width of the attached flange, from its free edge to the outer face
        of the web.
    depth : float
        Depth of the section, from the face of the attached flange on the
        skin to the outer face of the free flange.
    free_flange : float
        Width of the free flange, from the outer face of the web to its free
        edge.
    h : float
        Skin thickness; the web of the model starts at the middle surface
        of the skin.
    bend_radius : float
        Inner radius of the two bends; each bend of centre-line radius
        `r_c` = ``bend_radius`` `+ t_z/2` shortens the adjacent flange by
        `(2 - \pi/2) r_c`, so that the plates have the length of the centre
        line of the bent sheet; ``0`` gives square corners.

    Returns
    -------
    dict
        ``tz``, ``bf`` (width of the attached flange, from the middle
        surface of the web), ``Hw`` (width of the web, from the middle
        surface of the skin to the middle surface of the free flange) and
        ``bl`` (width of the free flange, from the middle surface of the
        web).

    """
    rc = bend_radius + tz/2 if bend_radius > 0 else 0.
    cut = (2 - pi/2)*rc
    return dict(tz=tz, bf=flange - tz/2 - cut, Hw=h/2 + depth - tz/2,
                bl=free_flange - tz/2 - cut)


def section_properties(sec, h, E, nu):
    r"""Properties of the stringer as modelled, per stringer

    The rectangles of the attached flange (centred at `h/2 + t_z/2` from
    the middle surface of the skin), of the web (from the middle surface of
    the skin, overlapping skin and flange) and of the free flange.
    Distances from the middle surface of the skin, positive away from the
    skin on the side of the stringer.

    Returns
    -------
    dict
        ``A`` area, ``zc`` centroid, ``Io`` moment of inertia about the
        middle surface of the skin (bending in the radial plane), ``J``
        torsion constant `\sum b t_z^3/3`, ``EA``, ``EIo`` and ``GJ``.

    """
    t = sec['tz']
    zf = h/2 + t/2
    Hw = sec['Hw']
    rects = [(sec['bf']*t, zf, sec['bf']*t**3/12),
             (Hw*t, Hw/2, t*Hw**3/12),
             (sec['bl']*t, Hw, sec['bl']*t**3/12)]
    A = sum(r[0] for r in rects)
    zc = sum(r[0]*r[1] for r in rects)/A
    Io = sum(r[2] + r[0]*r[1]**2 for r in rects)
    J = (sec['bf'] + Hw + sec['bl'])*t**3/3
    G = E/(2*(1 + nu))
    return dict(A=A, zc=zc, Io=Io, J=J, EA=E*A, EIo=E*Io, GJ=G*J)


def _isotropic(E, nu):
    G = E/(2*(1 + nu))
    return (E, E, nu, G, G, G)


def _set_end(p, end, flags):
    for k, v in flags.items():
        setattr(p, '%s%s' % (end, k), v)


def build_cylinder_z(R, h, L, E, nu, nstiff, sec, side='internal',
        flange_side=-1, bc='CC', mx=12, ny=6, nf=4, nw=4, nl=4,
        attached_flange=True, free_flange=True):
    r"""Whole cylinder with ``nstiff`` equally spaced Z stringers

    The circumference is divided, for each stringer `k`, into the strip of
    skin under its attached flange, of arc length `b_f`, and the bay up to
    the next strip. With ``flange_side=-1`` the strip of stringer `k` spans
    `\theta_k - b_f/R \le \theta \le \theta_k`, `\theta_k = 2 \pi (k +
    1)/n_s`, the web stands at its edge `\theta_k` and the free flange
    extends towards `+\theta`; with ``flange_side=1`` the strip spans
    `\theta_k \le \theta \le \theta_k + b_f/R`, `\theta_k = 2 \pi k/n_s`,
    and the free flange extends towards `-\theta`.

    Parameters
    ----------
    R, h, L : float
        Radius of the middle surface, skin thickness and length.
    E, nu : float
        Young's modulus and Poisson's ratio, skin and stringers.
    nstiff : int
        Number of stringers.
    sec : dict
        Middle lines of :func:`z_midlines`.
    side : str
        ``'internal'`` or ``'external'``.
    flange_side : int
        ``-1`` or ``1``, see above.
    bc : str
        The ends `x = 0` and `x = L`, two keys of :data:`END_FLAGS`.
    mx, ny, nf, nw, nl : int
        Terms along `x` (all domains) and across a bay, a strip, a web and
        a free flange.
    attached_flange, free_flange : bool
        ``False`` leaves out the attached flange (the strip has the skin
        thickness) or the free flange (the stringer is then an L or a
        blade).

    Returns
    -------
    md : :class:`panels.multidomain.MultiDomain`
        The assembly, ``conn_method='null-space'`` and
        ``pd_method='null-space'``.
    groups : dict
        Lists of the domains: ``'skin'`` (strips and bays, ordered in
        `\theta`), ``'strip'``, ``'web'`` and ``'lip'`` (the free flanges).

    """
    assert side in ('internal', 'external')
    assert flange_side in (-1, 1)
    lam = _isotropic(E, nu)
    tz = sec['tz']
    bf, Hw, bl = sec['bf'], sec['Hw'], sec['bl']
    pitch = 2*pi*R/nstiff
    bbay = pitch - bf
    assert bbay > 0
    (sk1, pl1), (sk2, pl2) = END_FLAGS[bc[0]], END_FLAGS[bc[1]]
    offset = -tz/2 if side == 'internal' else tz/2

    def panel(group, b, plyt, n, y0=0., r=None, off=0.):
        kw = dict(r=r) if r is not None else {}
        model = 'cylshell_clpt_sanders' if r is not None else 'plate_clpt_gl'
        p = Shell(group=group, x0=0., y0=y0, a=L, b=b, m=mx, n=n, plyt=plyt,
                  stack=[0.], laminaprop=lam, offset=off, model=model, **kw)
        skin_like = r is not None
        _set_end(p, 'x1', sk1 if skin_like else pl1)
        _set_end(p, 'x2', sk2 if skin_like else pl2)
        _set_end(p, 'y1', _FREE)
        _set_end(p, 'y2', _FREE)
        return p

    skin, strips, webs, lips = [], [], [], []
    y0 = 0.
    for k in range(nstiff):
        bay = panel('skin', bbay, h, ny, y0=y0, r=R)
        if attached_flange:
            strip = panel('strip', bf, h + tz, nf, r=R, off=offset)
        else:
            strip = panel('strip', bf, h, nf, r=R)
        if flange_side == -1:
            bay.y0 = y0
            strip.y0 = y0 + bbay
            skin += [bay, strip]
        else:
            strip.y0 = y0
            bay.y0 = y0 + bf
            skin += [strip, bay]
        strips.append(strip)
        y0 += pitch
    conns = []
    for p1, p2 in zip(skin, skin[1:] + skin[:1]):
        conns.append(dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0.))
    for strip in strips:
        web = panel('web', Hw, tz, nw)
        lip = panel('lip', bl, tz, nl)
        yweb = strip.b if flange_side == -1 else 0.
        conns.append(dict(p1=strip, p2=web, func='BFycte', ycte1=yweb,
                          ycte2=0. if side == 'internal' else Hw))
        ytip = Hw if side == 'internal' else 0.
        webs.append(web)
        if free_flange:
            conns.append(dict(p1=web, p2=lip, func='BFycte', ycte1=ytip,
                              ycte2=bl if flange_side == -1 else 0.))
            lips.append(lip)
    md = MultiDomain(skin + webs + lips, conns, conn_method='null-space',
                     pd_method='null-space')
    return md, dict(skin=skin, strip=strips, web=webs, lip=lips)


def skin_w(md, c, skin, R, nx=41, npt=7):
    r"""Normal displacement `w(x, \theta)` of the skin on a grid

    Returns ``x`` (``nx`` values), ``theta`` (increasing, rad) and ``w``
    with shape ``(ntheta, nx)``.
    """
    where = {}
    for group in sorted({p.group for p in skin}):
        res = md.uvw(c, group=group, gridx=nx, gridy=npt)
        members = [q for q in md.panels if q.group == group]
        for j, q in enumerate(members):
            where[id(q)] = (res, j)
    ths, ws = [], []
    for i, p in enumerate(skin):
        res, j = where[id(p)]
        th = (p.y0 + res['y'][j][:, 0])/R
        w = res['w'][j]
        ths.append(th[1:] if i > 0 else th)
        ws.append(w[1:] if i > 0 else w)
    x = res['x'][0][0, :]
    th = np.concatenate(ths)
    order = np.argsort(th, kind='stable')
    return x, th[order], np.vstack(ws)[order]


def classify(x, th, w, nmax=80):
    r"""Wave numbers ``(m, n)`` of a mode: ``n`` the harmonic of `\theta`
    (`\cos n\theta` and `\sin n\theta`) with the largest projection of `w`
    on the section of maximum `|w|`, ``m`` the number of sign changes plus
    one of `w` along `x` at the generator of maximum `|w|` (values below 5
    % of the maximum ignored)"""
    i, j = np.unravel_index(np.argmax(np.abs(w)), w.shape)
    wt = w[:, j]
    ns = np.arange(nmax + 1)
    pc = _trapz(wt[None, :]*np.cos(ns[:, None]*th[None, :]), th, axis=1)
    ps = _trapz(wt[None, :]*np.sin(ns[:, None]*th[None, :]), th, axis=1)
    proj = np.sqrt(pc**2 + ps**2)
    proj[0] /= 2
    n = int(np.argmax(proj))
    wx = w[i, :]
    wx = wx[np.abs(wx) > 0.05*np.abs(wx).max()]
    return int(np.sum(np.diff(np.sign(wx)) != 0)) + 1, n


def buckling_membrane(R, h, L, E, nu, nstiff, sec, side='internal',
        flange_side=-1, bc='CC', mx=12, ny=6, nf=4, nw=4, nl=4,
        attached_flange=True, free_flange=True, num_eigvalues=6,
        eps=1.e-4, classify_modes=True):
    r"""Axial buckling about the membrane pre-buckling state

    The pre-buckling state is the uniform axial strain ``-eps`` without
    Poisson expansion: `N_{xx} = -E h` ``eps`` in the bays, `-E (h + t_z)`
    ``eps`` in the strips and `-E t_z` ``eps`` in the webs and free
    flanges. ``bc='CC'``: both ends clamped; ``bc='SS'``: both ends simply
    supported (:data:`END_FLAGS`), the rigid axial translation removed by
    `u = 0` at the point `x = L/2` of the first skin domain, `y = 0`. The
    other parameters are those of :func:`build_cylinder_z`.

    Returns
    -------
    N : float
        Critical average force per unit circumference, `\bar{N} = P/(2 \pi
        R)`.
    modes : list
        ``[(N, m, n, c), ...]`` with `N > 0` in increasing order (``m`` and
        ``n`` of :func:`classify`, ``None`` with
        ``classify_modes=False``).
    md, groups :
        The assembly and the lists of :func:`build_cylinder_z`.
    ndof : int
        Number of degrees of freedom of the reduced problem.
    dt : float
        Run time, s.

    """
    t0 = time.time()
    md, groups = build_cylinder_z(R, h, L, E, nu, nstiff, sec, side=side,
            flange_side=flange_side, bc=bc, mx=mx, ny=ny, nf=nf, nw=nw,
            nl=nl, attached_flange=attached_flange, free_flange=free_flange)
    if bc == 'SS':
        groups['skin'][0].add_point_pd(L/2, 0., 1., 0., None, None, None,
                                       None)
    tz = sec['tz']
    P1 = 0.
    for p in md.panels:
        thick = {'skin': h, 'strip': h + tz if attached_flange else h
                 }.get(p.group, tz)
        p.Nxx = -E*thick*eps
        P1 += E*thick*eps*p.b
    N1 = P1/(2*pi*R)
    K = md.reduce(md.calc_kC(silent=True))
    KG = md.reduce(md.calc_kG(silent=True))
    eigvals, eigvecs = lb(K, KG, tol=0, sparse_solver=True, silent=True,
                          num_eigvalues=num_eigvalues)
    modes = []
    if classify_modes:
        eigvecs = md.expand(eigvecs)
    for k in range(len(eigvals)):
        if classify_modes:
            c = eigvecs[:, k].real
            x, th, w = skin_w(md, c, groups['skin'], R)
            m, n = classify(x, th, w)
        else:
            c, m, n = None, None, None
        modes.append((eigvals[k].real*N1, m, n, c))
    modes = sorted((mode for mode in modes if mode[0] > 0),
                   key=lambda mode: mode[0])
    return (modes[0][0], modes, md, groups, K.shape[0], time.time() - t0)
