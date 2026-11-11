r"""Stiffened cylinders in torsion of Moore and Wescoat (1944) with
:class:`panels.multidomain.MultiDomain`

Data and builders shared by the notebooks ``moore1944_ring_cylinders`` and
``moore1944_stiffened_cylinders``. The 24S-T cylinders of Moore and Wescoat
[1]_ have a mean diameter of 30.08 in, 35 5/8 in between the centres of the
end bulkheads, rings of 1/2 x 1/2 in solid bars inside the skin and, in the
stiffened specimens, 8 or 16 channel stringers outside it.

The model is a closed cylinder of :class:`panels.shell.Shell` domains:

- the skin is split along the axis at the ring stations (one bay per ring
  spacing) and along the circumference into sectors or, with stringers,
  into the strip under the base of each stringer and the bay between two
  stringers; consecutive domains are connected with ``'SSxcte'`` and
  ``'SSycte'``;
- each ring is a web of annular-sector domains (``'annular_clpt_donnell'``),
  one per skin domain of the circumference, of radial height and thickness
  1/2 in, whose outer edge is connected to the inner surface of the skin
  with ``'BFxcte'`` and ``face='lower'``, closed with ``'SSycte'``; or a
  rigid support, `v = w = 0` along the ring line;
- the base of a stringer is a layer of the laminate of the strip under it,
  on the outer face of the skin (offset `+t_s/2`, the reference surface of
  the strip remaining the middle surface of the skin); each web is a flat
  plate standing outside on an edge of the strip with ``'BFycte'``, from the
  middle surface of the skin to the middle plane of the lip, and each lip a
  flat plate on the free edge of its web with ``'BFycte'``, bent away from
  the channel; the curls of the lips are not represented, the lip width
  being chosen so that the section has the measured area. The webs and lips
  are split at the ring stations as the skin.

The connections are imposed exactly, with the default ``conn_method`` of
:class:`.MultiDomain`. The torsion is the constant shear flow `N_{xy}` of
the membrane state of a perfect cylinder, exact before buckling: the rings
and the stringers carry no stress, and the strips carry the same shear flow
as the skin. :func:`end_rotation_buckling` checks this with the rotation of
one end prescribed and the linear static pre-buckling state.

Units: in, lbf, psi; torques in ft-lb as in [1]_.

References
----------
.. [1] Moore, R. L., and Wescoat, C., "Torsion Tests of Stiffened Circular
   Cylinders", NACA ARR 4E31 (Wartime Report W-89), 1944.

"""
import time

import numpy as np
from structsolve import lb

from panels.shell import Shell
from panels.multidomain import MultiDomain

#: 24S-T, [1] p. 7
E, NU = 10.5e6, 1/3.
#: mean diameter and radius, in, [1] p. 3
D = 30.08
R = D/2.
#: ring bar, in, [1] p. 3 and Fig. 1
RING = 0.5
#: stringer sheet, in, and measured area of the section, in^2, [1] Fig. 30
T_STR = 0.032
A_STR = 0.072
#: base of the stringer between the outer faces of the webs, overall depth
#: and overall width of the section, in, [1] Figs. 1 and 30
BASE, DEPTH, WIDTH = 0.5, 0.64, 1.12

SKIN_MODEL = 'cylshell_clpt_sanders'
RING_MODEL = 'annular_clpt_donnell'
PLATE_MODEL = 'plate_clpt_donnell'

#: Specimens: sheet thickness (in), bays between the centres of the end
#: bulkheads and of the rings (in, [1] Figs. 5 and 11), number of stringers,
#: buckled section of the first buckling and observed shear-buckling stress
#: (psi, [1] Table I and Table II) and Moore's theoretical value (Table I,
#: Donnell, hinged edges) or computed values of Table II (ANC-5, Kromm,
#: Ebner, Donnell for an unstiffened cylinder of the bay length)
SPECIMENS = {
    '16': dict(t=0.020, bays=[35.625], nstr=0, panel='35.62',
               obs=1110, theory=1325),
    '17': dict(t=0.020, bays=[4.3125, 27., 4.3125], nstr=0, panel='27.00',
               obs=1280, theory=1520),
    '18': dict(t=0.020, bays=[4.3125, 13.5, 13.5, 4.3125], nstr=0,
               panel='13.50', obs=1710, theory=2210),
    '13': dict(t=0.020, bays=[8.8125, 9., 9., 8.8125], nstr=0,
               panel='8.81', obs=2310, theory=2870),
    '13A': dict(t=0.020, bays=[4.3125, 4.5, 9., 9., 4.5, 4.3125], nstr=0,
                panel='9.00', obs=2570, theory=2770),
    '19': dict(t=0.020, bays=[4.3125, 13.5, 13.5, 4.3125], nstr=8,
               panel='13.50 x 11.81', obs=2110,
               computed=(1620, 1810, 2320, 2210)),
    '14': dict(t=0.020, bays=[4.3125, 4.5, 9., 9., 4.5, 4.3125], nstr=8,
               panel='9.00 x 11.81', obs=2950,
               computed=(1750, 2700, 2960, 2770)),
    '20': dict(t=0.020, bays=[4.3125, 13.5, 13.5, 4.3125], nstr=16,
               panel='13.50 x 5.90', obs=2870,
               computed=(2070, 2600, 2570, 2210)),
    '15': dict(t=0.0195, bays=[8.8125, 9., 9., 8.8125], nstr=16,
               panel='8.81 x 5.90', obs=3370,
               computed=(2110, None, None, 2680)),
    '21': dict(t=0.0205, bays=[2.0625, 6.75, 9., 9., 6.75, 2.0625], nstr=16,
               panel='9.00 x 5.90', obs=4120,
               computed=(2240, None, None, 2910)),
}


def donnell_tau(L, t, d=D, e=E, nu=NU):
    r"""Shear-buckling stress of an unstiffened cylinder with hinged ends,
    Donnell's formula as given by Moore and Wescoat [1]_, p. 23, valid for
    `L^2 t/D^3 < 5.2`

    .. math::

        \tau_0 = \frac{E t^2}{(1 - \nu^2) L^2} \left[2.8 + \sqrt{2.6 + 1.4
        \left(\sqrt{1 - \nu^2}\frac{L^2}{t D}\right)^{3/2}}\right]

    """
    s = np.sqrt(1 - nu**2)*L**2/(t*d)
    return e*t**2/((1 - nu**2)*L**2)*(2.8 + np.sqrt(2.6 + 1.4*s**1.5))


def torque_ftlb(tau, t, d=D):
    r"""Torque (ft-lb) of the mean shear stress ``tau`` (psi), `T = \tau \pi
    D^2 t/2`, [1]_ p. 7"""
    return tau*np.pi*d**2*t/2./12.


def stringer_section(t, tz=T_STR, area=A_STR):
    r"""Middle-line dimensions of the channel stringer as modelled

    The strip under the base spans the middle planes of the two webs,
    `b_s = 1/2 - t_z`; each web goes from the middle surface of the skin to
    the middle plane of its lip, `H_w = t/2 + 0.64 - t_z/2`; the width of
    each lip, `b_l`, gives the section of the plates (base, two webs, two
    lips) the measured area ``area``, the curls and the bends not being
    represented. Returns a dict with ``bs``, ``Hw``, ``bl``, ``tz`` and the
    properties of the modelled section about its centroid, without the skin:
    ``A``, ``Ixx`` (about the axis parallel to the skin) and ``Iyy`` (about
    the axis normal to it), in and in^4.

    """
    bs = BASE - tz
    Hw = t/2 + DEPTH - tz/2
    bl = (area/tz - bs - 2*Hw)/2
    # rectangles (width along s, height along r, centroid s, centroid r);
    # base on the outer face of the skin
    z0 = t/2
    rects = [(bs, tz, 0., z0 + tz/2)]
    for sgn in (-1, 1):
        rects.append((tz, Hw - t/2, sgn*bs/2, z0 + (Hw - t/2)/2))
        rects.append((bl, tz, sgn*(bs/2 + bl/2), Hw))
    A = sum(w*h for w, h, _, _ in rects)
    zc = sum(w*h*z for w, h, _, z in rects)/A
    Ixx = sum(w*h**3/12 + w*h*(z - zc)**2 for w, h, _, z in rects)
    Iyy = sum(h*w**3/12 + w*h*s**2 for w, h, s, _ in rects)
    return dict(bs=bs, Hw=Hw, bl=bl, tz=tz, A=A, Ixx=Ixx, Iyy=Iyy,
                zc_from_skin_face=zc - t/2)


def _free(s):
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, edge + f, 1)
            setattr(s, edge + f + 'r', 1)
    return s


def _terms(length, per_in, n_min, n_max=30):
    return int(min(n_max, max(n_min, round(length*per_in))))


def build(spec, nsec=12, m_per_in=1.2, m_min=6, n=14, m_ring=6,
          n_strip=4, n_web=6, n_lip=4, rings='web', ends='hinged',
          u_end=0, stringers=True, t=None, skin_model=SKIN_MODEL,
          ring_model=RING_MODEL, lip_width=None, end_rotation=None,
          n_per_in=None):
    r"""Cylinder of the specimen ``spec`` of :data:`SPECIMENS`

    Parameters
    ----------
    spec : str
        Key of :data:`SPECIMENS`.
    nsec : int
        Sectors of the skin along the circumference, without stringers.
    m_per_in, m_min : float, int
        Terms along the axis of each bay, ``m_per_in`` times its length, at
        least ``m_min`` (at most 30).
    n : int
        Terms along the circumference of each sector, or of each bay
        between two stringers; ``n_strip``, ``n_web`` and ``n_lip`` across
        the strip under a base, a web and a lip.
    n_per_in : float or None
        With stringers: terms across each bay between two stringers,
        ``n_per_in`` times its width, at least 8 (at most 30), instead of
        ``n``.
    m_ring : int
        Terms across the height of the ring webs (along the circumference,
        those of the skin domain the ring sector is connected to).
    rings : str
        ``'web'``, the annular webs; ``'rigid'``, `v = w = 0` of the skin
        along the ring lines; ``'none'``.
    ends : str
        ``'hinged'``, `w_{,x}` free at the end bulkheads, or ``'clamped'``.
    u_end : int
        Flag of `u` at the end `x = L`: ``0`` restrained, ``1`` free (`u =
        0` at `x = 0` in both cases).
    stringers : bool
        Model the stringers of the specimen (``False``: rings only).
    t : float or None
        Skin thickness, that of the specimen when ``None``.
    lip_width : float or None
        Width of the lips, from :func:`stringer_section` when ``None``.
    end_rotation : float or None
        ``None``: the constant shear flow `N_{xy} = 1` lbf/in on the skin
        domains. A value `\theta`: the end `x = L` rotates about the axis by
        `\theta` (times the load factor), `v = \theta R`, the other end
        held, prescribed displacements imposed exactly (only without
        stringers).

    Returns
    -------
    md : :class:`.MultiDomain`
    parts : dict
        ``'skins'`` (list of lists, one per bay), ``'rings'``, ``'webs'``,
        ``'lips'``, ``'t'``, ``'L'``, ``'x_rings'``, ``'section'``.

    """
    data = SPECIMENS[spec]
    t = data['t'] if t is None else t
    bays = data['bays']
    nstr = data['nstr'] if stringers else 0
    if nstr and end_rotation is not None:
        raise NotImplementedError('end_rotation only without stringers')
    sec =stringer_section(t) if nstr else None
    if sec is not None and lip_width is not None:
        sec = dict(sec, bl=lip_width)
    # circumferential layout: (width, kind, terms)
    if nstr:
        pitch = 2*np.pi*R/nstr
        layout = []
        bbay = pitch - sec['bs']
        if n_per_in is not None:
            n = _terms(bbay, n_per_in, 8)
        for k in range(nstr):
            layout += [(sec['bs'], 'strip', n_strip), (bbay, 'bay', n)]
    else:
        layout = [(2*np.pi*R/nsec, 'bay', n)]*nsec
    wr = 1 if ends == 'hinged' else 0
    v_end = None if end_rotation is None else end_rotation*R
    skins, rings_, webs, lips, conn = [], [], [], [], []
    x0 = 0.
    nb = len(bays)
    for i, a in enumerate(bays):
        m = _terms(a, m_per_in, m_min)
        row, y0 = [], 0.
        for j, (b, kind, nj) in enumerate(layout):
            if kind == 'strip':
                p = Shell(model=skin_model, a=a, b=b, r=R, m=m, n=nj,
                          stack=[0, 0], plyts=[t, sec['tz']],
                          laminaprop=(E, NU), offset=sec['tz']/2,
                          x0=x0, y0=y0, group='strip')
            else:
                p = Shell(model=skin_model, a=a, b=b, r=R, m=m, n=nj,
                          stack=[0], plyt=t, laminaprop=(E, NU),
                          x0=x0, y0=y0, group='skin')
            _free(p)
            if end_rotation is None:
                p.Nxy = 1.
            if i == 0:
                p.x1u, p.x1v, p.x1w, p.x1wr = 0, 0, 0, wr
            if i == nb - 1:
                p.x2u, p.x2v, p.x2w, p.x2wr = u_end, 0, 0, wr
                if v_end is not None:
                    p.x2v = 1
                    p.add_distr_pd_fixed_x(a, kv=1., funcv=lambda s: v_end,
                                           cte=False)
            if rings == 'rigid':
                if i > 0:
                    p.x1v, p.x1w = 0, 0
                if i < nb - 1:
                    p.x2v, p.x2w = 0, 0
            if row:
                conn.append(dict(p1=row[-1], p2=p, func='SSycte',
                                 ycte1=row[-1].b, ycte2=0.))
            if i > 0:
                q = skins[-1][j]
                conn.append(dict(p1=q, p2=p, func='SSxcte', xcte1=q.a,
                                 xcte2=0.))
            row.append(p)
            y0 += b
        conn.append(dict(p1=row[-1], p2=row[0], func='SSycte',
                         ycte1=row[-1].b, ycte2=0.))
        # stringers of this bay
        if nstr:
            web_row, lip_row = [], []
            for j, p in enumerate(row):
                if p.group != 'strip':
                    continue
                for side, ycte1 in (('l', 0.), ('r', p.b)):
                    web = _free(Shell(model=PLATE_MODEL, a=a, b=sec['Hw'],
                                      m=m, n=n_web, stack=[0],
                                      plyt=sec['tz'], laminaprop=(E, NU),
                                      x0=x0, group='web'))
                    # the web stands outside: its edge y = Hw on the strip
                    conn.append(dict(p1=p, p2=web, func='BFycte',
                                     ycte1=ycte1, ycte2=web.b))
                    lip = _free(Shell(model=PLATE_MODEL, a=a, b=sec['bl'],
                                      m=m, n=n_lip, stack=[0],
                                      plyt=sec['tz'], laminaprop=(E, NU),
                                      x0=x0, group='lip'))
                    # tip of the web at its y = 0; the lip extends towards
                    # -y of the skin from its edge y = 0 (left web) and
                    # towards +y from its edge y = bl (right web)
                    conn.append(dict(p1=web, p2=lip, func='BFycte',
                                     ycte1=0., ycte2=0. if side == 'l'
                                     else lip.b))
                    web_row.append(web)
                    lip_row.append(lip)
            for ends_flag, edge in ((i == 0, 'x1'), (i == nb - 1, 'x2')):
                if ends_flag:
                    for s in web_row + lip_row:
                        # the plates are held by the bulkheads as the skin
                        setattr(s, edge + 'u', u_end if edge == 'x2'
                                else 0)
                        setattr(s, edge + 'v', 0)
                        setattr(s, edge + 'w', 0)
            if i > 0:
                for prev, cur in ((webs[-1], web_row), (lips[-1], lip_row)):
                    for q, s in zip(prev, cur):
                        conn.append(dict(p1=q, p2=s, func='SSxcte',
                                         xcte1=q.a, xcte2=0.))
            webs.append(web_row)
            lips.append(lip_row)
        # ring at the end x = a of this bay
        if rings == 'web' and i < nb - 1:
            ring_row = []
            r_out = R - t/2
            for p in row:
                w = _free(Shell(model=ring_model, a=RING, b=r_out*p.b/R,
                                r=r_out, m=m_ring, n=p.n, stack=[0],
                                plyt=RING, laminaprop=(E, NU),
                                group='ring'))
                conn.append(dict(p1=p, p2=w, func='BFxcte', xcte1=p.a,
                                 xcte2=0., face='lower'))
                if ring_row:
                    conn.append(dict(p1=ring_row[-1], p2=w, func='SSycte',
                                     ycte1=ring_row[-1].b, ycte2=0.))
                ring_row.append(w)
            conn.append(dict(p1=ring_row[-1], p2=ring_row[0], func='SSycte',
                             ycte1=ring_row[-1].b, ycte2=0.))
            rings_.append(ring_row)
        skins.append(row)
        x0 += a
    panels_ = ([p for row in skins for p in row]
               + [w for row in rings_ for w in row]
               + [s for row in webs for s in row]
               + [s for row in lips for s in row])
    kw = {} if end_rotation is None else dict(pd_method='null-space')
    md = MultiDomain(panels_, conn, **kw)
    return md, dict(skins=skins, rings=rings_, webs=webs, lips=lips, t=t,
                    L=sum(bays), x_rings=list(np.cumsum(bays)[:-1]),
                    section=sec, nstr=nstr, end_rotation=end_rotation)


def buckling(md, parts, num_eigvalues=4):
    r"""Linear buckling under the constant shear flow of :func:`build`

    Returns a dict with ``tau`` (psi), ``T`` (ft-lb), the load multipliers
    ``ev`` (critical shear flows, lbf/in), the modes ``evec`` (expanded),
    ``size``, ``size_r`` and ``seconds``.

    """
    t0 = time.time()
    Z = md.get_Z()
    kC_r = md.reduce(md.calc_kC(silent=True))
    kG_r = md.reduce(md.calc_kG(silent=True))
    ev, vec = lb(kC_r, kG_r, num_eigvalues=num_eigvalues, silent=True)
    ev = np.real(ev)
    tau = ev[0]/parts['t']
    return dict(tau=tau, T=torque_ftlb(tau, parts['t']), ev=ev,
                evec=md.expand(vec), size=md.get_size(), size_r=Z.shape[1],
                seconds=time.time() - t0)


def run_case(spec, kwargs, keep_mode=False):
    r"""Build the specimen ``spec`` with :func:`build` (``kwargs``) and run
    :func:`buckling` (or :func:`end_rotation_buckling` when ``kwargs`` has
    ``end_rotation``); returns a dict of plain values for a results file:
    ``tau``, ``T``, the first load multipliers ``ev``, the number of
    circumferential waves of the first mode ``waves``, ``size``,
    ``size_r``, ``domains``, ``seconds`` and, with ``keep_mode``, the
    radial displacement of the first mode on the unrolled skin, ``mode``
    (lists ``X``, ``S``, ``W`` per bay, see :func:`skin_w`)"""
    md, parts = build(spec, **kwargs)
    if kwargs.get('end_rotation') is not None:
        res = end_rotation_buckling(md, parts)
        res['size'] = md.get_size()
        res['size_r'] = md.Z.shape[1]
    else:
        res = buckling(md, parts)
    out = dict(spec=spec, kwargs=dict(kwargs), tau=float(res['tau']),
               T=float(res['T']), ev=[float(e) for e in np.real(res['ev'])],
               waves=circumferential_waves(parts, res['evec'][:, 0]),
               size=int(res['size']), size_r=int(res['size_r']),
               domains=len(md.panels), seconds=float(res['seconds']))
    if keep_mode:
        X, S, W = skin_w(parts, res['evec'][:, 0])
        out['mode'] = dict(X=[x.tolist() for x in X],
                           S=[s.tolist() for s in S],
                           W=[w.tolist() for w in W])
    return out


def end_rotation_buckling(md, parts, num_eigvalues=4):
    r"""Linear buckling about the linear static state of the end rotation of
    :func:`build` (``end_rotation`` given); the torque at load factor 1 is
    the reaction work-conjugate to the incremented prescribed displacements
    divided by the rotation, see :meth:`.MultiDomain.reaction_pd`"""
    t0 = time.time()
    c1 = md.linear_static(lbd=1., silent=True)
    P1 = md.reaction_pd(c1, lbd=1., NLgeom=False)['inc']
    ev, evec = md.linear_buckling(num_eigvalues=num_eigvalues, silent=True)
    T1 = P1/parts['end_rotation']            # lbf in at load factor 1
    T = ev[0]*T1/12.
    tau = 24*T/(np.pi*D**2*parts['t'])
    return dict(tau=tau, T=T, ev=ev, evec=evec, T1=T1, c1=c1,
                seconds=time.time() - t0)


def skin_w(parts, c, gridx=12, ny=12):
    r"""Radial displacement `w` of the skin on the unrolled surface: arrays
    ``X, S, W`` of shape ``(nx, ns)``, bays stacked along `x` and domains
    side by side along the arc `s`"""
    Xr, Sr, Wr = [], [], []
    for row in parts['skins']:
        Xs, Ss, Ws = [], [], []
        for p in row:
            xs = np.linspace(0, p.a, max(4, int(gridx*p.a/9.)))
            ys = np.linspace(0, p.b, max(3, int(ny*p.b/7.)))
            X, Y = np.meshgrid(xs, ys, indexing='ij')
            ci = c[p.col_start:p.col_end]
            _, f = p.uvw(ci, xs=X.ravel(), ys=Y.ravel())
            Xs.append(X + p.x0)
            Ss.append(Y + p.y0)
            Ws.append(np.real(f['w']).reshape(X.shape))
        Xr.append(np.concatenate(Xs, axis=1))
        Sr.append(np.concatenate(Ss, axis=1))
        Wr.append(np.concatenate(Ws, axis=1))
    return Xr, Sr, Wr


def circumferential_waves(parts, c, x=None, npts=720):
    r"""Number of full waves of `w` along the circumference at the axial
    station ``x`` (default: middle of the longest bay), from the dominant
    harmonic of its Fourier transform"""
    bays = [row[0].a for row in parts['skins']]
    i = int(np.argmax(bays)) if x is None else None
    if x is None:
        row = parts['skins'][i]
        xl = row[0].a/2
    else:
        x0s = np.cumsum([0.] + bays)
        i = int(np.searchsorted(x0s, x, side='right') - 1)
        row = parts['skins'][i]
        xl = x - x0s[i]
    s, w = [], []
    for p in row:
        ys = np.linspace(0, p.b, 40, endpoint=False)
        _, f = p.uvw(c[p.col_start:p.col_end], xs=np.full_like(ys, xl),
                     ys=ys)
        s.append(ys + p.y0)
        w.append(np.real(f['w']))
    s, w = np.concatenate(s), np.concatenate(w)
    si = np.linspace(0, 2*np.pi*R, npts, endpoint=False)
    wi = np.interp(si, s, w, period=2*np.pi*R)
    spec = np.abs(np.fft.rfft(wi))
    return int(np.argmax(spec[1:]) + 1)


def print_table(rows, headers):
    """Print a plain-text table"""
    cols = list(zip(*([headers] + [[str(c) for c in r] for r in rows])))
    widths = [max(len(c) for c in col) for col in cols]
    fmt = '  '.join('%%%ds' % w for w in widths)
    print(fmt % tuple(headers))
    print('  '.join('-'*w for w in widths))
    for r in rows:
        print(fmt % tuple(str(c) for c in r))


def rel(value, ref):
    """Relative difference in per cent"""
    return 100.*(value - ref)/ref
