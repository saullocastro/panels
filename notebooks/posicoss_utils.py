r"""Stringer-stiffened composite shells of the POSICOSS project with
:class:`panels.multidomain.MultiDomain`

Builders and post-processing shared by the notebooks of the
blade-stiffened curved panels of Abramovich and Weller (2009), under axial
compression, and of the cylinder with L-shaped stringers of Lanzi, Bisagni
and Cordisco (2003), under axial compression and torsion.

Every part of a stiffener is a :class:`panels.shell.Shell` domain: the skin
is split into strips along the circumference, each foot is a cylindrical
shell (``'cylshell_clpt_sanders'``) with its own laminate and offset, bonded
to the skin strip under it with the connection ``'SB'``, and each blade is a
flat plate standing on the edge of a foot with ``'BFycte'``. A blade is
loaded in its own plane, so its geometric stiffness needs the term `N_{xx}
v_{,x}^2` of the Green-Lagrange strains: the blades are
``'plate_clpt_gl'`` plates.
The connections and the prescribed displacements of the loaded ends are
imposed exactly (``conn_method='null-space'``, ``pd_method='null-space'``).
Lengths are in mm, forces in N and moduli in N/mm^2.

"""
import time

import numpy as np

from panels.shell import Shell
from panels.multidomain import MultiDomain

CURVED_MODEL = 'cylshell_clpt_sanders'
FLAT_MODEL = 'plate_clpt_gl'


def make_shell(a, b, r, stack, plyt, laminaprop, m, n, group, y0=0.,
               offset=0., model=None):
    """Shell domain with all the edges free

    The flags of the loaded ends are set afterwards with :func:`set_end`.
    ``r=None`` gives a flat domain, of the model ``FLAT_MODEL`` unless
    ``model`` is given.

    """
    if model is None:
        model = FLAT_MODEL if r is None else CURVED_MODEL
    s = Shell(a=a, b=b, r=r, stack=list(stack), plyt=plyt,
              laminaprop=laminaprop, m=m, n=n, offset=offset, model=model,
              group=group, x0=0., y0=y0)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for f in ('u', 'v', 'w'):
            setattr(s, edge + f, 1)
            setattr(s, edge + f + 'r', 1)
    return s


def set_end(s, edge, u=0, v=0, w=0, wr=0):
    """Flags of the end ``edge`` (``'x1'`` or ``'x2'``): ``0`` removes the
    value of the field (``wr``: the slope `w_{,x}`), ``1`` keeps it free"""
    setattr(s, edge + 'u', u)
    setattr(s, edge + 'v', v)
    setattr(s, edge + 'w', w)
    setattr(s, edge + 'wr', wr)


def _n_terms(width, n_per_mm, n_min=4, n_max=30):
    return int(min(n_max, max(n_min, round(width*n_per_mm))))


# ----------------------------------------------------------------------------
# Curved panel with blade stringers (Abramovich and Weller 2009, PSC panels)
# ----------------------------------------------------------------------------

def blade_panel(R, L, arc, y_stringers, skin_stack, blade_stack, blade_width,
                foot_steps, plyt, laminaprop, m, n_bay, n_strip=4,
                n_blade=6, unloaded='ss', foot_reference='foot',
                blade_model=None):
    r"""Curved panel stiffened by blade stringers with stepped feet

    Parameters
    ----------
    R, L, arc : float
        Radius of the skin mid-surface, free length between the clamped
        loaded edges and arc length of the panel.
    y_stringers : list of float
        Arc coordinates of the stringer mid-planes.
    skin_stack, blade_stack : list of float
        Ply angles from the inner (`-z`) to the outer (`+z`) face.
    blade_width : float
        Width of the blade domain, from the reference surface of the feet
        (of the skin without feet) to the blade tip.
    foot_steps : list of (float, list)
        From the blade outwards, the width of each step of each half of the
        foot and its ply angles from the inner face of the foot to the face
        bonded to the skin. An empty list gives blades standing on the
        skin.
    plyt, laminaprop :
        Ply thickness and lamina properties.
    m : int
        Terms along the axis, the same for all the domains.
    n_bay, n_strip, n_blade : int
        Terms across the widest bay (``n_bay``; the other bays get
        proportionally fewer, at least 4), across the strips of the feet
        and across the blade.
    unloaded : str
        ``'ss'``: the unloaded edges have `w = 0`, `u`, `v` and the
        rotation free; ``'clamped'``: also `w_{,y} = 0`.
    foot_reference : str
        Reference surface of all the foot domains, each with the offset that
        puts its plies on the inner face of the skin: ``'foot'``, the
        mid-surface of the thickest (central) step; ``'skin'``, the
        mid-surface of the skin, as a foot whose elements share the nodes
        of the skin. The steps are not connected to each other: each one
        follows the skin through ``'SB'``, which makes them continuous.
    blade_model : str or None
        Model of the blades, ``FLAT_MODEL`` when ``None``.

    Returns
    -------
    md : :class:`.MultiDomain`
        With the connections and the end-shortening `u = -1` of the edge `x
        = L` (multiplied by the load factor) of the skin and of the blades.
    parts : dict
        ``'skins'``, ``'feet'``, ``'blades'`` (lists of :class:`.Shell`) and
        ``'loaded'``, the prescriptions of the loaded edge for
        :meth:`.MultiDomain.reaction_resultant`.

    """
    # strips of the feet: (y_start, y_end, foot stack, stringer index); with
    # no foot, the skin is only split at the stringers
    steps = []
    for k, ys in enumerate(y_stringers):
        bounds = np.concatenate(([0.], np.cumsum([w for w, _ in foot_steps])))
        left, right = [], []
        for i, (_, stack) in enumerate(foot_steps):
            left.insert(0, (ys - bounds[i+1], ys - bounds[i], stack, k))
            right.append((ys + bounds[i], ys + bounds[i+1], stack, k))
        steps += left + right
    cuts = sorted(set([0., arc] + [s[0] for s in steps] + [s[1] for s in steps]
                      + list(y_stringers)))
    foot_at = {(round(s[0], 9), round(s[1], 9)): (s[2], s[3]) for s in steps}
    strips = []
    for y1, y2 in zip(cuts[:-1], cuts[1:]):
        stack, k = foot_at.get((round(y1, 9), round(y2, 9)), (None, None))
        strips.append((y1, y2, stack, k))
    bay_max = max(y2 - y1 for (y1, y2, st, k) in strips if st is None)

    t_skin = len(skin_stack)*plyt
    t_ref = max([len(st) for _, st in foot_steps], default=0)*plyt
    skins, feet, blades, conn = [], [], [], []
    for i, (y1, y2, stack, k) in enumerate(strips):
        b = y2 - y1
        if stack is None:
            n = max(4, int(round(n_bay*b/bay_max)))
        else:
            n = n_strip
        sk = make_shell(L, b, R, skin_stack, plyt, laminaprop, m, n,
                        group='skin', y0=y1)
        set_end(sk, 'x1')
        set_end(sk, 'x2', u=1)
        sk.add_distr_pd_fixed_x(L, ku=1., funcu=lambda y: -1., cte=False)
        if skins:
            conn.append(dict(p1=skins[-1], p2=sk, func='SSycte',
                             ycte1=skins[-1].b, ycte2=0.))
        skins.append(sk)
        base = sk
        foot = None
        if stack is not None:
            # offset of the foot: its face z = d + t/2 on the inner face of
            # the skin, at z = t_ref/2 ('foot') or -t_skin/2 ('skin')
            t = len(stack)*plyt
            if foot_reference == 'foot':
                d = (t_ref - t)/2.
            else:
                d = -(t_skin + t)/2.
            foot = make_shell(L, b, R, stack, plyt, laminaprop, m, n,
                              group='foot_%02d' % k, y0=y1, offset=d)
            set_end(foot, 'x1')
            set_end(foot, 'x2', u=1)
            # the skin is on the +z side of the foot
            conn.append(dict(p1=sk, p2=foot, func='SB'))
            feet.append(foot)
            base = foot
        on_stringer = [j for j, ys in enumerate(y_stringers)
                       if abs(y1 - ys) < 1e-9]
        if on_stringer:
            blade = make_shell(L, blade_width, None, blade_stack, plyt,
                               laminaprop, m, n_blade,
                               group='blade_%02d' % on_stringer[0], y0=0.,
                               model=blade_model)
            set_end(blade, 'x1')
            set_end(blade, 'x2', u=1)
            blade.add_distr_pd_fixed_x(L, ku=1., funcu=lambda y: -1.,
                                       cte=False)
            # blade normal to its base, towards -z (inside the panel)
            conn.append(dict(p1=base, p2=blade, func='BFycte', ycte1=0.,
                             ycte2=0.))
            blades.append(blade)
    # unloaded edges supported by the fixture plates
    for s, e in ((skins[0], 'y1'), (skins[-1], 'y2')):
        setattr(s, e + 'w', 0)
        if unloaded == 'clamped':
            setattr(s, e + 'wr', 0)
    panels_ = skins + feet + blades
    md = MultiDomain(panels_, conn, conn_method='null-space',
                     pd_method='null-space')
    loaded = [(p, 'x', L) for p in skins + blades]
    return md, dict(skins=skins, feet=feet, blades=blades, loaded=loaded,
                    strips=strips)


# ----------------------------------------------------------------------------
# Cylinder with L-shaped stringers (Lanzi, Bisagni and Cordisco 2003)
# ----------------------------------------------------------------------------

def l_cylinder(R, L, nstringers, skin_stack, reinf_stack, reinf_width,
               reinf_left, foot_stack, foot_width, blade_stack, blade_width,
               plyt, laminaprop, m, n_bay, n_reinf=4, n_foot=6, n_blade=6,
               load='torsion', bottom='pinned', blade_side='y1',
               theta=1.e-3, foot_reference='foot', blade_model=None):
    r"""Closed cylinder stiffened by L-shaped stringers

    Each stringer pitch, starting at the stringer, is split into a
    reinforced strip without foot (width ``reinf_left``), the reinforced
    strip under the foot (``foot_width``), the rest of the reinforced strip
    and the bare skin of the bay. The reinforcement plies are added on the
    outer face of the skin (laminate offset), the foot is bonded on the
    inner face with ``'SB'`` and the blade stands on the edge of the foot
    ``blade_side`` (``'y1'``: `y = 0`, the end closer to ``reinf_left``),
    pointing inwards, with ``'BFycte'``.

    The end `x = 0` has `u = v = w = 0` (``bottom='pinned'``, `w_{,x}` free,
    as the fixed translations of the finite element model of the source) or
    also `w_{,x} = 0` (``'clamped'``); the end `x = L` is a rigid plane with
    `w = w_{,x} = 0`, moved with ``load='axial'`` by the end-shortening `u =
    -1` (multiplied by the load factor) and with ``load='torsion'`` by the
    rotation ``theta`` (multiplied by the load factor) about the axis, in
    the sense of positive `v`, with `u = 0`. Under torsion the blade follows the rigid rotation
    through `w`, the displacement normal to its plane.

    The reference surface of the foot is its mid-surface
    (``foot_reference='foot'``) or, with an offset laminate, the mid-surface
    of the skin (``'skin'``, as a foot whose elements share the nodes of the
    skin); the blade stands on it, ``blade_width`` being measured from it.
    The blades are of the model ``blade_model``, ``FLAT_MODEL`` when
    ``None``.

    Returns
    -------
    md, parts : as in :func:`blade_panel`; ``parts`` also has ``'theta'``,
    the prescribed rotation at load factor 1.

    """
    pitch = 2*np.pi*R/nstringers
    reinf_right = reinf_width - reinf_left - foot_width
    bay = pitch - reinf_width
    assert reinf_right > 0 and bay > 0
    t_skin = len(skin_stack)*plyt
    t_reinf = (len(skin_stack) + len(reinf_stack))*plyt
    offset_reinf = (t_reinf - t_skin)/2.
    t_foot = len(foot_stack)*plyt
    # Sanders kinematics of 'SB': v(z) = (1 + z/r) v - z w,y; on the rigid end
    # w = 0, so the foot (reference radius R in the model) has
    # v_f (1 + z2/R) = v_s (1 + z1/R), z1 and z2 being the faces in contact
    z1 = offset_reinf - t_reinf/2.
    if foot_reference == 'foot':
        offset_foot = 0.
    else:
        offset_foot = z1 - t_foot/2.
    z2 = offset_foot + t_foot/2.
    v_skin = theta*R
    v_foot = v_skin*(1 + z1/R)/(1 + z2/R)

    skins, feet, blades, conn = [], [], [], []
    y = 0.
    for k in range(nstringers):
        widths = [(reinf_left, 'reinf', n_reinf), (foot_width, 'foot', n_foot),
                  (reinf_right, 'reinf', n_reinf), (bay, 'bay', n_bay)]
        for (b, kind, n) in widths:
            if kind == 'bay':
                stack, offset = skin_stack, 0.
            else:
                stack, offset = list(skin_stack) + list(reinf_stack), offset_reinf
            sk = make_shell(L, b, R, stack, plyt, laminaprop, m, n,
                            group='skin', y0=y, offset=offset)
            set_end(sk, 'x1', wr=0 if bottom == 'clamped' else 1)
            if load == 'axial':
                set_end(sk, 'x2', u=1)
                sk.add_distr_pd_fixed_x(L, ku=1., funcu=lambda s: -1.,
                                        cte=False)
            else:
                set_end(sk, 'x2', v=1)
                sk.add_distr_pd_fixed_x(L, kv=1., funcv=lambda s: v_skin,
                                        cte=False)
            if skins:
                conn.append(dict(p1=skins[-1], p2=sk, func='SSycte',
                                 ycte1=skins[-1].b, ycte2=0.))
            skins.append(sk)
            if kind == 'foot':
                foot = make_shell(L, b, R, foot_stack, plyt, laminaprop, m, n,
                                  group='foot_%02d' % k, y0=y,
                                  offset=offset_foot)
                conn.append(dict(p1=sk, p2=foot, func='SB'))
                feet.append(foot)
                blade = make_shell(L, blade_width, None, blade_stack, plyt,
                                   laminaprop, m, n_blade,
                                   group='blade_%02d' % k, y0=0.,
                                   model=blade_model)
                set_end(blade, 'x1', wr=0 if bottom == 'clamped' else 1)
                if load == 'axial':
                    set_end(blade, 'x2', u=1)
                    blade.add_distr_pd_fixed_x(L, ku=1., funcu=lambda s: -1.,
                                               cte=False)
                else:
                    set_end(blade, 'x2', w=1)
                    # rigid rotation of the blade about the cylinder axis,
                    # consistent with the foot at the root, w_b,y = -v_f/R
                    blade.add_distr_pd_fixed_x(L, kw=1.,
                        funcw=lambda s: v_foot*(1 - s/R), cte=False)
                ycte1 = 0. if blade_side == 'y1' else foot.b
                conn.append(dict(p1=foot, p2=blade, func='BFycte',
                                 ycte1=ycte1, ycte2=0.))
                blades.append(blade)
            y += b
    conn.append(dict(p1=skins[-1], p2=skins[0], func='SSycte',
                     ycte1=skins[-1].b, ycte2=0.))
    panels_ = skins + feet + blades
    md = MultiDomain(panels_, conn, conn_method='null-space',
                     pd_method='null-space')
    loaded = [(p, 'x', L) for p in skins + blades]
    return md, dict(skins=skins, feet=feet, blades=blades, loaded=loaded,
                    theta=theta, pitch=pitch, bay=bay)


# ----------------------------------------------------------------------------
# Analyses and post-processing
# ----------------------------------------------------------------------------

def linear_buckling(md, num_eigvalues=6):
    """Pre-buckling state for load factor 1, its generalized reaction and
    the linear buckling analysis about it

    Returns ``eigvals, eigvecs, c1, P1, seconds``, with ``P1`` the reaction
    work-conjugate to the incremented prescribed displacements, see
    :meth:`.MultiDomain.reaction_pd`.

    """
    t0 = time.time()
    c1 = md.linear_static(lbd=1., silent=True)
    P1 = md.reaction_pd(c1, lbd=1., NLgeom=False)['inc']
    eigvals, eigvecs = md.linear_buckling(num_eigvalues=num_eigvalues,
                                          silent=True)
    return eigvals, eigvecs, c1, P1, time.time() - t0


def skin_field(skins, c, field='w', gridx=60, n_per_strip=None, s_scale=1.):
    """Field of the skin strips on the unrolled surface

    Returns ``X, S, F``, arrays of shape ``(gridx, ns)``: axial coordinate,
    arc coordinate (times ``s_scale``) and the field, with the strips side
    by side.

    """
    Xs, Ss, Fs = [], [], []
    for p in skins:
        ny = n_per_strip or max(3, int(round(p.b/4.)))
        xs = np.linspace(0, p.a, gridx)
        ys = np.linspace(0, p.b, ny)
        X, Y = np.meshgrid(xs, ys, indexing='ij')
        ci = c[p.col_start:p.col_end]
        _, fields = p.uvw(ci, xs=X.ravel(), ys=Y.ravel())
        Xs.append(X)
        Ss.append((Y + p.y0)*s_scale)
        Fs.append(fields[field].reshape(X.shape))
    return (np.concatenate(Xs, axis=1), np.concatenate(Ss, axis=1),
            np.concatenate(Fs, axis=1))


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
