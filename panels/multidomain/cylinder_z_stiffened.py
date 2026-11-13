r"""Z-stiffened cylinders made of several domains
(:mod:`panels.multidomain.cylinder_z_stiffened`)

:func:`.create_cylinder_z_stiffened` builds a closed cylinder with equally
spaced Z-section stringers, each assembled from three plates. It is
imported in :mod:`panels.multidomain`.

"""
import numpy as np

from panels.shell import Shell
from panels.multidomain import MultiDomain


# default model of each group of domains
_DEFAULT_MODELS = {'skin': 'cylshell_clpt_sanders', 'stringer': 'plate_clpt_gl'}
# models accepted for each group
_ALLOWED_MODELS = {
    'skin': ('cylshell_clpt_sanders', 'cylshell_clpt_gl',
             'cylshell_clpt_tl', 'cylshell_clpt_donnell'),
    'stringer': ('plate_clpt_gl', 'plate_clpt_tl', 'plate_clpt_donnell'),
    }

_FREE = dict(u=1, ur=1, v=1, vr=1, w=1, wr=1)
# flags of an end, (skin, plates): 'C' clamped, 'S' simply supported
_END_FLAGS = {
    'C': (dict(_FREE, u=0, v=0, w=0, wr=0),
          dict(_FREE, u=0, v=0, vr=0, w=0, wr=0)),
    'S': (dict(_FREE, v=0, w=0), dict(_FREE, v=0, w=0)),
}


def _models(models):
    r"""The model of each group, ``models`` completed with
    ``_DEFAULT_MODELS``, see :func:`.create_cylinder_z_stiffened`"""
    models = {} if models is None else dict(models)
    unknown = sorted(set(models) - set(_DEFAULT_MODELS))
    if unknown:
        raise ValueError("unknown keys of models %s, the keys are %s"
                         % (unknown, sorted(_DEFAULT_MODELS)))
    out = dict(_DEFAULT_MODELS, **models)
    for group, model in out.items():
        if model not in _ALLOWED_MODELS[group]:
            raise ValueError("models[%r] must be one of %s, got %r"
                             % (group, _ALLOWED_MODELS[group], model))
    return out


def _set_end(p, end, flags):
    for k, v in flags.items():
        setattr(p, '%s%s' % (end, k), v)


# the plates of the Z profile, in the order of a sequence of stacks
_PROFILE = ('flange', 'web', 'free_flange')


def _flipped(stack):
    r"""The stack seen from the other face: the plies in reverse order,
    the angles changed in sign"""
    return [0. - angle for angle in reversed(stack)]


def _stacks_z(stack_z, side, flange_side):
    r"""The stack of each plate of the profile, in its own axes, from
    ``stack_z``, see :func:`.create_cylinder_z_stiffened`"""
    items = list(stack_z)
    if items and all(np.ndim(s) == 0 for s in items):
        # the bent sheet: the z axis of the attached flange is the normal of
        # the sheet, that of the free flange is opposite to it, that of the
        # web along it only for internal stringers with flange_side=-1 and
        # external stringers with flange_side=1
        web_flipped = (flange_side == -1) != (side == 'internal')
        sheet = [float(angle) for angle in items]
        return dict(flange=sheet,
                    web=_flipped(sheet) if web_flipped else list(sheet),
                    free_flange=_flipped(sheet))
    if (len(items) == len(_PROFILE)
            and all(np.ndim(s) == 1 and len(s) > 0 for s in items)):
        return {k: [float(angle) for angle in s]
                for k, s in zip(_PROFILE, items)}
    raise ValueError('stack_z must be one stack of angles or a sequence of '
                     'three stacks (attached flange, web, free flange), '
                     'got %r' % (stack_z, ))


def create_cylinder_z_stiffened(height, r, stack, plyt, laminaprop,
        nstiff, stack_z, plyt_z, width_flange, width_web, width_free_flange,
        side='internal', flange_side=-1, bc='CC', m=12, n=6, n_flange=4,
        n_web=4, n_free_flange=4, conn_method='null-space',
        pd_method='null-space', models=None, attached_flange=True,
        free_flange=True):
    r"""Create a cylinder multidomain assembly with Z-section stringers

    The cylinder of radius ``r`` (middle surface of the skin) and length
    ``height`` has ``nstiff`` identical stringers, equally spaced at the
    pitch `d = 2 \pi r/n_s`, ``nstiff`` `= n_s`. Each stringer is made of
    three plates, given by the widths of their middle lines:

    - the attached flange, of width ``width_flange`` `= b_f` from the
      middle surface of the web to its free edge, bonded to the skin: the
      strip of skin under it is one curved domain of the group ``'strip'``
      whose laminate is that of the skin and that of the flange;
    - the web, a flat domain of the group ``'web'`` normal to the skin, of
      width ``width_web`` `= H_w` from the surface of the strip on the side
      of the stringer, the face of the attached flange (the surface of the
      skin with ``attached_flange=False``), to the middle surface of the
      free flange;
    - the free flange, a flat domain of the group ``'free_flange'`` normal
      to the web at its free edge, of width ``width_free_flange`` `= b_l`
      from the middle surface of the web, on the side of the web opposite
      to the attached flange.

    The circumference is divided, for each stringer `k = 0, \dots, n_s -
    1`, into its strip and the bay of skin up to the next strip, a curved
    domain of the group ``'skin'`` of width `d - b_f`. With
    ``flange_side=-1`` the strip of stringer `k` spans `y_k - b_f \le y \le
    y_k`, `y_k = (k + 1) d`, the web stands at `y_k` and the free flange
    extends towards `+y`; with ``flange_side=1`` the strip spans `y_k \le y
    \le y_k + b_f`, `y_k = k d`, and the free flange extends towards `-y`,
    `y = r \theta` being the arc length along the circumference. The
    stringers stand inside the cylinder with ``side='internal'`` and
    outside with ``side='external'``.

    Consecutive skin domains (bays and strips) are connected with
    ``'SSycte'``, the last to the first, each strip to its web with
    ``'BFycte'`` along the line of the web, at the face of the strip on the
    side of the stringer (key ``'face'``, ``'lower'`` for internal and
    ``'upper'`` for external stringers), and each web to its free flange
    with ``'BFycte'`` along the free edge of the web, at the middle surface
    of the web. The ends `x = 0` and
    `x` = ``height`` have the boundary conditions ``bc[0]`` and ``bc[1]``:
    ``'C'``, clamped, `u = v = w = w_{,x} = 0` on the skin domains and `u =
    v = v_{,x} = w = w_{,x} = 0` on the plates; ``'S'``, simply supported,
    `v = w = 0` on every domain, `u` free. The longitudinal edges of every
    domain are free, the connections join them. With ``bc='SS'`` the
    assembly keeps the rigid axial translation, which the caller removes,
    e.g. with :meth:`.Shell.add_point_pd`.

    The assembly contains the skin domains, ordered along `y`, followed by
    the webs and the free flanges. The connections are stored in the
    assembly, ``md.conn``, and also returned.

    Parameters
    ----------
    height : float
        Cylinder length (along `x`), length.
    r : float
        Radius of the middle surface of the skin, length.
    stack : array-like
        Stacking sequence of the skin, the angle of each ply in degrees.
    plyt : float
        Ply thickness of the skin, length.
    laminaprop : list or tuple
        Orthotropic lamina properties of the skin and of the stringers:
        `E_1, E_2, \nu_{12}, G_{12}, G_{13}, G_{23}`, or `E, \nu` for an
        isotropic material. The moduli are forces per unit area, `\nu_{12}`
        is dimensionless.
    nstiff : int
        Number of stringers.
    stack_z : array-like, or sequence of three array-like
        Laminate of the stringers, the angle of each ply in degrees. One
        stack is the bent sheet: its plies in the order of the `z` axis of
        the skin at the attached flange, followed around the bends to the
        web and to the free flange, see the Notes. Three stacks give the
        laminate of each plate, in the order attached flange, web, free
        flange, each in the local axes of its plate (for the attached
        flange, those of the skin); the stack of a plate left out with
        ``attached_flange=False`` or ``free_flange=False`` is not used.
    plyt_z : float
        Ply thickness of the laminate of the stringers, the same for every
        ply of the three plates, length.
    width_flange : float
        Width `b_f` of the attached flange, from the middle surface of the
        web to its free edge, length.
    width_web : float
        Width `H_w` of the web, from the face of the strip on the side of
        the stringer, the face of the attached flange or, with
        ``attached_flange=False``, the surface of the skin, to the middle
        surface of the free flange, length.
    width_free_flange : float
        Width `b_l` of the free flange, from the middle surface of the web
        to its free edge, length.
    side : str, optional
        ``'internal'``, the default, for stringers inside the cylinder, or
        ``'external'`` for stringers outside.
    flange_side : int, optional
        ``-1``, the default, or ``1``: the side along `y` of the web where
        the attached flange lies, see above.
    bc : str, optional
        Boundary conditions of the ends `x = 0` and `x` = ``height``, two
        characters ``'C'`` or ``'S'``. Default ``'CC'``.
    m : int, optional
        Number of terms along `x` of every domain. Default ``12``.
    n : int, optional
        Number of terms along `y` of the bays of skin. Default ``6``.
    n_flange, n_web, n_free_flange : int, optional
        Number of terms along `y` of the strips, of the webs and of the free
        flanges. Default ``4``.
    conn_method, pd_method : str, optional
        How the connections and the prescribed displacements are imposed,
        ``'null-space'``, the default, or ``'penalty'``, see
        :class:`.MultiDomain`.
    models : dict, optional
        Model of the domains of each group: ``'skin'``, the bays and the
        strips, ``'cylshell_clpt_sanders'`` (default),
        ``'cylshell_clpt_gl'``, ``'cylshell_clpt_tl'`` or
        ``'cylshell_clpt_donnell'``, and ``'stringer'``, the webs and the
        free flanges, ``'plate_clpt_gl'`` (default), ``'plate_clpt_tl'`` or
        ``'plate_clpt_donnell'``, see :mod:`panels.models`. A missing key
        takes its default.
    attached_flange : bool, optional
        ``False`` leaves out the attached flange: the strips have the
        laminate of the skin. Default ``True``.
    free_flange : bool, optional
        ``False`` leaves out the free flange, the stringer is then an L or,
        with ``attached_flange=False``, a blade. Default ``True``.

    Returns
    -------
    md, conns, info : tuple
        The multidomain assembly, the list of connection dictionaries, the
        list stored in the assembly, ``md.conn``, and a dictionary with
        ``'skin'``,
        the list of the skin domains (bays and strips) ordered along `y`,
        ``'strips'``, ``'webs'`` and ``'free_flanges'``, the lists of these
        domains, one per stringer (``'free_flanges'`` empty with
        ``free_flange=False``), ``'pitch'``, the pitch `d`, and
        ``'models'``, the model of each group, ``models`` completed with the
        defaults, and ``'stacks_z'``, the stack of each plate in its local
        axes, a dictionary with the keys ``'flange'``, ``'web'`` and
        ``'free_flange'``.

    Raises
    ------
    ValueError
        When ``models`` has a key other than ``'skin'`` and ``'stringer'``
        or a model not listed above for its key, when ``side``,
        ``flange_side`` or ``bc`` has another value than those listed
        above, when ``stack_z`` is neither one stack nor three non-empty
        stacks, when ``nstiff < 1``, or when ``width_flange`` is not
        smaller than the pitch.

    Notes
    -----
    The local axes of a plate connected with ``'BFycte'`` are those of the
    domain it is connected to rotated by 90 degrees about `x`, see
    :meth:`.MultiDomain.get_kC_conn`: the `y` axis of a web points towards
    the axis of the cylinder, so a web stands inside the skin when its edge
    `y = 0` is connected and outside when its edge `y = H_w` is; the `y`
    axis of a free flange points along `-y` of the skin, so a free flange
    extends towards `+y` from its edge `y = b_l` and towards `-y` from its
    edge `y = 0`. The `z` axis of the skin points outwards, that of a web
    along `+y` of the skin and that of a free flange inwards. The root of a
    web is tied to the face of the strip, `z = d - h_s/2` (internal) or `z =
    d + h_s/2` (external), `h_s` and `d` being the thickness and the offset
    of the laminate of the strip, see
    :mod:`panels.multidomain.connections.kCBFycte_face`.

    A single ``stack_z`` is the sheet bent into the Z. Its normal `n` is
    the `z` axis of the skin at the attached flange and turns with the sheet
    around the bends, and its angles are measured from `x` towards the
    transverse direction `t` of the sheet, `x, t, n` right-handed (`t` is the
    `y` axis of the skin at the attached flange). `n` points along `+y` of
    the skin in the web of an internal stringer with ``flange_side=-1`` and
    of an external stringer with ``flange_side=1``, along `-y` in the web of
    the other two orientations, and outwards in the free flange. Where the
    `z` axis of a plate is opposite to `n`, its `y` axis is opposite to `t`,
    and the plate has the plies of the sheet in reverse order with the angles
    changed in sign. The axis `x` is common to the sheet and to the plate,
    and both triads are right-handed, so flipping `z` flips `y`: an angle
    measured from `x` towards `t` is the opposite angle measured from `x`
    towards `y`, e.g. a `+45` ply of the sheet is a `-45` ply of the plate,
    and reading the plies along `z` instead of `n` reverses their order: its ply `k` of `N` is the ply `N + 1 - k` of the sheet,
    at the angle `-\theta_{N + 1 - k}`. Its stiffness matrices are those of
    the sheet with the signs of `A_{16}, A_{26}`, of `B_{11}, B_{12}, B_{22},
    B_{66}` and of `D_{16}, D_{26}` changed. This holds for the free flange,
    and for the web in the two orientations where `n` points along `-y`. A
    symmetric sheet without angle plies, e.g. a cross-ply or an isotropic
    sheet, has the same laminate in the three plates. Three stacks give any
    laminate to each plate.

    The attached flange is bonded continuously to the skin (rivets or a
    bond line are not represented): the laminate of the strip is the
    laminate of the skin with the plies of the attached flange on the side
    of the stringer, of total thickness `h + t_z`, `h` and `t_z` the
    thicknesses of the skin and of the attached flange, with the offset
    `-t_z/2` (internal) or `+t_z/2` (external), so that the reference
    surface of the strip remains the middle surface of the skin and the
    strip and the bays share the displacements of that surface along their
    common edges. The web starts at the face of the strip, the face of the
    attached flange, `h/2 + t_z` from the middle surface of the skin, `t_z/2`
    beyond the middle surface of the attached flange, where the middle line
    of the sheet turns into the web: the corner, `t_z` by the thickness of
    the web, belongs to the attached flange, and the web does not overlap
    the skin or the flange. Its root follows the Kirchhoff-Love kinematics
    of the strip at that face, with the Sanders-Koiter term `z v/r`, see
    :mod:`panels.multidomain.connections.kCBFycte_face`. A stringer of
    overall depth `H` from the face of the attached flange on the skin to
    the outer face of the free flange, with the same thickness `t_z` in the
    three plates, has `H_w = H - 3 t_z/2`. The corners are square: the
    length of the bends is not represented.

    A Z section has no plane of symmetry through the axis of the cylinder,
    so the whole circumference is modelled. The free flange needs at least
    4 terms across: with fewer, :class:`.ConnectionTermsWarning` is issued
    and the connection to the web locks.

    The Z stringers of Peterson and Dow [1]_ (Fig. 2(b), 120 internal
    stringers on cylinders of 23.74 in radius) are represented with this
    assembly in the tests of the package.

    References
    ----------
    .. [1] Peterson, J. P., and Dow, M. B., "Compression Tests on Circular
       Cylinders Stiffened Longitudinally by Closely Spaced Z-Section
       Stringers", NASA Memo 2-12-59L, 1959.

    Examples
    --------
    A cylinder with 4 internal Z stringers, the groups of the domains and
    the connections:

    >>> from panels.multidomain import create_cylinder_z_stiffened
    >>> md, conns, info = create_cylinder_z_stiffened(height=0.5, r=0.25,
    ...     stack=[0], plyt=1.e-3, laminaprop=(70.e9, 0.3), nstiff=4,
    ...     stack_z=[0], plyt_z=1.e-3, width_flange=0.01, width_web=0.015,
    ...     width_free_flange=0.008, m=4, n=4)
    >>> print([len(info[k]) for k in ('skin', 'strips', 'webs', 'free_flanges')])
    [8, 4, 4, 4]
    >>> print(sorted(set(c['func'] for c in conns)), len(conns),
    ...       md.conn is conns)
    ['BFycte', 'SSycte'] 16 True

    A laminated bent sheet of two plies, and the stack of each plate in its
    local axes:

    >>> laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    >>> md, conns, info = create_cylinder_z_stiffened(height=0.5, r=0.25,
    ...     stack=[0, 90, 0], plyt=0.125e-3, laminaprop=laminaprop, nstiff=4,
    ...     stack_z=[0, 45], plyt_z=0.125e-3, width_flange=0.01,
    ...     width_web=0.015, width_free_flange=0.008, m=4, n=4)
    >>> print(info['stacks_z'])
    {'flange': [0.0, 45.0], 'web': [0.0, 45.0], 'free_flange': [-45.0, 0.0]}

    """
    models = _models(models)
    if side not in ('internal', 'external'):
        raise ValueError("side must be 'internal' or 'external', got %r"
                         % (side, ))
    if flange_side not in (-1, 1):
        raise ValueError('flange_side must be -1 or 1, got %r'
                         % (flange_side, ))
    if len(bc) != 2 or any(end not in _END_FLAGS for end in bc):
        raise ValueError("bc must be two characters 'C' or 'S', got %r"
                         % (bc, ))
    if nstiff < 1:
        raise ValueError('At least one stringer is needed')
    stack = list(stack)
    stacks_z = _stacks_z(stack_z, side, flange_side)
    tz = len(stacks_z['flange'])*plyt_z
    bf, Hw, bl = width_flange, width_web, width_free_flange
    pitch = 2*np.pi*r/nstiff
    bbay = pitch - bf
    if bbay <= 0:
        raise ValueError('width_flange must be smaller than the pitch')
    (sk1, pl1), (sk2, pl2) = _END_FLAGS[bc[0]], _END_FLAGS[bc[1]]

    def domain(group, b, n_y, y0=0., curved=False, strip=False):
        if strip and attached_flange:
            # the plies of the flange on the side of the stringer
            st_f = stacks_z['flange']
            if side == 'internal':
                st = st_f + stack
                plyts = [plyt_z]*len(st_f) + [plyt]*len(stack)
                offset = -tz/2
            else:
                st = stack + st_f
                plyts = [plyt]*len(stack) + [plyt_z]*len(st_f)
                offset = tz/2
            lam = dict(stack=st, plyts=plyts, plyt=plyt, offset=offset)
        elif curved:
            lam = dict(stack=stack, plyt=plyt)
        else:
            lam = dict(stack=stacks_z[group], plyt=plyt_z)
        kw = dict(r=r) if curved else {}
        model = models['skin'] if curved else models['stringer']
        p = Shell(group=group, x0=0., y0=y0, a=height, b=b, m=m, n=n_y,
                  laminaprop=laminaprop, model=model, **lam, **kw)
        _set_end(p, 'x1', sk1 if curved else pl1)
        _set_end(p, 'x2', sk2 if curved else pl2)
        _set_end(p, 'y1', _FREE)
        _set_end(p, 'y2', _FREE)
        return p

    skin, strips, webs, free_flanges = [], [], [], []
    for k in range(nstiff):
        y0 = k*pitch
        if flange_side == -1:
            bay = domain('skin', bbay, n, y0=y0, curved=True)
            strip = domain('strip', bf, n_flange, y0=y0 + bbay, curved=True,
                           strip=True)
            skin += [bay, strip]
        else:
            strip = domain('strip', bf, n_flange, y0=y0, curved=True,
                           strip=True)
            bay = domain('skin', bbay, n, y0=y0 + bf, curved=True)
            skin += [strip, bay]
        strips.append(strip)
    conns = []
    for p1, p2 in zip(skin, skin[1:] + skin[:1]):
        conns.append(dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0.))
    for strip in strips:
        web = domain('web', Hw, n_web)
        yweb = strip.b if flange_side == -1 else 0.
        conns.append(dict(p1=strip, p2=web, func='BFycte', ycte1=yweb,
                          ycte2=0. if side == 'internal' else Hw,
                          face='lower' if side == 'internal' else 'upper'))
        webs.append(web)
        if free_flange:
            lip = domain('free_flange', bl, n_free_flange)
            ytip = Hw if side == 'internal' else 0.
            conns.append(dict(p1=web, p2=lip, func='BFycte', ycte1=ytip,
                              ycte2=bl if flange_side == -1 else 0.))
            free_flanges.append(lip)
    md = MultiDomain(skin + webs + free_flanges, conns,
                     conn_method=conn_method, pd_method=pd_method)
    info = dict(skin=skin, strips=strips, webs=webs,
                free_flanges=free_flanges, pitch=pitch, models=models,
                stacks_z=stacks_z)
    return md, conns, info
