r"""Rectangular wing-box made of plate domains
(:mod:`panels.multidomain.wingbox`)

:func:`.create_wingbox` builds the untapered wing-box of Elalfy, De Breuker
and Castro, case High-AR-t5, with plate domains connected by penalty
stiffnesses, optionally with blade stringers, and
:func:`.wingbox_tip_force_lb` runs its linear buckling analysis under a
vertical tip force. Both are imported in :mod:`panels.multidomain`.

The defaults are those of the case High-AR-t5: chord ``W = 50.5`` mm,
half-span ``L = 1010`` mm, height ``H = 12.12`` mm, 8 ribs (9 bays), skins
1 mm thick, spars 5 mm thick, ``E = 73`` GPa, ``nu = 0.33``, ``rho = 2795``
kg/m\ :sup:`3`. The wing-box is clamped at the root and loaded by a
vertical tip force.

.. note::

    The ribs are connected to the skins with ``'BFxcte'`` and, by default,
    to the spars with ``'BFxycte'``, which joins the edge `x` = constant of
    a spar to the edge `y` = constant of a rib. The stringers pass through
    the ribs without a connection.

Notes
-----
The geometry and the material are those of [1]_, Table 1, p. 7, case
High-AR-t5, and Table 2, p. 7, with `E = 73` GPa and `G = 27.44` GPa, i.e.
`\nu = 0.33`. The first buckling load from Abaqus (S4R shells) is 173.11 N,
[1]_, Table 3, p. 8.

References
----------
.. [1] Elalfy, M. H., De Breuker, R., and Castro, S. G. P., Beam-like models
   for the post-buckling behaviour of wingbox structures, preprint, Delft
   University of Technology.

"""
import numpy as np
from scipy.sparse import csr_matrix
from structsolve import lb, solve

from panels.shell import Shell
from panels.multidomain.multidomain import MultiDomain


__all__ = ['create_wingbox', 'wingbox_tip_force_lb']

_X, _Y, _Z = np.eye(3)


def _plate(a, b, t, m, n, group, origin, ax, axy, E, nu, rho):
    r"""Isotropic plate domain with all the boundary condition flags free,
    at ``origin`` with its local axes `x` and `y` along ``ax`` and ``axy``"""
    p = Shell(a=a, b=b, r=None, stack=[0.], plyt=t, laminaprop=(E, nu),
              rho=rho, model='plate_clpt_donnell', m=m, n=n, group=group)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for dof in ('u', 'v', 'w', 'wr'):
            setattr(p, edge + dof, 1.)
    o = np.asarray(origin, float)
    p.x0, p.y0, p.z0 = o
    p.point_x = o + ax
    p.point_xy = o + axy
    return p


def create_wingbox(nstr=0, hs=3.e-3, ts=1.5e-3, L=1.010, W=50.5e-3,
        H=12.12e-3, t_skin=1.e-3, t_spar=5.e-3, t_rib=1.e-3, nbays=9,
        E=73.e9, nu=0.33, rho=2795., terms=None, P=1., dx_load=10.e-3,
        follower=False, tip_rib=True, stringers=True, ribs=True,
        conn_method='penalty', rib_spar=True):
    r"""Create the multidomain assembly of a rectangular wing-box

    The defaults are the case High-AR-t5 of Elalfy, De Breuker and Castro,
    see :mod:`panels.multidomain.wingbox`, without stringers. All the domains
    are isotropic ``'plate_clpt_donnell'`` :class:`.Shell` objects:

    - per bay, ``nstr + 1`` strips of each skin, of the groups
      ``'skin_top'`` and ``'skin_bot'``, at `z = \pm H/2`, connected to each
      other with ``'SSycte'``;
    - two spars, of the group ``'spar'``, at `y = 0` and `y = W`, with the
      height ``H`` between the mid-surfaces of the skins, connected to the
      skins with ``'BFycte'``, the skin being the base;
    - optionally ``nstr`` blade stringers per skin, of the group
      ``'stringer'``, equally spaced, inside the box, of height ``hs`` from
      the inner surface of the skin, i.e. ``hs + t_skin/2`` from its
      mid-surface, and thickness ``ts``, connected to the skins with
      ``'BFycte'``;
    - with ``ribs=True``, a rib of the group ``'rib'`` at the end of each bay,
      and with ``tip_rib=True`` also at the tip, `x = L`, split into strips
      like the skins, connected to the skins with ``'BFxcte'`` and, with
      ``rib_spar=True``, to the spars with ``'BFxycte'``, the edge `x = a`
      of the spar to the edge `y = 0` of the first rib strip, or `y` equal
      to the width of the strip for the last one. The stringers pass
      through the ribs without a connection.

    The skins, spars and stringers of consecutive bays are connected with
    ``'SSxcte'``, and those of the first bay are clamped at the root, `x =
    0`, with ``x1u = x1v = x1w = x1wr = 0``. The frames of the domains
    (global `X` along the span, `Y` along the chord, `Z` up) are: skins
    `(X, Y, Z)`; spars and stringers `(X, -Z, Y)`, their local `y` pointing
    down; ribs `(-Z, Y, X)`, following the conventions of ``'BFycte'`` and
    ``'BFxcte'``.

    The tip force ``P``, upwards, is a uniform pressure on the strips of the
    upper skin of the last bay over the last ``dx_load`` of the span, of
    resultant ``P``, added with :meth:`.Shell.add_pressure_load`, a follower
    load with ``follower=True``.

    The code of this function is illustrative of how it constructs the
    multidomain assembly:

    .. literalinclude:: ../../panels/multidomain/wingbox.py
        :pyobject: create_wingbox

    Parameters
    ----------
    nstr : int, optional
        Number of blade stringers per skin; the skin of each bay is split
        into ``nstr + 1`` strips.
    hs, ts : float, optional
        Height of the stringers from the inner surface of the skin, and
        their thickness, length.
    L, W, H : float, optional
        Half-span, chord (between the mid-surfaces of the spars) and height
        (between the mid-surfaces of the skins), length.
    t_skin, t_spar, t_rib : float, optional
        Thicknesses of the skins, the spars and the ribs, length. The
        default thickness of the ribs is that of the skins.
    nbays : int, optional
        Number of bays, ``nbays - 1`` ribs between the root and the tip.
    E, nu, rho : float, optional
        Young's modulus, force per unit area, Poisson's ratio and density,
        mass per unit volume.
    terms : dict, optional
        Number of terms ``(m, n)`` of the domains, by the keys ``'skin'``
        (default ``(8, 4)`` with stringers, ``(8, 8)`` without),
        ``'spar'`` (``(8, 5)``), ``'stringer'`` (``(8, 4)``) and ``'rib'``
        (``(4, 4)`` with stringers, ``(4, 6)`` without).
    P : float, optional
        Tip force, force.
    dx_load : float, optional
        Length of the loaded strip of the upper skin at the tip.
    follower : bool, optional
        Whether the tip load is a follower load, see
        :meth:`.Shell.add_pressure_load`.
    tip_rib : bool, optional
        Whether a rib is added at the tip.
    stringers : bool, optional
        Whether the stringers are added; with ``False`` and ``nstr > 0`` the
        skins are still split into ``nstr + 1`` strips.
    ribs : bool, optional
        Whether the ribs are added.
    rib_spar : bool, optional
        Whether the ribs are connected to the spars with ``'BFxycte'``; with
        ``False`` they are connected to the skins only.
    conn_method : str, optional
        How the connections are imposed, ``'penalty'``, the default, with
        the default stiffnesses of :func:`.calc_kt_kr`, or ``'null-space'``,
        see :class:`.MultiDomain`.

    Returns
    -------
    md, info : tuple
        The assembly, whose connections are stored in it, and a dictionary
        with ``'bays'``, a list with a dictionary per bay of the lists of
        domains ``'top'``, ``'bot'``, ``'spars'``, ``'st'`` (stringers) and
        ``'ribs'``, and ``'a'``, the length of a bay, ``'ws'``, the width of
        a skin strip, and ``'conn'``, the connections.

    Notes
    -----
    The geometry and the material of the defaults are those of the case
    High-AR-t5 of [1]_, Tables 1 and 2, p. 7. The thickness of the ribs is
    not given in [1]_, and the tip force is there a concentrated force at
    the tip, here distributed over the last ``dx_load`` of the upper skin.

    References
    ----------
    .. [1] Elalfy, M. H., De Breuker, R., and Castro, S. G. P., Beam-like
       models for the post-buckling behaviour of wingbox structures,
       preprint, Delft University of Technology.

    """
    terms = terms or {}
    ms, ns = terms.get('skin', (8, 4 if nstr else 8))
    msp, nsp = terms.get('spar', (8, 5))
    mst, nst = terms.get('stringer', (8, 4))
    mr, nr = terms.get('rib', (4, 4 if nstr else 6))
    mat = dict(E=E, nu=nu, rho=rho)
    a = L/nbays
    ws = W/(nstr + 1)
    hb = hs + t_skin/2  # stringer height from the skin mid-surface
    doms, conn, bays = [], [], []
    for i in range(nbays):
        x0 = i*a
        top = [_plate(a, ws, t_skin, ms, ns, 'skin_top', (x0, j*ws, H/2),
                      _X, _Y, **mat) for j in range(nstr + 1)]
        bot = [_plate(a, ws, t_skin, ms, ns, 'skin_bot', (x0, j*ws, -H/2),
                      _X, _Y, **mat) for j in range(nstr + 1)]
        spars = [_plate(a, H, t_spar, msp, nsp, 'spar', (x0, y, H/2), _X,
                        -_Z, **mat) for y in (0., W)]
        st_top, st_bot = [], []
        if stringers:
            st_top = [_plate(a, hb, ts, mst, nst, 'stringer',
                             (x0, (j + 1)*ws, H/2), _X, -_Z, **mat)
                      for j in range(nstr)]
            st_bot = [_plate(a, hb, ts, mst, nst, 'stringer',
                             (x0, (j + 1)*ws, -H/2 + hb), _X, -_Z, **mat)
                      for j in range(nstr)]
        for skin in (top, bot):
            conn += [dict(p1=p1, p2=p2, func='SSycte', ycte1=ws, ycte2=0.)
                     for p1, p2 in zip(skin[:-1], skin[1:])]
        conn += [dict(p1=top[0], p2=spars[0], func='BFycte', ycte1=0., ycte2=0.),
                 dict(p1=bot[0], p2=spars[0], func='BFycte', ycte1=0., ycte2=H),
                 dict(p1=top[-1], p2=spars[1], func='BFycte', ycte1=ws, ycte2=0.),
                 dict(p1=bot[-1], p2=spars[1], func='BFycte', ycte1=ws, ycte2=H)]
        conn += [dict(p1=top[j], p2=st, func='BFycte', ycte1=ws, ycte2=0.)
                 for j, st in enumerate(st_top)]
        conn += [dict(p1=bot[j], p2=st, func='BFycte', ycte1=ws, ycte2=hb)
                 for j, st in enumerate(st_bot)]
        bay_ribs = []
        if ribs and (i < nbays - 1 or tip_rib):
            bay_ribs = [_plate(H, ws, t_rib, mr, nr, 'rib',
                               (x0 + a, j*ws, H/2), -_Z, _Y, **mat)
                        for j in range(nstr + 1)]
            conn += [dict(p1=p1, p2=p2, func='SSycte', ycte1=ws, ycte2=0.)
                     for p1, p2 in zip(bay_ribs[:-1], bay_ribs[1:])]
            conn += [dict(p1=top[j], p2=r, func='BFxcte', xcte1=a, xcte2=0.)
                     for j, r in enumerate(bay_ribs)]
            conn += [dict(p1=bot[j], p2=r, func='BFxcte', xcte1=a, xcte2=H)
                     for j, r in enumerate(bay_ribs)]
            if rib_spar:
                conn += [dict(p1=spars[0], p2=bay_ribs[0], func='BFxycte',
                              xcte1=a, ycte2=0.),
                         dict(p1=spars[1], p2=bay_ribs[-1], func='BFxycte',
                              xcte1=a, ycte2=ws)]
        bay = dict(top=top, bot=bot, spars=spars, st=st_top + st_bot,
                   ribs=bay_ribs)
        if i == 0:
            for p in top + bot + spars + st_top + st_bot:
                p.x1u = p.x1v = p.x1w = p.x1wr = 0.
        else:
            prev = bays[-1]
            for k in ('top', 'bot', 'spars', 'st'):
                conn += [dict(p1=p1, p2=p2, func='SSxcte', xcte1=a, xcte2=0.)
                         for p1, p2 in zip(prev[k], bay[k])]
        bays.append(bay)
        doms += top + bot + spars + st_top + st_bot + bay_ribs
    # tip force: pressure on the upper skin of the last bay, resultant P
    for p in bays[-1]['top']:
        p.add_pressure_load(P/(W*dx_load), x1=a - dx_load, follower=follower,
                            cte=False)
    md = MultiDomain(doms, conn, conn_method=conn_method)
    return md, dict(bays=bays, a=a, ws=ws, conn=conn)


def wingbox_tip_force_lb(num_eigvalues=4, silent=True, **kwargs):
    r"""Linear buckling of the wing-box under a tip force

    The assembly comes from :func:`.create_wingbox`, with the tip force ``P
    = 1`` unless given in ``kwargs``. A linear static analysis gives the
    pre-buckling state: :meth:`.MultiDomain.calc_kC` and
    :meth:`.MultiDomain.calc_fext`, reduced with :meth:`.MultiDomain.reduce`
    and solved with ``structsolve.solve``, which removes the null rows and
    columns of the clamped root, and expanded with
    :meth:`.MultiDomain.expand`. With this state the geometric stiffness
    matrix :meth:`.MultiDomain.calc_kG` is computed, and the reduced matrices
    are passed to ``structsolve.lb``.

    Parameters
    ----------
    num_eigvalues : int, optional
        Number of eigenvalues to be extracted.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    kwargs : keyword arguments, optional
        Passed to :func:`.create_wingbox`.

    Returns
    -------
    md, c, eigvals, eigvecs : tuple
        Assembly, pre-buckling state (Ritz constants of the assembly at the
        tip force ``P``), eigenvalues and eigenvectors. ``eigvals`` are the
        load multipliers returned by ``structsolve.lb``, i.e. the buckling
        tip forces divided by ``P``, and ``eigvecs`` has one eigenvector of
        the assembly per column, with ``md.get_size()`` rows.

    Notes
    -----
    The first buckling mode of the wing-box without stringers is a local
    buckling of the upper skin in compression near the root, see [1]_,
    whose Abaqus model gives 173.11 N (Table 3, p. 8). With
    ``conn_method='null-space'`` the first buckling load converges to 176.4
    N, with skins of 12 x 12 terms, spars of 12 x 7 and ribs of 6 x 8;
    with the default terms and penalty connections it is 172.78 N, and
    161.9 N converged.

    References
    ----------
    .. [1] Elalfy, M. H., De Breuker, R., and Castro, S. G. P., Beam-like
       models for the post-buckling behaviour of wingbox structures,
       preprint, Delft University of Technology.

    Examples
    --------

    First buckling load, force, of the default wing-box under a tip force:

    >>> from panels.multidomain import wingbox_tip_force_lb
    >>> md, c, eigvals, eigvecs = wingbox_tip_force_lb(num_eigvalues=4)
    >>> round(float(eigvals[0]), 2)
    172.78

    """
    md, info = create_wingbox(**kwargs)
    k0 = md.reduce(md.calc_kC(silent=silent))
    c = md.expand(solve(k0, md.reduce(md.calc_fext(silent=silent)),
                        silent=silent))
    kG = md.reduce(md.calc_kG(c=c, silent=silent))
    eigvals, eigvecs = lb(csr_matrix(k0), csr_matrix(kG), silent=silent,
                          num_eigvalues=num_eigvalues)
    return md, c, eigvals, md.expand(eigvecs)
