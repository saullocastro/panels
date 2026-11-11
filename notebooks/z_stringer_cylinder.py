r"""Cylinders with Z-section stringers made of three plates

Helpers of the notebook ``peterson1959_blade_cylinder_buckling.ipynb``: the
middle-line dimensions of a Z section from the outer dimensions of the sheet
(:func:`z_midlines`), the properties of the section as modelled
(:func:`section_properties`), and the linear buckling under axial
compression about a membrane pre-buckling state (:func:`buckling_membrane`)
of the whole cylinder built with
:func:`panels.multidomain.create_cylinder_z_stiffened`, with the wave
numbers of the modes (:func:`skin_w`, :func:`classify`). The skin and the
stringers are of one isotropic material.

All quantities are in SI units or in any other consistent set;
:data:`IN`, :data:`LBF` and :data:`PSI` convert U.S. Customary values.

Notes
-----
The web of :func:`panels.multidomain.create_cylinder_z_stiffened` starts at
the face of the attached flange, `h/2 + t_z` from the middle surface of the
skin, and ends at the middle surface of the free flange; the section as
modelled is given by :func:`section_properties`.

"""
import time

import numpy as np
from numpy import pi
from structsolve import lb

from panels.multidomain import create_cylinder_z_stiffened

_trapz = getattr(np, 'trapezoid', None) or np.trapz

#: one inch, m
IN = 0.0254
#: one pound-force, N
LBF = 4.4482216152605
#: one psi, Pa
PSI = LBF/IN**2

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
        Skin thickness, not used: the web of the model starts at the face of
        the attached flange.
    bend_radius : float
        Inner radius of the two bends; each bend of centre-line radius
        `r_c` = ``bend_radius`` `+ t_z/2` shortens the adjacent flange by
        `(2 - \pi/2) r_c`, so that the plates have the length of the centre
        line of the bent sheet; ``0`` gives square corners.

    Returns
    -------
    dict
        ``tz``, ``bf`` (width of the attached flange, from the middle
        surface of the web), ``Hw`` (width of the web, from the face of the
        attached flange to the middle surface of the free flange) and
        ``bl`` (width of the free flange, from the middle surface of the
        web).

    """
    rc = bend_radius + tz/2 if bend_radius > 0 else 0.
    cut = (2 - pi/2)*rc
    return dict(tz=tz, bf=flange - tz/2 - cut, Hw=depth - 3*tz/2,
                bl=free_flange - tz/2 - cut)


def section_properties(sec, h, E, nu):
    r"""Properties of the stringer as modelled, per stringer

    The rectangles of the attached flange (centred at `h/2 + t_z/2` from
    the middle surface of the skin), of the web (from the face of the
    attached flange, `h/2 + t_z`, to the middle surface of the free flange)
    and of the free flange.
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
    z0 = h/2 + t
    rects = [(sec['bf']*t, zf, sec['bf']*t**3/12),
             (Hw*t, z0 + Hw/2, t*Hw**3/12),
             (sec['bl']*t, z0 + Hw, sec['bl']*t**3/12)]
    A = sum(r[0] for r in rects)
    zc = sum(r[0]*r[1] for r in rects)/A
    Io = sum(r[2] + r[0]*r[1]**2 for r in rects)
    J = (sec['bf'] + Hw + sec['bl'])*t**3/3
    G = E/(2*(1 + nu))
    return dict(A=A, zc=zc, Io=Io, J=J, EA=E*A, EIo=E*Io, GJ=G*J)


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
    supported (``bc`` of
    :func:`panels.multidomain.create_cylinder_z_stiffened`), the rigid
    axial translation removed by
    `u = 0` at the point `x = L/2` of the first skin domain, `y = 0`. ``R``,
    ``h``, ``L``: radius of the middle surface, skin thickness and length;
    ``E``, ``nu``: Young's modulus and Poisson's ratio; ``sec``: the middle
    lines of :func:`z_midlines`; ``mx``, ``ny``, ``nf``, ``nw``, ``nl``: the
    terms along `x` and across a bay, a strip, a web and a free flange; the
    other parameters are those of
    :func:`panels.multidomain.create_cylinder_z_stiffened`.

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
        The assembly and the dictionary ``info`` of
        :func:`panels.multidomain.create_cylinder_z_stiffened`.
    ndof : int
        Number of degrees of freedom of the reduced problem.
    dt : float
        Run time, s.

    """
    t0 = time.time()
    md, conns, groups = create_cylinder_z_stiffened(height=L, r=R,
            stack=[0.], plyt=h, laminaprop=(E, nu), nstiff=nstiff,
            stack_z=[0.], plyt_z=sec['tz'], width_flange=sec['bf'],
            width_web=sec['Hw'], width_free_flange=sec['bl'], side=side,
            flange_side=flange_side, bc=bc, m=mx, n=ny, n_flange=nf,
            n_web=nw, n_free_flange=nl, attached_flange=attached_flange,
            free_flange=free_flange)
    if bc == 'SS':
        groups['skin'][0].add_point_pd(L/2, 0., 1., 0., None, None, None,
                                       None)
    P1 = 0.
    for p in md.panels:
        thick = p.lam.h
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
