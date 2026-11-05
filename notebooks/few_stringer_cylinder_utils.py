r"""Utilities of the notebook egle1968_few_stringer_cylinder_vibration.ipynb

Freely supported steel cylinders with four or eight internal hat-section
(open channel) stringers, the shells of Egle and Sewall (1968) and of
Sewall and Naumann (1968, Fig. 12), modelled as a
:class:`panels.multidomain.MultiDomain` assembly of the whole cylinder (see
:func:`build_cylinder`):

- the skin, curved panels ``cylshell_clpt_sanders``: one narrow strip under
  each stringer base and one bay between two strips, connected with
  ``'SSycte'``;
- the base of each stringer, a curved panel ``cylshell_clpt_sanders`` bonded
  to the strip of skin under it with ``'SB'`` (``base='plate'``), or added
  to the laminate of that strip (``base='laminate'``);
- the two webs, flat plates ``plate_clpt_gl`` standing normal to the base
  (``'BFycte'``), and the two lips, flat plates ``plate_clpt_gl`` normal to
  the webs at their free edges (``'BFycte'``), pointing away from the
  channel.

The modes are classified by :func:`classify` with the dominant
circumferential harmonic `n`, the number of axial half-waves `m` and the
symmetry about the plane through the axis and the first stringer.

Quantities are in SI units; :data:`IN`, :data:`PSI` and :data:`LBS2IN4`
convert the U.S. Customary values of the papers.

"""
import time

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import norm as spnorm
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


def channel_midlines(t, base_width, height, lip, height_from='base_inner'):
    r"""Midline lengths of the hat-section stringer

    Parameters
    ----------
    t : float
        Wall thickness.
    base_width : float
        Outer width of the base (over the outer faces of the webs).
    height : float
        Height of the section up to the outer face of the lips, measured
        from the inner face of the base (``height_from='base_inner'``) or
        from its outer face, the face on the skin (``'base_outer'``).
    lip : float
        Width of a lip measured from the outer face of the web.

    Returns
    -------
    dict
        ``t``, ``base`` (distance between the midlines of the webs),
        ``web`` (from the midline of the base to the midline of the lips)
        and ``lip`` (from the midline of the web to the free edge).

    """
    total = height + (t if height_from == 'base_inner' else 0.)
    return dict(t=t, base=base_width - t, web=total - t, lip=lip + t/2)


def section_properties(sec, h, base='plate'):
    r"""Section properties of the stringer as modelled

    The section is the union of the rectangles of the plates of the model,
    each of thickness ``t`` along its midline: the base, centred at the
    distance `h/2 + t/2` from the middle surface of the skin, two webs and
    two lips. With ``base='laminate'`` the base is part of the laminate of
    the skin strip and each web starts at the middle surface of the skin, as
    connected in the model, so that it overlaps the skin and the base over
    `h/2 + t/2`.

    Returns a dict with the area ``A``, the distance of the centroid from
    the middle surface of the skin ``z``, the radius of gyration about the
    middle surface of the skin ``d`` (`d^2 A = \int z^2 dA`, the moment of
    inertia of the stringer strain energy of Egle and Sewall), the polar
    radius of gyration about the line of the middle surface under the
    stringer ``p`` (`p^2 A = \int (z^2 + y^2) dA`), the torsion constant of
    the open thin-walled section ``J`` `= \sum b t^3/3` and the moments of
    inertia about the centroid, ``Iyc`` (bending in the radial plane) and
    ``Izc`` (bending in the tangent plane).
    """
    t = sec['t']
    zb = h/2 + t/2
    zl = zb + sec['web']
    z0w = zb if base == 'plate' else 0.
    yw = sec['base']/2
    # (area, y, z, I about own horizontal axis, I about own vertical axis)
    rects = [(sec['base']*t, 0., zb, sec['base']*t**3/12, t*sec['base']**3/12)]
    lw = zl - z0w
    for s in (-1, 1):
        rects.append((lw*t, s*yw, (z0w + zl)/2, t*lw**3/12, lw*t**3/12))
        rects.append((sec['lip']*t, s*(yw + sec['lip']/2), zl,
                      sec['lip']*t**3/12, t*sec['lip']**3/12))
    A = sum(r[0] for r in rects)
    z = sum(r[0]*r[2] for r in rects)/A
    I0 = sum(r[3] + r[0]*r[2]**2 for r in rects)
    Iz = sum(r[4] + r[0]*r[1]**2 for r in rects)
    Iyc = I0 - A*z**2
    J = (sec['base'] + 2*lw + 2*sec['lip'])*t**3/3
    return dict(A=A, z=z, d=np.sqrt(I0/A), p=np.sqrt((I0 + Iz)/A), J=J,
                Iyc=Iyc, Izc=Iz)


_FREE = dict(u=1, ur=1, v=1, vr=1, w=1, wr=1)
#: freely supported end: skin and base `v = w = 0`; webs and lips, whose
#: local `v` and `w` are the two displacements in the plane of the
#: cross-section, `v = w = 0`; `u` free everywhere
FREELY_SUPPORTED = dict(_FREE, v=0, w=0)


def _set_end(p, end, flags):
    for k, v in flags.items():
        setattr(p, '%s%s' % (end, k), v)


def _panel(model, a, b, plyt, lam, rho, m, n, r=None, offset=0., y0=0.,
           group='skin'):
    kw = dict(r=r) if r is not None else {}
    p = Shell(group=group, x0=0., y0=y0, a=a, b=b, m=m, n=n, plyt=plyt,
              stack=[0.], laminaprop=lam, rho=rho, offset=offset,
              model=model, **kw)
    _set_end(p, 'x1', FREELY_SUPPORTED)
    _set_end(p, 'x2', FREELY_SUPPORTED)
    _set_end(p, 'y1', _FREE)
    _set_end(p, 'y2', _FREE)
    return p


def build_cylinder(R, h, L, E, nu, rho, nstiff, sec, base='plate',
        stiffened=True, mx=12, ny=16, ns=4, nw=4, nl=4):
    r"""Freely supported cylinder with ``nstiff`` internal hat stringers

    The whole circumference is modelled. Stringer `k` is centred at `\theta_k
    = \theta_0 + 2 \pi k/` ``nstiff``, `\theta_0 = ` ``sec['base']``/`(2R)`:
    the skin starts at `\theta = 0` with the strip under the base of the
    first stringer, of arc length ``sec['base']`` (the distance between the
    midlines of the webs), followed by the bay up to the next strip, and so
    on, the last bay connected to the first strip. With
    ``stiffened=False`` the cylinder is unstiffened, divided in the same
    panels.

    With ``base='plate'`` the base is a curved panel of thickness ``t``
    covering the strip, connected to it with ``'SB'`` (the lower face of the
    strip to the upper face of the base, `z` pointing outwards), and the
    webs start at the middle surface of the base, with the width
    ``sec['web']``. With ``base='laminate'`` the strip has the thickness `h
    + t` with the offset `-t/2`, its reference surface remaining the middle
    surface of the skin, and the webs start at that surface, with the width
    `h/2 + t/2 +` ``sec['web']``. The webs are connected with ``'BFycte'``
    at the edges `y = 0` and `y = b` of the base (or strip), with
    ``ycte2 = 0``: their local `y` axis points inwards (see
    :mod:`panels.multidomain.connections.kCBFycte`). Each lip is connected
    with ``'BFycte'`` at the free edge of its web; its local `y` axis is
    then the circumferential direction `-\theta`, so the lip of the web at
    `y = 0` is connected at its edge `y = 0` and that of the web at `y = b`
    at its edge `y =` ``sec['lip']``, both pointing away from the channel.

    Both ends are freely supported (:data:`FREELY_SUPPORTED`), the
    supported end of Egle and Sewall (1968): `v = w = 0` and `u` free for
    the skin and for every plate of the stringers.

    Parameters
    ----------
    R, h, L : float
        Radius of the middle surface, skin thickness and length, m.
    E, nu, rho : float
        Young's modulus, Pa, Poisson's ratio and density, kg/m^3, of the
        skin and of the stringers.
    nstiff : int
        Number of stringers.
    sec : dict
        Midlines of :func:`channel_midlines`, m.
    base : str
        ``'plate'`` or ``'laminate'``.
    stiffened : bool
        ``False`` for the unstiffened cylinder.
    mx, ny, ns, nw, nl : int
        Terms along `x` (all panels), across a bay, across a strip and its
        base, across a web and across a lip.

    Returns
    -------
    md : :class:`panels.multidomain.MultiDomain`
        The assembly, ``conn_method='null-space'``.
    skin : list
        The skin panels ordered in `\theta`.
    theta0 : float
        Angle of the first stringer, rad.

    """
    lam = isotropic(E, nu)
    bs = sec['base']
    bbay = 2*np.pi*R/nstiff - bs
    skin = []
    y0 = 0.
    for k in range(nstiff):
        if stiffened and base == 'laminate':
            strip = _panel('cylshell_clpt_sanders', L, bs, h + sec['t'], lam,
                           rho, mx, ns, r=R, offset=-sec['t']/2, y0=y0)
        else:
            strip = _panel('cylshell_clpt_sanders', L, bs, h, lam, rho, mx, ns,
                           r=R, y0=y0)
        bay = _panel('cylshell_clpt_sanders', L, bbay, h, lam, rho, mx, ny, r=R,
                     y0=y0 + bs)
        skin += [strip, bay]
        y0 += bs + bbay
    conns = []
    for p1, p2 in zip(skin, skin[1:] + skin[:1]):
        conns.append(dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0.))
    others = []
    if stiffened:
        t = sec['t']
        for k in range(nstiff):
            strip = skin[2*k]
            if base == 'plate':
                pb = _panel('cylshell_clpt_sanders', L, bs, t, lam, rho, mx, ns,
                            r=R, y0=strip.y0, group='base')
                conns.append(dict(p1=strip, p2=pb, func='SB'))
                others.append(pb)
                bw = sec['web']
            else:
                pb = strip
                bw = h/2 + t/2 + sec['web']
            for yb, ylip in ((0., 0.), (bs, sec['lip'])):
                web = _panel('plate_clpt_gl', L, bw, t, lam, rho, mx, nw,
                             group='web')
                lip = _panel('plate_clpt_gl', L, sec['lip'], t, lam, rho, mx,
                             nl, group='lip')
                conns.append(dict(p1=pb, p2=web, func='BFycte', ycte1=yb,
                                  ycte2=0.))
                conns.append(dict(p1=web, p2=lip, func='BFycte', ycte1=bw,
                                  ycte2=ylip))
                others += [web, lip]
    md = MultiDomain(skin + others, conns, conn_method='null-space')
    return md, skin, bs/(2*R)


def skin_w(md, c, skin, R, theta0, nx=41, npt=31):
    r"""Normal displacement `w(x, \theta)` of the skin on a grid

    Returns ``x`` (``nx`` values, m), ``theta`` (increasing, measured from
    the first stringer, `-\theta_0 \le \theta < 2\pi - \theta_0`, rad) and
    ``w`` with shape ``(ntheta, nx)``. Each bay is sampled at ``npt``
    points, each strip at 3.
    """
    ths, ws = [], []
    x = None
    for p in skin:
        npy = 3 if p.b < 0.5*skin[1].b else npt
        res = md.uvw(c, group=None, eval_panel=p, gridx=nx, gridy=npy)
        th = (p.y0 + res['y'][0][:, 0])/R - theta0
        ths.append(th[:-1])
        ws.append(res['w'][0][:-1])
        x = res['x'][0][0, :]
    return x, np.concatenate(ths), np.vstack(ws)


def harmonics(x, th, w, nmax=40):
    r"""Fourier coefficients of `w(x, \theta)` in `\theta`

    Returns ``a`` and ``b`` with shape ``(nmax + 1, nx)``, the coefficients
    of `\cos n\theta` and `\sin n\theta` at each `x`, `w = \sum_n (a_n \cos
    n\theta + b_n \sin n\theta)` (trapezoidal rule over the closed
    circumference).
    """
    thc = np.concatenate((th, [th[0] + 2*np.pi]))
    wc = np.vstack((w, w[:1]))
    ns = np.arange(nmax + 1)
    a = _trapz(np.cos(ns[:, None]*thc[None, :])[:, :, None]*wc[None], thc,
               axis=1)/np.pi
    b = _trapz(np.sin(ns[:, None]*thc[None, :])[:, :, None]*wc[None], thc,
               axis=1)/np.pi
    a[0] /= 2
    return a, b


def classify(x, a, b, L):
    r"""Wave numbers, symmetry and coupling ratio of a mode

    ``n`` is the harmonic with the largest amplitude
    `\sum_x (a_n^2 + b_n^2)` (`a_0` counted twice, the same norm as the other
    harmonics), as in Egle and Sewall (1968, p. 522); ``m`` is the sine
    `\sin(m\pi x/L)` with the largest projection of the dominant harmonic
    along `x`; ``sym`` is the fraction `\sum a_n^2/\sum (a_n^2 + b_n^2)` of
    the cosine terms, 1 for a mode symmetric about the plane of the first
    stringer and 0 for an antisymmetric one; ``ratio`` is the second largest
    to the largest amplitude of the harmonics, as defined by Egle and Sewall
    (p. 524-525).
    """
    amp2 = (a**2).sum(axis=1) + (b**2).sum(axis=1)
    amp2[0] = 4*(a[0]**2).sum()
    order = np.argsort(amp2)[::-1]
    n = int(order[0])
    ratio = float(np.sqrt(amp2[order[1]]/amp2[order[0]]))
    cn = a[n] if (a[n]**2).sum() >= (b[n]**2).sum() else b[n]
    ms = np.arange(1, 9)
    proj = np.abs(_trapz(cn[None, :]*np.sin(ms[:, None]*np.pi*x[None, :]/L),
                         x, axis=1))
    m = int(ms[np.argmax(proj)])
    sa, sb = (a**2).sum(), (b**2).sum()
    return m, n, float(sa/(sa + sb)), ratio


def _symmetrize_clusters(freqs, coefs, vecs, rtol=1.e-5):
    r"""Rotate each cluster of equal frequencies to symmetric and
    antisymmetric modes

    Within a cluster of (numerically) equal frequencies the eigensolver
    returns any orthogonal combination of the modes; the combination is
    rotated to the eigenvectors of the matrix of the products of the cosine
    coefficients, which separates the symmetric and antisymmetric modes.
    ``coefs`` is a list of ``(a, b)``, ``vecs`` the list of Ritz vectors.
    """
    i = 0
    nm = len(freqs)
    while i < nm:
        j = i + 1
        while j < nm and abs(freqs[j] - freqs[i]) <= rtol*freqs[i]:
            j += 1
        if j - i > 1:
            A = np.array([coefs[k][0].ravel() for k in range(i, j)])
            B = np.array([coefs[k][1].ravel() for k in range(i, j)])
            S = A @ A.T
            T = A @ A.T + B @ B.T
            # generalized problem S q = mu T q: mu = fraction of cosines
            Lc = np.linalg.cholesky(T)
            Li = np.linalg.inv(Lc)
            mu, Q = np.linalg.eigh(Li @ S @ Li.T)
            Q = Li.T @ Q
            shape = coefs[i][0].shape
            newA = Q.T @ A
            newB = Q.T @ B
            V = np.array([vecs[k] for k in range(i, j)])
            newV = Q.T @ V
            for q, k in enumerate(range(i, j)):
                coefs[k] = (newA[q].reshape(shape), newB[q].reshape(shape))
                vecs[k] = newV[q]
        i = j
    return coefs, vecs


def skin_inplane_dofs(md, skin):
    r"""Indices of the in-surface DOFs `u, v` of the skin panels

    The panels with the classical laminated plate theory have 3 DOFs per
    term, ordered `u, v, w` (see :mod:`panels.models`), from the row
    ``row_start`` of each panel in the assembly.
    """
    idx = []
    for p in skin:
        start = p.row_start
        nterms = p.m*p.n
        base = start + 3*np.arange(nterms)
        idx += [base, base + 1]
    return np.sort(np.concatenate(idx))


def frequencies(md, skin, R, L, theta0, num_eigvalues=300, fmin=1.,
        fmax=np.inf, no_skin_inplane_inertia=False, nmax=40):
    r"""Natural frequencies of the assembly, classified

    With ``no_skin_inplane_inertia=True`` the rows and columns of the
    in-surface DOFs `u, v` of the skin panels in the mass matrix
    (:func:`skin_inplane_dofs`) are multiplied by `10^{-3}`, which scales
    the in-surface inertia of the skin by `10^{-6}`, the simplification of
    the numerical results of Egle and Sewall (1968, p. 522), the inertia of
    the stringers being kept. (Zero instead of `10^{-6}` would leave the
    axial rigid-body translation of the unstiffened skin without stiffness
    and without mass.)

    Returns a list of dicts ``f`` (Hz), ``m``, ``n``, ``sym``, ``ratio`` (see
    :func:`classify`), ``a``, ``b`` (:func:`harmonics`) and ``c`` (the Ritz
    vector), in increasing frequency, the modes with ``f < fmin`` or ``f >
    fmax`` excluded (the last of the ``num_eigvalues`` computed eigenpairs
    converge less tightly: ``fmax`` must lie well below the highest);
    the grid ``x``, the run time, s, the largest relative residual
    `\|(\boldsymbol{K} - \omega^2 \boldsymbol{M}) \boldsymbol{\phi}\| /
    \|\boldsymbol{K} \boldsymbol{\phi}\|` and the largest backward error
    `\|(\boldsymbol{K} - \omega^2 \boldsymbol{M}) \boldsymbol{\phi}\| /
    ((\|\boldsymbol{K}\|_F + \omega^2 \|\boldsymbol{M}\|_F)
    \|\boldsymbol{\phi}\|)` of the returned modes, and the highest computed
    frequency, Hz. The eigenpairs are accepted by ``structsolve.freq`` up
    to a relative residual of 1 (``check_rtol``): the reduced matrices of
    the null-space method are ill-conditioned, and accurate eigenpairs
    (backward errors of the order of `10^{-14}`) may have relative
    residuals of `10^{-2}` to `10^{-1}`; both measures are returned.
    """
    t0 = time.time()
    Kfull = md.calc_kC(silent=True)
    Mfull = md.calc_kM(silent=True)
    if no_skin_inplane_inertia:
        keep = np.ones(Mfull.shape[0])
        keep[skin_inplane_dofs(md, skin)] = 1.e-3
        Mskin = None
        for p in skin:
            Mp = p.calc_kM(size=md.get_size(), row0=p.row_start, col0=p.col_start,
                           silent=True)
            Mskin = Mp if Mskin is None else Mskin + Mp
        D = diags(keep)
        Mfull = csr_matrix(Mfull - Mskin + D @ Mskin @ D)
    K = md.reduce(Kfull)
    M = md.reduce(Mfull)
    eigvals, eigvecs = freq(K, M, tol=0, sparse_solver=True, silent=True,
                            num_eigvalues=num_eigvalues, symmetric=True,
                            check_rtol=1.)
    normK, normM = spnorm(K), spnorm(M)
    omegas = np.sqrt(np.abs(eigvals.real))
    order = [k for k in np.argsort(omegas) if fmin <= omegas[k]/(2*np.pi) <= fmax]
    fs, coefs, vecs = [], [], []
    rres = berr = 0.
    x = None
    for k in order:
        phi = eigvecs[:, k].real
        Kphi = K @ phi
        r = np.linalg.norm(Kphi - omegas[k]**2*(M @ phi))
        rres = max(rres, r/np.linalg.norm(Kphi))
        berr = max(berr, r/((normK + omegas[k]**2*normM)*np.linalg.norm(phi)))
        c = md.expand(phi)
        x, th, w = skin_w(md, c, skin, R, theta0)
        coefs.append(harmonics(x, th, w, nmax=nmax))
        vecs.append(c)
        fs.append(omegas[k]/(2*np.pi))
    coefs, vecs = _symmetrize_clusters(fs, coefs, vecs)
    out = []
    for f, (a, b), c in zip(fs, coefs, vecs):
        m, n, sym, ratio = classify(x, a, b, L)
        out.append(dict(f=f, m=m, n=n, sym=sym, ratio=ratio, a=a, b=b, c=c))
    fhigh = omegas.max()/(2*np.pi)
    return out, x, time.time() - t0, rres, berr, fhigh


def lowest(modes, symmetry=None, tol=0.1):
    r"""Lowest frequency of each ``(m, n)``, a dict of mode dicts

    ``symmetry='sym'`` keeps the modes with ``sym > 1 - tol``,
    ``'antisym'`` those with ``sym < tol``, ``None`` all modes.
    """
    out = {}
    for mode in modes:
        if symmetry == 'sym' and mode['sym'] < 1 - tol:
            continue
        if symmetry == 'antisym' and mode['sym'] > tol:
            continue
        key = (mode['m'], mode['n'])
        if key not in out:
            out[key] = mode
    return out


def donnell_ss(R, h, L, E, nu, rho, m, n, inplane_inertia=True):
    r"""Frequency, Hz, of the freely supported unstiffened cylinder, Donnell

    The modes `u = A \cos(m\pi x/L) \cos n\theta`, `v = B \sin(m\pi x/L)
    \sin n\theta`, `w = C \sin(m\pi x/L) \cos n\theta` satisfy the
    equations of motion of the Donnell shell with `\lambda = m \pi R/L`,
    `k = h^2/(12R^2)` and the frequency parameter `\Delta = \rho (1 -
    \nu^2) R^2 \omega^2/E` (Leissa, Vibration of Shells, NASA SP-288,
    1973, Ch. 2): `\Delta` is the smallest eigenvalue of the symmetric
    matrix

    .. math::

        \begin{bmatrix}
        \lambda^2 + \frac{1 - \nu}{2} n^2 & \frac{1 + \nu}{2} \lambda n &
        \nu \lambda \\
        & \frac{1 - \nu}{2} \lambda^2 + n^2 & n \\
        \text{sym.} & & 1 + k (\lambda^2 + n^2)^2
        \end{bmatrix}

    and, without the in-surface inertia (``inplane_inertia=False``), the
    Schur complement of the last diagonal term, `\Delta = (1 -
    \nu^2)\lambda^4/(\lambda^2 + n^2)^2 + k (\lambda^2 + n^2)^2`.
    """
    lbd = m*np.pi*R/L
    k = h**2/(12*R**2)
    A = np.array([[lbd**2 + (1 - nu)/2*n**2, (1 + nu)/2*lbd*n, nu*lbd],
                  [(1 + nu)/2*lbd*n, (1 - nu)/2*lbd**2 + n**2, n],
                  [nu*lbd, n, 1 + k*(lbd**2 + n**2)**2]])
    if inplane_inertia:
        delta = np.linalg.eigvalsh(A)[0]
    else:
        delta = A[2, 2] - A[2, :2] @ np.linalg.solve(A[:2, :2], A[:2, 2])
    return np.sqrt(delta*E/(rho*(1 - nu**2)*R**2))/(2*np.pi)
