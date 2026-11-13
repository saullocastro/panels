r"""Torsional buckling of the stiffened cylinders of Moore and Wescoat (1944)

Closed 24S-T cylinders of mean radius `R = 15.04` in under the constant
shear flow `N_{xy}` of a torque, hinged ends (`u = v = w = 0`, `w_{,x}`
free), assembled from :class:`.Shell` domains with the default
``conn_method`` of :class:`.MultiDomain`. Verified: specimen 16 of Moore
and Wescoat (1944), without rings, with `u` free at the end `x = L`, the
first buckling stress against Donnell's formula for hinged ends as given in
Moore and Wescoat (1944), p. 23; specimen 17, with two rings of 1/2 x 1/2 in
solid bars as webs of ``'annular_clpt_donnell'`` domains connected to the
inner surface of the skin with ``'BFxcte'`` and ``face='lower'``, on a
coarse mesh, against values of the code. The builder :func:`_cylinder` also
adds channel stringers outside the skin (base as a second layer of the skin
strip under it, webs and lips as flat plates connected with ``'BFycte'``).

Notes
-----
In the membrane state of a perfect cylinder in torsion the skin carries the
uniform shear flow `N_{xy} = T/(2 \pi R^2)` and the rings and stringers no
stress, so that the constant `N_{xy}` given to the skin domains (and to the
strips under the stringer bases, which carry the same shear flow) is the
exact pre-buckling state. The critical shear flow is the lowest load
multiplier of `(\boldsymbol{K}_r + \lambda \boldsymbol{K}_{G,r})
\boldsymbol{\phi}_r = \boldsymbol{0}`, the matrices reduced with the null
space of the connections, and `\tau_{cr} = \lambda/t`.

Donnell's formula [2]_ for an unstiffened cylinder of length `L` with
hinged ends, as given by Moore and Wescoat [1]_, p. 23, for `L^2 t/D^3 <
5.2`, is

.. math::

    \tau_0 = \frac{E t^2}{(1 - \nu^2) L^2} \left[2.8 + \sqrt{2.6 + 1.4
    \left(\sqrt{1 - \nu^2}\frac{L^2}{t D}\right)^{3/2}}\right]

which gives 1323 psi for specimen 16 (`L = 35.62` in, `t = 0.020` in, `D =
30.08` in, `E = 10.5 \times 10^6` psi, `\nu = 1/3`), Table I printing 1325
psi.

The model of specimen 16 converges to 1351.8 psi with `u` free at `x =
L`, 2.2 % above Donnell's value; the tolerance, 3 %, covers this
difference of the two solutions and the coarser mesh of the test.

References
----------
.. [1] Moore, R. L., and Wescoat, C., "Torsion Tests of Stiffened Circular
   Cylinders", NACA ARR 4E31 (Wartime Report W-89), 1944, Tables I and II,
   pp. 25-26, appendix p. 23, Figs. 1, 5, 11 and 30.
.. [2] Donnell, L. H., "Stability of Thin-Walled Tubes under Torsion", NACA
   Report 479, 1933.

"""
import numpy as np
import pytest
from structsolve import lb

from panels.shell import Shell
from panels.multidomain import MultiDomain

E, NU = 10.5e6, 1/3.
D = 30.08
R = D/2.
RING = 0.5
TZ = 0.032
#: Donnell's formula against the model, specimen 16, `u` free at `x = L`
DONNELL_RTOL = 0.03


def donnell_tau(L, t):
    r"""Donnell's shear-buckling stress of a cylinder with hinged ends, see
    the module ``Notes``"""
    s = np.sqrt(1 - NU**2)*L**2/(t*D)
    return E*t**2/((1 - NU**2)*L**2)*(2.8 + np.sqrt(2.6 + 1.4*s**1.5))


def _free(s):
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, edge + f, 1)
            setattr(s, edge + f + 'r', 1)
    return s


def _cylinder(t, bays, layout, m_per_in, m_min=6, m_ring=6, u_end=0,
              n_web=6, n_lip=4, lip=None):
    r"""Closed cylinder of :class:`.Shell` domains, rings between the bays

    ``layout`` lists the domains along the circumference, ``(width, kind,
    terms)``, ``kind`` being ``'bay'`` (skin) or ``'strip'`` (skin and base of
    a stringer, with a web on each edge and a lip on each web, the lip of
    width ``lip``). Returns the assembly and the skin thickness.

    """
    skins, conn, panels_ = [], [], []
    webs_prev = None
    nb = len(bays)
    x0 = 0.
    for i, a in enumerate(bays):
        m = int(min(30, max(m_min, round(a*m_per_in))))
        row, y0 = [], 0.
        webs = []
        for j, (b, kind, n) in enumerate(layout):
            if kind == 'strip':
                p = Shell(model='cylshell_clpt_sanders', a=a, b=b, r=R, m=m,
                          n=n, stack=[0, 0], plyts=[t, TZ],
                          laminaprop=(E, NU), offset=TZ/2, x0=x0, y0=y0)
            else:
                p = Shell(model='cylshell_clpt_sanders', a=a, b=b, r=R, m=m,
                          n=n, stack=[0], plyt=t, laminaprop=(E, NU), x0=x0,
                          y0=y0)
            _free(p)
            p.Nxy = 1.
            if i == 0:
                p.x1u, p.x1v, p.x1w = 0, 0, 0
            if i == nb - 1:
                p.x2u, p.x2v, p.x2w = u_end, 0, 0
            if row:
                conn.append(dict(p1=row[-1], p2=p, func='SSycte',
                                 ycte1=row[-1].b, ycte2=0.))
            if i > 0:
                q = skins[-1][j]
                conn.append(dict(p1=q, p2=p, func='SSxcte', xcte1=q.a,
                                 xcte2=0.))
            row.append(p)
            y0 += b
            if kind == 'strip':
                Hw = t/2 + 0.64 - TZ/2
                for ycte1, ycte2_lip in ((0., 0.), (p.b, lip)):
                    web = _free(Shell(model='plate_clpt_donnell', a=a, b=Hw,
                                      m=m, n=n_web, stack=[0], plyt=TZ,
                                      laminaprop=(E, NU), x0=x0))
                    conn.append(dict(p1=p, p2=web, func='BFycte',
                                     ycte1=ycte1, ycte2=web.b))
                    lp = _free(Shell(model='plate_clpt_donnell', a=a, b=lip,
                                     m=m, n=n_lip, stack=[0], plyt=TZ,
                                     laminaprop=(E, NU), x0=x0))
                    conn.append(dict(p1=web, p2=lp, func='BFycte', ycte1=0.,
                                     ycte2=ycte2_lip))
                    webs += [web, lp]
        conn.append(dict(p1=row[-1], p2=row[0], func='SSycte',
                         ycte1=row[-1].b, ycte2=0.))
        for k, edge in ((0, 'x1'), (nb - 1, 'x2')):
            if i == k:
                for s in webs:
                    setattr(s, edge + 'u', u_end if edge == 'x2' else 0)
                    setattr(s, edge + 'v', 0)
                    setattr(s, edge + 'w', 0)
        if webs_prev is not None:
            for q, s in zip(webs_prev, webs):
                conn.append(dict(p1=q, p2=s, func='SSxcte', xcte1=q.a,
                                 xcte2=0.))
        webs_prev = webs
        panels_ += row + webs
        if i < nb - 1:
            ring = []
            r_out = R - t/2
            for p in row:
                w = _free(Shell(model='annular_clpt_donnell', a=RING,
                                b=r_out*p.b/R, r=r_out, m=m_ring, n=p.n,
                                stack=[0], plyt=RING, laminaprop=(E, NU)))
                conn.append(dict(p1=p, p2=w, func='BFxcte', xcte1=p.a,
                                 xcte2=0., face='lower'))
                if ring:
                    conn.append(dict(p1=ring[-1], p2=w, func='SSycte',
                                     ycte1=ring[-1].b, ycte2=0.))
                ring.append(w)
            conn.append(dict(p1=ring[-1], p2=ring[0], func='SSycte',
                             ycte1=ring[-1].b, ycte2=0.))
            panels_ += ring
        skins.append(row)
        x0 += a
    return MultiDomain(panels_, conn)


def _tau(md, t):
    md.get_Z()
    kC_r = md.reduce(md.calc_kC(silent=True))
    kG_r = md.reduce(md.calc_kG(silent=True))
    ev, _ = lb(kC_r, kG_r, num_eigvalues=4, silent=True)
    return np.real(ev[0])/t


def test_specimen16_donnell():
    r"""Unstiffened specimen 16, `u` free at the end `x = L`, 8 sectors:
    first buckling stress against Donnell's formula, see the module
    ``Notes``"""
    t = 0.020
    md = _cylinder(t, [35.625], [(2*np.pi*R/8, 'bay', 12)]*8, m_per_in=0.3,
                   m_min=10, u_end=1)
    tau = _tau(md, t)
    tau0 = donnell_tau(35.625, t)
    assert abs(tau/tau0 - 1) < DONNELL_RTOL


@pytest.mark.slow
def test_specimen17_rings_coarse():
    r"""Specimen 17, two ring webs on the inner surface of the skin
    (``'annular_clpt_donnell'``, ``'BFxcte'`` with ``face='lower'``), 8
    sectors of 8 terms, 0.2 terms per inch along the axis: the rings raise
    the first buckling stress above that of the cylinder without rings and
    the same mesh; both are values of the code with this coarse mesh"""
    t = 0.020
    layout = [(2*np.pi*R/8, 'bay', 8)]*8
    tau0 = _tau(_cylinder(t, [35.625], layout, m_per_in=0.2), t)
    tau = _tau(_cylinder(t, [4.3125, 27., 4.3125], layout, m_per_in=0.2), t)
    assert np.isclose(tau0, 1552.19, rtol=1e-4)
    assert np.isclose(tau, 2151.90, rtol=1e-4)
    assert tau > 1.2*tau0
