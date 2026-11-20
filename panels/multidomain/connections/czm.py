r"""
Mixed-mode multilinear cohesive interface between two domains (``'CZM'``)

The connection ``'CZM'`` bonds the lower face of a domain ``p1`` (top) to the
upper face of a domain ``p2`` (bottom) over their common area with a
cohesive law, for every model of :mod:`panels.modelDB`: the classical
laminated plate theory and the first- and third-order shear deformation
theories, with the Donnell or the Sanders-Koiter kinematics. It replaces the
mode I connection ``'SB_TSL'`` for mixed-mode loading:

- the separations are the displacement jumps of the perfect bond ``'SB'``,
  :func:`.kCsdt._sb_components`, evaluated at the Gauss-Legendre points of
  the interface with :func:`.nullspace.pd_rows`, such that a pristine
  interface with an infinite stiffness is the bond ``'SB'``;
- the law is a superposition of mixed-mode bilinear laws,
  :class:`.BilinearLaw`, each one the law of Turon et al. (2018) [1]_ with a
  mode-dependent penalty stiffness, the Benzeggagh-Kenane criterion with
  its own exponent and its own damage variable, as proposed by Abdel-Monsef
  et al. (2023) [2]_ for multilinear laws;
- the normal stiffness of each law is kept undamaged for a negative normal
  separation (contact);
- the tangent stiffness is the consistent one, including the derivative of
  the damage variables with respect to the separations;
- each interface keeps its own history, the damage threshold of each law at
  each point, updated only with converged states, see
  :meth:`.CohesiveInterface.commit`.

Notes
-----
With the separations `\Delta_u`, `\Delta_v` (tangential) and `\Delta_w`
(normal) at a point, the shear norm `\Delta_{sh} = \sqrt{\Delta_u^2 +
\Delta_v^2}` and the positive part `\langle \Delta_w \rangle`, a bilinear
law with the penalty stiffnesses `k_I` (normal) and `k_{sh}` (shear) has,
from Turon et al. (2018) [1]_, Eqs. (12), (17), (18):

.. math::

    \Delta_{eq} = \frac{k_{sh} \Delta_{sh}^2 + k_I \langle \Delta_w \rangle^2}
        {\sqrt{k_{sh}^2 \Delta_{sh}^2 + k_I^2 \langle \Delta_w \rangle^2}},
    \qquad B = \frac{k_{sh} \Delta_{sh}^2}{k_{sh} \Delta_{sh}^2 + k_I
        \langle \Delta_w \rangle^2}, \qquad k_B = k_I (1 - B) + k_{sh} B

the energies at onset and at failure of the Benzeggagh-Kenane form,
Eqs. (23), (25),

.. math::

    G_o(B) = G_{oI} + (G_{oII} - G_{oI}) B^{\eta_{BK}}, \qquad
    G_c(B) = G_{Ic} + (G_{IIc} - G_{Ic}) B^{\eta_{BK}}

with `G_{oI} = \tau_{o,I}^2/(2 k_I)` and `G_{oII} = \tau_{o,II}^2/(2
k_{sh})`, the separations at onset and at failure `\Delta_o = \sqrt{2
G_o/k_B}`, `\Delta_f = 2 G_c/(k_B \Delta_o)`, Eqs. (24), (26), the loading
function `H = (\Delta_{eq} - \Delta_o)/(\Delta_f - \Delta_o)`, the threshold
`r = \max(r_{old}, \min(H, 1))` and the damage, Eq. (22),

.. math::

    d = \frac{r \Delta_f}{r \Delta_f + (1 - r) \Delta_o}

The tractions are `\tau_u = (1 - d) k_{sh} \Delta_u`, `\tau_v = (1 - d)
k_{sh} \Delta_v` and `\tau_w = (1 - d) k_I \langle \Delta_w \rangle - k_I
\langle -\Delta_w \rangle`, Eqs. (7), (8).

The law dissipates no energy while the damage does not grow, and the
damage does not decrease when the mode ratio changes, provided that
`\partial d/\partial B = 0`, which holds when `\Delta_f/\Delta_o = G_c/G_o`
does not depend on `B`, i.e. `G_{Ic}/G_{oI} = G_{IIc}/G_{oII}`, Eq. (1) of
[1]_:

.. math::

    \frac{k_{sh}}{k_I} = \frac{G_{Ic}}{G_{IIc}} \left( \frac{\tau_{o,II}}
        {\tau_{o,I}} \right)^2

which :class:`.BilinearLaw` uses to set `k_{sh}` when it is not given. The
dissipated energy is then bounded by the pure-mode toughnesses of the law
for any loading path.

The superposed laws use, by default, ``mode_ratio='per-law'``: each law
computes `\Delta_{eq}` and `B` with its own stiffnesses, and is a law of
[1]_ on its own. With ``mode_ratio='summed'`` they use the sums of the
stiffnesses of all the laws, as written by Abdel-Monsef et al. (2023) [2]_,
Eqs. (4), (5). Both are the same when all the laws have the same ratio
`k_{sh}/k_I`, see :func:`.law_with_ratio`, the recommended choice; with
different ratios the summed option does not dissipate the sum of the
Benzeggagh-Kenane toughnesses under a constant mode ratio.

A Benzeggagh-Kenane exponent below 1 makes the derivative of `B^{\eta_{BK}}`
unbounded at `B = 0`, and the factor `B^{\eta_{BK}}` nearly a switch: such a
law loses a large part of its toughness for round-off shear separations. A
:class:`UserWarning` is issued for `\eta_{BK} < 1`, and the derivative is
taken as zero at `B = 0`.

References
----------
.. [1] Turon, A., Gonzalez, E. V., Sarrado, C., Guillamet, G., Maimi, P.,
   Accurate simulation of delamination under mixed-mode loading using a
   cohesive model with a mode-dependent penalty stiffness, Composite
   Structures, 184, 506-511, 2018.
   https://doi.org/10.1016/j.compstruct.2017.10.017
.. [2] Abdel-Monsef, S., Tijs, B. H. A. H., Renart, J., Turon, A., Accurate
   simulation of delamination under mixed-mode loading using a multilinear
   cohesive law, Engineering Fracture Mechanics, 284, 109233, 2023.
   https://doi.org/10.1016/j.engfracmech.2023.109233

"""
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import csr_matrix, diags

from panels import DOUBLE
from panels.multidomain.connections import nullspace, kCsdt


class BilinearLaw(object):
    r"""Mixed-mode bilinear cohesive law with a mode-dependent stiffness

    Parameters
    ----------
    GIc, GIIc : float
        Fracture toughnesses in mode I and in shear (modes II and III),
        force/length. ``GIIc`` may be zero for a law that represents a
        mechanism active in mode I only, e.g. fibre bridging, together with
        ``tauII = 0`` and an explicit ``KII``.
    tauI, tauII : float
        Strengths, the tractions at damage onset in pure mode, stress.
    KI : float
        Penalty stiffness in mode I, force/length**3.
    eta : float
        Exponent of the Benzeggagh-Kenane criterion.
    KII : float or None, optional
        Penalty stiffness in shear. ``None`` for the value of Eq. (1) of
        Turon et al. (2018), see the module notes.
    name : str, optional
        Label.

    Raises
    ------
    ValueError
        When a toughness is not larger than the elastic energy at onset in
        the same mode, which would require a snap-back of the law, or when
        ``KII`` cannot be found from Eq. (1).

    """
    def __init__(self, GIc, GIIc, tauI, tauII, KI, eta, KII=None, name=''):
        self.GIc = float(GIc)
        self.GIIc = float(GIIc)
        self.tauI = float(tauI)
        self.tauII = float(tauII)
        self.KI = float(KI)
        self.eta = float(eta)
        self.name = name
        if KII is None:
            if self.GIIc == 0. or self.tauII == 0.:
                raise ValueError('A law without shear toughness or strength '
                                 'needs KII')
            KII = self.KI*(self.GIc/self.GIIc)*(self.tauII/self.tauI)**2
        self.KII = float(KII)
        self.GoI = self.tauI**2/(2*self.KI)
        self.GoII = self.tauII**2/(2*self.KII) if self.KII > 0 else 0.
        if not self.GIc > self.GoI:
            raise ValueError('G_Ic = {0} must exceed the elastic energy at '
                             'onset tau_I**2/(2 K_I) = {1}'.format(self.GIc,
                                                                   self.GoI))
        if self.GIIc > 0 and not self.GIIc > self.GoII:
            raise ValueError('G_IIc = {0} must exceed tau_II**2/(2 K_II) = '
                             '{1}'.format(self.GIIc, self.GoII))
        if self.eta < 1.:
            warnings.warn('Benzeggagh-Kenane exponent {0} < 1: the law loses '
                          'a large part of its toughness for small shear '
                          'separations, see the notes of '
                          'panels.multidomain.connections.czm'.format(
                              self.eta), UserWarning, stacklevel=2)

    def __repr__(self):
        return ('BilinearLaw(GIc={0}, GIIc={1}, tauI={2}, tauII={3}, KI={4}, '
                'eta={5}, KII={6})'.format(self.GIc, self.GIIc, self.tauI,
                                           self.tauII, self.KI, self.eta,
                                           self.KII))

    def consistency_error(self):
        r"""Relative violation of `G_{Ic}/G_{oI} = G_{IIc}/G_{oII}`, Eq. (1)
        of Turon et al. (2018); zero for a law without shear toughness and
        strength"""
        rI = self.GIc/self.GoI
        if self.GoII == 0.:
            return 0. if self.GIIc == 0. else np.inf
        return abs(rI - self.GIIc/self.GoII)/rI

    def Gc(self, B):
        return self.GIc + (self.GIIc - self.GIc)*B**self.eta

    def Go(self, B):
        return self.GoI + (self.GoII - self.GoI)*B**self.eta


def law_with_ratio(GIc, GIIc, tauI, KI, eta, kappa, name=''):
    r"""Bilinear law with the ratio of stiffnesses ``kappa`` and the shear
    strength of Eq. (1) of Turon et al. (2018)

    Returns the :class:`.BilinearLaw` with `k_{sh} = \kappa k_I` and
    `\tau_{o,II} = \tau_{o,I} \sqrt{\kappa G_{IIc}/G_{Ic}}`, zero for
    ``GIIc = 0``. Superposed laws built with the same ``kappa`` share the
    equivalent separation and the mode ratio, see the module notes.

    """
    tauII = tauI*np.sqrt(kappa*GIIc/GIc)
    return BilinearLaw(GIc, GIIc, tauI, tauII, KI, eta, KII=kappa*KI,
                       name=name)


def _dbeta(B, eta):
    r"""Derivative of `B^\eta`, zero at `B = 0` for `\eta \ne 1`"""
    out = np.zeros_like(B)
    pos = B > 0
    out[pos] = eta*B[pos]**(eta - 1.)
    if eta == 1.:
        out[~pos] = 1.
    return out


def evaluate_laws(laws, du, dv, dw, r_old, mode_ratio='per-law',
                  tangent=True):
    r"""Tractions, tangent, thresholds and damage of superposed laws at
    many points

    Parameters
    ----------
    laws : list of :class:`.BilinearLaw`
        The superposed laws.
    du, dv, dw : np.ndarray
        Separations at the points, 1-D arrays of the same size.
    r_old : np.ndarray
        Thresholds of the last converged state, shape ``(len(laws),
        npts)``.
    mode_ratio : str, optional
        ``'per-law'`` (default) or ``'summed'``, see the module notes.
    tangent : bool, optional
        Whether the tangent is computed.

    Returns
    -------
    tau : np.ndarray
        Tractions, shape ``(3, npts)``, along `u`, `v`, `w`.
    T : np.ndarray or None
        Consistent tangent `\partial \tau_i/\partial \Delta_j`, shape ``(3,
        3, npts)``, not symmetric where the damage grows.
    r_new, d : np.ndarray
        Thresholds and damage variables of the state, shape ``(len(laws),
        npts)``.

    """
    du = np.asarray(du, dtype=DOUBLE)
    dv = np.asarray(dv, dtype=DOUBLE)
    dw = np.asarray(dw, dtype=DOUBLE)
    npts = du.size
    S = du*du + dv*dv
    wp = np.maximum(dw, 0.)
    N = wp*wp
    opening = dw > 0
    tau = np.zeros((3, npts), dtype=DOUBLE)
    T = np.zeros((3, 3, npts), dtype=DOUBLE) if tangent else None
    r_new = np.empty((len(laws), npts), dtype=DOUBLE)
    d = np.empty((len(laws), npts), dtype=DOUBLE)
    sep = np.vstack((du, dv, wp))
    KIsum = sum(l.KI for l in laws)
    KIIsum = sum(l.KII for l in laws)
    for i, law in enumerate(laws):
        if mode_ratio == 'per-law':
            a, b = law.KII, law.KI
        elif mode_ratio == 'summed':
            a, b = KIIsum, KIsum
        else:
            raise ValueError("mode_ratio must be 'per-law' or 'summed'")
        P = a*S + b*N
        Q = a*a*S + b*b*N
        sep_nz = P > 0
        lam = np.zeros(npts)
        B = np.zeros(npts)
        sQ = np.sqrt(Q[sep_nz])
        lam[sep_nz] = P[sep_nz]/sQ
        B[sep_nz] = a*S[sep_nz]/P[sep_nz]
        Bn = B**law.eta
        Go = law.GoI + (law.GoII - law.GoI)*Bn
        Gc = law.GIc + (law.GIIc - law.GIc)*Bn
        KB = law.KI*(1. - B) + law.KII*B
        no_capacity = Go <= 0.
        rho = np.full(npts, np.inf)
        cap = ~no_capacity
        rho[cap] = Gc[cap]/Go[cap]
        lo = np.zeros(npts)
        lo[cap] = np.sqrt(2*Go[cap]/KB[cap])
        H = np.zeros(npts)
        act = cap & sep_nz
        H[act] = (lam[act]/lo[act] - 1.)/(rho[act] - 1.)
        H[no_capacity] = np.inf
        Hc = np.clip(H, 0., 1.)
        ri = np.maximum(r_old[i], Hc)
        r_new[i] = ri
        di = np.ones(npts)
        part = ri < 1.
        rr, rh = ri[part], rho[part]
        di[part] = rr*rh/(rr*rh + 1. - rr)
        d[i] = di
        Keff = np.vstack((np.full(npts, law.KII), np.full(npts, law.KII),
                          np.where(opening, law.KI, 0.)))
        tau += (1. - di)*Keff*sep
        # contact: undamaged normal stiffness for dw < 0
        tau[2] += np.where(dw < 0, law.KI*dw, 0.)
        if not tangent:
            continue
        for j in range(3):
            T[j, j] += (1. - di)*Keff[j]
        T[2, 2] += np.where(dw < 0, law.KI, 0.)
        loading = act & (H > r_old[i]) & (H > 0.) & (H < 1.)
        if not np.any(loading):
            continue
        L = loading
        # derivatives of lambda and B with respect to S and N
        sQl = np.sqrt(Q[L])
        PL = P[L]
        dlam_dS = a/sQl - PL*a*a/(2*Q[L]*sQl)
        dlam_dN = b/sQl - PL*b*b/(2*Q[L]*sQl)
        dB_dS = a*b*N[L]/PL**2
        dB_dN = -a*S[L]*b/PL**2
        dBn = _dbeta(B[L], law.eta)
        dGo = (law.GoII - law.GoI)*dBn
        dGc = (law.GIIc - law.GIc)*dBn
        dKB = law.KII - law.KI
        GoL, GcL, KBL, loL, rhoL = Go[L], Gc[L], KB[L], lo[L], rho[L]
        dlo_dB = (dGo*KBL - GoL*dKB)/(KBL*KBL)/loL
        drho_dB = (dGc*GoL - GcL*dGo)/(GoL*GoL)
        lamL = lam[L]
        dH_dlam = 1./(loL*(rhoL - 1.))
        dH_dB = (-lamL/loL**2*dlo_dB)/(rhoL - 1.) \
            - (lamL/loL - 1.)/(rhoL - 1.)**2*drho_dB
        dH_dS = dH_dlam*dlam_dS + dH_dB*dB_dS
        dH_dN = dH_dlam*dlam_dN + dH_dB*dB_dN
        # dS/d(du, dv, dw) = (2du, 2dv, 0), dN/d(du, dv, dw) = (0, 0, 2<dw>)
        dH = np.vstack((2*du[L]*dH_dS, 2*dv[L]*dH_dS, 2*wp[L]*dH_dN))
        rL = ri[L]
        dd_dr = rhoL/(rL*rhoL + 1. - rL)**2
        t_undamaged = Keff[:, L]*sep[:, L]
        for j in range(3):
            for k in range(3):
                T[j, k, L] -= t_undamaged[j]*dd_dr*dH[k]
    return tau, T, r_new, d


def contact_traction(dw, k, eps):
    r"""Normal traction and its derivative of a contact-only interface

    `\tau_w = -k\, p(-\Delta_w)` with the smoothed ramp `p(x) = 0` for `x
    \le 0`, `x^2/(2\epsilon)` for `0 < x < \epsilon` and `x - \epsilon/2`
    beyond: no traction for an opening, the penalty `k` for a penetration
    larger than `\epsilon`, and a continuous tangent, which keeps the
    quadratic convergence of the Newton-Raphson iterations when points of
    the interface close or open. A sharp switch makes the iterations
    oscillate between the states of the points near `\Delta_w = 0`. The
    traction differs from `k \Delta_w` by at most `k \epsilon/2`.

    """
    x = np.maximum(-np.asarray(dw, dtype=DOUBLE), 0.)
    small = x < eps
    p = np.where(small, x*x/(2*eps), x - 0.5*eps)
    dp = np.where(small, x/eps, 1.)
    return -k*p, k*dp


class CohesiveInterface(object):
    r"""Cohesive interface of a ``'CZM'`` connection

    Parameters
    ----------
    p_top, p_bot : :class:`.Shell`
        Domains on the positive and negative sides of the interface along
        their common `z` axis, ``p1`` and ``p2`` of the connection, with the
        same ``x0``, ``y0``, ``a`` and ``b`` and with ``col_start`` defined.
    laws : list of :class:`.BilinearLaw`
        Superposed laws. An empty list gives a contact-only interface, with
        the normal stiffness ``k_contact``.
    nr_x_gauss, nr_y_gauss : int
        Gauss-Legendre points along `x` and `y`.
    size : int
        Size of the assembly.
    mode_ratio : str, optional
        ``'per-law'`` (default) or ``'summed'``, see the module notes.
    k_contact : float or None, optional
        Normal stiffness of a contact-only interface (no laws).
    contact_eps : float, optional
        Penetration over which the contact stiffness of a contact-only
        interface is phased in, see :func:`.contact_traction`, length.

    Attributes
    ----------
    r : np.ndarray
        Damage thresholds of the last converged state, shape ``(len(laws),
        nr_y_gauss*nr_x_gauss)``.
    d : np.ndarray
        Damage variables of the last converged state, same shape.
    dissipated : float
        Energy dissipated up to the last converged state, force*length,
        `\sum_g w_g \sum_\ell \int G_{c,\ell}(B) \, \mathrm{d} r_\ell`, with
        `G_c` at the mode ratio of the end of each increment, see the
        Notes.
    Bu, Bv, Bw : scipy.sparse.csr_matrix
        Separation operators, ``Bu @ c`` the separation `\Delta_u` at the
        points, shape ``(nr_y_gauss*nr_x_gauss, size)``.
    weights : np.ndarray
        Integration weights including the Jacobian `ab/4`.
    x, y : np.ndarray
        Coordinates of the points in the domain, relative to its corner.

    Notes
    -----
    The separations are those of the perfect bond ``'SB'``,

    .. math::

        \Delta_u = u^t(z^t) - u^b(z^b), \qquad \Delta_v = v^t(z^t) -
        v^b(z^b), \qquad \Delta_w = w^t - w^b

    with `z^t = d_1 - h_1/2` the lower face of ``p_top`` and `z^b = d_2 +
    h_2/2` the upper face of ``p_bot``, `d` being the offsets, and `u(z)`,
    `v(z)` the displacements of the model of each domain at these faces,
    :func:`.kCsdt._surface_terms`: `u - z w_{,x}` for the classical
    laminated plate theory, `u + z \phi_x` for the first-order shear
    deformation theory, plus the cubic terms of the third-order theory and
    the term `z v/r` of the Sanders-Koiter kinematics. The rows at the
    Gauss-Legendre points are those of :func:`.nullspace.pd_rows`.

    The internal force vector and the consistent tangent are

    .. math::

        \boldsymbol{F}^{coh} = \sum_i \boldsymbol{B}_i^\top \boldsymbol{W}
        \boldsymbol{\tau}_i, \qquad
        \boldsymbol{K}_T^{coh} = \sum_{i,j} \boldsymbol{B}_i^\top \boldsymbol{W}
        \mathrm{diag}\left(\frac{\partial \tau_i}{\partial \Delta_j}\right)
        \boldsymbol{B}_j

    with `i, j = u, v, w` and `\boldsymbol{W}` the diagonal of the weights.

    The energy dissipated by a law at a point is `\int Y \, \mathrm{d}d`,
    with the thermodynamic force `Y = \frac{1}{2} (k_{sh} \Delta_{sh}^2 +
    k_I \langle \Delta_w \rangle^2) = \frac{1}{2} k_B \Delta_{eq}^2`. While
    the damage grows, `\Delta_{eq} = \Delta_o (1 + r(\rho - 1))` with `\rho
    = \Delta_f/\Delta_o = G_c/G_o`, and `\mathrm{d}d/\mathrm{d}r = \rho/(1 +
    r(\rho - 1))^2`, such that `Y \, \mathrm{d}d = G_o \rho \, \mathrm{d}r =
    G_c \, \mathrm{d}r`: the threshold `r` is the fraction of `G_c`
    dissipated, as noted by Turon et al. (2018) after their Eq. (20). With
    `\partial d/\partial B = 0` this holds for any path, the mode ratio
    entering through `G_c(B)`, taken at the end of each increment.

    """
    def __init__(self, p_top, p_bot, laws, nr_x_gauss, nr_y_gauss, size,
                 mode_ratio='per-law', k_contact=None, contact_eps=1e-3):
        for attr in ('x0', 'y0', 'a', 'b'):
            if not np.isclose(getattr(p_top, attr), getattr(p_bot, attr)):
                raise ValueError("The domains of a 'CZM' connection must "
                                 "have the same {0}".format(attr))
        if p_top.col_start is None or p_bot.col_start is None:
            raise ValueError('col_start of the domains must be defined')
        self.p_top = p_top
        self.p_bot = p_bot
        self.laws = list(laws)
        if not self.laws and not k_contact:
            raise ValueError('A contact-only interface needs k_contact')
        self.k_contact = k_contact
        self.contact_eps = contact_eps
        self.mode_ratio = mode_ratio
        self.nr_x_gauss = int(nr_x_gauss)
        self.nr_y_gauss = int(nr_y_gauss)
        self.size = int(size)
        xis, wxi = leggauss(self.nr_x_gauss)
        etas, weta = leggauss(self.nr_y_gauss)
        z_top = kCsdt._face(p_top, -1)
        z_bot = kCsdt._face(p_bot, +1)
        B = []
        for comp in (nullspace.U, nullspace.V, nullspace.W):
            Bt = nullspace.pd_rows(p_top, comp, xis, etas, z_top,
                                   p_top.col_start, size)
            Bb = nullspace.pd_rows(p_bot, comp, xis, etas, z_bot,
                                   p_bot.col_start, size)
            B.append((Bt - Bb).tocsr())
        self.Bu, self.Bv, self.Bw = B
        self.weights = np.outer(weta, wxi).ravel()*p_top.a*p_top.b/4
        xs, ys = np.meshgrid((xis + 1)*p_top.a/2, (etas + 1)*p_top.b/2)
        self.x = xs.ravel()
        self.y = ys.ravel()
        self.npts = self.weights.size
        for law in self.laws:
            err = law.consistency_error()
            if err > 1e-6:
                warnings.warn('Law {0!r} violates Eq. (1) of Turon et al. '
                              '(2018) by {1:.1e}: its damage may decrease '
                              'when the mode ratio changes'.format(law, err),
                              UserWarning, stacklevel=2)
        ratios = [l.KII/l.KI for l in self.laws]
        if (mode_ratio == 'summed' and ratios
                and max(ratios) - min(ratios) > 1e-9*max(ratios)):
            warnings.warn("mode_ratio='summed' with different ratios K_II/K_I "
                          "of the laws: the energy dissipated under a constant "
                          "mode ratio is not the sum of the B-K toughnesses",
                          UserWarning, stacklevel=2)
        self.reset()

    def reset(self):
        r"""Pristine interface: thresholds and damage zero, nothing
        dissipated"""
        self.r = np.zeros((len(self.laws), self.npts), dtype=DOUBLE)
        self.d = np.zeros_like(self.r)
        self.dissipated = 0.

    def separations(self, c):
        r"""Separations `\Delta_u, \Delta_v, \Delta_w` at the points"""
        return self.Bu @ c, self.Bv @ c, self.Bw @ c

    def state(self, c, tangent=True):
        r"""Tractions, tangent, thresholds and damage at ``c`` from the last
        converged state, see :func:`.evaluate_laws`; for a contact-only
        interface, the normal contact traction of :func:`.contact_traction`"""
        du, dv, dw = self.separations(c)
        if not self.laws:
            tau = np.zeros((3, self.npts))
            tw, kw = contact_traction(dw, self.k_contact, self.contact_eps)
            tau[2] = tw
            T = None
            if tangent:
                T = np.zeros((3, 3, self.npts))
                T[2, 2] = kw
            empty = np.zeros((0, self.npts))
            return tau, T, empty, empty
        return evaluate_laws(self.laws, du, dv, dw, self.r,
                             self.mode_ratio, tangent=tangent)

    def fint(self, c):
        r"""Internal force vector of the interface at ``c``"""
        tau, _, _, _ = self.state(c, tangent=False)
        w = self.weights
        return (self.Bu.T @ (w*tau[0]) + self.Bv.T @ (w*tau[1])
                + self.Bw.T @ (w*tau[2]))

    def kT(self, c):
        r"""Consistent tangent stiffness matrix of the interface at ``c``,
        not symmetric where the damage grows"""
        _, T, _, _ = self.state(c, tangent=True)
        return self._assemble(T)

    def kC0(self):
        r"""Secant stiffness of the last converged state at zero
        separation, with the normal stiffness of an opening, used by the
        linear analyses"""
        T = np.zeros((3, 3, self.npts))
        for i, law in enumerate(self.laws):
            s = 1. - self.d[i]
            T[0, 0] += s*law.KII
            T[1, 1] += s*law.KII
            T[2, 2] += s*law.KI
        if not self.laws:
            T[2, 2] = self.k_contact
        return self._assemble(T)

    def _assemble(self, T):
        Bs = (self.Bu, self.Bv, self.Bw)
        K = csr_matrix((self.size, self.size))
        for i in range(3):
            for j in range(3):
                tij = T[i, j]
                if not np.any(tij):
                    continue
                K = K + Bs[i].T @ diags(self.weights*tij) @ Bs[j]
        return K.tocsr()

    def _Gc_laws(self, c):
        r"""`G_c(B)` of each law at the points, with the mode ratio of
        ``mode_ratio``"""
        du, dv, dw = self.separations(c)
        S = du*du + dv*dv
        N = np.maximum(dw, 0.)**2
        KIs = sum(l.KI for l in self.laws)
        KIIs = sum(l.KII for l in self.laws)
        out = np.empty((len(self.laws), self.npts))
        for i, law in enumerate(self.laws):
            a, b = ((law.KII, law.KI) if self.mode_ratio == 'per-law'
                    else (KIIs, KIs))
            P = a*S + b*N
            B = np.where(P > 0, a*S/np.where(P > 0, P, 1.), 0.)
            out[i] = law.Gc(B)
        return out

    def commit(self, c):
        r"""Updates the history with the converged state ``c``

        The thresholds and the damage variables become those of ``c``, and
        the energy dissipated since the last converged state is added to
        ``dissipated``.

        """
        if not self.laws:
            return
        _, _, r_new, d_new = self.state(c, tangent=False)
        Gc = self._Gc_laws(c)
        dr = r_new - self.r
        self.dissipated += float(np.sum(self.weights*np.sum(Gc*dr, axis=0)))
        self.r = r_new
        self.d = d_new

    def energy(self, c):
        r"""Free energy stored in the interface at ``c`` with the damage of
        the last converged state, Eqs. (3), (4) of Turon et al. (2018)"""
        du, dv, dw = self.separations(c)
        S = du*du + dv*dv
        N = np.maximum(dw, 0.)**2
        Nc = np.minimum(dw, 0.)**2
        psi = np.zeros(self.npts)
        for i, law in enumerate(self.laws):
            psi += 0.5*(1. - self.d[i])*(law.KII*S + law.KI*N)
            psi += 0.5*law.KI*Nc
        if not self.laws:
            x = np.maximum(-dw, 0.)
            e = self.contact_eps
            psi += self.k_contact*np.where(x < e, x**3/(6*e),
                                           0.5*x*x - 0.5*e*x + e*e/6)
        return float(np.sum(self.weights*psi))

    def resultants(self, c):
        r"""Resultants of the tractions on ``p_top`` at ``c``, along `x`,
        `y`, `z`, with the sign of the force applied by ``p_bot``"""
        tau, _, _, _ = self.state(c, tangent=False)
        return -(tau @ self.weights)
