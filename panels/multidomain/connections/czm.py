r"""
Mixed-mode multilinear cohesive interface between two domains (``'CZM'``)

The connection ``'CZM'`` bonds the lower face of a domain ``p1`` (top) to the
upper face of a domain ``p2`` (bottom) over their common area with a
cohesive law, for every model of :mod:`panels.modelDB`: the classical
laminated plate theory and the first- and third-order shear deformation
theories, with the Donnell or the Sanders-Koiter kinematics. It is the
connection for mixed-mode loading, the connection ``'SB_TSL'`` being for
mode I only:

- the separations are the displacement jumps of the perfect bond ``'SB'``,
  :func:`.kCsdt._sb_components`, evaluated at the Gauss-Legendre points of
  the interface with :func:`.nullspace.pd_rows`, such that a pristine
  interface with an infinite stiffness is the bond ``'SB'``;
- the law is a superposition of mixed-mode bilinear laws,
  :class:`.BilinearLaw`, each one with a mode-dependent penalty stiffness,
  the Benzeggagh-Kenane criterion with its own exponent and its own damage
  variable, which together make a multilinear law;
- the normal stiffness of each law is kept undamaged for a negative normal
  separation (contact);
- the tangent stiffness is the consistent one, including the derivative of
  the damage variables with respect to the separations;
- each interface keeps its own history, the damage threshold of each law at
  each point, updated only with converged states, see
  :meth:`.CohesiveInterface.commit`.

Notes
-----
Each bilinear law is the law of Turon et al. (2018) [1]_, and the laws are
superposed as proposed by Abdel-Monsef et al. (2023) [2]_ for multilinear
laws.

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

    One law of a connection ``'CZM'``: the tractions grow linearly with the
    separations up to the damage onset and then decrease linearly to zero
    at failure, with the onset and the toughness interpolated between mode
    I and shear by the Benzeggagh-Kenane criterion, and one damage variable.
    The object holds the properties of the law only: the laws of an
    interface are given as a list to :class:`.CohesiveInterface`, which
    keeps their history and evaluates them with :func:`.evaluate_laws`. The
    flags ``open_only``, ``slip_offset``, ``opening_driven`` and
    ``wake_offset`` change the separations that drive the damage, not the
    undamaged response. A :class:`UserWarning` is issued for ``eta < 1``,
    and for a law with a shear stiffness and without shear strength that
    is neither ``open_only`` nor ``opening_driven``.

    Parameters
    ----------
    GIc, GIIc : float
        Fracture toughnesses in mode I and in shear (modes II and III),
        force/length. ``GIIc`` may be zero for a law that represents a
        mechanism active in mode I only, e.g. fibre bridging, together with
        ``tauII = 0`` and an explicit ``KII``. Such a law fails at once
        wherever the interface is closed and slides, its onset separation
        being zero there; round-off sliding suffices, e.g. ahead of the
        crack tip of a symmetric DCB. A small positive ``GIIc`` with the
        ``tauII`` of :func:`.law_with_ratio` avoids this (a warning is
        issued).
    tauI, tauII : float
        Strengths, the tractions at damage onset in pure mode, stress.
    KI : float
        Penalty stiffness in mode I, force/length**3.
    eta : float
        Exponent of the Benzeggagh-Kenane criterion, dimensionless.
    KII : float or None, optional
        Penalty stiffness in shear, force/length**3. ``None``, the default,
        for ``KI*(GIc/GIIc)*(tauII/tauI)**2``, with which the damage does
        not decrease when the mode ratio changes.
    name : str, optional
        Label. Default ``''``.
    open_only : bool, optional
        If ``True``, the damage of the law grows only where the interface
        is open, `\Delta_w > 0`: a mechanism that needs an open crack, e.g.
        fibre bridging, is not damaged by the sliding of a closed interface
        ahead of the crack tip. The thresholds are frozen there, so that no
        energy is dissipated and the damage does not decrease. Default
        ``False``.
    slip_offset : bool, optional
        Only with ``open_only=True``. If ``True``, the damage of the law is
        driven by the opening and by the slip gained since the point was
        last closed, the slip less the offset of
        :attr:`.CohesiveInterface.s0`: a point that slid while closed and
        reopens starts in mode I, instead of nearly in pure shear, where a
        law without shear strength has no capacity. The tractions keep the
        total slip, so that the undamaged response is unchanged. Default
        ``False``.
    opening_driven : bool, optional
        If ``True``, the damage of the law is driven by the opening alone,
        with the onset and the toughness of mode I, e.g. fibre bridging,
        which needs an open crack and has no shear toughness: the sliding
        of a closed or open interface neither damages the law nor changes
        its mode ratio, and the damage does not grow while `\Delta_w \le
        0`. Both traction components keep the factor `1 - d` on the
        stiffnesses of the law, so that the undamaged response is
        unchanged. With ``KII = 0`` the law carries a normal traction only.
        Default ``False``.
    wake_offset : bool, optional
        Only with ``open_only=True``. As ``slip_offset``, but the offset,
        :attr:`.CohesiveInterface.s0w`, follows the slip of the point, open
        or closed, until all the laws without ``wake_offset`` (the
        crack-tip laws) have failed there, `d = 1`, and is fixed
        afterwards: the damage of the law is driven by the opening and by
        the slip gained since the wake of the crack reached the point, for
        a mechanism that exists in the wake only, e.g. fibre bridging. The
        tractions keep the total slip. Default ``False``.

    Attributes
    ----------
    GIc, GIIc, tauI, tauII, KI, eta, name
        As given.
    KII : float
        Penalty stiffness in shear, given or computed, force/length**3.
    GoI, GoII : float
        Elastic energies at damage onset in mode I and in shear,
        ``tauI**2/(2*KI)`` and ``tauII**2/(2*KII)``, the second one zero
        for ``KII = 0``, force/length.
    open_only, slip_offset, opening_driven, wake_offset : bool
        As given.

    Raises
    ------
    ValueError
        When a toughness is not larger than the elastic energy at onset in
        the same mode, which would require a snap-back of the law; when
        ``KII`` is ``None`` and ``GIIc`` or ``tauII`` is zero; when
        ``slip_offset`` or ``wake_offset`` is given without ``open_only``;
        or when more than one of ``slip_offset``, ``opening_driven`` and
        ``wake_offset`` is given.

    Notes
    -----
    The law is the one of Turon et al. (2018) [1]_, with `k_I` = ``KI``,
    `k_{sh}` = ``KII``, `\tau_{o,I}` = ``tauI``, `\tau_{o,II}` = ``tauII``
    and `\eta_{BK}` = ``eta``. The energies at onset in pure mode are
    `G_{oI} = \tau_{o,I}^2/(2 k_I)` and `G_{oII} = \tau_{o,II}^2/(2
    k_{sh})`. At the mode ratio `B`, the separations at onset and at failure
    are `\Delta_o = \sqrt{2 G_o(B)/k_B}` and `\Delta_f = 2 G_c(B)/(k_B
    \Delta_o)`, such that `\Delta_f/\Delta_o = G_c/G_o`: the law softens
    without a snap-back only for `G_{Ic} > G_{oI}` and `G_{IIc} > G_{oII}`,
    which is checked.

    With ``KII=None`` the shear stiffness is that of Eq. (1) of [1]_,

    .. math::

        k_{sh} = k_I \frac{G_{Ic}}{G_{IIc}} \left( \frac{\tau_{o,II}}
            {\tau_{o,I}} \right)^2

    which gives `G_{Ic}/G_{oI} = G_{IIc}/G_{oII}`: `\Delta_f/\Delta_o` then
    does not depend on `B`, and the damage does not decrease when the mode
    ratio changes, see :meth:`consistency_error`.

    A law without shear strength and with `k_{sh} > 0` has `G_o = 0` at `B
    = 1`, the mode ratio of a closed interface that slides: its onset
    separation is zero there, and it fails without dissipating energy.
    ``open_only`` freezes the threshold of the law where `\Delta_w \le 0`.

    With ``slip_offset`` and ``wake_offset``, the shear norm of the
    equivalent separation `\Delta_{eq}` and of `B` is `\Delta_{sh} =
    |\boldsymbol{\Delta}_s - \boldsymbol{\Delta}_{s0}|`, with
    `\boldsymbol{\Delta}_s = (\Delta_u, \Delta_v)` and the offset
    `\boldsymbol{\Delta}_{s0}` of the interface: a point that slid while
    closed and reopens starts at `B = 0`, instead of `B \approx 1` and `G_o
    \approx 0` for a law without shear strength.

    With ``opening_driven``, `\Delta_{eq} = \langle \Delta_w \rangle` and
    `B = 0`. The energy released by the damage, `\int Y \mathrm{d}d` with
    `Y = (k_{sh} \Delta_{sh}^2 + k_I \langle \Delta_w \rangle^2)/2`, then
    exceeds the `G_{Ic}` of the bookkeeping of
    :meth:`.CohesiveInterface.commit` by the shear part, `\int k_{sh}
    \Delta_{sh}^2/2 \, \mathrm{d}d`, unless the interface does not slide or
    `k_{sh} = 0`: with `k_{sh} = 0` the law dissipates exactly `G_{Ic}` for
    any path, which a law with `k_{sh} = 0` and the per-law mode ratio does
    without this flag.

    References
    ----------
    .. [1] Turon, A., Gonzalez, E. V., Sarrado, C., Guillamet, G., Maimi,
       P., Accurate simulation of delamination under mixed-mode loading
       using a cohesive model with a mode-dependent penalty stiffness,
       Composite Structures, 184, 506-511, 2018.
       https://doi.org/10.1016/j.compstruct.2017.10.017

    """
    def __init__(self, GIc, GIIc, tauI, tauII, KI, eta, KII=None, name='',
                 open_only=False, slip_offset=False, opening_driven=False,
                 wake_offset=False):
        self.open_only = bool(open_only)
        self.slip_offset = bool(slip_offset)
        self.opening_driven = bool(opening_driven)
        if self.slip_offset and not self.open_only:
            raise ValueError('slip_offset requires open_only=True')
        self.wake_offset = bool(wake_offset)
        if self.wake_offset and not self.open_only:
            raise ValueError('wake_offset requires open_only=True')
        if (self.slip_offset + self.opening_driven + self.wake_offset) > 1:
            raise ValueError('slip_offset, opening_driven and wake_offset '
                             'exclude each other')
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
        if (self.GoII == 0. and self.KII > 0. and not self.open_only
                and not self.opening_driven):
            warnings.warn('Law without shear strength: its onset separation '
                          'is zero at B = 1, so any sliding of a closed '
                          'interface, round-off included, fails it without '
                          'dissipating energy, e.g. ahead of a crack tip. '
                          'Prefer a small positive GIIc and tauII from '
                          'Eq. (1), see law_with_ratio', UserWarning,
                          stacklevel=2)

    def __repr__(self):
        return ('BilinearLaw(GIc={0}, GIIc={1}, tauI={2}, tauII={3}, KI={4}, '
                'eta={5}, KII={6}{7}'.format(
                    self.GIc, self.GIIc, self.tauI, self.tauII, self.KI,
                    self.eta, self.KII,
                    ', open_only=True' if self.open_only else '')
                + (', slip_offset=True' if self.slip_offset else '')
                + (', opening_driven=True' if self.opening_driven else '')
                + (', wake_offset=True' if self.wake_offset else '')
                + ')')

    def consistency_error(self):
        r"""Relative error of the law in the condition on its stiffnesses

        Measures how far the shear stiffness of the law is from the one
        that ``KII=None`` gives, with which the damage does not decrease
        when the mode ratio changes. :class:`.CohesiveInterface` warns
        about the laws with an error above ``1e-6``.

        Returns
        -------
        err : float
            Relative error, dimensionless: zero, to the round-off, for a
            law built with ``KII=None`` or by :func:`.law_with_ratio`, and
            zero for a law without shear toughness whose elastic energy at
            onset in shear ``GoII`` is zero; ``np.inf`` for a law with a
            shear toughness and ``GoII = 0``.

        Notes
        -----
        The condition is `G_{Ic}/G_{oI} = G_{IIc}/G_{oII}`, equivalent to
        Eq. (1) of Turon et al. (2018) [1]_, and the error is

        .. math::

            e = \frac{\left| G_{Ic}/G_{oI} - G_{IIc}/G_{oII} \right|}
                {G_{Ic}/G_{oI}}

        References
        ----------
        .. [1] Turon, A., Gonzalez, E. V., Sarrado, C., Guillamet, G.,
           Maimi, P., Accurate simulation of delamination under mixed-mode
           loading using a cohesive model with a mode-dependent penalty
           stiffness, Composite Structures, 184, 506-511, 2018.
           https://doi.org/10.1016/j.compstruct.2017.10.017

        """
        rI = self.GIc/self.GoI
        if self.GoII == 0.:
            return 0. if self.GIIc == 0. else np.inf
        return abs(rI - self.GIIc/self.GoII)/rI

    def Gc(self, B):
        r"""Fracture toughness of the law at a mode ratio

        Interpolates between the toughnesses ``GIc`` and ``GIIc``; used for
        the dissipated energy of :meth:`.CohesiveInterface.commit`.

        Parameters
        ----------
        B : float or np.ndarray
            Mode ratio, from 0 (mode I) to 1 (shear), dimensionless.

        Returns
        -------
        Gc : float or np.ndarray
            Fracture toughness, force/length, with the shape of ``B``.

        Notes
        -----
        Benzeggagh-Kenane form, `G_c(B) = G_{Ic} + (G_{IIc} - G_{Ic})
        B^{\eta_{BK}}`.

        """
        return self.GIc + (self.GIIc - self.GIc)*B**self.eta

    def Go(self, B):
        r"""Elastic energy of the law at damage onset at a mode ratio

        Interpolates between the energies at onset ``GoI`` and ``GoII`` as
        :meth:`Gc` does between the toughnesses.

        Parameters
        ----------
        B : float or np.ndarray
            Mode ratio, from 0 (mode I) to 1 (shear), dimensionless.

        Returns
        -------
        Go : float or np.ndarray
            Energy at onset, force/length, with the shape of ``B``.

        Notes
        -----
        `G_o(B) = G_{oI} + (G_{oII} - G_{oI}) B^{\eta_{BK}}`, with `G_{oI} =
        \tau_{o,I}^2/(2 k_I)` and `G_{oII} = \tau_{o,II}^2/(2 k_{sh})`.

        """
        return self.GoI + (self.GoII - self.GoI)*B**self.eta


def law_with_ratio(GIc, GIIc, tauI, KI, eta, kappa, name='',
                   open_only=False, slip_offset=False, opening_driven=False,
                   shear_free=False, wake_offset=False):
    r"""Bilinear law with a given ratio of the shear to the normal stiffness

    Builds the :class:`.BilinearLaw` with the shear stiffness ``kappa*KI``
    and the shear strength for which the damage does not decrease when the
    mode ratio changes, zero for ``GIIc = 0``. Superposed laws built with
    the same ``kappa`` share the equivalent separation and the mode ratio,
    with ``mode_ratio='per-law'`` and ``'summed'`` alike.

    Parameters
    ----------
    GIc, GIIc : float
        Fracture toughnesses in mode I and in shear, force/length.
    tauI : float
        Strength in mode I, stress.
    KI : float
        Penalty stiffness in mode I, force/length**3.
    eta : float
        Exponent of the Benzeggagh-Kenane criterion, dimensionless.
    kappa : float
        Ratio of the shear to the normal penalty stiffness, dimensionless.
    name : str, optional
        Label. Default ``''``.
    open_only, slip_offset, opening_driven, wake_offset : bool, optional
        Flags of :class:`.BilinearLaw`. Default ``False``.
    shear_free : bool, optional
        If ``True``, only with ``GIIc = 0``, the shear stiffness is zero
        instead of ``kappa*KI``: a law with a normal traction only, see
        ``opening_driven`` of :class:`.BilinearLaw`. Default ``False``.

    Returns
    -------
    law : :class:`.BilinearLaw`
        The law.

    Raises
    ------
    ValueError
        When ``shear_free`` is given with ``GIIc`` different from zero, and
        in the cases of :class:`.BilinearLaw`.

    Notes
    -----
    With `k_{sh} = \kappa k_I`, Eq. (1) of Turon et al. (2018) [1]_ gives
    the shear strength

    .. math::

        \tau_{o,II} = \tau_{o,I} \sqrt{\kappa \frac{G_{IIc}}{G_{Ic}}}

    The equivalent separation and the mode ratio depend on the stiffnesses
    through `k_{sh}/k_I` only, such that laws with the same `\kappa`, and
    without ``shear_free``, have the same ones.

    References
    ----------
    .. [1] Turon, A., Gonzalez, E. V., Sarrado, C., Guillamet, G., Maimi,
       P., Accurate simulation of delamination under mixed-mode loading
       using a cohesive model with a mode-dependent penalty stiffness,
       Composite Structures, 184, 506-511, 2018.
       https://doi.org/10.1016/j.compstruct.2017.10.017

    """
    tauII = tauI*np.sqrt(kappa*GIIc/GIc)
    KII = kappa*KI
    if shear_free:
        if GIIc != 0.:
            raise ValueError('shear_free requires GIIc = 0')
        KII = 0.
    return BilinearLaw(GIc, GIIc, tauI, tauII, KI, eta, KII=KII,
                       name=name, open_only=open_only,
                       slip_offset=slip_offset,
                       opening_driven=opening_driven,
                       wake_offset=wake_offset)


def _dbeta(B, eta):
    r"""Derivative of the Benzeggagh-Kenane factor with respect to the mode
    ratio

    Parameters
    ----------
    B : np.ndarray
        Mode ratios, dimensionless.
    eta : float
        Exponent of the Benzeggagh-Kenane criterion, dimensionless.

    Returns
    -------
    out : np.ndarray
        Derivatives, with the shape of ``B``.

    Notes
    -----
    `\mathrm{d} B^\eta/\mathrm{d} B = \eta B^{\eta - 1}` for `B > 0`. At `B
    = 0` the derivative is 1 for `\eta = 1` and zero for `\eta > 1`; for
    `\eta < 1` it is unbounded and is taken as zero.

    """
    out = np.zeros_like(B)
    pos = B > 0
    out[pos] = eta*B[pos]**(eta - 1.)
    if eta == 1.:
        out[~pos] = 1.
    return out


def evaluate_laws(laws, du, dv, dw, r_old, mode_ratio='per-law',
                  tangent=True, s0=None, s0_wake=None):
    r"""Tractions, tangent, thresholds and damage of superposed laws at
    many points

    Evaluates every law at the given separations, starting from the
    thresholds of the last converged state, and sums the tractions and the
    tangents of the laws. The arguments are not modified: the history is
    updated by the caller, see :meth:`.CohesiveInterface.commit`. The
    normal stiffness of every law is kept undamaged for a negative normal
    separation (contact).

    Parameters
    ----------
    laws : list of :class:`.BilinearLaw`
        The superposed laws.
    du, dv, dw : np.ndarray
        Tangential and normal separations at the points, length, 1-D arrays
        of the same size ``npts``.
    r_old : np.ndarray
        Thresholds of the last converged state, between 0 and 1,
        dimensionless, shape ``(len(laws), npts)``.
    mode_ratio : str, optional
        ``'per-law'`` (default), each law with the equivalent separation
        and the mode ratio of its own stiffnesses, or ``'summed'``, all the
        laws with those of the sums of the stiffnesses of the laws.
    tangent : bool, optional
        Whether the tangent is computed. Default ``True``.
    s0 : np.ndarray or None, optional
        Slip offsets `(\Delta_{u0}, \Delta_{v0})` of the last converged
        state, length, shape ``(2, npts)``, used by the laws with
        ``slip_offset``: their damage is driven by the shear separations
        `\Delta_u - \Delta_{u0}`, `\Delta_v - \Delta_{v0}`; the tractions
        keep `\Delta_u`, `\Delta_v`. ``None``, the default, for zero
        offsets.
    s0_wake : np.ndarray or None, optional
        The same for the laws with ``wake_offset``, see
        :attr:`.CohesiveInterface.s0w`. Default ``None``.

    Returns
    -------
    tau : np.ndarray
        Tractions along `u`, `v`, `w`, stress, shape ``(3, npts)``.
    T : np.ndarray or None
        Consistent tangent `\partial \tau_i/\partial \Delta_j`,
        force/length**3, shape ``(3, 3, npts)``, not symmetric where the
        damage grows; ``None`` with ``tangent=False``.
    r_new, d : np.ndarray
        Thresholds and damage variables of the state, dimensionless, shape
        ``(len(laws), npts)``.

    Raises
    ------
    ValueError
        When ``mode_ratio`` is neither ``'per-law'`` nor ``'summed'``.

    Notes
    -----
    For each law, with `S = \Delta_u^2 + \Delta_v^2`, `N = \langle \Delta_w
    \rangle^2` and the stiffnesses `a = k_{sh}`, `b = k_I` of the law
    (``'per-law'``) or their sums over the laws (``'summed'``), the
    equivalent separation and the mode ratio of Turon et al. (2018) [1]_
    are

    .. math::

        \Delta_{eq} = \frac{a S + b N}{\sqrt{a^2 S + b^2 N}}, \qquad
        B = \frac{a S}{a S + b N}

    both zero for zero `S` and `N`. `S` takes the separations less the
    offsets for the laws with ``slip_offset`` or ``wake_offset``, and is
    zero for the laws with ``opening_driven``. With the properties of the
    law, `G_o(B)`, `G_c(B)` and `k_B = k_I (1 - B) + k_{sh} B` give the
    separation at onset `\Delta_o = \sqrt{2 G_o/k_B}` and the ratio `\rho =
    \Delta_f/\Delta_o = G_c/G_o`, and then the loading function, the
    threshold and the damage

    .. math::

        H = \frac{\Delta_{eq}/\Delta_o - 1}{\rho - 1}, \qquad
        r = \max(r_{old}, \min(\max(H, 0), 1)), \qquad
        d = \frac{r \rho}{r \rho + 1 - r}

    Where `G_o = 0`, i.e. `B = 1` for a law without shear strength, the law
    has no capacity: `H = \infty` and `\rho` takes the value `G_{Ic}/G_{oI}`,
    its limit for a law that satisfies `G_{Ic}/G_{oI} = G_{IIc}/G_{oII}`.
    For a law with ``open_only``, `H = -\infty` where `\Delta_w \le 0`,
    which keeps the threshold.

    The tractions are the sums over the laws of

    .. math::

        \tau_u = (1 - d) k_{sh} \Delta_u, \qquad
        \tau_v = (1 - d) k_{sh} \Delta_v, \qquad
        \tau_w = (1 - d) k_I \langle \Delta_w \rangle - k_I \langle
        -\Delta_w \rangle

    and the tangent the sum of

    .. math::

        \frac{\partial \tau_i}{\partial \Delta_j} = (1 - d) k_i \delta_{ij}
        - k_i \Delta_i \frac{\partial d}{\partial r} \frac{\partial H}
        {\partial \Delta_j}, \qquad \frac{\partial d}{\partial r} =
        \frac{\rho}{(r \rho + 1 - r)^2}

    with `k_u = k_v = k_{sh}`, and `k_w = k_I`, `\Delta_w` for an opening.
    For `\Delta_w \le 0` the normal term is the undamaged `k_I`: at
    `\Delta_w = 0` the contact stiffness is taken, such that the tangent of
    a pristine interface at zero separation is `k_I`. The second term
    exists only where the law is loading, `H > r_{old}` and `0 < H < 1`,
    and `\partial H/\partial \Delta_j` follows from the chain rule through
    `\Delta_{eq}` and `B`, including `\partial \Delta_o/\partial B` and
    `\partial \rho/\partial B`, with `\partial S/\partial \Delta_j = (2
    \Delta_u, 2 \Delta_v, 0)` and `\partial N/\partial \Delta_j = (0, 0, 2
    \langle \Delta_w \rangle)`. The derivative of `d` with respect to
    `\rho` at a constant `r` is not included: it multiplies `\partial
    \rho/\partial B`, which is zero for the laws that satisfy
    `G_{Ic}/G_{oI} = G_{IIc}/G_{oII}`.

    References
    ----------
    .. [1] Turon, A., Gonzalez, E. V., Sarrado, C., Guillamet, G., Maimi,
       P., Accurate simulation of delamination under mixed-mode loading
       using a cohesive model with a mode-dependent penalty stiffness,
       Composite Structures, 184, 506-511, 2018.
       https://doi.org/10.1016/j.compstruct.2017.10.017

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
    S_tot, su, sv = S, du, dv
    zero = np.zeros(npts)
    if s0 is not None and any(getattr(l, 'slip_offset', False)
                              for l in laws):
        su_rel = du - s0[0]
        sv_rel = dv - s0[1]
        S_rel = su_rel*su_rel + sv_rel*sv_rel
    if s0_wake is not None and any(getattr(l, 'wake_offset', False)
                                   for l in laws):
        su_w = du - s0_wake[0]
        sv_w = dv - s0_wake[1]
        S_w = su_w*su_w + sv_w*sv_w
    for i, law in enumerate(laws):
        if mode_ratio == 'per-law':
            a, b = law.KII, law.KI
        elif mode_ratio == 'summed':
            a, b = KIIsum, KIsum
        else:
            raise ValueError("mode_ratio must be 'per-law' or 'summed'")
        #NOTE slip_offset: the damage driver uses the slip since the last
        #     closed converged state; the offset is constant within the
        #     increment, and where the point is closed the damage of an
        #     open_only law does not grow, so the offset does not matter
        if getattr(law, 'opening_driven', False):
            # damage driven by <dw> alone: no slip in Delta_eq and B
            S, su, sv = zero, zero, zero
        elif s0 is not None and getattr(law, 'slip_offset', False):
            S, su, sv = S_rel, su_rel, sv_rel
        elif s0_wake is not None and getattr(law, 'wake_offset', False):
            S, su, sv = S_w, su_w, sv_w
        else:
            S, su, sv = S_tot, du, dv
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
        #NOTE where G_o = 0 (B = 1 for a law without shear strength),
        #     G_c/G_o is 0/0; its limit for a law that satisfies Eq. (1) is
        #     the constant G_Ic/G_oI, which gives the damage of a frozen
        #     threshold (open_only)
        rho = np.full(npts, law.GIc/law.GoI)
        cap = ~no_capacity
        rho[cap] = Gc[cap]/Go[cap]
        lo = np.zeros(npts)
        lo[cap] = np.sqrt(2*Go[cap]/KB[cap])
        H = np.zeros(npts)
        act = cap & sep_nz
        H[act] = (lam[act]/lo[act] - 1.)/(rho[act] - 1.)
        H[no_capacity] = np.inf
        if law.open_only:
            # no damage growth where the interface is closed
            H[~opening] = -np.inf
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
        #NOTE at dw = 0 the traction is zero from both sides; the contact
        #     stiffness is taken there, such that the tangent of the
        #     pristine interface, e.g. at c = 0, is K_I and not zero
        T[2, 2] += np.where(dw <= 0, law.KI, 0.)
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
        # dS/d(du, dv, dw) = (2du, 2dv, 0), dN/d(du, dv, dw) = (0, 0, 2<dw>),
        # du, dv less the constant offsets for slip_offset
        dH = np.vstack((2*su[L]*dH_dS, 2*sv[L]*dH_dS, 2*wp[L]*dH_dN))
        rL = ri[L]
        dd_dr = rhoL/(rL*rhoL + 1. - rL)**2
        t_undamaged = Keff[:, L]*sep[:, L]
        for j in range(3):
            for k in range(3):
                T[j, k, L] -= t_undamaged[j]*dd_dr*dH[k]
    return tau, T, r_new, d


def contact_traction(dw, k, eps):
    r"""Normal traction and its derivative of a contact-only interface

    No traction for an opening, the penalty stiffness ``k`` for a
    penetration larger than ``eps``, and a smooth transition in between,
    such that the derivative is continuous.

    Parameters
    ----------
    dw : float or np.ndarray
        Normal separations, negative for a penetration, length.
    k : float
        Contact stiffness, force/length**3.
    eps : float
        Penetration over which the stiffness is phased in, length.

    Returns
    -------
    tw : np.ndarray
        Normal tractions, negative in compression, stress, with the shape
        of ``dw``.
    kw : np.ndarray
        Derivatives of the tractions with respect to ``dw``,
        force/length**3, with the shape of ``dw``.

    Notes
    -----
    `\tau_w = -k\, p(-\Delta_w)` with the smoothed ramp

    .. math::

        p(x) = \begin{cases} 0 & x \le 0 \\ x^2/(2 \epsilon) & 0 < x <
        \epsilon \\ x - \epsilon/2 & x \ge \epsilon \end{cases}

    The continuous tangent `\partial \tau_w/\partial \Delta_w = k\, p'(
    -\Delta_w)` keeps the quadratic convergence of the Newton-Raphson
    iterations when points of the interface close or open. A sharp switch
    makes the iterations oscillate between the states of the points near
    `\Delta_w = 0`. The traction differs from `k \Delta_w` by at most `k
    \epsilon/2`.

    """
    x = np.maximum(-np.asarray(dw, dtype=DOUBLE), 0.)
    small = x < eps
    p = np.where(small, x*x/(2*eps), x - 0.5*eps)
    dp = np.where(small, x/eps, 1.)
    return -k*p, k*dp


class CohesiveInterface(object):
    r"""Cohesive interface of a ``'CZM'`` connection

    Holds the operators that give the separations at the Gauss-Legendre
    points of the common area of two domains, the laws and the damage
    history of one connection, and computes the internal force vector, the
    tangent and the secant stiffness matrices, the stored energy and the
    resultants of the interface. The history changes only with
    :meth:`commit` and :meth:`reset`. :class:`.MultiDomain` creates one
    interface per connection ``'CZM'``, see
    :meth:`.MultiDomain.cohesive_interfaces`. A :class:`UserWarning` is
    issued for a law whose :meth:`.BilinearLaw.consistency_error` exceeds
    ``1e-6``, and for ``mode_ratio='summed'`` with laws of different ratios
    ``KII/KI``.

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
        Size of the assembly, the number of its Ritz constants.
    mode_ratio : str, optional
        ``'per-law'`` (default) or ``'summed'``, see
        :func:`.evaluate_laws`.
    k_contact : float or None, optional
        Normal stiffness of a contact-only interface (no laws),
        force/length**3. Default ``None``.
    contact_eps : float, optional
        Penetration over which the contact stiffness of a contact-only
        interface is phased in, see :func:`.contact_traction`, length.
        Default ``1e-3``.

    Raises
    ------
    ValueError
        When the domains differ in ``x0``, ``y0``, ``a`` or ``b``, when
        ``col_start`` of a domain is not defined, or when there are no laws
        and no ``k_contact``.

    Attributes
    ----------
    p_top, p_bot, laws, mode_ratio, k_contact, contact_eps
        As given.
    npts : int
        Number of points, ``nr_y_gauss*nr_x_gauss``, ordered as
        ``l*nr_x_gauss + k`` with ``k`` along `x` and ``l`` along `y`.
    dofs : np.ndarray
        Columns of the Ritz constants of ``p_top`` followed by those of
        ``p_bot`` in the assembly.
    r : np.ndarray
        Damage thresholds of the last converged state, shape ``(len(laws),
        nr_y_gauss*nr_x_gauss)``.
    d : np.ndarray
        Damage variables of the last converged state, same shape.
    s0 : np.ndarray
        Slip offsets `(\Delta_{u0}, \Delta_{v0})` of the laws with
        ``slip_offset``, shape ``(2, nr_y_gauss*nr_x_gauss)``: the shear
        separations of the last converged state in which the point was
        closed, `\Delta_w \le 0`, zero for a pristine interface. One offset
        per point serves all such laws, since it depends on the separations
        only. Updated by :meth:`commit`; unused without such laws.
    s0w : np.ndarray
        Slip offsets of the laws with ``wake_offset``, same shape: the shear
        separations of the last converged state in which some law without
        ``wake_offset`` was not failed at the point, `d < 1`. Updated by
        :meth:`commit`; unused without such laws.
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
    dissipated, as noted by Turon et al. (2018) [1]_ after their Eq. (20).
    With `\partial d/\partial B = 0` this holds for any path, the mode
    ratio entering through `G_c(B)`, taken at the end of each increment.

    References
    ----------
    .. [1] Turon, A., Gonzalez, E. V., Sarrado, C., Guillamet, G., Maimi,
       P., Accurate simulation of delamination under mixed-mode loading
       using a cohesive model with a mode-dependent penalty stiffness,
       Composite Structures, 184, 506-511, 2018.
       https://doi.org/10.1016/j.compstruct.2017.10.017

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
        #NOTE dense operators restricted to the degrees of freedom of the
        #     two domains, for the assembly of the tangent with dense matrix
        #     products: every point involves all the terms of both domains
        self.dofs = np.concatenate((np.arange(p_top.col_start, p_top.col_end),
                                    np.arange(p_bot.col_start, p_bot.col_end)))
        self._Bl = [Bi[:, self.dofs].toarray() for Bi in B]
        nl = self.dofs.size
        self._rows_g = np.repeat(self.dofs, nl)
        self._cols_g = np.tile(self.dofs, nl)
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
        self._slip_offset = any(l.slip_offset for l in self.laws)
        self._wake_offset = any(l.wake_offset for l in self.laws)
        self._tip = [i for i, l in enumerate(self.laws) if not l.wake_offset]
        self.reset()

    def reset(self):
        r"""Returns the interface to the pristine state

        Sets the thresholds ``r``, the damage variables ``d``, the slip
        offsets ``s0`` and ``s0w`` and the energy ``dissipated`` to zero.
        Called by the constructor, and by
        :meth:`.MultiDomain.reset_cohesive` for all the interfaces of an
        assembly.

        """
        self.r = np.zeros((len(self.laws), self.npts), dtype=DOUBLE)
        self.d = np.zeros_like(self.r)
        self.s0 = np.zeros((2, self.npts), dtype=DOUBLE)
        self.s0w = np.zeros((2, self.npts), dtype=DOUBLE)
        self.dissipated = 0.

    def separations(self, c):
        r"""Separations at the points of the interface

        The displacement jumps between the faces of the two domains in
        contact at the Gauss-Legendre points, from the operators ``Bu``,
        ``Bv`` and ``Bw``.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, shape ``(size,)``.

        Returns
        -------
        du, dv, dw : np.ndarray
            Tangential separations along `x` and `y` and normal separation,
            positive for an opening, length, shape ``(npts,)``.

        Notes
        -----
        `\Delta_i = \boldsymbol{B}_i \boldsymbol{c}`, `i = u, v, w`, with
        the displacement of the lower face of ``p_top`` less the one of the
        upper face of ``p_bot``, see the notes of the class.

        """
        return self.Bu @ c, self.Bv @ c, self.Bw @ c

    def state(self, c, tangent=True):
        r"""Tractions, tangent, thresholds and damage at a configuration

        Evaluates the laws at the separations of ``c``, starting from the
        thresholds and the slip offsets of the last converged state, with
        :func:`.evaluate_laws`; for a contact-only interface, the normal
        traction of :func:`.contact_traction`. The history is not changed.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, shape ``(size,)``.
        tangent : bool, optional
            Whether the tangent is computed. Default ``True``.

        Returns
        -------
        tau : np.ndarray
            Tractions along `u`, `v`, `w`, stress, shape ``(3, npts)``.
        T : np.ndarray or None
            Consistent tangent of the tractions with respect to the
            separations, force/length**3, shape ``(3, 3, npts)``; ``None``
            with ``tangent=False``.
        r_new, d : np.ndarray
            Thresholds and damage variables at ``c``, shape ``(len(laws),
            npts)``; for a contact-only interface, shape ``(0, npts)``.

        Notes
        -----
        The thresholds at ``c`` are `r = \max(r_{old}, \min(H, 1))`, with
        `r_{old}` of the last converged state and the loading function `H`
        of each law at ``c``: an iteration of the non-linear analysis does
        not accumulate damage on the previous iterations. For a
        contact-only interface only `\tau_w` and `\partial \tau_w/\partial
        \Delta_w` are not zero.

        """
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
                             self.mode_ratio, tangent=tangent,
                             s0=self.s0 if self._slip_offset else None,
                             s0_wake=self.s0w if self._wake_offset else None)

    def fint(self, c):
        r"""Internal force vector of the interface

        The tractions of :meth:`state` integrated over the interface with
        the separation operators; the history is not changed.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, shape ``(size,)``.

        Returns
        -------
        fint : np.ndarray
            Internal forces of the interface at ``c``, with the history of
            the last converged state, shape ``(size,)``.

        Notes
        -----
        `\boldsymbol{F}^{coh} = \sum_i \boldsymbol{B}_i^\top \boldsymbol{W}
        \boldsymbol{\tau}_i`, `i = u, v, w`, with the tractions of
        :meth:`state` and `\boldsymbol{W}` the diagonal of ``weights``.

        """
        tau, _, _, _ = self.state(c, tangent=False)
        w = self.weights
        return (self.Bu.T @ (w*tau[0]) + self.Bv.T @ (w*tau[1])
                + self.Bw.T @ (w*tau[2]))

    def kT(self, c):
        r"""Consistent tangent stiffness matrix of the interface

        The derivative of :meth:`fint` with respect to the Ritz constants,
        not symmetric where the damage grows.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, shape ``(size,)``.

        Returns
        -------
        kT : scipy.sparse.csr_matrix
            Tangent stiffness matrix at ``c``, shape ``(size, size)``, with
            entries only in the rows and columns of the two domains.

        Notes
        -----
        `\boldsymbol{K}_T^{coh} = \sum_{i,j} \boldsymbol{B}_i^\top
        \boldsymbol{W} \mathrm{diag}(\partial \tau_i/\partial \Delta_j)
        \boldsymbol{B}_j`, `i, j = u, v, w`, with the tangent of
        :meth:`state`. It is assembled as the matrix of the pristine
        interface, computed once, plus the products over the points whose
        tangent differs from the pristine one.

        """
        _, T, _, _ = self.state(c, tangent=True)
        return self._assemble(T)

    def kC0(self):
        r"""Secant stiffness of the last converged state at zero
        separation, with the normal stiffness of an opening, used by the
        linear analyses

        A contact-only interface has no stiffness in a linear analysis, the
        faces being taken as open: the linear analyses cannot represent
        contact, and treating the faces as bonded would glue a pre-crack
        shut. Use a non-linear analysis where contact matters.

        Returns
        -------
        kC0 : scipy.sparse.csr_matrix
            Secant stiffness matrix, shape ``(size, size)``.

        Notes
        -----
        `\boldsymbol{K}_{C0}^{coh} = \sum_i \boldsymbol{B}_i^\top
        \boldsymbol{W} \mathrm{diag}(k_i^{sec}) \boldsymbol{B}_i`, `i = u,
        v, w`, with `k_u^{sec} = k_v^{sec} = \sum_\ell (1 - d_\ell)
        k_{sh,\ell}` and `k_w^{sec} = \sum_\ell (1 - d_\ell) k_{I,\ell}`,
        `d_\ell` being the damage of the law `\ell` at the last converged
        state.

        """
        T = np.zeros((3, 3, self.npts))
        for i, law in enumerate(self.laws):
            s = 1. - self.d[i]
            T[0, 0] += s*law.KII
            T[1, 1] += s*law.KII
            T[2, 2] += s*law.KI
        return self._assemble(T)

    def _T0(self):
        r"""Tangent of the pristine interface at the points

        The same for an opening and for a closure; zero for a contact-only
        interface.

        Returns
        -------
        T0 : np.ndarray
            Tangent of the tractions with respect to the separations,
            force/length**3, shape ``(3, 3, npts)``.

        Notes
        -----
        `\mathrm{diag}(k_{sh}, k_{sh}, k_I)` summed over the laws, the
        tangent of :func:`.evaluate_laws` for `d = 0` without damage
        growth.

        """
        T0 = np.zeros((3, 3, self.npts))
        if self.laws:
            T0[0, 0] = sum(l.KII for l in self.laws)
            T0[1, 1] = T0[0, 0]
            T0[2, 2] = sum(l.KI for l in self.laws)
        return T0

    def _K0(self):
        r"""Local stiffness matrix of the pristine interface, cached"""
        if getattr(self, '_K0_cache', None) is None:
            self._K0_cache = self._local(self._T0())
        return self._K0_cache

    def _local(self, T, rows=None):
        r"""Dense `\sum_{i,j} \boldsymbol{B}_i^\top \boldsymbol{W}
        \mathrm{diag}(T_{ij}) \boldsymbol{B}_j` on the degrees of freedom of
        the two domains, over the points ``rows`` (all by default), with
        three matrix products"""
        Bl = self._Bl
        w = self.weights
        if rows is not None:
            Bl = [B[rows] for B in Bl]
            w = w[rows]
        nl = self.dofs.size
        K = np.zeros((nl, nl))
        for i in range(3):
            H = None
            for j in range(3):
                tij = T[i, j]
                if not np.any(tij):
                    continue
                term = (w*tij)[:, None]*Bl[j]
                H = term if H is None else H + term
            if H is not None:
                K += Bl[i].T @ H
        return K

    def _to_global(self, K):
        from scipy.sparse import coo_matrix
        return coo_matrix((K.ravel(), (self._rows_g, self._cols_g)),
                          shape=(self.size, self.size)).tocsr()

    def _assemble(self, T):
        r"""`\sum_{i,j} \boldsymbol{B}_i^\top \boldsymbol{W} \mathrm{diag}(T_{ij})
        \boldsymbol{B}_j`, computed as the cached pristine matrix plus the
        products restricted to the points where `T` differs from the pristine
        tangent, i.e. the points that are damaged, loading or, for a
        contact-only interface, in contact"""
        if not self.laws:
            rows = np.nonzero(np.any(T != 0, axis=(0, 1)))[0]
            if rows.size == 0:
                return csr_matrix((self.size, self.size))
            return self._to_global(self._local(T[:, :, rows], rows))
        dT = T - self._T0()
        rows = np.nonzero(np.any(dT != 0, axis=(0, 1)))[0]
        K = self._K0()
        if rows.size:
            K = K + self._local(dT[:, :, rows], rows)
        return self._to_global(K)

    def kTv(self, c, v):
        r"""Product of the consistent tangent stiffness matrix by a vector

        The product of :meth:`kT` at ``c`` by ``v``, without assembling the
        matrix.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, shape ``(size,)``.
        v : np.ndarray
            Vector, shape ``(size,)``.

        Returns
        -------
        kTv : np.ndarray
            Product, shape ``(size,)``.

        Notes
        -----
        `\boldsymbol{K}_T^{coh} \boldsymbol{v} = \sum_i \boldsymbol{B}_i^\top
        \boldsymbol{W} \sum_j (\partial \tau_i/\partial \Delta_j)
        (\boldsymbol{B}_j \boldsymbol{v})`, `i, j = u, v, w`.

        """
        _, T, _, _ = self.state(c, tangent=True)
        Bs = (self.Bu, self.Bv, self.Bw)
        dv = [B @ v for B in Bs]
        w = self.weights
        out = np.zeros(self.size)
        for i in range(3):
            s = sum(T[i, j]*dv[j] for j in range(3))
            if np.any(s):
                out += Bs[i].T @ (w*s)
        return out

    def _Gc_laws(self, c):
        r"""Fracture toughness of each law at the points

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, shape ``(size,)``.

        Returns
        -------
        Gc : np.ndarray
            Toughnesses at the mode ratios of ``c``, force/length, shape
            ``(len(laws), npts)``.

        Notes
        -----
        `G_c(B)` of :meth:`.BilinearLaw.Gc`, with the mode ratio `B = a
        S/(a S + b N)` of :func:`.evaluate_laws` for the option
        ``mode_ratio`` and the slip offsets of the interface, `B = 0` at
        zero separation; `G_{Ic}` for the laws with ``opening_driven``.

        """
        du, dv, dw = self.separations(c)
        S_tot = du*du + dv*dv
        S_rel = (du - self.s0[0])**2 + (dv - self.s0[1])**2
        S_w = (du - self.s0w[0])**2 + (dv - self.s0w[1])**2
        N = np.maximum(dw, 0.)**2
        KIs = sum(l.KI for l in self.laws)
        KIIs = sum(l.KII for l in self.laws)
        out = np.empty((len(self.laws), self.npts))
        for i, law in enumerate(self.laws):
            a, b = ((law.KII, law.KI) if self.mode_ratio == 'per-law'
                    else (KIIs, KIs))
            if law.opening_driven:
                out[i] = law.GIc
                continue
            S = (S_rel if law.slip_offset else S_w if law.wake_offset
                 else S_tot)
            P = a*S + b*N
            B = np.where(P > 0, a*S/np.where(P > 0, P, 1.), 0.)
            out[i] = law.Gc(B)
        return out

    def commit(self, c):
        r"""Updates the history with the converged state ``c``

        The thresholds and the damage variables become those of ``c``, and
        the energy dissipated since the last converged state is added to
        ``dissipated``. For laws with ``slip_offset``, the offsets of the
        points closed at ``c``, `\Delta_w \le 0`, become their shear
        separations at ``c``, after the thresholds are evaluated with the
        previous offsets, as in the iterations; for laws with
        ``wake_offset``, those of the points where a law without
        ``wake_offset`` has not failed at ``c``. Nothing is done for a
        contact-only interface.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly at a converged state, shape
            ``(size,)``.

        Notes
        -----
        The energy dissipated in the increment is

        .. math::

            \Delta D = \sum_g w_g \sum_\ell G_{c,\ell}(B_g) \left( r_{\ell,
            g} - r_{\ell, g}^{old} \right)

        over the points `g` with the weights `w_g` and the laws `\ell`,
        with the toughness `G_c` at the mode ratio of ``c``, the end of the
        increment: the threshold `r` of a law is the fraction of its `G_c`
        already dissipated, see the notes of the class.

        """
        if not self.laws:
            return
        _, _, r_new, d_new = self.state(c, tangent=False)
        Gc = self._Gc_laws(c)
        dr = r_new - self.r
        self.dissipated += float(np.sum(self.weights*np.sum(Gc*dr, axis=0)))
        self.r = r_new
        self.d = d_new
        if self._slip_offset:
            du, dv, dw = self.separations(c)
            closed = dw <= 0
            self.s0[0, closed] = du[closed]
            self.s0[1, closed] = dv[closed]
        if self._wake_offset:
            du, dv, dw = self.separations(c)
            if self._tip:
                follow = ~np.all(self.d[self._tip] >= 1., axis=0)
            else:
                follow = np.zeros(self.npts, bool)
            self.s0w[0, follow] = du[follow]
            self.s0w[1, follow] = dv[follow]

    def energy(self, c):
        r"""Free energy stored in the interface

        The elastic energy of the separations of ``c`` with the damage of
        the last converged state, the contact energy of a negative normal
        separation included.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, shape ``(size,)``.

        Returns
        -------
        energy : float
            Stored energy, force*length.

        Notes
        -----
        `\sum_g w_g \psi_g`, with the free energy per unit area of Eqs.
        (3), (4) of Turon et al. (2018) [1]_, summed over the laws,

        .. math::

            \psi = \sum_\ell \left[ \frac{1}{2} (1 - d_\ell) \left(
            k_{sh,\ell} \Delta_{sh}^2 + k_{I,\ell} \langle \Delta_w
            \rangle^2 \right) + \frac{1}{2} k_{I,\ell} \langle -\Delta_w
            \rangle^2 \right]

        For a contact-only interface, `\psi = k \int_0^x p(s) \,
        \mathrm{d}s` with `x = \langle -\Delta_w \rangle` and the ramp `p`
        of :func:`.contact_traction`: `k x^3/(6 \epsilon)` for `x <
        \epsilon` and `k (x^2/2 - \epsilon x/2 + \epsilon^2/6)` beyond.

        References
        ----------
        .. [1] Turon, A., Gonzalez, E. V., Sarrado, C., Guillamet, G.,
           Maimi, P., Accurate simulation of delamination under mixed-mode
           loading using a cohesive model with a mode-dependent penalty
           stiffness, Composite Structures, 184, 506-511, 2018.
           https://doi.org/10.1016/j.compstruct.2017.10.017

        """
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
        r"""Resultants of the tractions of the interface

        The tractions of :meth:`state` integrated over the area of the
        interface; the history is not changed.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, shape ``(size,)``.

        Returns
        -------
        R : np.ndarray
            Resultants of the tractions on ``p_top`` at ``c``, along `x`,
            `y`, `z`, with the sign of the force applied by ``p_bot``,
            force, shape ``(3,)``.

        Notes
        -----
        `R_i = -\sum_g w_g \tau_{i,g}`, `i = u, v, w`, the tractions of
        :meth:`state` integrated over the interface: a positive traction,
        which opposes a positive separation, pulls ``p_top`` along the
        negative axis.

        """
        tau, _, _, _ = self.state(c, tangent=False)
        return -(tau @ self.weights)
