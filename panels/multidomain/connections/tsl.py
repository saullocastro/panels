r"""Discretisation of a cohesive domain (``'SB_TSL'`` connection)

The rules below come from the convergence study of the cohesive zone on a
mode I double cantilever beam (DCB), see :ref:`cohesive_zone`. The opening
ahead of the process zone decays as that of two beams connected by an elastic
foundation of stiffness `k_o`: with the opening `\Delta = w_1 - w_2` and the
bending stiffnesses `D_{11}^{(1)}`, `D_{11}^{(2)}` of the two panels,
`\Delta_{,xxxx} + 4 \lambda^4 \Delta = 0` with the rate

.. math::

    \lambda = \left[ \frac{k_o}{4} \left( \frac{1}{D_{11}^{(1)}} +
    \frac{1}{D_{11}^{(2)}} \right) \right]^{1/4}

which is `\lambda = (k_o/2 D_{11})^{1/4}` for the symmetric DCB and tends to
`(k_o/4 D_{11}^{(1)})^{1/4}` when the second panel is much stiffer than the
first. The cohesive zone has the length, Rice's estimate as used by Turon et
al. (2007):

.. math::

    l_{cz} = 0.88 \frac{E_2 G_{Ic}}{\tau_o^2}

The Ritz basis along `x` of the cohesive domain must resolve the decay, and
the Gauss-Legendre points along `x` must sample the cohesive zone. With `m`
terms and `n_x` points over a cohesive domain of length `L_{tsl}`, the loads
of the DCB are converged to 0.1% from `m = 1.0 \lambda L_{tsl}`, and to
0.02% with `n_x = 10 L_{tsl}/l_{cz}`; with `n_x = 7 L_{tsl}/l_{cz}` the error
is 0.1-0.2%, and with fewer points the error has a floor of about 0.6-0.9%
and the solution needs bisections. The recommended values are
`m = 1.25 \lambda L_{tsl}` and `n_x = 10 L_{tsl}/l_{cz}`.

Before version 0.13.11 the rate was `(k_o/4 D_{11})^{1/4}`, that of one beam
on a rigid foundation, `2^{-1/4}` times the rate of the symmetric DCB, and
the factors of the rules were 1.2 and 1.5 instead of 1.0 and 1.25.

"""
import numpy as np

from .penalties import build_panel_lam


def tsl_resolution(p1, k_o, tau_o, G1c, E2=None, p2=None):
    r"""Decay rate of the opening and length of the cohesive zone

    The laminates of ``p1`` and, when given, ``p2`` are rebuilt with
    :func:`.build_panel_lam`, and `D_{11}` is read from them. The rate is
    ``(k_o*(1/D11_1 + 1/D11_2)/4)**0.25`` and the length
    ``0.88*E2*G1c/tau_o**2``, see the module docstring.

    Parameters
    ----------
    p1 : :class:`.Shell`
        Panel ``p1`` of the ``'SB_TSL'`` connection, whose bending stiffness
        `D_{11}` is used, that of the laminate ``p1.lam``, also when the
        attribute ``ABDnxny`` is set.
    k_o : float
        Pristine stiffness of the interface, force/length**3.
    tau_o : float
        Traction at damage onset, force/length**2.
    G1c : float
        Critical energy release rate in mode I, force/length.
    E2 : float, optional
        Transverse modulus used in `l_{cz}`, force/length**2. By default the
        second value of the first lamina of ``p1.laminaprops``.
    p2 : :class:`.Shell`, optional
        Panel ``p2`` of the connection, whose `D_{11}` is used in the same
        way. By default that of ``p1``, the symmetric DCB, `\lambda =
        (k_o/2 D_{11})^{1/4}`.

    Returns
    -------
    lam, lcz : float
        `\lambda`, 1/length, and `l_{cz}`, length.

    """
    build_panel_lam(p1)
    D11_1 = p1.lam.D[0, 0]
    if p2 is None:
        D11_2 = D11_1
    else:
        build_panel_lam(p2)
        D11_2 = p2.lam.D[0, 0]
    if E2 is None:
        E2 = p1.laminaprops[0][1]
    lam = (k_o*(1/D11_1 + 1/D11_2)/4)**0.25
    lcz = 0.88*E2*G1c/tau_o**2
    return lam, lcz


def tsl_margin(lam, lcz):
    r"""Length of the cohesive domain needed ahead of the crack front

    `3 l_{cz} + 8/\lambda`: the process zone, and the region ahead of it
    where the opening oscillates and decays as `e^{-\lambda x}`, the
    solution of beams on an elastic foundation (Kanninen 1973, Int. J.
    Fract. 9, 83-92), here the two panels connected by `k_o`. Beyond
    `8/\lambda` the opening, and with it the traction outside the cohesive
    domain, is below `e^{-8} = 3.4 \times 10^{-4}` of its value at the end
    of the process zone. A cohesive domain shorter than the crack growth plus
    this margin lets the compressive lobe and the tractions ahead of it reach
    the perfect bond: the loads are hardly affected, but the integral of the
    cohesive tractions no longer balances the load. On the DCB validation
    runs this difference is 0.25% at most with the margin, against up to
    10% with `2\pi/\lambda + 3 l_{cz}`, when the basis follows
    :func:`tsl_discretization` and the perfect bond is not wider than the
    bonded strip.

    Parameters
    ----------
    lam : float
        Decay rate `\lambda` of the opening, 1/length, see
        :func:`tsl_resolution`.
    lcz : float
        Length of the cohesive zone `l_{cz}`, length, see
        :func:`tsl_resolution`.

    Returns
    -------
    margin : float
        ``3*lcz + 8/lam``, length.

    """
    return 3*lcz + 8/lam


def tsl_discretization(lam, lcz, L_tsl, terms_per_decay=1.25, points_per_lcz=10.,
                       m_max=25):
    r"""Terms and Gauss points along `x` of a cohesive domain

    - `m = \lceil t \lambda L_{tsl} \rceil`, with `t` given by
      ``terms_per_decay``, at most ``m_max``;
    - `n_x = p L_{tsl}/l_{cz}`, with `p` given by ``points_per_lcz``, but not
      less than `4 m` to integrate the products of the approximation
      functions, rounded up to a multiple of 10.

    The Gauss-Legendre points and weights are computed by
    :func:`scipy.special.roots_legendre`, without an upper limit for `n_x`.
    The Bardell functions go up to 30 terms, so a cohesive domain longer than
    `30/(1.0 \lambda)` is not resolved.

    Parameters
    ----------
    lam, lcz : float
        See :func:`tsl_resolution`.
    L_tsl : float
        Length of the cohesive domain along `x`, length.
    terms_per_decay, points_per_lcz : float, optional
        Factors of the rules, dimensionless. Default ``1.25`` and ``10.``.
    m_max : int, optional
        Maximum number of terms. Default ``25``.

    Returns
    -------
    m, nx : int
        Terms and Gauss points along `x`.

    """
    m = int(min(m_max, np.ceil(terms_per_decay*lam*L_tsl)))
    nx = max(points_per_lcz*L_tsl/lcz, 4*m)
    nx = int(np.ceil(nx/10)*10)
    return m, nx
