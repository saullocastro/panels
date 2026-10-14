r"""Discretisation of a cohesive domain (``'SB_TSL'`` connection)

The rules below come from the convergence study of the cohesive zone on a
mode I double cantilever beam (DCB), see :ref:`cohesive_zone`. The opening
ahead of the process zone decays as that of a beam on an elastic foundation,
with the rate

.. math::

    \lambda = \left( \frac{k_o}{4 D_{11}} \right)^{1/4}

and the cohesive zone has the length, Rice's estimate as used by Turon et
al. (2007):

.. math::

    l_{cz} = 0.88 \frac{E_2 G_{Ic}}{\tau_o^2}

The Ritz basis along `x` of the cohesive domain must resolve the decay, and
the Gauss-Legendre points along `x` must sample the cohesive zone. With `m`
terms and `n_x` points over a cohesive domain of length `L_{tsl}`, the loads
of the DCB are converged to 0.1% from `m = 1.2 \lambda L_{tsl}`, and to
0.02% with `n_x = 10 L_{tsl}/l_{cz}`; with `n_x = 7 L_{tsl}/l_{cz}` the error
is 0.1-0.2%, and with fewer points the error has a floor of about 0.6-0.9%
and the solution needs bisections. The recommended values are
`m = 1.5 \lambda L_{tsl}` and `n_x = 10 L_{tsl}/l_{cz}`.

"""
import numpy as np

from .penalties import build_panel_lam


def tsl_resolution(p1, k_o, tau_o, G1c, E2=None):
    r"""Decay rate of the opening and length of the cohesive zone

    Parameters
    ----------
    p1 : :class:`.Shell`
        Panel ``p1`` of the ``'SB_TSL'`` connection, whose bending stiffness
        `D_{11}` is used.
    k_o : float
        Pristine stiffness of the interface, force/length**3.
    tau_o : float
        Traction at damage onset, force/length**2.
    G1c : float
        Critical energy release rate in mode I, force/length.
    E2 : float, optional
        Transverse modulus used in `l_{cz}`. By default the second value of
        the first lamina of ``p1.laminaprops``.

    Returns
    -------
    lam, lcz : float
        `\lambda`, 1/length, and `l_{cz}`, length.

    """
    build_panel_lam(p1)
    D11 = p1.lam.D[0, 0]
    if E2 is None:
        E2 = p1.laminaprops[0][1]
    lam = (k_o/(4*D11))**0.25
    lcz = 0.88*E2*G1c/tau_o**2
    return lam, lcz


def tsl_margin(lam, lcz):
    r"""Length of the cohesive domain needed ahead of the crack front

    `2\pi/\lambda + 3 l_{cz}`, which contains the process zone and the
    compressive lobe of the opening ahead of it. A cohesive domain shorter
    than the crack growth plus this margin lets the compressive lobe reach
    the end of the cohesive domain.

    """
    return 2*np.pi/lam + 3*lcz


def tsl_discretization(lam, lcz, L_tsl, terms_per_decay=1.5, points_per_lcz=10.,
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
    `30/(1.2 \lambda)` is not resolved.

    Parameters
    ----------
    lam, lcz : float
        See :func:`tsl_resolution`.
    L_tsl : float
        Length of the cohesive domain along `x`.
    terms_per_decay, points_per_lcz : float, optional
        Factors of the rules.
    m_max : int, optional
        Maximum number of terms.

    Returns
    -------
    m, nx : int
        Terms and Gauss points along `x`.

    """
    m = int(min(m_max, np.ceil(terms_per_decay*lam*L_tsl)))
    nx = max(points_per_lcz*L_tsl/lcz, 4*m)
    nx = int(np.ceil(nx/10)*10)
    return m, nx
