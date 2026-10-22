r"""Penalty constants of the connections and prescribed displacements

The functions compute the penalty constants from the laminates of the
panels: :func:`.calc_kt_kr` for the connections between two panels,
:func:`.calc_ku_kv_kw_point_pd`, :func:`.calc_ku_kv_kw_line_pd_xcte` and
:func:`.calc_ku_kv_kw_line_pd_ycte` for the prescribed displacements, and
:func:`.calc_kw_tsl` for the out-of-plane stiffness and the damage of a
connection with a traction-separation law. All of them are imported in
:mod:`panels.multidomain.connections`.

"""
import numpy as np

# Penalties for prescribed displs
# Used to enforce connections
def calc_ku_kv_kw_point_pd(p):
    r"""Penalty constants of a displacement prescribed at a point

    The constants are the membrane stiffnesses of the laminate ``p.lam``,
    which must already exist: ``ku = A11``, ``kv = A22`` and ``kw = A11``.
    The laminate is not rebuilt.

    Parameters
    ----------
    p : :class:`.Shell`
        Panel with the prescribed displacement.

    Returns
    -------
    ku, kv, kw : float
        Penalty constants of `u`, `v` and `w`, force per unit length.

    See Also
    --------
    calc_ku_kv_kw_line_pd_xcte, calc_ku_kv_kw_line_pd_ycte

    Notes
    -----
    The constants multiply the penalty energy of the prescribed displacement
    at the point `(x_p, y_p)`, `\frac{1}{2} (k_u u^2 + k_v v^2 + k_w w^2)`,
    whose stiffness matrix is computed by the kernel ``fkCpd`` of
    :mod:`panels.multidomain.connections.kCpd`. Taking the membrane
    stiffnesses of the laminate makes the penalty scale with the stiffness
    of the panel, as for :func:`.calc_kt_kr`, instead of a fixed high value.

    TODO(author): why is `k_w` also `A_{11}`, and not a constant related to
    the bending stiffness of the laminate?

    """
    A11 = p.lam.A[0, 0]
    A22 = p.lam.A[1, 1]
    ku = A11
    kv = A22
    kw = A11
    # print('penalties in function', ku, kv, kw)
    return ku, kv, kw


def calc_ku_kv_kw_line_pd_xcte(p):
    r"""Penalty constants of a displacement prescribed along a line of
    constant `x`

    The constants are the membrane stiffnesses of the laminate ``p.lam``,
    which must already exist, divided by the width ``p.b`` of the panel:
    ``ku = A11/b``, ``kv = A22/b`` and ``kw = A11/b``. The laminate is not
    rebuilt.

    Parameters
    ----------
    p : :class:`.Shell`
        Panel with the prescribed displacement.

    Returns
    -------
    ku, kv, kw : float
        Penalty constants of `u`, `v` and `w`, force per unit area.

    See Also
    --------
    calc_ku_kv_kw_point_pd, calc_ku_kv_kw_line_pd_ycte

    Notes
    -----
    The constants multiply the penalty energy of the displacement prescribed
    along the line `x = x_p`, `\frac{1}{2} \int_y (k_u u^2 + k_v v^2 + k_w
    w^2) dy`, whose stiffness matrix is computed by the kernel
    ``fkCld_xcte`` of :mod:`panels.multidomain.connections.kCpd`. The
    division by the width ``b`` gives the constants per unit length of the
    line, see :func:`.calc_ku_kv_kw_point_pd`.

    """
    A11 = p.lam.A[0, 0]
    A22 = p.lam.A[1, 1]
    ku = A11 / p.b
    kv = A22 / p.b
    kw = A11 / p.b
    return ku, kv, kw


def calc_ku_kv_kw_line_pd_ycte(p):
    r"""Penalty constants of a displacement prescribed along a line of
    constant `y`

    The constants are the membrane stiffnesses of the laminate ``p.lam``,
    which must already exist, divided by the length ``p.a`` of the panel:
    ``ku = A11/a``, ``kv = A22/a`` and ``kw = A11/a``. The laminate is not
    rebuilt.

    Parameters
    ----------
    p : :class:`.Shell`
        Panel with the prescribed displacement.

    Returns
    -------
    ku, kv, kw : float
        Penalty constants of `u`, `v` and `w`, force per unit area.

    See Also
    --------
    calc_ku_kv_kw_point_pd, calc_ku_kv_kw_line_pd_xcte

    Notes
    -----
    The constants multiply the penalty energy of the displacement prescribed
    along the line `y = y_p`, `\frac{1}{2} \int_x (k_u u^2 + k_v v^2 + k_w
    w^2) dx`, whose stiffness matrix is computed by the kernel
    ``fkCld_ycte`` of :mod:`panels.multidomain.connections.kCpd`. The
    division by the length ``a`` gives the constants per unit length of the
    line, see :func:`.calc_ku_kv_kw_point_pd`.

    """
    A11 = p.lam.A[0, 0]
    A22 = p.lam.A[1, 1]
    ku = A11 / p.a
    kv = A22 / p.a
    kw = A11 / p.a
    return ku, kv, kw

def build_panel_lam(panel):
    r"""Build the laminate ``panel.lam`` with :meth:`.Shell._rebuild`, which
    raises a ``ValueError`` when the stacking sequence, the ply thicknesses
    or the properties of the laminae are missing

    The laminate is rebuilt at every call. :meth:`.Shell._rebuild` also sets
    the other attributes derived from the input of the panel, e.g.
    ``plyts``, ``laminaprops``, ``rhos`` and ``size``. Nothing is returned.

    Parameters
    ----------
    panel : :class:`.Shell`
        Panel whose laminate is built.

    """
    panel._rebuild()

# Penalties for connections
def calc_kt_kr(p1, p2, connection_type):
    r"""Calculate translation and rotation penalty constants

    The laminates of both panels are first rebuilt with
    :func:`.build_panel_lam`. The constants are computed from the
    membrane and bending stiffnesses `A_{11}`, `A_{22}`, `D_{11}`, `D_{22}`
    of the laminates and from their thicknesses ``h``: with ``'xcte'`` the
    terms `A_{11}`, `D_{11}` of both panels are used, with ``'ycte'`` the
    terms `A_{22}`, `D_{22}`, with ``'xcte-ycte'`` the terms 11 of ``p1``
    and 22 of ``p2``, and with ``'ycte-xcte'`` the opposite. With
    ``'bot-top'`` the translation constant uses `A_{11}` and is also
    divided by ``min(p1.a, p1.b)``, and no rotation constant is computed.
    The comparison with ``connection_type`` is case-insensitive.

    Parameters
    ----------
    p1 : :class:`.Panel`
        First panel.
    p2 : :class:`.Panel`
        Second panel.
    connection_type : str
        One of the types:
            - 'xcte'
            - 'ycte'
            - 'bot-top': ``'SB'``, ``p1`` and ``p2`` on the positive and
              negative sides of the interface along `z`
            - 'xcte-ycte': to a 90° connection
            - 'ycte-xcte': to a 90° connection

    Returns
    -------
    kt, kr : tuple
        A tuple with both values. ``kt`` is a force per unit area, and a
        force per unit volume with ``'bot-top'``; ``kr`` is a force, and
        ``None`` with ``'bot-top'``. The function returns ``None``, not a
        tuple, when ``connection_type`` is none of the types above.

    Notes
    -----
    Theoretically, the penalty stiffnesses kt and kr can be arbitrarily high in
    order to impose the energy penalty. However, the use of high values is
    associated with numerical instabilities such that one should choose the
    penalty stiffnesses that are just high enough to impose the proper
    penalties, but not excessively high. In the current study it is proposed to
    calculate kt and kr based on laminate properties of the panels being
    connected, instead of using fixed high values, a common practice in the
    literature [1]_ .

    For details on how to derive these equations, see [1]_ (MD paper eqn
    34-40). With the stiffnesses `A^{(1)}, A^{(2)}` and `D^{(1)}, D^{(2)}`
    of ``p1`` and ``p2`` selected by ``connection_type``, e.g. `A_{11}` and
    `D_{11}` for ``'xcte'``, and the thicknesses `h_1, h_2`:

    .. math::

        k_t = \frac{4 A^{(1)} A^{(2)}}{(A^{(1)} + A^{(2)}) (h_1 + h_2)},
        \qquad
        k_r = \frac{4 D^{(1)} D^{(2)}}{(D^{(1)} + D^{(2)}) (h_1 + h_2)}

    i.e. the harmonic mean of the stiffnesses of the two laminates divided
    by their mean thickness `(h_1 + h_2)/2`. The translation constant
    penalizes a difference of displacement along a line, a force per unit
    area, and the rotation constant a difference of rotation along a line,
    a force. For ``'bot-top'``, the penalty is integrated over an area, and
    `k_t` is further divided by `\min(a_1, b_1)` of ``p1``, a force per unit
    volume.

    TODO(author): confirm the choice of `\min(a_1, b_1)` as the length
    scale of the area connection ``'bot-top'``.

    The constants are a scaling of the penalty terms, calculated from the
    laminates ``p1.lam`` and ``p2.lam``, not from the stiffness integrated by
    the kernels: they ignore ``Shell.force_orthotropic_laminate`` and
    ``Shell.ABDnxny``. They depend only on the diagonal terms `A_{11}`,
    `A_{22}`, `D_{11}`, `D_{22}` and on the thickness, which forcing the
    orthotropy does not change, whereas with ``ABDnxny`` the penalty
    constants may have to be scaled, e.g. by multiplying the returned values.

    References
    ----------
    .. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
       models to Address Linear Buckling and Vibration of Stiffened Composite
       Panels with Debonding Defect, Composite Structures, 2017.
       https://doi.org/10.1016/j.compstruct.2016.10.026

    """

    build_panel_lam(p1)
    build_panel_lam(p2)

    A11_p1 = p1.lam.A[0, 0]
    A11_p2 = p2.lam.A[0, 0]
    D11_p1 = p1.lam.D[0, 0]
    D11_p2 = p2.lam.D[0, 0]
    A22_p1 = p1.lam.A[1, 1]
    A22_p2 = p2.lam.A[1, 1]
    D22_p1 = p1.lam.D[1, 1]
    D22_p2 = p2.lam.D[1, 1]
    hp1 = p1.lam.h
    hp2 = p2.lam.h
    if connection_type.lower() == 'xcte':
        kt = 4*A11_p1*A11_p2/((A11_p1 + A11_p2)*(hp1 + hp2))
        kr = 4*D11_p1*D11_p2/((D11_p1 + D11_p2)*(hp1 + hp2))
        return kt, kr
    elif connection_type.lower() == 'ycte':
        kt = 4*A22_p1*A22_p2/((A22_p1 + A22_p2)*(hp1 + hp2))
        kr = 4*D22_p1*D22_p2/((D22_p1 + D22_p2)*(hp1 + hp2))
        return kt, kr
    elif connection_type.lower() == 'bot-top':
        kt = 4*A11_p1*A11_p2/((A11_p1 + A11_p2)*(hp1 + hp2)) / min(p1.a, p1.b)
        kr = None
        return kt, kr
    elif connection_type.lower() == 'xcte-ycte':
        kt = 4*A11_p1*A22_p2 / ((A11_p1+A22_p2)*(hp1+hp2))
        kr = 4*D11_p1*D22_p2 / ((D11_p1+D22_p2)*(hp1+hp2))
        return kt, kr
    elif connection_type.lower() == 'ycte-xcte':
        kt = 4*A22_p1*A11_p2 / ((A22_p1+A11_p2)*(hp1+hp2))
        kr = 4*D22_p1*D11_p2 / ((D22_p1+D11_p2)*(hp1+hp2))
        return kt, kr


def calc_kw_tsl(pA, pB=None, tsl_type='bilinear', k_i=None, del_d=None,
                tau_o=None, G1c=None):
    r"""Stiffness and damage of a bilinear traction-separation law

    The damage at each separation of ``del_d`` is computed from the
    separation at damage onset ``tau_o/k_i`` and the separation at complete
    failure ``2*G1c/tau_o``, and the stiffness is ``k_i*(1 - damage)``.
    Negative separations keep the pristine stiffness ``k_i``; there is no
    additional stiffness against interpenetration. The damage history is not
    considered here, see :meth:`.MultiDomain.calc_k_dmg`.

    The panels ``pA`` and ``pB`` are not used in the computation. Where
    ``del_d`` is close to zero the stiffness is ``k_i`` and the damage zero.
    Above the separation at complete failure the stiffness is zero and the
    damage one.

    Parameters
    ----------
    pA : :class:`.Shell`
        Panel on the positive side of the interface along the `z` axis of
        both panels, ``p1`` of the connection.
    pB : :class:`.Shell`, optional
        Panel on the negative side of the interface along `z`, ``p2``.
        Default ``None``.
    tsl_type : str, optional
        Type of the traction-separation law, only ``'bilinear'`` is
        available. Default ``'bilinear'``.
    k_i : float
        Pristine stiffness of the interface, `k_o`, force/length**3.
        Required, the default ``None`` raises a ``ValueError``.
    del_d : np.ndarray
        Normal separation between the panels at each point of the grid,
        length. Array of any shape. Required, the default ``None`` raises a
        ``ValueError``.
    tau_o : float
        Traction at damage onset, force/length**2. Required, the default
        ``None`` raises a ``ValueError``.
    G1c : float
        Critical energy release rate in mode I, force/length. Required, the
        default ``None`` raises a ``ValueError``.

    Returns
    -------
    kw_tsl : np.ndarray
        Stiffness at each point of the grid, force/length**3, with the shape
        of ``del_d``.
    dmg_index : np.ndarray
        Damage at each point of the grid, between ``0`` and ``1``,
        dimensionless, with the shape of ``del_d``.

    Raises
    ------
    ValueError
        When ``tsl_type`` is not ``'bilinear'``; when ``k_i``, ``tau_o`` or
        ``G1c`` is ``None``; when ``del_d`` is ``None``.

    Notes
    -----
    In the bilinear law the traction grows linearly with the stiffness `k_i`
    up to the traction `\tau_o`, at the separation `\Delta_o = \tau_o/k_i`,
    and then decreases linearly to zero at the separation `\Delta_f = 2
    G_{Ic}/\tau_o`, such that the area under the law is `G_{Ic}` [1]_. With
    the stiffness `k_i (1 - \hat d)`, the damage of the bilinear law at the
    separation `\Delta` is:

    .. math::

        \hat d(\Delta) = \begin{cases}
            0 & \Delta \le \Delta_o \\
            \frac{\Delta_f (\Delta - \Delta_o)}{(\Delta_f - \Delta_o) \Delta}
              & \Delta_o < \Delta < \Delta_f \\
            1 & \Delta \ge \Delta_f
        \end{cases}

    and the stiffness of the softening branch is `k_i \Delta_o (\Delta_f -
    \Delta)/((\Delta_f - \Delta_o) \Delta)`. The damage is driven by the
    normal separation only, in mode I, with the critical energy release
    rate `G_{Ic}`. The irreversibility of the damage, `d = \max(d^h, \hat
    d)` with the damage `d^h` of the last converged increment, is applied by
    :meth:`.MultiDomain.calc_k_dmg`.

    References
    ----------
    .. [1] D'Souza, N., Multi-Domain Semi-Analytical Cohesive Zone Approach,
       MSc thesis, Delft University of Technology, 2024.
       https://resolver.tudelft.nl/uuid:4709f888-c032-4d9f-97e3-402dfdd47cf1

    """
    if tsl_type != 'bilinear':
        raise ValueError(f'tsl_type {tsl_type!r} not available, use "bilinear"')
    if k_i is None or tau_o is None or G1c is None:
        raise ValueError('k_i, tau_o and G1c are required')
    if del_d is None:
        raise ValueError('Out of plane separation field is required')

    # Cohesive Law Parameters
    del_o = tau_o/k_i
    del_f = 2*G1c/tau_o # [mm] - Separation at complete failure (damage variable = 1)

    # Stiffness when there is damage
    # Ignoring div by zero error when del_d is 0 for calc of damage. Its overwritten below so its fine :)
    with np.errstate(divide='ignore'):
        k_dmg = k_i*del_o*np.divide((del_f - del_d), (del_f - del_o)*del_d)

    # Overwriting stiffnesses where separation is zero
    k_dmg[np.isclose(del_d, 0)] = k_i

    # Filter maps/mask to enable the original stiffness to be added at the same time
    f_i = del_d < del_o         # Filter mask for initial stiffness (before damage)
    f_dmg = del_d >= del_o      #                 degraded stiffness (after damage is created)
    f_f = del_d <= del_f        #                 failure - ensures k = 0 after failure

    kw_tsl = np.multiply(f_f, f_i*k_i + np.multiply(f_dmg, k_dmg))

    # Calculating Damage Index
    with np.errstate(divide='ignore'):
        dmg_index_after_dmg = np.divide((del_d - del_o)*del_f , (del_f - del_o)*del_d)

    dmg_index_after_dmg[np.isclose(del_d, 0)] = 0

    # Applies a filter mask to remove the terms when del_d < del_o, setting them to 0
    dmg_index = np.multiply(f_dmg, dmg_index_after_dmg)
    dmg_index[del_d > del_f] = 1

    return kw_tsl, dmg_index
