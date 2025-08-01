from composites import laminated_plate
import numpy as np

# Penalties for prescribed displs
# Used to enforce connections
def calc_ku_kv_kw_point_pd(p):
    A11 = p.lam.A[0, 0]
    A22 = p.lam.A[1, 1]
    ku = A11
    kv = A22
    kw = A11
    # print('penalties in function', ku, kv, kw)
    return ku, kv, kw


def calc_ku_kv_kw_line_pd_xcte(p):
    A11 = p.lam.A[0, 0]
    A22 = p.lam.A[1, 1]
    ku = A11 / p.b
    kv = A22 / p.b
    kw = A11 / p.b
    return ku, kv, kw


def calc_ku_kv_kw_line_pd_ycte(p):
    A11 = p.lam.A[0, 0]
    A22 = p.lam.A[1, 1]
    ku = A11 / p.a
    kv = A22 / p.a
    kw = A11 / p.a
    return ku, kv, kw

def build_panel_lam(panel):
    panel._rebuild()
    if panel.lam is not None:
        return
    if panel.stack is None:
        raise ValueError('Panel defined without stacking sequence')
    if panel.plyts is None:
        raise ValueError('Panel defined without ply thicknesses')
    if panel.laminaprops is None:
        raise ValueError('Panel defined without laminae properties')
    panel.lam = laminated_plate(panel.stack, plyts=panel.plyts,
            laminaprops=panel.laminaprops)
    return

# Penalties for connections
def calc_kt_kr(p1, p2, connection_type):
    """Calculate translation and rotation penalty constants

    For details on how to derive these equations, see
    [castro2017Multidomain]_ (MD paper eqn 34-40)

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
            - 'bot-top'
            - 'xcte-ycte': to a 90° connection
            - 'ycte-xcte': to a 90° connection

    Returns
    -------
    kt, kr : tuple
        A tuple with both values.

    Note
    ----
    Theoretically, the penalty stiffnesses kt and kr can be arbitrarily high in
    order to impose the energy penalty. However, the use of high values is
    associated with numerical instabilities such that one should choose the
    penalty stiffnesses that are just high enough to impose the proper
    penalties, but not excessively high. In the current study it is proposed to
    calculate kt and kr based on laminate properties of the panels being
    connected, instead of using fixed high values, a common practice in the
    literature [castro2017Multidomain]_ .

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


def calc_kw_tsl(pA, pB=None, tsl_type=None, k_i=1.e4, k_ipen=1.e6, del_d=None,
                tau_o=67, G1c=1.12):
    r"""Calculate out of plane stiffness of the damaged region (where the traction separation law exists)

    Parameters
    ----------
    pA : :class:`.Shell`
        Top panel.
    pB : :class:`.Shell`
        Bottom panel.
    tsl_type : string
        Type of TSL to be used. Possible options are: 'linear' (no softening),
        or 'bilinear' (with linear softening).
    k_i : float
        Pristine out-of-plane stiffness, in other words, when there is no damage.
    k_ipen : float
        Out-of-plane penalty stiffness to prevent interpenetration between
        ``pA`` and ``pB``.
    del_d : np.ndarray
        A 2D np.array() corresponding to out-of-plane separation field. This is
        the out of plane separation between panels, at each point of the input
        grid.
    tau_o : float
        Traction at damage onset. The default value is in MPa units.
    G1c : float
        Critical fracture energy release rate in mode I. The default value is
        in kJ/m^2 = N/mm units.

    Returns
    -------
    kw_tsl : float
        The out of plane stiffness, when ``tsl_type == 'bilinear'``, this is
        a map of stiffnesses at each point of the input grid.
    dmg_index : np.ndarray
        The calculated damage at each point across the input grid. Note that
        this is only valid when ``tsl_type == 'bilinear'``.

    """
    if tsl_type == 'linear':
        tsl_input = 1e4 # N/mm^3 (from Fig 8a - https://doi.org/10.1016/j.compositesa.2022.107101 - faulty !!!!)
        kw_tsl = tsl_input  # N/mm^3 (same as kt for SB connection)

        return kw_tsl, None

    elif tsl_type == 'bilinear':
        if del_d is None:
            raise ValueError('Out of plane separation field is required')

        # Cohesive Law Parameters
        del_o = tau_o/k_i
        del_f = 2*G1c/tau_o # [mm] - Separation at complete failure (damage variable = 1)

        k_dmg = np.zeros_like(del_d)
        # Stiffness when there is damage
        # Ignoring div by zero error when del_d is 0 for calc of damage. Its overwritten below so its fine :)
        with np.errstate(divide='ignore'):
            k_dmg = k_i*del_o*np.divide((del_f - del_d), (del_f - del_o)*del_d)

        # Overwriting stiffnesses where separation is zero
        k_dmg[np.isclose(del_d, 0)] = k_i

        # But it should not matter because youre only checking cases when del_d = 0 i.e. initial value so its already init by 0
        # So no numerical error will exist

        # Filter maps/mask to enable the original stiffness to be added at the same time
        f_i = del_d < del_o         # Filter mask for initial stiffness (before damage)
        f_dmg = del_d >= del_o      #                 degraded stiffness (after damage is created)
            # Can also be used as a damage map - True = damage exists
        f_f = del_d <= del_f        #                 failure - ensures k = 0 after failure

        # Prevention of interpenetration between panels
            # Modification in TSL - negative separation has very high positive k
            #    /\             ^ traction
            #   /  \            -> separation
            #  |
            #  |
            #  |

        # Filter mask for interpenetration stiffness
        f_ipen = del_d < 0

        # Overall k that takes into account inital k as well
        kw_tsl = np.multiply(np.invert(f_ipen), np.multiply(f_f, f_i*k_i + np.multiply(f_dmg,k_dmg))) + f_ipen*k_ipen

        # Calculating Damage Index
        with np.errstate(divide='ignore'):
            dmg_index_after_dmg = np.divide((del_d - del_o)*del_f , (del_f - del_o)*del_d)

        dmg_index_after_dmg[np.isclose(del_d, 0)] = 0

        # Applies a filter mask to remove the terms when del_d < del_o, setting them to 0
        dmg_index = np.multiply(f_dmg, dmg_index_after_dmg)
        dmg_index[del_d > del_f] = 1

        return kw_tsl, dmg_index

    else:
        raise ValueError('Invalid TSL type')
