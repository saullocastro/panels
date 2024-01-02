from composites import laminated_plate


def calc_ku_kv_kw_point_pd(p):
    A11 = p.lam.A[0, 0]
    A22 = p.lam.A[1, 1]
    ku = A11 / p.lam.h / p.a / p.b
    kv = A22 / p.lam.h / p.a / p.b
    kw = A11 / p.lam.h / p.b / p.b
    return ku, kv, kw


def calc_ku_kv_kw_line_pd_xcte(p):
    A11 = p.lam.A[0, 0]
    A22 = p.lam.A[1, 1]
    ku = A11 / p.lam.h / p.b
    kv = A22 / p.lam.h / p.b
    kw = A11 / p.lam.h / p.b
    return ku, kv, kw


def calc_ku_kv_kw_line_pd_ycte(p):
    A11 = p.lam.A[0, 0]
    A22 = p.lam.A[1, 1]
    ku = A11 / p.lam.h / p.a
    kv = A22 / p.lam.h / p.a
    kw = A11 / p.lam.h / p.a
    return ku, kv, kw


def calc_kt_kr(p1, p2, connection_type):
    """Calculate translation and rotation penalty constants

    For details on how to derive these equations, see
    [castro2017AssemblyModels]_.

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
    Theoretically, the penalty stiffnesses kt and kr can be arbitrarily high in order to impose the energy penalty. However, the
    use of high values is associated with numerical instabilities such that one should choose the penalty stiffnesses that are just high
    enough to impose the proper penalties, but not excessively high. In the current study it is proposed to calculate kt and kr based
    on laminate properties of the panels being connected, instead of using fixed high values, a common practice in the literature.
    [castro2017AssemblyModels]
    """
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

