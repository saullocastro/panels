r"""Linear buckling of annular plates under a torque,
``'annular_clpt_donnell'``

A full annulus loaded by a tangential force on the outer edge, the inner
edge fixed in its plane: the membrane state of the linear static case
against `N_{\rho\theta} = T/(2 \pi \rho^2)`, and the buckling torques of
the linear buckling about it, ``calc_kG(c)``, against Irie, Yamada and
Tsujino (1983), Table 1, as reprinted by Bagheri, Kiani and Eslami (2019),
Table 1, for the eight combinations of clamped, simply supported and free
edges and the radius ratios `\beta = 0.3` and `0.5`; and the clamped
annulus for the three annular models.

Notes
-----
A torque `T` applied to one edge of an annulus and reacted on the other
edge gives the membrane state `N_{\rho\rho} = N_{\theta\theta} = 0` and

.. math::

    N_{\rho\theta} = \frac{T}{2 \pi \rho^2}

from the equilibrium of the moments about the centre of the part of the
plate inside the radius `\rho`, whatever the in-plane edge conditions,
Bagheri, Kiani and Eslami [2]_, Eqs. (10) and (12). In panels `N_{xy} =
-N_{\rho\theta}`, the axis `x` pointing inwards. The force per unit length
`q = T/(2 \pi a^2)` along `\boldsymbol{e}_\theta` on the outer edge of
radius `a`, where `y` is the arc length, applies the torque `T`; the inner
edge, of radius `b = \beta a`, has `u = v = 0`, as in [2]_, Eq. (10).

Irie, Yamada and Tsujino [1]_ tabulate the buckling torque in the
dimensionless form

.. math::

    \tau_B = \frac{T_B}{2 \pi D}, \qquad D = \frac{E h^3}{12 (1 - \nu^2)}

for `\nu = 0.3`, with the edge conditions named inner-outer: clamped (C,
`w = w_{,\rho} = 0`), simply supported (SS, `w = M_\rho = 0`) and free
(F, `M_\rho = V_\rho = 0`). Bagheri, Kiani and Eslami [2]_, Table 1, reprint
these values for `\beta \le 0.7` beside their own (generalized differential
quadrature, `h/a = 0.001`), which agree with them within 0.4 % for `\beta
= 0.3` and `0.5`. The buckling modes are spirals with `n` waves along the
circumference; the full annulus is made of two half annuli with free
radial edges connected with ``'SSycte'``. The linear static case and the
geometric stiffness of its membrane state, ``calc_kG(c)``, are computed
by :meth:`.MultiDomain.linear_buckling`; the critical torques come in
pairs of opposite sign, and the least absolute value is taken.

References
----------
.. [1] Irie, T., Yamada, G., and Tsujino, M., "Buckling loads of annular
   plates subjected to a torque", Journal of Sound and Vibration, Vol. 86,
   No. 1, 1983, pp. 145-146, Table 1.
   https://doi.org/10.1016/0022-460X(83)90951-3
.. [2] Bagheri, H., Kiani, Y., and Eslami, M. R., "Buckling of moderately
   thick annular plates subjected to torque", Archive of Mechanical
   Engineering, Vol. 66, No. 2, 2019, pp. 209-227, Eqs. (10), (12) and
   Table 1. https://doi.org/10.24425/ame.2019.128445

"""
import numpy as np
import pytest

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels import modelDB


E, NU, H = 70e9, 0.3, 1e-3
A = 1.
D = E*H**3/(12*(1 - NU**2))
#: flags (w, w_rho) of the clamped (C), simply supported (SS) and free (F)
#: edges, 0 for a zero value
EDGE = {'C': (0, 0), 'SS': (0, 1), 'F': (1, 1)}
#: Irie, Yamada and Tsujino (1983), Table 1, `\tau_B` for `\beta = 0.3` and
#: `0.5`, edges (inner, outer), `\nu = 0.3`
IRIE = {
    ('SS', 'F'): (7.319, 17.26),
    ('F', 'SS'): (7.330, 19.03),
    ('F', 'C'): (10.60, 32.05),
    ('C', 'F'): (12.80, 35.80),
    ('SS', 'SS'): (37.10, 110.6),
    ('SS', 'C'): (47.18, 142.8),
    ('C', 'SS'): (51.06, 150.1),
    ('C', 'C'): (61.97, 186.1),
    }
BETAS = (0.3, 0.5)
CASES = [(edges, beta) for edges in IRIE for beta in BETAS]
MODELS = [k for k in modelDB.db if k.startswith('annular_clpt_')]


def annulus(beta, inner, outer, T, model='annular_clpt_donnell', m=10,
            n=16):
    r"""Full annulus, two half annuli connected with ``'SSycte'``, with the
    torque `T` applied on the outer edge"""
    halves = []
    for _ in range(2):
        s = Shell(model=model, a=A*(1 - beta), b=np.pi*A, r=A, m=m, n=n,
                  plyt=H, stack=[0], laminaprop=(E, NU))
        for e in ('y1', 'y2'):
            for f in ('u', 'v', 'w', 'ur', 'vr', 'wr'):
                setattr(s, e + f, 1)
        s.x1u = s.x1v = 1
        s.x2u = s.x2v = 0
        s.x1w, s.x1wr = EDGE[outer]
        s.x2w, s.x2wr = EDGE[inner]
        s.add_distr_load_fixed_x(0., funcy=lambda y: T/(2*np.pi*A**2)
                                 + 0*y)
        halves.append(s)
    p1, p2 = halves
    conn = [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0),
            dict(p1=p1, p2=p2, func='SSycte', ycte1=0, ycte2=p2.b)]
    return MultiDomain([p1, p2], conn)


def tau_B(beta, inner, outer, **kwargs):
    r"""Least `\tau_B = T_B/(2 \pi D)` of the annulus"""
    md = annulus(beta, inner, outer, T=2*np.pi*D, **kwargs)
    eigvals, _ = md.linear_buckling(num_eigvalues=6, silent=True)
    return np.abs(eigvals).min()


def test_membrane_state():
    r"""The linear static case gives `N_{xy} = -T/(2 \pi \rho^2)`, within
    0.5 %, and `N_{xx}`, `N_{yy}` below 0.5 % of it"""
    T = 1.
    beta = 0.3
    md = annulus(beta, 'C', 'C', T)
    c = md.linear_static()
    s = md.panels[0]
    xs, ys = np.meshgrid(np.linspace(0.1, 0.9, 5)*s.a,
                         np.linspace(0.1, 0.9, 5)*s.b)
    fields = s.stress(c[s.col_start: s.col_end], xs=xs, ys=ys,
                      NLgeom=False)[1]
    ref = -T/(2*np.pi*(A - xs)**2)
    assert np.allclose(fields['Nxy'], ref, rtol=5e-3)
    assert np.abs(fields['Nxx']).max() < 5e-3*np.abs(ref).max()
    assert np.abs(fields['Nyy']).max() < 5e-3*np.abs(ref).max()


@pytest.mark.parametrize('edges, beta', CASES)
def test_irie1983_table1(edges, beta):
    r"""Least `\tau_B` against Irie, Yamada and Tsujino (1983), Table 1,
    within 1 %"""
    ref = IRIE[edges][BETAS.index(beta)]
    assert np.isclose(tau_B(beta, *edges), ref, rtol=1e-2)


@pytest.mark.parametrize('model', MODELS)
def test_clamped_models(model):
    r"""Clamped annulus, `\beta = 0.3`, for the three annular models,
    against Irie, Yamada and Tsujino (1983), Table 1, `\tau_B = 61.97`,
    within 1 %"""
    assert np.isclose(tau_B(0.3, 'C', 'C', model=model), 61.97, rtol=1e-2)
