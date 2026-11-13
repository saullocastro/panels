r"""Geometric stiffness matrix from a constant or a computed prestress

Linear buckling of a simply supported symmetric cross-ply plate under
uniaxial compression for five models, with the geometric stiffness of a
constant ``Nxx`` and with that of the pre-buckling static solution of a load
distributed along ``x = a`` (``calc_kG(c=c0)``), also when the laminate
stiffness ``ABDnxny`` is given at the integration points. The in-plane
boundary conditions leave the pre-buckling state uniform, so every first
load multiplier equals the closed-form buckling load of a simply supported
plate of rectangular orthotropy (Leissa, 1969, eqs. (9.24), (10.1) and
(10.43)-(10.46)).

Notes
-----
The laminate ``[0, 90, 0]s`` is symmetric with `D_{16} = D_{26} = 0`, a
plate of rectangular orthotropy with `D_x = D_{11}`, `D_y = D_{22}` and
`D_{xy} = D_{12} + 2 D_{66}` (eq. (9.24) of [1]_). With ``u = 0`` at ``x =
0``, ``v = 0`` at ``y = 0`` and the other in-plane displacements free, the
load `N_{xx} = -1` applied at ``x = a`` gives the uniform pre-buckling state
`N_{xx} = -1`, `N_{yy} = N_{xy} = 0`, with `u` and `v` linear in `x` and
`y`, represented exactly. The deflection `w = \sin(m \pi x/a) \sin(n \pi
y/b)` (eq. (10.43)) solves eq. (10.1) of [1]_ with `\omega = 0` when

.. math::

    \lambda N_{xx} \left(\frac{m \pi}{a}\right)^2 + D_{11}
    \left(\frac{m \pi}{a}\right)^4 + 2 D_{xy} \left(\frac{m \pi}{a}\right)^2
    \left(\frac{n \pi}{b}\right)^2 + D_{22} \left(\frac{n \pi}{b}\right)^4 = 0

the orthotropic form of eq. (10.46); the reference is the lowest `\lambda`
over `m, n`, `\lambda = 3.8433` here (`m = 2`, `n = 1`). With ``m = 8``,
``n = 9`` terms all models are within `2 \times 10^{-5}` of it (``m = n =
10``: `10^{-6}`); the tolerance is `10^{-4}`. The models with transverse
shear differ from the classical value by less than `10^{-6}` (`a/h =
10^4`). The two cylinders have ``r = 1e8`` and give the plate values.

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, National
   Aeronautics and Space Administration, Washington, DC, 1969.
   https://ntrs.nasa.gov/citations/19700009156

"""
import sys
sys.path.append('../..')

import pytest
import numpy as np

from structsolve import solve, lb
from panels.shell import Shell


def lb_ss_orthotropic(p, Nxx, Nyy, mmax=10):
    """Closed-form buckling multiplier of a simply supported plate of
    rectangular orthotropy under constant ``Nxx`` and ``Nyy``"""
    D = p.ABD[3:6, 3:6]
    D11, D22, Dxy = D[0, 0], D[1, 1], D[0, 1] + 2*D[2, 2]
    lambdas = []
    for m in range(1, mmax + 1):
        for n in range(1, mmax + 1):
            alpha = m*np.pi/p.a
            beta = n*np.pi/p.b
            K = D11*alpha**4 + 2*Dxy*alpha**2*beta**2 + D22*beta**4
            G = Nxx*alpha**2 + Nyy*beta**2
            if G < 0:
                lambdas.append(-K/G)
    return min(lambdas)


def build_plate(model):
    p = Shell()
    p.a = 8.
    p.b = 4.
    p.r = 1.e8
    p.stack = [0, 90, 0, 0, 90, 0]
    p.plyt = 1e-3*0.125
    p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    p.model = model
    p.m = 8
    p.n = 9
    # u = 0 at x = 0, v = 0 at y = 0, other in-plane displacements free
    p.x2u = 1
    p.y1u = 1
    p.y2u = 1
    p.x1v = 1
    p.x2v = 1
    p.y2v = 1
    return p


def test_panel_fkG_num():
    for model in ['plate_clpt_donnell',
                  'cylshell_clpt_donnell',
                  'cylshell_clpt_sanders',
                  'plate_fsdt_donnell',
                  'plate_tsdt_donnell']:
        print('Checking calc_kG(c=c0) for model {0}'.format(model))
        p = build_plate(model)
        Nxx = -1.

        # from constant Nxx
        p.Nxx = Nxx
        k0 = p.calc_kC()
        kG0 = p.calc_kG()
        lambda_ref = lb_ss_orthotropic(p, Nxx, 0.)
        eigvals, eigvecs = lb(k0, kG0, silent=True)
        assert np.isclose(eigvals[0], lambda_ref, rtol=1e-4)

        # from pre-buckling static solution
        p.Nxx = 0.
        p.add_distr_load_fixed_x(p.a, lambda y: Nxx, None, None)
        fext = p.calc_fext()
        c0 = solve(k0, fext)
        kG = p.calc_kG(c=c0)
        eigvals, eigvecs = lb(k0, kG, silent=True)
        assert np.isclose(eigvals[0], lambda_ref, rtol=1e-4)


@pytest.mark.slow
def test_panel_fkG_num_ABDnxny():
    for model in ['plate_clpt_donnell',
                  'cylshell_clpt_donnell',
                  'cylshell_clpt_sanders',
                  'plate_fsdt_donnell',
                  'plate_tsdt_donnell']:
        print('Checking calc_kG(c=c0, ABDnxny) for model {0}'.format(model))
        p = build_plate(model)
        Nxx = -1.

        # from pre-buckling static solution
        p.Nxx = 0.
        p.add_distr_load_fixed_x(p.a, lambda y: Nxx, None, None)
        fext = p.calc_fext()
        k0 = p.calc_kC()
        c0 = solve(k0, fext)
        lambda_ref = lb_ss_orthotropic(p, Nxx, 0.)
        kG = p.calc_kG(c=c0)
        eigvals, eigvecs = lb(k0, kG, silent=True)
        assert np.isclose(eigvals[0], lambda_ref, rtol=1e-4)

        nx = 9
        ny = 9
        ABDnxny = p.ABD
        kG = p.calc_kG(c=c0, ABDnxny=ABDnxny, nx=nx, ny=ny)
        eigvals, eigvecs = lb(k0, kG, silent=True)
        assert np.isclose(eigvals[0], lambda_ref, rtol=1e-4)

        nx = 12
        ny = 10
        ABDnxny = np.array([[p.ABD]*ny]*nx)
        kG = p.calc_kG(c=c0, ABDnxny=ABDnxny, nx=nx, ny=ny)
        eigvals, eigvecs = lb(k0, kG, silent=True)
        assert np.isclose(eigvals[0], lambda_ref, rtol=1e-4)


#TODO  test_panel_fkG_num_ckL()


if __name__ == '__main__':
    test_panel_fkG_num()
    test_panel_fkG_num_ABDnxny()
