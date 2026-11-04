r"""Linear buckling with the prestress of a prescribed edge displacement

The displacement of the edge ``x = a`` (``u``) or ``y = b`` (``v``) of a
simply supported symmetric cross-ply plate is prescribed with a penalty
stiffness of the multi-domain connections, the linear static solution gives
the prestress and the buckling analysis runs. The in-plane boundary
conditions leave the pre-buckling state uniform, so the first load
multiplier equals the closed-form buckling load of a simply supported plate
of rectangular orthotropy (Leissa, 1969, eqs. (9.24), (10.1) and
(10.43)-(10.46)) divided by the membrane force that the prescribed
displacement produces through the penalty spring.

Notes
-----
The laminate ``[0, 90, 0]s`` is symmetric with `D_{16} = D_{26} = 0`, a
plate of rectangular orthotropy with `D_x = D_{11}`, `D_y = D_{22}` and
`D_{xy} = D_{12} + 2 D_{66}` (eq. (9.24) of [1]_). For the edge ``x = a``,
``u = 0`` at ``x = 0``, ``v = 0`` at ``y = 0`` and the other in-plane
displacements are free; the prescribed `u_p` at ``x = a`` acts through the
penalty constant `k_u = A_{11}/b` per unit length
(:func:`.calc_ku_kv_kw_line_pd_xcte`) in series with the plate, whose
uniform state `\varepsilon_{xx} = a_{11} N_{xx}`, `\boldsymbol{a} =
\boldsymbol{A}^{-1}`, gives `u(a) = a_{11} N_{xx} a`. Equilibrium of the
spring, `N_{xx} = k_u (u_p - u(a))`, gives

.. math::

    N_{xx} = \frac{u_p}{1/k_u + a_{11} a}

and the first load multiplier is `\lambda = N_{cr}/|N_{xx}|`, with `N_{cr}`
the lowest root over `m, n` of

.. math::

    N_{cr} \left(\frac{m \pi}{a}\right)^2 = D_{11}
    \left(\frac{m \pi}{a}\right)^4 + 2 D_{xy} \left(\frac{m \pi}{a}\right)^2
    \left(\frac{n \pi}{b}\right)^2 + D_{22} \left(\frac{n \pi}{b}\right)^4

the orthotropic form of eq. (10.46) of [1]_, from eq. (10.1) with the
deflection of eq. (10.43). The edge ``y = b`` is the same with `x` and `y`,
`k_v = A_{22}/a` (:func:`.calc_ku_kv_kw_line_pd_ycte`) and `a_{22} b`.
With ``m = 8``, ``n = 9`` terms the models are within `2 \times 10^{-5}` of
the reference (``m = n = 10``: `10^{-7}`); the tolerance is `10^{-4}`. The
cylinder has ``r = 1e8`` and gives the plate values.

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
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.multidomain.connections import calc_ku_kv_kw_line_pd_xcte, calc_ku_kv_kw_line_pd_ycte
from panels.multidomain.connections import fkCld_xcte, fkCld_ycte
from panels.plot_shell import plot_shell


def Ncr_ss_orthotropic(p, direction, mmax=10):
    """Closed-form buckling load of a simply supported plate of rectangular
    orthotropy under uniaxial compression along ``x`` or ``y``"""
    D = p.ABD[3:6, 3:6]
    D11, D22, Dxy = D[0, 0], D[1, 1], D[0, 1] + 2*D[2, 2]
    Ns = []
    for m in range(1, mmax + 1):
        for n in range(1, mmax + 1):
            alpha = m*np.pi/p.a
            beta = n*np.pi/p.b
            K = D11*alpha**4 + 2*Dxy*alpha**2*beta**2 + D22*beta**4
            Ns.append(K/(alpha**2 if direction == 'x' else beta**2))
    return min(Ns)


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
    return p


@pytest.mark.slow
def test_shell_pd_xcte():
    for model in ['plate_clpt_donnell',
                  'cylshell_clpt_donnell']:
        print('Checking prescribed u at x = a for model {0}'.format(model))
        p = build_plate(model)
        # u = 0 at x = 0 and prescribed at x = a, v = 0 at y = 0, other
        # in-plane displacements free
        p.x1u = 0
        p.x2u = 1
        p.y1u = 1
        p.y2u = 1
        p.x1v = 1
        p.x2v = 1
        p.y1v = 0
        p.y2v = 1

        # from pre-buckling static solution
        up = -0.001
        k0 = p.calc_kC()
        ku, kv, kw = calc_ku_kv_kw_line_pd_xcte(p)
        kCp = fkCld_xcte(ku, 0., 0., p, p.a, p.get_size(), 0, 0)
        kCp = finalize_symmetric_matrix(kCp)
        k0 = k0 + kCp
        p.add_distr_pd_fixed_x(p.a, ku, None, None, lambda y: up, None, None)
        fext = p.calc_fext()
        c0 = solve(k0, fext)
        kG = p.calc_kG(c=c0)
        eigvals, eigvecs = lb(k0, kG, silent=True, num_eigvalues=10)
        print(eigvals)
        plot_shell(p, eigvecs[:, 0], vec='w', filename='test_lb_pd_xcte.png')

        a11 = np.linalg.inv(p.ABD[:3, :3])[0, 0]
        Nxx = up/(1/ku + a11*p.a)
        lambda_ref = Ncr_ss_orthotropic(p, 'x')/abs(Nxx)
        assert np.isclose(eigvals[0], lambda_ref, rtol=1e-4)


@pytest.mark.slow
def test_shell_pd_ycte():
    for model in ['plate_clpt_donnell',
                  'cylshell_clpt_donnell']:
        print('Checking prescribed v at y = b for model {0}'.format(model))
        p = build_plate(model)
        # v = 0 at y = 0 and prescribed at y = b, u = 0 at x = 0, other
        # in-plane displacements free
        p.x1u = 0
        p.x2u = 1
        p.y1u = 1
        p.y2u = 1
        p.x1v = 1
        p.x2v = 1
        p.y1v = 0
        p.y2v = 1

        # from pre-buckling static solution
        vp = -0.001
        k0 = p.calc_kC()
        ku, kv, kw = calc_ku_kv_kw_line_pd_ycte(p)
        kCp = fkCld_ycte(0., kv, 0., p, p.b, p.get_size(), 0, 0)
        kCp = finalize_symmetric_matrix(kCp)
        k0 = k0 + kCp
        p.add_distr_pd_fixed_y(p.b, None, kv, None, None, lambda x: vp, None)
        fext = p.calc_fext()
        c0 = solve(k0, fext)
        kG = p.calc_kG(c=c0)
        eigvals, eigvecs = lb(k0, kG, silent=True, num_eigvalues=10)
        print(eigvals)
        plot_shell(p, eigvecs[:, 0], vec='w', filename='test_lb_pd_ycte.png')

        a22 = np.linalg.inv(p.ABD[:3, :3])[1, 1]
        Nyy = vp/(1/kv + a22*p.b)
        lambda_ref = Ncr_ss_orthotropic(p, 'y')/abs(Nyy)
        assert np.isclose(eigvals[0], lambda_ref, rtol=1e-4)


if __name__ == '__main__':
    test_shell_pd_xcte()
    test_shell_pd_ycte()
