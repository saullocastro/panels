r"""Natural frequencies of a laminated plate, with and without prestress

For nine plate and cylindrical-shell models with a large radius, the first
natural frequency of a simply supported symmetric cross-ply plate, with the
geometric stiffness of ``Nxx = -30``, ``Nyy = -2.5`` added or not, against
the closed-form frequency of a simply supported plate of rectangular
orthotropy with in-plane forces (Leissa, 1969, eqs. (9.24), (9.29), (10.1)
and (10.43)-(10.44)).

Notes
-----
The laminate ``[0, 90]s`` is symmetric with `D_{16} = D_{26} = 0`, a plate
of rectangular orthotropy with `D_x = D_{11}`, `D_y = D_{22}` and `D_{xy} =
D_{12} + 2 D_{66}` (eq. (9.24) of [1]_). The mode `w = \sin(m \pi x/a)
\sin(n \pi y/b) \cos \omega t` (eq. (10.43)) solves eq. (10.1) of [1]_ when

.. math::

    \rho h\, \omega^2 = D_{11} \left(\frac{m \pi}{a}\right)^4 + 2 D_{xy}
    \left(\frac{m \pi}{a}\right)^2 \left(\frac{n \pi}{b}\right)^2 + D_{22}
    \left(\frac{n \pi}{b}\right)^4 + N_{xx} \left(\frac{m \pi}{a}\right)^2 +
    N_{yy} \left(\frac{n \pi}{b}\right)^2

with `\rho h` the mass per unit area: the orthotropic form of eq. (10.44),
which reduces to eq. (9.29) without in-plane forces. The reference is the
lowest `\omega` over `m, n` (`m = n = 1`), 31.517 rad/s without and 19.646
rad/s with prestress, 61 % of the buckling load. The flat plate
decouples the membrane and bending problems, so the in-plane boundary
conditions do not enter.

With ``m = n = 8`` the classical models are within `10^{-6}` of the
reference; the models with transverse shear are within `1.2 \times 10^{-5}`,
the effect of the transverse shear and of the rotary inertia that the
classical theory neglects (`a/h = 2000`); the tolerance is `10^{-4}`. The
cylinders have ``r = 1e8`` and give the plate values.

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, National
   Aeronautics and Space Administration, Washington, DC, 1969.
   https://ntrs.nasa.gov/citations/19700009156

"""
import sys
sys.path.append('../..')

import numpy as np
from structsolve import freq

from panels.shell import Shell


models = ['plate_clpt_donnell', 'cylshell_clpt_donnell',
          'cylshell_clpt_sanders', 'plate_fsdt_donnell', 'plate_tsdt_donnell',
          'cylshell_fsdt_donnell', 'cylshell_fsdt_sanders',
          'cylshell_tsdt_donnell', 'cylshell_tsdt_sanders']


def omega_ss_orthotropic(p, Nxx, Nyy, mmax=10):
    """Closed-form first natural frequency of a simply supported plate of
    rectangular orthotropy under constant ``Nxx`` and ``Nyy``"""
    D = p.ABD[3:6, 3:6]
    D11, D22, Dxy = D[0, 0], D[1, 1], D[0, 1] + 2*D[2, 2]
    rhoh = p.rho*sum(p.plyts)
    omegas = []
    for m in range(1, mmax + 1):
        for n in range(1, mmax + 1):
            alpha = m*np.pi/p.a
            beta = n*np.pi/p.b
            K = D11*alpha**4 + 2*Dxy*alpha**2*beta**2 + D22*beta**4
            G = Nxx*alpha**2 + Nyy*beta**2
            omegas.append(np.sqrt((K + G)/rhoh))
    return min(omegas)


def test_panel_freq():
    for model in models:
        for prestress in [True, False]:
            print('Frequency Analysis, prestress={0}, model={1}'.format(
                  prestress, model))
            p = Shell()
            p.model = model
            p.a = 1.
            p.b = 0.5
            p.r = 1.e8
            p.stack = [0, 90, 90, 0]
            p.plyt = 1e-3*0.125
            p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
            p.rho = 1.3e3
            p.m = 8
            p.n = 8
            p.Nxx = -30.
            p.Nyy = -2.5
            k0 = p.calc_kC(silent=True)
            M = p.calc_kM(silent=True)
            if prestress:
                kG0 = p.calc_kG(silent=True)
                k0 += kG0
            eigvals, eigvecs = freq(k0, M, sparse_solver=True, silent=True)
            omegan = np.sqrt(-eigvals)
            if prestress:
                omegan_ref = omega_ss_orthotropic(p, p.Nxx, p.Nyy)
            else:
                omegan_ref = omega_ss_orthotropic(p, 0., 0.)
            assert np.isclose(omegan[0], omegan_ref, rtol=1e-4)


if __name__ == '__main__':
    test_panel_freq()
