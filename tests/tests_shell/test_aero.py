r"""Flutter of a panel with the aerodynamic stiffness of the piston theory

For each model, a simply supported isotropic square plate without and with
the in-plane prestress of its biaxial buckling load: the aerodynamic
pressure parameter ``beta`` is increased until two natural frequencies of
``kC + kG + kA``, with the mass matrix ``kM``, coalesce and become complex,
the flutter onset, found by bisection. The non-dimensional onset `\lambda =
\beta a^3/D` is compared with Singha and Ganapathi (2005), Table 2 (`a/h =
100`, without prestress) and Table 3 (`a/h = 240`, biaxial compression `P =
P_{cr}`).

Notes
-----
With the linear piston theory of :meth:`.Shell.calc_kA`, the aerodynamic
pressure is `q = \beta w_{,x}`, `\beta = \rho_a U_a^2/\sqrt{M^2 - 1}`, the
dynamic pressure parameter of eq. (10) of [1]_, made non-dimensional as
`\lambda = \beta a^3/D`, `D = E h^3/(12 (1 - \nu^2))`. Without damping, two
eigenvalues `\omega^2` of `(\boldsymbol{K}_C + \boldsymbol{K}_G +
\boldsymbol{K}_A) \boldsymbol{c} = \omega^2 \boldsymbol{M} \boldsymbol{c}`
approach each other as `\beta` grows and coalesce at the critical `\lambda`,
beyond which they are complex conjugates [1]_; :func:`flutter_onset` finds
the first `\beta` at which one of the four lowest `\omega^2` has an
imaginary part larger than `10^{-6} |\omega^2|`, by bisection to a relative
width of `10^{-5}`.

References of [1]_ for the simply supported square plate:

- Table 2, `a/b = 1`, `a/h = 100`, skew angle 0: `\lambda_{cr}/\pi^4 =
  5.27` (present FE results of [1]_; 5.12 and 5.25 in the two references
  cited there), `\lambda_{cr} = 513.35`;
- Table 3, `a/h = 240`, `\nu = 0.3`, biaxial compression `P/P_{cr} = 1`:
  `\lambda_{cr} = 343.15` (342.0 and 343.45 in the references cited there).
  Here `P_{cr}` is the biaxial buckling load of each model, from
  ``Nxx = Nyy = -1``.

For a classical plate the critical `\lambda` depends neither on `\nu` nor on
the density. With ``m = n = 10`` the classical models give 512.51 (Table 2,
`-0.16 \%`) and 343.35 (Table 3, `+0.06 \%`), the models with transverse
shear 512.04 (`-0.25 \%`) and 343.26 (`+0.03 \%`), unchanged for ``m = n =
12``; the tolerances are `5 \times 10^{-3}` (Table 2) and `2 \times 10^{-3}`
(Table 3), the scatter of the values tabulated in [1]_. The cylinders have
``r = 1e8`` and give the plate values.

References
----------
.. [1] Singha, M. K., Ganapathi, M., A parametric study on supersonic
   flutter behavior of laminated composite skew flat panels. Composite
   Structures 69 (2005) 55-63. https://doi.org/10.1016/j.compstruct.2004.04.018

"""
import sys
sys.path.append('../..')

import numpy as np

from structsolve import freq, lb
from panels.shell import Shell


def flutter_onset(s, beta_low, beta_high, rtol=1e-5):
    """Bisection of the first coalescence of the natural frequencies"""
    kC = s.calc_kC()
    kG = s.calc_kG()
    kM = s.calc_kM()

    def unstable(beta):
        s.beta = beta
        kA = s.calc_kA()
        eigvals, eigvecs = freq(kC + kG + kA, kM, sparse_solver=True,
                                silent=True, num_eigvalues=10)
        omegan2 = -np.asarray(eigvals, dtype=complex)[:4]
        return np.any(np.abs(omegan2.imag) > 1e-6*np.abs(omegan2))

    assert not unstable(beta_low)
    assert unstable(beta_high)
    while (beta_high - beta_low) > rtol*beta_high:
        beta = 0.5*(beta_low + beta_high)
        if unstable(beta):
            beta_high = beta
        else:
            beta_low = beta
    return 0.5*(beta_low + beta_high)


def test_aero():
    for model in ['plate_clpt_donnell',
                  'cylshell_clpt_donnell',
                  'cylshell_clpt_sanders',
                  'plate_fsdt_donnell',
                  'plate_tsdt_donnell']:
        for pre_stress in [False, True]:
            print('Flutter Analysis Piston Theory, consider_pre_stress={0}, model={1}'.
                  format(pre_stress, model))
            s = Shell()
            s.model = model
            s.a = 1.
            s.b = 1.
            s.r = 1.e8
            E = 70e9
            nu = 0.3
            h = s.a/240 if pre_stress else s.a/100
            D = E*h**3/(12*(1 - nu**2))
            s.stack = [0]
            s.plyt = h
            s.laminaprop = (E, nu)
            s.rho = 2.7e3
            s.m = 10
            s.n = 10

            if pre_stress:
                # biaxial buckling load of the model
                s.Nxx = -1.
                s.Nyy = -1.
                eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
                s.Nxx = -eigvals[0]
                s.Nyy = -eigvals[0]
            else:
                s.Nxx = 0
                s.Nyy = 0

            beta = flutter_onset(s, 100*D/s.a**3, 1000*D/s.a**3)
            lambda_cr = beta*s.a**3/D
            if pre_stress:
                assert np.isclose(lambda_cr, 343.15, rtol=2e-3)
            else:
                assert np.isclose(lambda_cr, 5.27*np.pi**4, rtol=5e-3)


if __name__ == '__main__':
    test_aero()
