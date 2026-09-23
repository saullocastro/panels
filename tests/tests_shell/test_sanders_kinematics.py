r"""Cylindrical shells with Sanders-Koiter against Donnell kinematics

The references are:

- Sanders, J. L., "An improved first-approximation theory for thin shells",
  NASA TR R-24, 1959: all strains of Sanders' theory vanish for rigid-body
  motions, which is not the case for Love's or Donnell's theories.

- Leissa, A. W., "Vibration of shells", NASA SP-288, 1973, with the
  frequency parameter `\Omega = \omega R \sqrt{\rho (1 - \nu^2)/E}` of
  circular cylindrical shells supported by shear diaphragms (SD). According
  to Leissa (p. 158), an open panel of angle `\theta_0` supported on all
  four edges by shear diaphragms has the same frequencies as the closed
  shell with `n = \pi/\theta_0` circumferential waves.

  - Table 2.8: frequencies of 3D elasticity, and Tables 2.6 and 2.7:
    percent differences of each shell theory with respect to them.
  - Tables 2.46 and 2.47: deep open shells, `\theta_0 > \pi`.

- Loy, C. T., Lam, K. Y. and Shu, C., "Analysis of cylindrical shells using
  generalized differential quadrature", Shock and Vibration, 4, 193-198,
  1997, with Love's theory, as reproduced in Table 1 of Li, S.-R., Fu, X.-H.
  and Batra, R. C., Mech. Res. Commun., 37, 577-580, 2010.

- Batdorf, S. B., NACA TR 874, 1947, and NASA SP-8007-2020/REV 2, Sec.
  4.1.1.1: axial buckling of cylindrical shells and panels with Donnell's
  equations.

Leissa's values neglect the rotary inertia, which is included in the mass
matrices of this package. The mass matrix without it is obtained exactly
from two mass matrices, since the rotary inertia scales with `h^3` and the
translational inertia with `h`: evaluating ``calc_kM()`` with ``(h, rho)``
and ``(2 h, rho/2)`` gives `M_1 = M_t + M_r` and `M_2 = M_t + 4 M_r`, such
that `M_t = (4 M_1 - M_2)/3`.

The exact solutions of the Donnell and Sanders kinematics are also computed
herein, with the Navier solution of the SD-SD panel, which Leissa's tables
confirm to the printed digits.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from structsolve.sparseutils import remove_null_cols

from panels.shell import Shell
from panels.models.clpt_field import fg

E = 70.e9
NU = 0.3
RHO = 2700.


def make_panel(model, R, h, L, theta0, m=12, n=12, stack=None,
               laminaprop=None):
    r"""Panel of angle ``theta0`` supported by shear diaphragms

    The SD support is `v = w = N_x = M_x = 0` along the curved edges and
    `u = w = N_\theta = M_\theta = 0` along the straight edges.

    """
    s = Shell()
    s.model = model
    s.a = L
    s.b = R*theta0
    s.r = R
    if stack is None:
        s.stack = [0.]
        s.plyt = h
        s.laminaprop = (E, NU)
    else:
        s.stack = stack
        s.plyt = h/len(stack)
        s.laminaprop = laminaprop
    s.rho = RHO
    s.m = m
    s.n = n
    for edge in ('x1', 'x2'):
        setattr(s, edge + 'u', 1.)
        setattr(s, edge + 'v', 0.)
        setattr(s, edge + 'w', 0.)
    for edge in ('y1', 'y2'):
        setattr(s, edge + 'u', 0.)
        setattr(s, edge + 'v', 1.)
        setattr(s, edge + 'w', 0.)
    s._rebuild()
    return s


def translational_mass(s):
    """Mass matrix without rotary inertia, see the module docstring"""
    h = s.lam.h
    rho = s.lam.intrho/h
    shape = (s.nx, s.ny)
    M1 = s.calc_kM(h_nxny=np.full(shape, h), rho_nxny=np.full(shape, rho))
    M2 = s.calc_kM(h_nxny=np.full(shape, 2*h), rho_nxny=np.full(shape, rho/2))
    return (4*M1 - M2)/3


def modal_amplitude(s, vec, jx=1, jy=1):
    r"""Projection of the mode onto `\sin(j_x \pi x/L)\sin(j_y \pi y/b)`"""
    xs = np.linspace(0, s.a, 31)
    ys = np.linspace(0, s.b, 31)
    xs, ys = np.meshgrid(xs, ys)
    _, fields = s.uvw(np.ascontiguousarray(vec.real), xs=xs, ys=ys)
    w = fields['w'].ravel()
    ref = (np.sin(jx*np.pi*xs/s.a)*np.sin(jy*np.pi*ys/s.b)).ravel()
    return abs(w @ ref)/(np.linalg.norm(w)*np.linalg.norm(ref))


def frequency_parameter(s, jx=1, jy=1, num_eigvalues=12):
    r"""`\Omega` of the mode with ``jx`` and ``jy`` half-waves in `w`"""
    K = s.calc_kC()
    M = translational_mass(s)
    K, used = remove_null_cols(K)
    M, _ = remove_null_cols(M)
    eigvals, eigvecs = eigsh(K, M=M, k=num_eigvalues, sigma=-1., which='LM')
    vecs = np.zeros((s.get_size(), num_eigvalues))
    vecs[used, :] = eigvecs
    amps = [modal_amplitude(s, vecs[:, i], jx, jy) for i in range(num_eigvalues)]
    i = int(np.argmax(amps))
    assert amps[i] > 0.9, 'mode (%d, %d) not found' % (jx, jy)
    omega = np.sqrt(eigvals[i])
    return omega*s.r*np.sqrt(RHO*(1 - NU**2)/E)


def navier_frequency(model, R, h, L, n, ABD=None, rhoh=None, m=1):
    r"""Lowest frequency of the SD-SD panel or shell with the Navier solution

    With `\alpha = m \pi/L` and `\beta = n/R`, the strains of the modes
    `u = A \cos \alpha x \sin n \theta`, `v = B \sin \alpha x \cos n \theta`,
    `w = C \sin \alpha x \sin n \theta` are proportional to the columns of
    the rows below, applied to `(A, B, C)`.

    Returns `\omega`.

    """
    a = m*np.pi/L
    b = n/R
    Ematrix = np.array([
        [-a, 0, 0],
        [0, b, 1/R],
        [-b, a, 0],
        [0, 0, a**2],
        [0, b/R, b**2] if model == 'sanders' else [0, 0, b**2],
        [b/(2*R), 3*a/(2*R), 2*a*b] if model == 'sanders' else [0, 0, 2*a*b],
    ])
    if ABD is None:
        D = E*h**3/(12*(1 - NU**2))
        A = E*h/(1 - NU**2)
        ABD = np.zeros((6, 6))
        ABD[:3, :3] = A*np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU)/2]])
        ABD[3:, 3:] = D/A*ABD[:3, :3]
        rhoh = RHO*h
    K = Ematrix.T @ ABD @ Ematrix
    return np.sqrt(eigh(K, rhoh*np.eye(3), eigvals_only=True)[0])


def navier_Omega(model, R, h, L, n):
    omega = navier_frequency(model, R, h, L, n)
    return omega*R*np.sqrt(RHO*(1 - NU**2)/E)


# (R/h, L/R, n = pi/theta0, Donnell, Sanders, source)
LEISSA = [
    # Leissa Tables 2.6 and 2.8, R/h = 20: 3D elasticity gives 0.121249
    (20, 4, 2, 0.127128, 0.121268, 'Leissa T2.6/T2.8, 3D = 0.121249'),
    # 3D elasticity gives 0.0161063, Donnell overestimates it by 18.9 %
    (20, 20, 1, 0.0191493, 0.0161029, 'Leissa T2.6/T2.8, 3D = 0.0161063'),
    # 3D elasticity gives 0.129881
    (20, 4, 3, 0.143327, 0.130244, 'Leissa T2.6/T2.8, 3D = 0.129881'),
    # Love's theory by Loy et al. (1997) gives 0.009382
    (100, 20, 2, 0.011623, 0.009378, 'Loy et al. (1997), Love = 0.009382'),
]


@pytest.mark.parametrize('Rh, LR, n, Omega_donnell, Omega_sanders, source',
                         LEISSA)
def test_navier_reproduces_the_references(Rh, LR, n, Omega_donnell,
                                          Omega_sanders, source):
    R = 1.
    h = R/Rh
    L = LR*R
    assert np.isclose(navier_Omega('donnell', R, h, L, n), Omega_donnell,
                      rtol=5e-5)
    assert np.isclose(navier_Omega('sanders', R, h, L, n), Omega_sanders,
                      rtol=5e-5)


@pytest.mark.parametrize('Rh, LR, n, Omega_donnell, Omega_sanders, source',
                         LEISSA)
def test_frequency_sd_panel(Rh, LR, n, Omega_donnell, Omega_sanders, source):
    """Leissa's panel of angle pi/n against the closed shell with n waves"""
    R = 1.
    h = R/Rh
    L = LR*R
    theta0 = np.pi/n
    Omegas = {}
    for model in ('cylshell_clpt_donnell', 'cylshell_clpt_sanders'):
        s = make_panel(model, R, h, L, theta0)
        Omegas[model] = frequency_parameter(s)
    print(source, Omegas)
    assert np.isclose(Omegas['cylshell_clpt_donnell'], Omega_donnell, rtol=1e-4)
    assert np.isclose(Omegas['cylshell_clpt_sanders'], Omega_sanders, rtol=1e-4)


def test_frequency_deep_open_panel():
    r"""Leissa Table 2.46: `R/h = 20`, `\theta_0 = 270^\circ`, `L/R = 100`

    Donnell's value, 0.00374949, is from Leissa's table, whereas Sanders'
    theory is compared with Flugge's value of the same table, 0.00459872,
    which is 0.1 % below its exact Navier solution, 0.00460365. Donnell's
    theory is 18 % below both.
    """
    R = 1.
    h = R/20
    L = 100*R
    theta0 = 3*np.pi/2
    s = make_panel('cylshell_clpt_donnell', R, h, L, theta0)
    assert np.isclose(frequency_parameter(s), 0.00374949, rtol=1e-4)
    s = make_panel('cylshell_clpt_sanders', R, h, L, theta0)
    Omega = frequency_parameter(s)
    assert np.isclose(Omega, 0.00460365, rtol=1e-4)
    assert np.isclose(Omega, 0.00459872, rtol=2e-3)


@pytest.mark.parametrize('n, Omega_donnell, Omega_sanders', [
    (2, 0.197278, 0.196963),
    (4, 0.111525, 0.108761),
    ])
def test_frequency_laminated_sd_panel(n, Omega_donnell, Omega_sanders):
    r"""Cross-ply `[0, 90, 0]` panel, `R/h = 100`, `L/R = 5`

    Material with `E_1/E_2 = 40`, `G_{12}/E_2 = 0.6`, `\nu_{12} = 0.25`,
    with `\Omega^* = \omega R \sqrt{\rho/E_2}`, against the exact Navier
    solution.
    """
    R = 1.
    h = R/100
    L = 5*R
    E2 = 1.e9
    laminaprop = (40*E2, E2, 0.25, 0.6*E2, 0.6*E2, 0.5*E2)
    for model, ref in (('donnell', Omega_donnell), ('sanders', Omega_sanders)):
        s = make_panel('cylshell_clpt_' + model, R, h, L, np.pi/n,
                       stack=[0, 90, 0], laminaprop=laminaprop)
        omega_navier = navier_frequency(model, R, h, L, n, ABD=s.lam.ABD,
                                        rhoh=RHO*h)
        assert np.isclose(omega_navier*R*np.sqrt(RHO/E2), ref, rtol=5e-6)
        Omega = frequency_parameter(s)*np.sqrt(E/(1 - NU**2)/E2)
        assert np.isclose(Omega, ref, rtol=1e-4)


def batdorf_panel(R, h, L, b, m, j):
    r"""Axial buckling load of an SD panel with Donnell's equations"""
    D = E*h**3/(12*(1 - NU**2))
    alpha = m*np.pi/L
    beta = j*np.pi/b
    k2 = alpha**2 + beta**2
    return D*k2**2/alpha**2 + E*h/R**2*alpha**2/k2**2


def test_axial_buckling_sd_panel():
    r"""Half-cylinder panel, `R/h = 100`, `L/R = 2`, uniform `N_x`

    Donnell's buckling load is the classical `\sigma_{cr} = 0.6052 E h/R` of
    Batdorf, minimized over the Navier modes, whereas Sanders' kinematics
    reach a lower load, 0.58559, with the mode `(1, 5)` of wider
    circumferential waves. With `N_{cr}` per unit length the parameter is
    `N_{cr} R/(E h^2)`.
    """
    R = 1.
    h = R/100
    L = 2*R
    theta0 = np.pi
    b = R*theta0
    ref_donnell = min(batdorf_panel(R, h, L, b, m, j)
                      for m in range(1, 40) for j in range(1, 40))*R/(E*h**2)
    assert np.isclose(ref_donnell, 0.60523, rtol=1e-4)
    from structsolve import lb
    loads = {}
    for model in ('cylshell_clpt_donnell', 'cylshell_clpt_sanders'):
        # Donnell's critical mode has 9 circumferential half-waves
        s = make_panel(model, R, h, L, theta0, m=20, n=20)
        s.Nxx = -1.
        eigvals, _ = lb(s.calc_kC(), s.calc_kG(), silent=True)
        loads[model] = eigvals[0]*R/(E*h**2)
    print(loads)
    assert np.isclose(loads['cylshell_clpt_donnell'], ref_donnell, rtol=2e-4)
    assert np.isclose(loads['cylshell_clpt_sanders'], 0.58559, rtol=1e-3)


RIGID_BODY_MOTIONS = {
    # translations c and rotations omega, U = c + omega x X, where
    # X = (x, R sin(theta), R cos(theta)), with theta measured from the center
    # of the panel, such that w is along e_r = (0, sin(theta), cos(theta)) and
    # v along e_theta = (0, cos(theta), -sin(theta))
    'translation x': ((1, 0, 0), (0, 0, 0)),
    'translation y': ((0, 1, 0), (0, 0, 0)),
    'translation z': ((0, 0, 1), (0, 0, 0)),
    'rotation x': ((0, 0, 0), (1, 0, 0)),
    'rotation y': ((0, 0, 0), (0, 1, 0)),
    'rotation z': ((0, 0, 0), (0, 0, 1)),
}


def rigid_body_ritz_constants(s, name, amplitude):
    """Least-squares projection of a rigid-body motion onto the Ritz basis"""
    cvec, omega = RIGID_BODY_MOTIONS[name]
    theta0 = s.b/s.r
    G = []
    rhs = []
    size = s.get_size()
    for x in np.linspace(0, s.a, 21):
        for y in np.linspace(0, s.b, 21):
            theta = y/s.r - theta0/2
            X = np.array([x, s.r*np.sin(theta), s.r*np.cos(theta)])
            U = amplitude*(np.array(cvec) + np.cross(omega, X))
            ex = np.array([1., 0, 0])
            et = np.array([0, np.cos(theta), -np.sin(theta)])
            er = np.array([0, np.sin(theta), np.cos(theta)])
            g = np.zeros((5, size))
            fg(g, x, y, s)
            G += [g[0], g[1], g[2]]
            rhs += [U @ ex, U @ et, U @ er]
    return np.linalg.lstsq(np.array(G), np.array(rhs), rcond=None)[0]


def make_free_panel(model, R=0.4, h=1e-3, L=0.3, theta0=0.5):
    s = make_panel(model, R, h, L, theta0)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for field in 'uvw':
            setattr(s, edge + field, 1.)
            setattr(s, edge + field + 'r', 1.)
    s._rebuild()
    return s


@pytest.mark.parametrize('name', list(RIGID_BODY_MOTIONS))
def test_rigid_body_motions_are_strain_free(name):
    """Sanders (1959): all strains vanish for any rigid-body motion

    Donnell's changes of curvature do not vanish for the translations normal
    to the axis, `kyy = A cos(theta)/R**2` (see the kinematics in the theory
    folder), nor for the rotations about the axes normal to it, e.g. `kxy =
    -2 w,xy` for the rotation about the z axis.
    """
    amplitude = 1.e-3
    maxima = {}
    for model in ('cylshell_clpt_donnell', 'cylshell_clpt_sanders'):
        s = make_free_panel(model)
        c = rigid_body_ritz_constants(s, name, amplitude)
        _, fields = s.strain(c, gridx=15, gridy=15, NLgeom=False)
        maxima[model] = {k: np.abs(fields[k]).max()
                         for k in ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy')}
    # the reference curvature of the spurious strains
    scale = amplitude/s.r**2
    for k, v in maxima['cylshell_clpt_sanders'].items():
        assert v < 1.e-8*scale, (k, v)
    donnell = maxima['cylshell_clpt_donnell']
    if name in ('translation y', 'translation z', 'rotation y', 'rotation z'):
        assert max(donnell['kyy'], donnell['kxy']) > 0.1*scale
    else:
        assert max(donnell.values()) < 1.e-8*scale


def test_rigid_translation_strain_energy():
    r"""Closed-form strain energy of Donnell's kinematics, zero for Sanders'

    For a translation `d` along the radial direction at the center of the
    panel, `w = d \cos\theta`, `v = -d \sin\theta`, Donnell's change of
    curvature `\kappa_{\theta} = d \cos\theta/R^2` is the only non-zero
    strain, such that

    .. math::

        U_D = \frac{D_{22} L d^2}{2 R^3}\left(\frac{\theta_0}{2}
              + \frac{\sin\theta_0}{2}\right)

    whereas `U_S = 0` for Sanders' kinematics.
    """
    d = 1.e-3
    energies = {}
    for model in ('cylshell_clpt_donnell', 'cylshell_clpt_sanders'):
        s = make_free_panel(model)
        c = rigid_body_ritz_constants(s, 'translation z', d)
        K0 = s.calc_kC()
        energies[model] = 0.5*c @ (K0 @ c)
    R = s.r
    theta0 = s.b/R
    D22 = s.lam.ABD[4, 4]
    U_D = D22*s.a*d**2/(2*R**3)*(theta0/2 + np.sin(theta0)/2)
    assert np.isclose(energies['cylshell_clpt_donnell'], U_D, rtol=1e-6)
    assert abs(energies['cylshell_clpt_sanders']) < 1e-8*U_D


def test_sanders_approaches_donnell_for_short_waves():
    r"""Donnell's kinematics are accurate for short wavelengths

    Leissa Table 2.8, `R/h = 500`, `n = 4`, `l/mR = 1`: 3D elasticity gives
    `\Omega = 0.354118`, Donnell 0.354145 and Sanders 0.35412, i.e. both
    within 0.01 %.
    """
    R = 1.
    h = R/500
    L = R
    for model, ref in (('cylshell_clpt_donnell', 0.354145),
                       ('cylshell_clpt_sanders', 0.35412)):
        s = make_panel(model, R, h, L, np.pi/4)
        assert np.isclose(frequency_parameter(s), ref, rtol=2e-5)
    assert np.isclose(navier_Omega('sanders', R, h, L, 4), 0.35412, rtol=2e-5)


if __name__ == '__main__':
    for case in LEISSA:
        test_frequency_sd_panel(*case)
    test_frequency_deep_open_panel()
    test_axial_buckling_sd_panel()
    for name in RIGID_BODY_MOTIONS:
        test_rigid_body_motions_are_strain_free(name)
    test_rigid_translation_strain_energy()
