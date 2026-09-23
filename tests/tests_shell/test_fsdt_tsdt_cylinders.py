r"""Cylindrical shells using the FSDT and the TSDT

The models ``'cylshell_fsdt_donnell'``, ``'cylshell_fsdt_sanders'``,
``'cylshell_tsdt_donnell'`` and ``'cylshell_tsdt_sanders'``, see
``theory/shells/fsdt_tsdt/fsdt_tsdt.py``. The references are:

- Leissa, A. W., "Vibration of shells", NASA SP-288, 1973, Table 2.8:
  frequencies of 3D elasticity of circular cylindrical shells supported by
  shear diaphragms (SD), with the frequency parameter `\Omega = \omega R
  \sqrt{\rho (1 - \nu^2)/E}`. An open panel of angle `\pi/n` supported by SD
  on all edges has the frequencies of the closed shell with `n`
  circumferential waves (Leissa, p. 158).

- Batdorf, S. B., NACA TR 874, 1947: axial buckling of cylindrical panels
  with Donnell's equations.

- Sanders, J. L., NASA TR R-24, 1959: the strains of Sanders' theory vanish
  for any rigid-body motion.

The Navier solution of the SD-SD panel of the kinematics of
``theory/shells/fsdt_tsdt``, implemented independently herein, verifies the
Ritz solution, whose CLPT counterpart is verified against the same
references in ``test_sanders_kinematics.py``.
"""
import os
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from structsolve import lb
from structsolve.sparseutils import remove_null_cols, finalize_symmetric_matrix

from panels import modelDB
from panels.shell import Shell
from panels.models.fsdt_tsdt_field import fg, strain_names
from panels.multidomain import MultiDomain

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_sanders_kinematics import (E, NU, RHO, LEISSA, RIGID_BODY_MOTIONS,
        make_panel, modal_amplitude, batdorf_panel)

MODELS = ['cylshell_fsdt_donnell', 'cylshell_fsdt_sanders',
          'cylshell_tsdt_donnell', 'cylshell_tsdt_sanders']
SANDERS = ['cylshell_fsdt_sanders', 'cylshell_tsdt_sanders']
LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.9e9)


def theory_kinematics(model):
    _, theory, kinematics = model.split('_')
    return theory, kinematics


def random_panel(model, seed=1):
    s = Shell(model=model, a=0.4, b=0.3, r=0.35, stack=[0, 45, -45, 90, 30],
              plyt=0.4e-3, laminaprop=LAMINAPROP, rho=1600., m=6, n=7)
    rng = np.random.default_rng(seed)
    for e in ('x1', 'x2', 'y1', 'y2'):
        for d in ('u', 'v', 'w', 'phix', 'phiy'):
            for r in ('', 'r'):
                setattr(s, e + d + r, float(rng.integers(0, 2)))
    return s


def rel(A, B):
    A = A.toarray() if hasattr(A, 'toarray') else np.asarray(A)
    B = B.toarray() if hasattr(B, 'toarray') else np.asarray(B)
    return np.abs(A - B).max()/np.abs(B).max()


@pytest.mark.parametrize('model', MODELS)
def test_analytical_and_numerical_matrices_agree(model):
    s = random_panel(model)
    size = s.get_size()
    mats = modelDB.db[model]['matrices']
    num = modelDB.db[model]['matrices_num']
    nx, ny = 2*s.m + 4, 2*s.n + 4
    fin = finalize_symmetric_matrix
    z = np.zeros(size)
    assert rel(fin(num.fkC_num(z, s.ABD, s, size, 0, 0, nx, ny, 0)),
               fin(mats.fk0(s, size, 0, 0))) < 1e-12
    assert rel(fin(num.fkG_num(z, s.ABD, s, size, 0, 0, nx, ny, 10., -20., 5.)),
               fin(mats.fkG0(10., -20., 5., s, size, 0, 0))) < 1e-12
    assert rel(fin(num.fkM_num(s, 0.1e-3, np.array([s.lam.h, s.rho]), size, 0,
                               0, nx, ny)),
               fin(mats.fkM(s, 0.1e-3, size, 0, 0))) < 1e-12
    s.beta, s.gamma = 1.3e5, 2.1e5
    assert rel(num.fkAx_num(s, size, 0, 0, nx, ny),
               mats.fkAx(s.beta, s.gamma, s, size, 0, 0)) < 1e-12


@pytest.mark.parametrize('model', MODELS)
def test_tangent_is_the_jacobian_of_fint(model):
    s = random_panel(model)
    size = s.get_size()
    num = modelDB.db[model]['matrices_num']
    nx, ny = 2*s.m + 4, 2*s.n + 4
    fin = finalize_symmetric_matrix
    rng = np.random.default_rng(0)
    c = 1e-3*rng.standard_normal(size)
    fint = lambda cc: np.asarray(num.calc_fint(cc, s.ABD, s, size, 0, nx, ny))
    kT = (fin(num.fkC_num(c, s.ABD, s, size, 0, 0, nx, ny, 1))
          + fin(num.fkG_num(c, s.ABD, s, size, 0, 0, nx, ny)))
    k0 = fin(num.fkC_num(np.zeros(size), s.ABD, s, size, 0, 0, nx, ny, 0))
    e = rng.standard_normal(size)
    h = 1e-7
    fd = (fint(c + h*e) - fint(c - h*e))/(2*h)
    assert np.linalg.norm(kT @ e - fd)/np.linalg.norm(fd) < 1e-9
    # the non-linear terms matter at this amplitude
    assert np.linalg.norm(k0 @ e - fd)/np.linalg.norm(fd) > 1e-4


def navier(model, R, h, L, n, ABD, rho, m=1):
    r"""Lowest frequency of the SD-SD panel or shell with the Navier solution

    With `\alpha = m \pi/L` and `\beta = n/R`, the modes are `u = A \cos\alpha
    x \sin\beta y`, `v = B \sin \cos`, `w = C \sin \sin`, `\phi_x = D \cos
    \sin` and `\phi_y = F \sin \cos`. The rows below give the amplitudes of
    the generalized strains, and the mass matrix is integrated through the
    thickness, including the rotary inertia. Returns `\omega`.

    """
    theory, kinematics = theory_kinematics(model)
    a = m*np.pi/L
    b = n/R
    sk = 1. if kinematics == 'sanders' else 0.
    c1 = 4/(3*h**2) if theory == 'tsdt' else 0.
    eps0 = [[-a, 0, 0, 0, 0],
            [0, -b, 1/R, 0, 0],
            [b, a, 0, 0, 0]]
    eps1 = [[0, 0, 0, -a, 0],
            [0, -sk*b/R, 0, 0, -b],
            [-sk*b/(2*R), sk*3*a/(2*R), 0, b, a]]
    eps3 = -c1*np.array([[0, 0, -a**2, -a, 0],
                         [0, 0, -b**2, 0, -b],
                         [0, 0, 2*a*b, b, a]])
    gam0 = np.array([[0, 0, b, 0, 1],
                     [0, 0, a, 1, 0]])
    if theory == 'fsdt':
        Em = np.vstack([eps0, eps1, gam0])
    else:
        Em = np.vstack([eps0, eps1, eps3, gam0, -3*c1*gam0])
    K = Em.T @ ABD @ Em
    M = np.zeros((5, 5))
    zs, wz = np.polynomial.legendre.leggauss(12)
    for z, wg in zip(zs*h/2, wz*h/2):
        # u(z), v(z), w(z) of the displacement field
        Uz = np.array([1, 0, -c1*z**3*a, z - c1*z**3, 0])
        Vz = np.array([0, 1 + sk*z/R, -c1*z**3*b, 0, z - c1*z**3])
        Wz = np.array([0, 0, 1, 0, 0])
        M += rho*wg*(np.outer(Uz, Uz) + np.outer(Vz, Vz) + np.outer(Wz, Wz))
    return np.sqrt(eigh(K, M, eigvals_only=True)[0])


def ritz_frequency(s, num_eigvalues=10):
    r"""`\omega` of the mode with one half-wave along each direction"""
    K, used = remove_null_cols(s.calc_kC(), silent=True)
    M = s.calc_kM().tocsr()[used][:, used]
    eigvals, eigvecs = eigsh(K, M=M, k=num_eigvalues, sigma=-1., which='LM')
    vecs = np.zeros((s.get_size(), num_eigvalues))
    vecs[used, :] = eigvecs
    amps = [modal_amplitude(s, vecs[:, i]) for i in range(num_eigvalues)]
    i = int(np.argmax(amps))
    assert amps[i] > 0.9
    return np.sqrt(eigvals[i])


@pytest.mark.parametrize('n', [2, 3])
@pytest.mark.parametrize('model', MODELS)
def test_navier_thick_cross_ply(model, n):
    r"""Cross-ply `[0, 90, 0]`, `R/h = 10`, `L/R = 2`, `E_1/E_2 = 40`

    The transverse shear deformation lowers the frequency of the CLPT by 3 to
    4 %.
    """
    R = 1.
    h = R/10
    L = 2*R
    E2 = 1.e9
    laminaprop = (40*E2, E2, 0.25, 0.6*E2, 0.6*E2, 0.5*E2)
    s = make_panel(model, R, h, L, np.pi/n, m=14, n=14, stack=[0, 90, 0],
                   laminaprop=laminaprop)
    omega = ritz_frequency(s)
    assert np.isclose(omega, navier(model, R, h, L, n, s.ABD, RHO), rtol=1e-6)
    clpt = make_panel(model.replace('fsdt', 'clpt').replace('tsdt', 'clpt'),
                      R, h, L, np.pi/n, m=14, n=14, stack=[0, 90, 0],
                      laminaprop=laminaprop)
    assert 0.94 < omega/ritz_frequency(clpt) < 0.985


@pytest.mark.parametrize('Rh, LR, n, Omega_donnell, Omega_sanders, source',
                         LEISSA[:3])
def test_frequency_leissa_3d_elasticity(Rh, LR, n, Omega_donnell,
                                        Omega_sanders, source):
    r"""Leissa Table 2.8, `R/h = 20`, 3D elasticity

    With the rotary inertia and the transverse shear deformation, the FSDT and
    TSDT with Sanders' kinematics are within 0.07 % of the 3D elasticity
    solution. For `L/R = 4` and `n = 3`, where the CLPT with Sanders'
    kinematics is 0.21 % above it, they are three times closer.
    """
    ref = float(source.split('= ')[-1])
    R = 1.
    h = R/Rh
    L = LR*R
    factor = R*np.sqrt(RHO*(1 - NU**2)/E)
    for model in SANDERS:
        s = make_panel(model, R, h, L, np.pi/n, m=14, n=14)
        Omega = ritz_frequency(s)*factor
        assert np.isclose(Omega, navier(model, R, h, L, n, s.ABD, RHO)*factor,
                          rtol=1e-6)
        assert abs(Omega/ref - 1) < 7e-4
        if n == 3:
            clpt = make_panel('cylshell_clpt_sanders', R, h, L, np.pi/n, m=14,
                              n=14)
            Omega_clpt = ritz_frequency(clpt)*factor
            assert abs(Omega_clpt/ref - 1) > 3*abs(Omega/ref - 1)


def test_axial_buckling_thin_panel():
    r"""Half-cylinder panel, `R/h = 100`, `L/R = 2`, uniform `N_x`

    Donnell's buckling load of the FSDT and TSDT is 0.35 % below Batdorf's
    classical value, `N_{cr} R/(E h^2) = 0.6052`, of the CLPT, due to the
    transverse shear deformation, and that of Sanders' kinematics is within
    0.05 % of the CLPT with Sanders' kinematics, see
    ``test_sanders_kinematics.test_axial_buckling_sd_panel``.
    """
    R = 1.
    h = R/100
    L = 2*R
    b = R*np.pi
    ref_donnell = min(batdorf_panel(R, h, L, b, m, j)
                      for m in range(1, 40) for j in range(1, 40))*R/(E*h**2)
    for model in MODELS:
        s = make_panel(model, R, h, L, np.pi, m=20, n=20)
        s.Nxx = -1.
        eigvals, _ = lb(s.calc_kC(), s.calc_kG(), silent=True)
        load = eigvals[0]*R/(E*h**2)
        if 'donnell' in model:
            assert 0.994 < load/ref_donnell < 0.998
        else:
            assert 0.9995 < load/0.58559 < 1.


def rigid_body_ritz_constants(s, name, amplitude):
    r"""Least-squares projection of a rigid-body motion onto the Ritz basis

    For the rotation `\omega`, the normal `e_r` rotates by `\omega \times
    e_r`, whose components along `e_x` and `e_\theta` are the rotations
    ``phix`` and ``phiy`` of :func:`.fg`, which is `\phi_y + v/r` for the
    Sanders' kinematics.
    """
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
            dn = amplitude*np.cross(omega, er)
            g = np.zeros((5, size))
            fg(g, x, y, s)
            G += list(g)
            rhs += [U @ ex, U @ et, U @ er, dn @ ex, dn @ et]
    return np.linalg.lstsq(np.array(G), np.array(rhs), rcond=None)[0]


def free_panel(model, R=0.4, h=1e-3, L=0.3, theta0=0.5):
    s = make_panel(model, R, h, L, theta0)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for field in ('u', 'v', 'w', 'phix', 'phiy'):
            setattr(s, edge + field, 1.)
            setattr(s, edge + field + 'r', 1.)
    s._rebuild()
    return s


@pytest.mark.parametrize('theory', ['fsdt', 'tsdt'])
@pytest.mark.parametrize('name', list(RIGID_BODY_MOTIONS))
def test_rigid_body_motions_are_strain_free(name, theory):
    """Sanders (1959): all strains vanish for any rigid-body motion, which
    for the FSDT and TSDT includes the transverse shear strains, whereas
    Donnell's kinematics give spurious strains for the translations normal to
    the axis and the rotations about the axes normal to it, as for the CLPT"""
    amplitude = 1.e-3
    maxima = {}
    for kinematics in ('donnell', 'sanders'):
        model = 'cylshell_{0}_{1}'.format(theory, kinematics)
        s = free_panel(model)
        c = rigid_body_ritz_constants(s, name, amplitude)
        _, fields = s.strain(c, gridx=15, gridy=15, NLgeom=False)
        #NOTE the higher-order strains of the TSDT carry the factor c1 =
        #     4/(3 h^2), which amplifies the error of the projection
        c1 = 4/(3*s.lam.h**2)
        maxima[kinematics] = {k: np.abs(fields[k]).max()/(
                              c1 if k[-1] in '23' else 1.)
                              for k in strain_names(s)}
    scale = amplitude/s.r**2
    for k, v in maxima['sanders'].items():
        assert v < 1.e-8*scale, (k, v)
    donnell = max(maxima['donnell'].values())
    if name in ('translation y', 'translation z', 'rotation y', 'rotation z'):
        # the spurious strains of the FSDT and TSDT are also the transverse
        # shear strains, e.g. gyz = w,y for the translations
        assert donnell > 0.01*scale


@pytest.mark.parametrize('model', SANDERS)
def test_rigid_translation_strain_energy(model):
    s = free_panel(model)
    c = rigid_body_ritz_constants(s, 'translation z', 1.e-3)
    K0 = s.calc_kC()
    donnell = free_panel(model.replace('sanders', 'donnell'))
    c_d = rigid_body_ritz_constants(donnell, 'translation z', 1.e-3)
    U_D = 0.5*c_d @ (donnell.calc_kC() @ c_d)
    assert U_D > 0
    assert abs(0.5*c @ (K0 @ c)) < 1e-8*U_D


def split_cylinder(model, ny):
    r"""SD panel of `\pi/2`, `R/h = 50`, `L/R = 2`, in ``ny`` domains along
    the circumference, connected with ``'SSycte'``"""
    R = 1.
    h = R/50
    L = 2*R
    b = R*np.pi/2
    panels = []
    for k in range(ny):
        p = make_panel(model, R, h, L, np.pi/2/ny, m=10, n=10)
        p.y0 = k*b/ny
        if k > 0:
            for field in ('u', 'v', 'w', 'phix', 'phiy'):
                setattr(p, 'y1' + field, 1.)
                setattr(p, 'y1' + field + 'r', 1.)
        if k < ny - 1:
            for field in ('u', 'v', 'w', 'phix', 'phiy'):
                setattr(p, 'y2' + field, 1.)
                setattr(p, 'y2' + field + 'r', 1.)
        panels.append(p)
    conn = []
    for pA, pB in zip(panels[:-1], panels[1:]):
        from panels.multidomain.connections import calc_kt_kr
        kt, kr = calc_kt_kr(pA, pB, 'ycte')
        conn.append(dict(p1=pA, p2=pB, func='SSycte', ycte1=pA.b, ycte2=0,
                         kt=1e3*kt, kr=1e3*kr))
    return panels, conn


@pytest.mark.parametrize('model', MODELS)
def test_multidomain_split_cylinder(model):
    r"""Two domains along the circumference reproduce the single domain

    The edge connection penalizes `u, v, w`, `\phi_x` and the rotation of the
    normal `\Phi_y`, which is `\phi_y + v/r` for Sanders' kinematics.
    """
    panels, conn = split_cylinder(model, 2)
    md = MultiDomain(panels, conn)
    K, used = remove_null_cols(md.calc_kC(), silent=True)
    M = md.calc_kM(silent=True).tocsr()[used][:, used]
    w_md = np.sqrt(eigsh(K, M=M, k=1, sigma=-1., which='LM')[0][0])
    single = make_panel(model, 1., 1/50, 2., np.pi/2, m=10, n=16)
    K, used = remove_null_cols(single.calc_kC(), silent=True)
    M = single.calc_kM().tocsr()[used][:, used]
    w_single = np.sqrt(eigsh(K, M=M, k=1, sigma=-1., which='LM')[0][0])
    assert np.isclose(w_md, w_single, rtol=2e-3)


@pytest.mark.parametrize('theory', ['fsdt', 'tsdt'])
def test_bf_rigid_rotation_of_sanders_skin(theory):
    r"""Rigid rotation `\theta` of a Sanders skin with a flange about the axis

    The skin has `v_1 = r \theta` and the rotation of the normal `\Phi_y =
    \phi_y + v/r = \theta`, with `\phi_y = 0`, and the flange, a plate along
    `-z_1` from its edge `y_2 = 0`, has `w_2 = (r - y_2) \theta` and `\phi_y =
    -w_{2,y} = \theta`, see :mod:`panels.multidomain.connections.kCBFycte`.
    Neither the panels nor the connection store energy.
    """
    R = 0.4
    base = Shell(group='base', model='cylshell_%s_sanders' % theory, x0=0,
                 y0=0, a=0.6, b=0.3, r=R, m=6, n=7, stack=[0, 45, -45, 90],
                 plyt=0.2e-3, laminaprop=LAMINAPROP)
    flange = Shell(group='flange', model='plate_%s_donnell' % theory, x0=0,
                   y0=0, a=0.6, b=0.05, m=6, n=5, stack=[0, 90, 90, 0],
                   plyt=0.2e-3, laminaprop=LAMINAPROP)
    for p in (base, flange):
        for e in ('x1', 'x2', 'y1', 'y2'):
            for d in ('u', 'v', 'w', 'phix', 'phiy'):
                for r in ('', 'r'):
                    setattr(p, e + d + r, 1.)
    kt, kr = 1.e8, 1.e4
    md = MultiDomain(panels=[base, flange],
                     conn=[dict(p1=base, p2=flange, func='BFycte',
                                ycte1=base.b/2, ycte2=0., kt=kt, kr=kr)])
    theta = 1.e-3

    def fit(p, targets):
        xs, ys = np.meshgrid(np.linspace(0, p.a, 12), np.linspace(0, p.b, 10))
        xs, ys = xs.ravel(), ys.ravel()
        size = p.get_size()
        G = np.zeros((5*xs.size, size))
        for k, (x, y) in enumerate(zip(xs, ys)):
            g = np.zeros((5, size))
            fg(g, x, y, p)
            G[5*k:5*k + 5] = g
        t = np.ravel([[f(x, y) for f in targets] for x, y in zip(xs, ys)])
        c = np.linalg.lstsq(G, t, rcond=None)[0]
        assert np.allclose(G @ c, t, atol=1e-10*np.abs(t).max())
        return c

    zero = lambda x, y: 0.
    c = np.zeros(md.get_size())
    # skin: u, v, w, phix and Phiy = phiy + v/r, the rows of fg
    c[base.col_start:base.col_end] = fit(base, [zero, lambda x, y: R*theta,
                                               zero, zero, lambda x, y: theta])
    # flange: w2 = (R - y2) theta, phiy = -w2,y
    c[flange.col_start:flange.col_end] = fit(flange, [
        zero, zero, lambda x, y: (R - y)*theta, zero, lambda x, y: theta])
    U_ref = 0.5*kr*base.a*theta**2
    assert abs(0.5*c @ md.get_kC_conn() @ c) < 1e-9*U_ref
    assert abs(0.5*c @ md.calc_kC(silent=True) @ c) < 1e-9*U_ref
