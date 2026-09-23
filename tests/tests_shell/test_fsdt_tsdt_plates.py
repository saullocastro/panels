r"""Plates with the first-order (FSDT) and Reddy's third-order (TSDT) shear
deformation theories against the literature

Unless stated otherwise, the plates are square and simply supported with the
SS-1 condition of Reddy, for which the Navier solutions of the references
are exact:

- along `x = 0, a`: `v = w = \phi_y = 0`, with `u` and `\phi_x` free
- along `y = 0, b`: `u = w = \phi_x = 0`, with `v` and `\phi_y` free

The FSDT uses the shear correction factor `k = 5/6`, the default of
``Shell.fsdt_shear_correction``, unless stated otherwise. The mass matrices
include the rotary and the higher-order inertia, consistently with the
displacement field of each theory.

References:

- Noor, A. K., "Free vibrations of multilayered composite plates", AIAA J.,
  11(7), 1038-1039, 1973 (3D elasticity), and Reddy, J. N., "A simple
  higher-order theory for laminated composite plates", J. Appl. Mech., 51,
  745-752, 1984 (TSDT), as reproduced in Table 3 of Adim, B., Daouadji, T. H.
  and Rabahi, A., Int. J. Adv. Struct. Eng., 8, 103-117, 2016.

- Noor, A. K., "Stability of multilayered composite plates", Fibre Sci.
  Technol., 8(2), 81-89, 1975 (3D elasticity), with the TSDT of Reddy
  (1984), as reproduced in Table 7 of Adim et al. (2016).

- Whitney, J. M. and Pagano, N. J., "Shear deformation in heterogeneous
  anisotropic plates", J. Appl. Mech., 37, 1031-1036, 1970 (FSDT), and Reddy
  (1984) (TSDT), as reproduced in Table 6 of Natarajan, S., Nguyen-Xuan,
  H., Ferreira, A. J. M. and Carrera, E., arXiv:1312.4032.

- Pagano, N. J., "Exact solutions for rectangular bidirectional composites
  and sandwich plates", J. Compos. Mater., 4, 20-34, 1970 (3D elasticity),
  with the TSDT of Reddy (1984) and the FSDT of Reddy and Chao (1981), as
  reproduced in Tables 1 and 2 of arXiv:1312.4032.

- Liew, K. M., Xiang, Y. and Kitipornchai, S., "Transverse vibration of
  thick rectangular plates", Comput. Struct., 49, 1-29, 1993 (FSDT, `k =
  5/6`), and Hashemi, S. H. and Arsanjani, M., "Exact characteristic
  equations for some of classical boundary conditions of vibrating
  moderately thick rectangular plates", Int. J. Solids Struct., 42, 819-853,
  2005 (FSDT, `k = 0.86667`), as reproduced in Table 2 of arXiv:2009.05597.

- Leissa, A. W., "The free vibration of rectangular plates", J. Sound
  Vib., 31, 257-293, 1973 (classical plate theory).

The 3D elasticity solutions are given in the comments, for reference, since
none of the equivalent single-layer theories reproduces them exactly.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.special import roots_legendre
from scipy.sparse.linalg import eigsh
from structsolve import lb, solve
from structsolve.sparseutils import remove_null_cols

from panels import modelDB
from panels.shell import Shell

E2 = 1.e9
RHO = 1500.

CLPT = 'plate_clpt_donnell'
FSDT = 'plate_fsdt_donnell'
TSDT = 'plate_tsdt_donnell'


def set_edge(s, edge, kind):
    r"""Boundary condition of one edge

    ``kind`` is ``'S'`` for simply supported (SS-1), ``'C'`` for clamped or
    ``'F'`` for free. The tangential displacement and rotation of an edge of
    constant `x` are `v` and `\phi_y`, and of constant `y`, `u` and
    `\phi_x`.
    """
    tangential, normal = ('v', 'u') if edge.startswith('x') else ('u', 'v')
    rot_tangential, rot_normal = (('phiy', 'phix') if edge.startswith('x')
                                  else ('phix', 'phiy'))
    for field in ('u', 'v', 'w', 'phix', 'phiy'):
        setattr(s, edge + field, 1.)
        setattr(s, edge + field + 'r', 1.)
    if kind == 'S':
        setattr(s, edge + tangential, 0.)
        setattr(s, edge + 'w', 0.)
        setattr(s, edge + rot_tangential, 0.)
    elif kind == 'C':
        for field in ('u', 'v', 'w', 'phix', 'phiy'):
            setattr(s, edge + field, 0.)
        # the classical plate theory clamps the slope of w, whereas the
        # shear deformation theories clamp the rotations, with w,x free
        if 'clpt' in s.model:
            setattr(s, edge + 'wr', 0.)
    elif kind != 'F':
        raise ValueError(kind)


def make_plate(model, a, h, stack, laminaprop, bcs='SSSS', m=12, n=12,
               b=None):
    """Plate with the edges x = 0, y = 0, x = a and y = b given by ``bcs``"""
    s = Shell()
    s.model = model
    s.a = a
    s.b = a if b is None else b
    s.stack = stack
    s.plyt = h/len(stack)
    s.laminaprop = laminaprop
    s.rho = RHO
    s.m = m
    s.n = n
    for edge, kind in zip(('x1', 'y1', 'x2', 'y2'), bcs):
        set_edge(s, edge, kind)
    s._rebuild()
    return s


def lowest_frequencies(s, k=1, flexural=False):
    r"""The ``k`` lowest natural frequencies

    With ``flexural=True`` the in-plane modes are discarded, which is only
    meaningful for laminates without extension-bending coupling, whose modes
    are either purely flexural or purely in-plane.
    """
    K = s.calc_kC()
    M = s.calc_kM()
    K, used = remove_null_cols(K)
    M, _ = remove_null_cols(M)
    if not flexural:
        eigvals = eigsh(K, M=M, k=k, sigma=-1., which='LM',
                        return_eigenvectors=False)
        return np.sqrt(np.sort(eigvals))
    #NOTE the dense symmetric solver, because the eigenvectors of ARPACK
    #     proved unreliable for the clusters of flexural and in-plane modes
    num = k + 12
    eigvals, eigvecs = eigh(K.toarray(), M.toarray(),
                            subset_by_index=[0, num - 1])
    vecs = np.zeros((s.get_size(), num))
    vecs[used, :] = eigvecs
    if flexural:
        dofs = modelDB.db[s.model]['dofs']
        inplane = np.linalg.norm(vecs[0::dofs], axis=0) + np.linalg.norm(vecs[1::dofs], axis=0)
        w = np.linalg.norm(vecs[2::dofs], axis=0)
        eigvals = eigvals[inplane < 1e-6*w]
    assert len(eigvals) >= k
    return np.sqrt(eigvals[:k])


def orthotropic(E1E2, G12=0.6, G13=0.6, G23=0.5, nu12=0.25):
    return (E1E2*E2, E2, nu12, G12*E2, G13*E2, G23*E2)


def cross_ply(num_layers, symmetric=False):
    if symmetric:
        half = [0 if i % 2 == 0 else 90 for i in range(num_layers//2)]
        return half + half[::-1]
    return [0 if i % 2 == 0 else 90 for i in range(num_layers)]


# E1/E2: (TSDT of Reddy (1984), FSDT with k = 5/6 by the Navier solution)
NOOR_1973 = {
    # antisymmetric [0/90], a/h = 5, 3D elasticity of Noor (1973):
    # 6.2578, 6.9845, 7.6745, 8.1763, 8.5625
    2: {3: (6.2169, 6.20855), 10: (6.9887, 6.93924), 20: (7.8210, 7.70599),
        30: (8.5050, 8.32112), 40: (9.0871, 8.83331)},
    # antisymmetric [0/90]_2, a/h = 5, 3D elasticity of Noor (1973):
    # 6.5455, 8.1445, 9.4055, 10.1650, 10.679
    4: {3: (6.5008, 6.50425), 10: (8.1954, 8.2246), 20: (9.6265, 9.68846),
        30: (10.5348, 10.61976), 40: (11.1716, 11.27077)},
}


@pytest.mark.parametrize('num_layers', [2, 4])
@pytest.mark.parametrize('E1E2', [3, 10, 20, 30, 40])
def test_frequency_antisymmetric_cross_ply_noor(num_layers, E1E2):
    r"""Noor (1973), `a/h = 5`, `\bar\omega = \omega a^2/h \sqrt{\rho/E_2}`"""
    a = 1.
    h = a/5
    tsdt, fsdt = NOOR_1973[num_layers][E1E2]
    for model, ref, rtol in ((TSDT, tsdt, 2e-5), (FSDT, fsdt, 2e-6)):
        s = make_plate(model, a, h, cross_ply(num_layers),
                       orthotropic(E1E2))
        omega = lowest_frequencies(s)[0]
        assert np.isclose(omega*a**2/h*np.sqrt(RHO/E2), ref, rtol=rtol), model


# a/h: (FSDT of Whitney and Pagano (1970), TSDT of Reddy (1984))
WHITNEY_PAGANO = {2: (5.4998, 5.5065), 4: (9.3949, 9.3235),
                  10: (15.1426, 15.1073),
                  #NOTE for a/h = 20 the published TSDT value is 17.6457, which
                  #     differs from the Navier solution of the TSDT, 17.64657,
                  #     by a transposition of the last two digits, whereas all
                  #     the other values agree with the Navier solution to the
                  #     printed digits
                  20: (17.6596, 17.64657),
                  50: (18.6742, 18.6718), 100: (18.8362, 18.8356)}


@pytest.mark.parametrize('ah', list(WHITNEY_PAGANO))
def test_frequency_symmetric_cross_ply_vs_thickness(ah):
    r"""`[0/90/90/0]`, `E_1/E_2 = 40`, from thick to thin plates

    The classical plate theory, with rotary inertia, gives 15.9024 for `a/h
    = 2` and 18.8898 for `a/h = 100`, which the shear deformation theories
    approach as the plate becomes thin.
    """
    a = 1.
    h = a/ah
    fsdt, tsdt = WHITNEY_PAGANO[ah]
    for model, ref in ((FSDT, fsdt), (TSDT, tsdt)):
        s = make_plate(model, a, h, [0, 90, 90, 0], orthotropic(40))
        # for a/h = 2 an in-plane mode is the lowest one
        omega = lowest_frequencies(s, flexural=True)[0]
        assert np.isclose(omega*a**2/h*np.sqrt(RHO/E2), ref, rtol=2e-5), model


# layers: (TSDT of Reddy (1984), FSDT with k = 5/6 and CLPT by the Navier
# solution)
NOOR_1975 = {
    # 3D elasticity of Noor (1975): 21.2796
    4: (22.5790, 22.8060, 30.3591),
    # 3D elasticity of Noor (1975): 23.6689
    6: (24.4596, 24.5777, 33.5817),
    # 3D elasticity of Noor (1975): 24.9636
    10: (25.4225, 25.4500, 35.2316),
}


@pytest.mark.parametrize('num_layers', list(NOOR_1975))
def test_buckling_antisymmetric_cross_ply_noor(num_layers):
    r"""Noor (1975), uniaxial compression, `a/h = 10`, `E_1/E_2 = 40`

    With `\bar{N} = N_{cr} a^2/(E_2 h^3)`.
    """
    a = 1.
    h = a/10
    refs = NOOR_1975[num_layers]
    for model, ref in zip((TSDT, FSDT, CLPT), refs):
        s = make_plate(model, a, h, cross_ply(num_layers), orthotropic(40))
        s.Nxx = -1.
        eigvals, _ = lb(s.calc_kC(), s.calc_kG(), silent=True)
        assert np.isclose(eigvals[0]*a**2/(E2*h**3), ref, rtol=2e-5), model


def sinusoidal_load(s, q0):
    r"""External force vector of the pressure `q_0 \sin(\pi x/a) \sin(\pi y/b)`"""
    fg = modelDB.db[s.model]['field'].fg
    size = s.get_size()
    fext = np.zeros(size)
    g = np.zeros((5, size))
    points, weights = roots_legendre(2*max(s.m, s.n))
    for xi, wx in zip(points, weights):
        x = (xi + 1)*s.a/2
        for eta, wy in zip(points, weights):
            y = (eta + 1)*s.b/2
            g[:] = 0
            fg(g, x, y, s)
            q = q0*np.sin(np.pi*x/s.a)*np.sin(np.pi*y/s.b)
            fext += wx*wy*(s.a/2)*(s.b/2)*q*g[2]
    return fext


# a/h: {model: normalized central deflection}
PAGANO = {
    # 3D elasticity of Pagano (1970): 1.9540
    4: {TSDT: 1.8937},
    # 3D elasticity of Pagano (1970): 0.7430
    10: {TSDT: 0.7147, FSDT: 0.6628, CLPT: 0.43125},
    # 3D elasticity of Pagano (1970): 0.4347
    100: {TSDT: 0.4343, FSDT: 0.4337},
}


@pytest.mark.parametrize('ah', list(PAGANO))
def test_bending_symmetric_cross_ply_pagano(ah):
    r"""Pagano (1970), `[0/90/90/0]` under a sinusoidal pressure

    Material with `E_1 = 25 E_2`, `G_{12} = G_{13} = 0.5 E_2`, `G_{23} = 0.2
    E_2`, `\nu_{12} = 0.25`, and the central deflection normalized as
    `\bar{w} = 100 w E_2 h^3/(q_0 a^4)`. The classical plate theory gives
    0.43125 for any `a/h`.
    """
    a = 1.
    h = a/ah
    q0 = 1.e3
    laminaprop = orthotropic(25, G12=0.5, G13=0.5, G23=0.2)
    for model, ref in PAGANO[ah].items():
        s = make_plate(model, a, h, [0, 90, 90, 0], laminaprop)
        c = solve(s.calc_kC(), sinusoidal_load(s, q0), silent=True)
        _, fields = s.uvw(c, xs=np.array([a/2]), ys=np.array([a/2]))
        wbar = 100*fields['w'][0]*E2*h**3/(q0*a**4)
        # the references are given with 4 significant digits, except for the
        # FSDT with a/h = 10, 0.6628, which is one unit of the last digit
        # above its Navier solution with k = 5/6, 0.66271
        assert np.isclose(wbar, ref, rtol=2e-4), model


@pytest.mark.parametrize('ha, k, ref', [
    # Liew, Xiang and Kitipornchai (1993)
    (0.1, 5/6, 19.0651),
    (0.2, 5/6, 17.4485),
    # Hashemi and Arsanjani (2005)
    (0.1, 0.86667, 19.0840),
    (0.2, 0.86667, 17.5055),
    ])
def test_frequency_isotropic_mindlin_plate(ha, k, ref):
    r"""Square SSSS isotropic plate, `\lambda = \omega a^2 \sqrt{\rho h/D}`

    The classical plate theory gives `2 \pi^2 = 19.7392` without rotary
    inertia. The shear correction factor alone changes the result by 0.4 % for
    `h/a = 0.2`.
    """
    a = 1.
    h = ha*a
    E = 70.e9
    nu = 0.3
    s = make_plate(FSDT, a, h, [0], (E, nu))
    s.fsdt_shear_correction = k
    s._rebuild()
    D = E*h**3/(12*(1 - nu**2))
    omega = lowest_frequencies(s)[0]
    assert np.isclose(omega*a**2*np.sqrt(RHO*h/D), ref, rtol=1e-5)


def test_higher_modes_isotropic_thick_plate():
    r"""Square SSSS isotropic plate, `h/a = 0.1`, `\nu = 0.3`

    With `\bar\omega = \omega h \sqrt{\rho/G}`, for the modes `(1, 1)`,
    `(1, 2)`, `(2, 1)`, `(2, 2)`, `(1, 3)` and `(3, 1)`, the FSDT with `k =
    5/6` gives 0.0930, 0.2219, 0.2219, 0.3406, 0.4149 and 0.4149, as
    reproduced for the isotropic plate in Table 1 of "Free vibration
    analysis of advanced composite plates with porosities", Algerian Journal
    of Research and Technology (ajrt.dz). The 3D elasticity solution of Srinivas, Rao and Rao
    (1970) is about 0.0932, 0.2226, 0.3421 and 0.4171 for the distinct modes.
    """
    a = 1.
    h = 0.1*a
    E = 70.e9
    nu = 0.3
    G = E/(2*(1 + nu))
    s = make_plate(FSDT, a, h, [0], (E, nu), m=14, n=14)
    # discarding the in-plane modes, e.g. the shear mode 0.31416 = pi*h/a
    omegas = lowest_frequencies(s, k=6, flexural=True)*h*np.sqrt(RHO/G)
    refs = [0.0930, 0.2219, 0.2219, 0.3406, 0.4149, 0.4149]
    # the references are given with 3 significant digits
    assert np.allclose(omegas, refs, rtol=6e-4)


# Leissa (1973), with the edges x = 0, y = 0, x = a and y = b in this order
LEISSA_PLATES = {'SSSS': 19.739, 'SCSC': 28.951, 'SCSS': 23.646,
                 'SCSF': 12.687, 'SSSF': 11.685, 'SFSF': 9.631}


@pytest.mark.parametrize('model', [CLPT, FSDT, TSDT])
@pytest.mark.parametrize('bcs', list(LEISSA_PLATES))
def test_frequency_thin_isotropic_plate_leissa(bcs, model):
    r"""Leissa (1973), square plates, `\lambda = \omega a^2 \sqrt{\rho h/D}`

    For a thin plate, `h/a = 0.001`, the shear deformation theories must
    recover the classical plate theory for any boundary condition.
    """
    a = 1.
    h = 0.001*a
    E = 70.e9
    nu = 0.3
    s = make_plate(model, a, h, [0], (E, nu), bcs=bcs)
    # removing the in-plane rigid-body motions, which do not change the
    # flexural modes of the isotropic plate
    s.x1u = s.x2u = 0.
    D = E*h**3/(12*(1 - nu**2))
    omega = lowest_frequencies(s)[0]
    # the references are given with 5 significant digits
    assert np.isclose(omega*a**2*np.sqrt(RHO*h/D), LEISSA_PLATES[bcs],
                      rtol=5e-5)


@pytest.mark.parametrize('model', [FSDT, TSDT])
def test_analytical_and_numerical_matrices_agree(model):
    """The closed-form and the numerically integrated matrices"""
    s = make_plate(model, 0.3, 0.01, [30, -45, 0, 90], orthotropic(20),
                   bcs='SCFS', m=6, n=7, b=0.2)
    s.offset = 0.1e-3
    s._rebuild()
    size = s.get_size()
    s.Nxx, s.Nyy, s.Nxy = -3., 2., 5.
    k0 = s.calc_kC().toarray()
    kG = s.calc_kG().toarray()
    kM = s.calc_kM().toarray()
    #NOTE giving c or h_nxny forces the numerical integration
    k0_num = s.calc_kC(c=np.zeros(size)).toarray()
    kG_num = s.calc_kG(c=np.zeros(size)).toarray()
    kM_num = s.calc_kM(h_nxny=np.full((s.nx, s.ny), s.lam.h),
                       rho_nxny=np.full((s.nx, s.ny), RHO)).toarray()
    for A, B in ((k0, k0_num), (kG, kG_num), (kM, kM_num)):
        assert np.abs(A - B).max() <= 1e-12*np.abs(A).max()


@pytest.mark.parametrize('model', [FSDT, TSDT])
def test_rigid_body_modes_of_free_plate(model):
    """A free plate has 3 in-plane and 3 out-of-plane rigid-body modes"""
    s = make_plate(model, 0.3, 0.01, [30, -45, 0, 90], orthotropic(20),
                   bcs='FFFF', m=6, n=6, b=0.2)
    eigvals = np.linalg.eigvalsh(s.calc_kC().toarray())
    # the zero eigenvalues are at the round-off level of the membrane
    # stiffness, whereas the lowest flexural one is about 1e-12 of it
    assert np.all(np.abs(eigvals[:6]) < 1e-15*eigvals[-1])
    assert eigvals[6] > 1e3*np.abs(eigvals[:6]).max()


if __name__ == '__main__':
    test_frequency_antisymmetric_cross_ply_noor(2, 40)
    test_buckling_antisymmetric_cross_ply_noor(4)
    test_bending_symmetric_cross_ply_pagano(10)
    test_frequency_isotropic_mindlin_plate(0.1, 5/6, 19.0651)


def test_shear_correction_options():
    """A float k scales the uncorrected transverse shear stiffness, whereas a
    string selects the method of the composites module"""
    kwargs = dict(a=0.3, h=0.01, stack=[0, 45, 90], laminaprop=orthotropic(20),
                  m=4, n=4)
    s = make_plate(FSDT, **kwargs)
    Abar = s.lam.Abar_ts
    assert np.allclose(s.ABD[6:, 6:], 5/6*Abar)
    s.fsdt_shear_correction = 'constant'
    s._rebuild()
    assert np.allclose(s.ABD[6:, 6:], 5/6*Abar)
    s.fsdt_shear_correction = None
    s._rebuild()
    assert np.allclose(s.ABD[6:, 6:], Abar)
    s.fsdt_shear_correction = 'rohwer'
    s._rebuild()
    assert np.allclose(s.ABD[6:, 6:], s.lam.Ats)
    assert not np.allclose(s.ABD[6:, 6:], 5/6*Abar)
    # the TSDT needs no shear correction
    s = make_plate(TSDT, **kwargs)
    assert np.allclose(s.ABD[9:11, 9:11], Abar)
    assert np.allclose(s.ABD[9:11, 11:13], s.lam.Dtrans)
    assert np.allclose(s.ABD[11:13, 11:13], s.lam.Ftrans)


@pytest.mark.parametrize('model', [FSDT, TSDT])
def test_stiffpanelbay_rejects_5_dof_models(model):
    r"""The stiffener kernels assume the 3 DOFs of the classical laminated
    plate theory, the multi-domain models are tested in
    ``tests/multidomain/test_multidomain_fsdt_tsdt.py``"""
    from panels.stiffpanelbay import StiffPanelBay
    bay = StiffPanelBay()
    bay.a = 0.3
    bay.b = 0.2
    bay.model = model
    bay.stack = [0, 90]
    bay.plyt = 0.005
    bay.laminaprop = orthotropic(20)
    bay.m = bay.n = 4
    bay.add_panel(0, 0.2)
    with pytest.raises(NotImplementedError, match='3 DOFs'):
        bay.calc_kC(silent=True)
