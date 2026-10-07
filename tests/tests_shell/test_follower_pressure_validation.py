r"""Validation of the follower pressure: rings and long sandwich cylinders

A long cylinder under a uniform pressure buckles as a ring, in generalized
plane strain. It is modelled with a quarter of the circumference, ``b = pi
r/2``, with symmetry planes at the straight edges (``v = 0``, ``w,y = 0``) and
at the ends (``u = 0``, ``v,x = w,x = 0``), which contains the ring modes
``w = cos(n y/r)`` with ``n = 2, 4, ...``. The linear buckling problem is

.. math::
    ([K_0] + \lambda ([K_G(c_0)] + [K_{C_{follower}}]))\{c\} = \{0\}

with `c_0` the linear static solution of a unit external pressure, solved by
:func:`structsolve.lb` with the unsymmetric solvers when needed.

1. Closed forms for the ring, :func:`ring_closed_form`, the same as
   ``ring_buckling`` of ``theory/shells/follower_pressure/follower_pressure.py``
   (sympy): with Sanders' kinematics and ``n = 2``, ``p R**3/D = -3`` for the
   follower (hydrostatic) pressure, ``-4`` for a pressure of constant
   direction (dead load) and ``-4.5`` for a centrally directed load, in the
   inextensional limit, the analytical values of Table 3 of Schweizerhof and
   Ramm (1984), *Comput. Struct.* 18:1099-1114. With Donnell's kinematics the
   follower pressure gives ``-16/5``, the known inaccuracy of Donnell's theory
   for low ``n``.

2. The long sandwich cylinders of Han, Kardomateas and Simitses (2004),
   *Compos. B* 35:591-598, Table 1 materials, faces ``f = 0.1 in`` with the
   fibres along the hoop direction, core ``c = 1 in``, ``R0/h = 15...120``.
   The FSDT ring with the follower pressure is exactly their shell formula
   ``p = 3 (EI)eq/(B R0**3 (1 + 4 ks))`` in the inextensional limit, with the
   transverse shear stiffness ``C/B`` of the core only, Eq. (14a), or of the
   effective modulus ``Gbar``, Eqs. (14b) and (14c), which is checked against
   their Table 2. The comparison with their elasticity solution, with the
   pressure on the outer surface (``zp = h/2``), is printed by running this
   module.

3. Finite cylinders with shear-diaphragm ends under lateral or hydrostatic
   pressure, against the exact solution of the same Sanders or Donnell
   strains with trigonometric modes, :func:`cylinder_closed_form`, whose
   modes depend on ``x`` and test the terms of ``u`` of the follower load.

4. Thick orthotropic and isotropic rings of Kardomateas (1993), *J. Appl.
   Mech.* 60:195-202, Tables 1 and 2, and the homogeneous graphite/epoxy ring
   of Kardomateas and Simitses (2003), Table 4, against their shell formulas
   and 3D elasticity solutions.

5. Batdorf's formula for simply supported cylinders under lateral and
   hydrostatic pressure, NASA/SP-8007-2020/REV 2, the Donnell solution with a
   pressure of constant direction, :func:`batdorf_lateral_pressure`.

6. The cylinders of Schweizerhof and Ramm (1984), Table 4, under a lateral
   pressure of constant direction (``k_c``) or follower (``k_f``), hinged at
   both ends or cantilevers with a free upper edge, B1 and B2, for which the
   load stiffness is unsymmetric.

Units: N, m, Pa.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.linalg import eig
from scipy.special import roots_legendre
from structsolve import solve, lb

from panels import modelDB
from panels.shell import Shell


def ring_closed_form(kinematics, load, n, A, D, R, S=None):
    r"""Buckling pressure of a ring in plane strain, mode ``cos(n y/R)``

    Numerical version of ``ring_buckling`` in
    ``theory/shells/follower_pressure/follower_pressure.py``: the second
    variation of the total potential of the amplitudes ``(W, V)``, plus ``P``
    of ``phiy`` for the FSDT (``S``, the transverse shear stiffness, not
    ``None``), is ``H0 + p H1`` and the pressure of smallest modulus that
    makes it singular is returned, negative for an external pressure.

    """
    k = n/R
    kv = 1/R if kinematics == 'sanders' else 0.
    fsdt = S is not None
    nd = 3 if fsdt else 2
    # coefficient vectors of the amplitudes (W, V, P) of each strain, whose
    # integral over the circumference is pi R times the square
    eyy = np.array([1/R, k, 0.])
    if fsdt:
        kyy = np.array([0., kv*k, k])
        gyz = np.array([-k, 0., 1.])
    else:
        kyy = np.array([k**2, kv*k, 0.])
        gyz = np.zeros(3)
    byy = np.array([-k, -kv, 0.])
    H0 = A*np.outer(eyy, eyy) + D*np.outer(kyy, kyy)
    if fsdt:
        H0 += S*np.outer(gyz, gyz)
    H1 = R*np.outer(byy, byy)
    if load == 'follower':
        L = np.array([[1/R, k, 0.], [k, kv, 0.], [0., 0., 0.]])
    elif load == 'dead':
        L = np.zeros((3, 3))
    elif load == 'central':
        L = np.diag([0., 1/R, 0.])
    else:
        raise ValueError(load)
    H1 -= L
    H0 = H0[:nd, :nd]
    H1 = H1[:nd, :nd]
    ps = eig(H0, -H1, right=False)
    ps = ps[np.isfinite(ps)]
    ps = ps[np.abs(ps.imag) <= 1e-10*np.abs(ps)].real
    ps = ps[ps != 0]
    return ps[np.argmin(np.abs(ps))]


def ring_shell(model, R, stack, plyts, laminaprops, m=4, n=15, **kwargs):
    r"""Quarter ring in generalized plane strain, see the module notes"""
    s = Shell(model=model, **kwargs)
    s.r = R
    s.a = 0.5*R
    s.b = np.pi/2*R
    s.stack = list(stack)
    s.plyts = list(plyts)
    s.laminaprops = list(laminaprops)
    s.m = m
    s.n = n
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, e + f, 1.)
            setattr(s, e + f + 'r', 1.)
    # symmetry planes at the ends: generalized plane strain
    s.x1u = s.x2u = 0.
    s.x1vr = s.x2vr = 0.
    s.x1wr = s.x2wr = 0.
    # symmetry planes at the straight edges, modes cos(n y/r), n even
    s.y1v = s.y2v = 0.
    s.y1wr = s.y2wr = 0.
    # rotations: phiy free at the ends, phix zero there (symmetry), phiy
    # zero at the straight edges (it varies as sin(n y/r))
    s.x1phiy = s.x2phiy = 1.
    s.x1phix = s.x2phix = 0.
    s.y1phix = s.y2phix = 1.
    s.y1phiy = s.y2phiy = 0.
    s._rebuild()
    return s


def central_load_stiffness(s, p):
    r"""Load stiffness of a centrally directed load, `-\partial F/\partial c`
    with `F_v = \int p v/r N_v dA`, integrated independently of the kernels"""
    fg = modelDB.db[s.model]['field'].fg
    size = s.get_size()
    g = np.zeros((5, size))
    K = np.zeros((size, size))
    xs, wx = roots_legendre(s.nx)
    ys, wy = roots_legendre(s.ny)
    for xi, wxi in zip(xs, wx):
        for eta, wyi in zip(ys, wy):
            fg(g, (xi + 1)*s.a/2, (eta + 1)*s.b/2, s)
            K -= wxi*wyi*s.a*s.b/4*p/s.r*np.outer(g[1], g[1])
    return K


def ring_buckling_pressure(s, load='follower', zp=0., return_mode=False):
    r"""Critical external pressure of the ring ``s`` by linear buckling, as a
    positive value, the load multiplier of a unit external pressure"""
    s.clear_loads()
    s.add_pressure_load(-1., cte=False, follower=(load == 'follower'), zp=zp)
    k0 = s.calc_kC()
    c0 = solve(k0, s.calc_fext(), silent=True)
    K = s.calc_kG(c=c0)
    if load == 'follower':
        K = K + s.calc_kCfollower()
    elif load == 'central':
        K = K + central_load_stiffness(s, -1.*(1 + zp/s.r))
    eigvals, eigvecs = lb(k0, K, silent=True, num_eigvalues=4)
    if return_mode:
        return eigvals[0], eigvecs[:, 0]
    return eigvals[0]


E = 70.e9
NU = 0.3
G = E/(2*(1 + NU))
ISO = (E, E, NU, G, G, G)


@pytest.mark.parametrize('load', ['follower', 'dead', 'central'])
@pytest.mark.parametrize('model', ['cylshell_clpt_sanders',
                                   'cylshell_clpt_donnell',
                                   'cylshell_fsdt_sanders',
                                   'cylshell_fsdt_donnell'])
def test_isotropic_ring(model, load):
    R = 1.
    h = 0.01
    s = ring_shell(model, R, [0.], [h], [ISO], fsdt_shear_correction=5/6)
    A22 = s.ABD[1, 1]
    D22 = s.ABD[4, 4]
    S = s.ABD[6, 6] if 'fsdt' in model else None
    kin = 'sanders' if 'sanders' in model else 'donnell'
    ref = -ring_closed_form(kin, load, 2, A22, D22, R, S)
    p_cr = ring_buckling_pressure(s, load)
    # the discretization error of 15 terms along y is below 1e-8
    assert np.isclose(p_cr, ref, rtol=1.e-6), (p_cr, ref)
    inextensional = {('sanders', 'follower'): 3., ('sanders', 'dead'): 4.,
                     ('sanders', 'central'): 4.5,
                     ('donnell', 'follower'): 16/5, ('donnell', 'dead'): 4.,
                     ('donnell', 'central'): 64/15}
    # r/h = 100: extensibility and transverse shear of order (h/r)**2
    assert np.isclose(p_cr*R**3/D22, inextensional[(kin, load)], rtol=1e-3)


def test_isotropic_ring_tsdt():
    R = 1.
    h = 0.01
    for kin, ref in (('sanders', 3.), ('donnell', 16/5)):
        s = ring_shell('cylshell_tsdt_' + kin, R, [0.], [h], [ISO])
        assert np.isclose(ring_buckling_pressure(s)*R**3/s.ABD[4, 4], ref,
                          rtol=1e-3)


def test_follower_load_stiffness_is_symmetric_for_the_ring():
    s = ring_shell('cylshell_fsdt_sanders', 1., [0.], [0.01], [ISO])
    s.add_pressure_load(-1., follower=True)
    K = s.calc_kCfollower().toarray()
    assert np.linalg.norm(K - K.T) <= 1e-12*np.linalg.norm(K)


def test_mode_is_n2():
    s = ring_shell('cylshell_clpt_sanders', 1., [0.], [0.01], [ISO])
    p, mode = ring_buckling_pressure(s, return_mode=True)
    ys = np.linspace(0, s.b, 41)
    _, fields = s.uvw(mode, xs=np.full_like(ys, s.a/2), ys=ys)
    w = fields['w'].ravel()
    w /= w[np.argmax(np.abs(w))]
    assert np.allclose(w, np.cos(2*ys/s.r)*np.sign(w[0]), atol=1e-6)


def cylinder_closed_form(kinematics, load, m, n, L, R, ABD, Nxx, Nyy, p):
    r"""Load multiplier of a cylinder with shear-diaphragm ends

    Mode ``u = U cos(l x) cos(n t)``, ``v = V sin(l x) sin(n t)``, ``w = W
    sin(l x) cos(n t)``, ``t = y/R``, ``l = m pi/L``, under the membrane
    prestate ``lam (Nxx, Nyy)`` and the pressure ``lam p``, a ``'follower'``
    pressure with the first-order area vector or a ``'dead'`` one, with the
    CLPT strains of the kernels, Sanders or Donnell. The fields are
    evaluated at quadrature points that integrate the trigonometric products
    exactly, and ``ABD`` must be orthotropic (no 16, 26 or B terms). Returns
    the smallest positive ``lam``.

    """
    l = m*np.pi/L
    k = n/R
    kv = 1/R if kinematics == 'sanders' else 0.
    xg, wg = np.polynomial.legendre.leggauss(2*m + 8)
    xs = (xg + 1)*L/2
    nt = 4*n + 8
    ts = np.arange(nt)*2*np.pi/nt
    X, T = np.meshgrid(xs, ts, indexing='ij')
    W8 = np.outer(wg*L/2, np.full(nt, 2*np.pi*R/nt))
    cx, sx, cn, sn = np.cos(l*X), np.sin(l*X), np.cos(n*T), np.sin(n*T)
    zero = np.zeros_like(X)
    names = ('u', 'ux', 'uy', 'v', 'vx', 'vy', 'w', 'wx', 'wy', 'wxx', 'wyy',
             'wxy')
    F = [dict.fromkeys(names, zero) for _ in range(3)]
    F[0].update(u=cx*cn, ux=-l*sx*cn, uy=-k*cx*sn)
    F[1].update(v=sx*sn, vx=l*cx*sn, vy=k*sx*cn)
    F[2].update(w=sx*cn, wx=l*cx*cn, wy=-k*sx*sn, wxx=-l**2*sx*cn,
                wyy=-k**2*sx*cn, wxy=-l*k*cx*sn)

    def strains(f):
        kyy = -f['wyy']
        kxy = -2*f['wxy']
        if kinematics == 'sanders':
            kyy = kyy + f['vy']/R
            kxy = kxy + 1.5*f['vx']/R - 0.5*f['uy']/R
        return np.array([f['ux'], f['vy'] + f['w']/R, f['uy'] + f['vx'],
                         -f['wxx'], kyy, kxy])

    E = [strains(f) for f in F]
    H0 = np.zeros((3, 3))
    H1 = np.zeros((3, 3))
    for i, fi in enumerate(F):
        for j, fj in enumerate(F):
            H0[i, j] = np.sum(W8*np.einsum('a...,ab,b...->...', E[i], ABD,
                                           E[j]))
            byi = fi['wy'] - kv*fi['v']
            byj = fj['wy'] - kv*fj['v']
            H1[i, j] = np.sum(W8*(Nxx*fi['wx']*fj['wx'] + Nyy*byi*byj))
            if load == 'follower':
                a1 = (-fj['wx'], -fj['wy'] + kv*fj['v'],
                      fj['ux'] + fj['vy'] + fj['w']/R)
                H1[i, j] -= np.sum(W8*p*(fi['u']*a1[0] + fi['v']*a1[1]
                                         + fi['w']*a1[2]))
    lam = eig(H0, -H1, right=False)
    lam = lam[np.isfinite(lam)]
    lam = lam[np.abs(lam.imag) <= 1e-8*np.abs(lam)].real
    return lam[lam > 0].min()


def cylinder_shell(model, L, R, h, m=8, n=25):
    r"""Half length and half circumference of a cylinder with
    shear-diaphragm ends, ``v = w = 0``, with symmetry planes at the
    mid-length and at ``t = 0, pi``, which contain the modes of
    :func:`cylinder_closed_form` with ``m`` odd and any ``n``"""
    s = Shell(model=model)
    s.r = R
    s.a = L/2
    s.b = np.pi*R
    s.stack = [0.]
    s.plyt = h
    s.laminaprop = (E, NU)
    s.m = m
    s.n = n
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, e + f, 1.)
            setattr(s, e + f + 'r', 1.)
    s.x1v = s.x1w = 0.
    s.x2u = 0.
    s.x2vr = s.x2wr = 0.
    s.y1v = s.y2v = 0.
    s.y1wr = s.y2wr = 0.
    s._rebuild()
    return s


@pytest.mark.parametrize('model,load,prestate', [
    ('cylshell_clpt_sanders', 'follower', 'lateral'),
    ('cylshell_clpt_sanders', 'follower', 'hydrostatic'),
    ('cylshell_clpt_sanders', 'dead', 'hydrostatic'),
    ('cylshell_clpt_donnell', 'follower', 'lateral'),
    ])
def test_finite_cylinder_shear_diaphragm(model, load, prestate):
    r"""Finite cylinder, ``L/R = 1.5``, ``R/h = 500``, under a lateral
    (``Nxx = 0``) or hydrostatic (``Nxx = p R/2``, the end caps as a dead
    axial load) external pressure, with the membrane prestate given by
    ``Nxx`` and ``Nyy`` and the pressure as a follower or dead load. The
    modes depend on ``x``, testing the terms of ``u`` of the follower load"""
    L, R, h = 1.5, 1., 0.002
    p = -1.
    s = cylinder_shell(model, L, R, h)
    Nxx = 0. if prestate == 'lateral' else p*R/2
    s.Nxx = Nxx
    s.Nyy = p*R
    s.add_pressure_load(p, cte=False, follower=(load == 'follower'))
    K = s.calc_kG()
    if load == 'follower':
        K = K + s.calc_kCfollower()
    lam = lb(s.calc_kC(), K, silent=True, num_eigvalues=4)[0][0]
    kin = 'sanders' if 'sanders' in model else 'donnell'
    ref = min(cylinder_closed_form(kin, load, 1, n, L, R, s.ABD, Nxx, p*R, p)
              for n in range(2, 16))
    assert np.isclose(lam, ref, rtol=1e-4), (lam, ref)


# Han, Kardomateas and Simitses (2004), Table 1, with 1 = r, 2 = theta and
# 3 = z. As laminaprop of a ply with the fibres along the hoop direction
# (90 deg): E11 = E2, E22 = E3, nu12 = nu23, G12 = G23 (theta-z),
# G13 = G12 (r-theta, the transverse shear of the ring), G23 = G31
GPa = 1.e9
FACES = {
    'boron': (221.0*GPa, 20.7*GPa, 0.23, 5.79*GPa, 5.79*GPa, 3.29*GPa),
    'graphite': (181.0*GPa, 10.3*GPa, 0.28, 7.17*GPa, 7.17*GPa, 5.96*GPa),
    'kevlar': (75.9*GPa, 5.52*GPa, 0.34, 2.28*GPa, 2.28*GPa, 1.89*GPa),
}
# the isotropic core, G = E/(2 (1 + nu)) = 0.017256 GPa, which Table 1 rounds
# to 0.0173 GPa; with it the shell formulas reproduce Tables 2 and 3 within
# 5e-5, see test_han2004_formulas_reproduce_tables
GCORE = 0.0459*GPa/(2*(1 + 0.33))
CORE = (0.0459*GPa, 0.0459*GPa, 0.33, GCORE, GCORE, GCORE)
INCH = 0.0254
F = 0.1*INCH
C = 1.0*INCH
H = 2*F + C
RATIOS = (15, 30, 60, 120)
# Table 2 and Table 4: elasticity, shell with the core shear only, Eq.
# (14a), shell with the effective shear modulus, Eqs. (14b) and (14c), and
# finite elements S8R and C3D20R, in Pa
HAN2004 = {
    'boron': [(741773, 651125, 899768, 809208, 783556),
              (277305, 253721, 323361, 298839, 287722),
              (70416, 67383, 76087, 73863, 72373),
              (11817, 11717, 12203, 12189, 12106)],
    'graphite': [(720842, 637826, 874654, 790739, 762926),
                 (258549, 238236, 298643, 278606, 268754),
                 (61528, 59207, 65825, 64381, 63244),
                 (9918, 9829, 10168, 10184, 10126)],
    'kevlar': [(605472, 551668, 719856, 673845, 647488),
               (171351, 162433, 188347, 184090, 179624),
               (31418, 30712, 32397, 32633, 32341),
               (4476, 4403, 4470, 4533, 4522)],
}


# Table 2, classical shell formula, Eq. (12), in Pa
HAN2004_CLASSICAL = {'boron': (6898740, 862343, 107793, 13474),
                     'graphite': (5650460, 706307, 88288, 11036),
                     'kevlar': (2370590, 296324, 37040, 4630)}
# Table 3, thicker faces, f = 0.3 in, c = 0.6 in, graphite/epoxy faces:
# elasticity, classical shell, shell with the core shear only and with the
# effective modulus Gbar, in Pa
HAN2004_TABLE3 = [(15, 1244010, 11731900, 416091, 1501160),
                  (30, 393573, 1466490, 188038, 542378),
                  (60, 105699, 183311, 67900, 128553),
                  (120, 19297, 22914, 16081, 20709)]


def sandwich_ring(face, ratio, model='cylshell_fsdt_sanders', shear=None,
                  f=F, c=C):
    r"""Ring of Han et al. (2004), faces ``f`` and core ``c``, with ``shear``
    the transverse shear stiffness ``C/B`` of the shell formulas, or ``None``
    for the default of the model"""
    s = ring_shell(model, ratio*(2*f + c), [90., 0., 90.], [f, c, f],
                   [FACES[face], CORE, FACES[face]])
    if shear is not None:
        Abar44 = s.lam.Abar_ts[0, 0]
        s.fsdt_shear_correction = shear/Abar44
        s._rebuild()
        assert np.isclose(s.ABD[6, 6], shear)
    return s


def han_shear_stiffness(face, kind, f=F, c=C):
    r"""``C/B`` of Eq. (14a), ``'core'``, or Eqs. (14b, 14c), ``'Gbar'``"""
    Gf = FACES[face][4]
    Gc = CORE[4]
    if kind == 'core':
        return c*Gc
    Gbar = (2*f + c)/(2*f/Gf + c/Gc)
    return (2*f + c)*Gbar


def han_shell_formulas(face, ratio, f=F, c=C):
    r"""Eqs. (12)-(14) of Han et al. (2004): classical, core-only and
    effective-modulus shell formulas per unit width, ``(EI)eq/B`` with the
    hoop modulus ``Ef`` of the faces"""
    Ef = FACES[face][0]
    Ec = CORE[0]
    R0 = ratio*(2*f + c)
    EI = Ef*f**3/6 + 2*Ef*f*(f/2 + c/2)**2 + Ec*c**3/12
    p_cl = 3*EI/R0**3
    out = [p_cl]
    for kind in ('core', 'Gbar'):
        ks = EI/(han_shear_stiffness(face, kind, f, c)*R0**2)
        out.append(p_cl/(1 + 4*ks))
    return out


def test_han2004_formulas_reproduce_tables():
    r"""The shell formulas of Han et al. (2004) reproduce their Tables 2 and
    3 (to the rounding of the tables), such that the definitions of the
    stiffnesses used here are theirs"""
    for face in FACES:
        for i, ratio in enumerate(RATIOS):
            elast, core_only, gbar = HAN2004[face][i][:3]
            cl, co, gb = han_shell_formulas(face, ratio)
            assert np.allclose([cl, co, gb], [HAN2004_CLASSICAL[face][i],
                               core_only, gbar], rtol=2e-4)
    f, c = 0.3*INCH, 0.6*INCH
    for ratio, elast, classical, core_only, gbar in HAN2004_TABLE3:
        assert np.allclose(han_shell_formulas('graphite', ratio, f, c),
                           [classical, core_only, gbar], rtol=2e-4)


@pytest.mark.parametrize('face', ['boron', 'graphite', 'kevlar'])
def test_han2004_shell_formulas(face):
    r"""The FSDT ring reproduces the shell formulas of Table 2"""
    for i, ratio in enumerate(RATIOS):
        elast, core_only, gbar, s8r, c3d20 = HAN2004[face][i]
        for kind, ref in (('core', core_only), ('Gbar', gbar)):
            S = han_shear_stiffness(face, kind)
            s = sandwich_ring(face, ratio, shear=S)
            p_cr = ring_buckling_pressure(s)
            # the closed form of the same discrete theory
            closed = -ring_closed_form('sanders', 'follower', 2, s.ABD[1, 1],
                                       s.ABD[4, 4], s.r, S)
            assert np.isclose(p_cr, closed, rtol=1e-6)
            # the shell formula of Han et al. uses (EI)eq with Ef, panels the
            # plane-strain D22, E/(1 - nu12 nu21) is 0.5% larger for boron
            # and less for the other faces; the extensibility of the ring is
            # also neglected by the formula
            assert np.isclose(p_cr, ref, rtol=1e-2), (face, ratio, kind,
                                                     p_cr, ref)


def batdorf_lateral_pressure(L, R, t, E, nu, prestate):
    r"""Buckling pressure of Batdorf, NASA/SP-8007-2020/REV 2, Eqs.
    (19)-(22) with ``gamma = 1`` (no knockdown), the minimum over the integer
    circumferential wave numbers ``n = beta pi R/L``. It is the Donnell
    solution of a simply supported cylinder under a pressure of constant
    direction, ``'lateral'`` or ``'hydrostatic'`` (axial load ``p R/2``)"""
    D = E*t**3/(12*(1 - nu**2))
    Z = L**2/(R*t)*np.sqrt(1 - nu**2)
    fac = 0. if prestate == 'lateral' else 0.5
    ks = []
    for n in range(2, 60):
        beta = n*L/(np.pi*R)
        ks.append(((1 + beta**2)**2 + 12*Z**2/(np.pi**4*(1 + beta**2)**2))
                  /(beta**2 + fac))
    return min(ks)*np.pi**2*D/(R*L**2)


@pytest.mark.parametrize('LR,Rt,prestate', [(1., 100., 'lateral'),
                                            (3., 100., 'hydrostatic')])
def test_nasa_sp8007_batdorf(LR, Rt, prestate):
    r"""The Donnell model with the dead load reproduces Batdorf's formula of
    NASA/SP-8007-2020/REV 2, the follower load lowers it, by 1.3% and 3.6%
    here"""
    R = 1.
    t = R/Rt
    L = LR*R
    p = -1.
    s = cylinder_shell('cylshell_clpt_donnell', L, R, t, m=8, n=30)
    Nxx = 0. if prestate == 'lateral' else p*R/2
    ref = batdorf_lateral_pressure(L, R, t, E, NU, prestate)
    lams = []
    for follower in (False, True):
        s.clear_loads()
        s.Nxx = Nxx
        s.Nyy = p*R
        s.add_pressure_load(p, cte=False, follower=follower)
        K = s.calc_kG()
        if follower:
            K = K + s.calc_kCfollower()
        lams.append(lb(s.calc_kC(), K, silent=True, num_eigvalues=4)[0][0])
    assert np.isclose(lams[0], ref, rtol=1e-4), (lams[0], ref)
    assert 0.9*ref < lams[1] < 0.995*ref


# Kardomateas (1993), 1 = r, 2 = theta, 3 = z: E1 = E3 = 14, E2 = 57,
# G12 = G23 = 5.7, G31 = 5.0 GPa, nu23 = 0.277; as laminaprop of a ply with
# the fibres along the hoop direction (90 deg): E11 = E2, E22 = E3, nu12 =
# nu23, G12 = G23 (theta-z), G13 = G12 (r-theta), G23 = G31 (z-r)
KARD1993_ORTHO = (57.*GPa, 14.*GPa, 0.277, 5.7*GPa, 5.7*GPa, 5.0*GPa)
KARD1993_ISO = (57.*GPa, 57.*GPa, 0.3, 57.*GPa/2.6, 57.*GPa/2.6, 57.*GPa/2.6)
# R2/R1, elasticity and shell, Eq. (29a), p_cr R2**3/(E2 h**3), with R1 = 1 m
KARD1993 = {
    'ortho': [(1.10, 0.2728, 0.2930), (1.15, 0.2768, 0.3119),
              (1.20, 0.2784, 0.3308), (1.25, 0.2780, 0.3495),
              (1.30, 0.2762, 0.3681), (1.35, 0.2733, 0.3864),
              (1.40, 0.2696, 0.4046)],
    'iso': [(1.10, 0.2999, 0.3159), (1.15, 0.3109, 0.3363),
            (1.20, 0.3209, 0.3567), (1.25, 0.3301, 0.3769),
            (1.30, 0.3384, 0.3969), (1.35, 0.3459, 0.4167),
            (1.40, 0.3528, 0.4363)],
}


def kardomateas1993_ring(kind, ratio, model='cylshell_clpt_sanders', zp=0.):
    r"""``p_cr R2**3/(E2 h**3)`` of the ring with ``R2/R1 = ratio``"""
    R1 = 1.
    R2 = ratio*R1
    h = R2 - R1
    mat = KARD1993_ORTHO if kind == 'ortho' else KARD1993_ISO
    s = ring_shell(model, (R1 + R2)/2, [90.], [h], [mat])
    return s, ring_buckling_pressure(s, zp=zp*h)*R2**3/(57.*GPa*h**3)


@pytest.mark.parametrize('kind', ['ortho', 'iso'])
def test_kardomateas1993_shell_formula(kind):
    r"""The CLPT ring against Eq. (29a), ``p = E2 (n**2 - 1) h**3/(12 (1 -
    nu23 nu32) R**3)``, which neglects the extensibility of the ring, of
    relative order ``h**2/(12 R**2)``, below 1% up to ``R2/R1 = 1.4``"""
    for ratio, elast, shell in KARD1993[kind]:
        s, p = kardomateas1993_ring(kind, ratio)
        h = s.plyts[0]
        closed = -ring_closed_form('sanders', 'follower', 2, s.ABD[1, 1],
                                   s.ABD[4, 4], s.r, None)
        assert np.isclose(p, closed*(ratio)**3/(57.*GPa*h**3), rtol=1e-6)
        assert p < shell
        assert np.isclose(p, shell, rtol=1.2*h**2/(12*s.r**2) + 1e-3), (
            ratio, p, shell)


@pytest.mark.parametrize('ratio,elast,classical', [(30, 1641360, 1675930),
                                                   (60, 208228, 209491),
                                                   (120, 26180, 26186)])
def test_kardomateas2003_homogeneous_ring(ratio, elast, classical):
    r"""Homogeneous graphite/epoxy ring, ``h = 30.48 mm``, of Table 4 of
    Kardomateas and Simitses (2003): the FSDT and the TSDT with the pressure
    on the outer surface agree with the elasticity solution within 0.3%,
    and the CLPT with the classical formula ``3 E2 h**3/(12 R**3)`` within the
    plane-strain factor ``1/(1 - nu12 nu21)``"""
    mat = FACES['graphite']
    R = ratio*H
    s = ring_shell('cylshell_clpt_sanders', R, [90.], [H], [mat])
    p = ring_buckling_pressure(s)
    nu21 = mat[2]*mat[1]/mat[0]
    assert np.isclose(p, classical/(1 - mat[2]*nu21), rtol=2e-3)
    for model in ('cylshell_fsdt_sanders', 'cylshell_tsdt_sanders'):
        s = ring_shell(model, R, [90.], [H], [mat])
        p = ring_buckling_pressure(s, zp=H/2)
        assert np.isclose(p, elast, rtol=3e-3), (model, p, elast)


SCHWEIZERHOF_E = 2.1e6
SCHWEIZERHOF_NU = 0.3
# Table 4 of Schweizerhof and Ramm (1984): (shell, L/R, R/t, boundary
# conditions, n, k_c, k_f), with p_cr = k E t**3/(12 (1 - nu**2) R**3); the
# values of k_f of the cantilevers are those with the symmetrized load
# stiffness of their Eq. (63), the circumferential wave number n is the one
# with the lowest load of panels, the same as theirs except for the hinged
# shell 3 (7 or 8 in the reference)
SCHWEIZERHOF1984 = [
    (1, 6., 100., 'hinged', 4, 18.76, 17.62),
    (1, 6., 100., 'B1', 3, 10.89, 9.70),
    (1, 6., 100., 'B2', 3, 9.11, 8.10),
    (2, 3., 100., 'hinged', 5, 35.25, 34.28),
    (2, 3., 100., 'B1', 4, 20.85, 19.58),
    (2, 3., 100., 'B2', 4, 16.43, 15.42),
    (3, 3., 500., 'hinged', 7, None, 77.48),
    (3, 3., 500., 'B2', 6, 36.48, 35.48),
]


def schweizerhof_sector(L, R, t, n, bc, model='cylshell_clpt_sanders', m=14,
                        nt=8):
    r"""Half wave of the cylinder, ``b = pi R/n``, with symmetry planes at the
    straight edges, under a lateral pressure:

    - ``'hinged'``: both ends with ``v = w = 0``, ``u`` free (one point fixed);
    - ``'B1'``: lower end with ``u = v = w = 0``, upper end free;
    - ``'B2'``: lower end with ``v = w = 0``, ``u`` free (one point fixed),
      upper end free.

    The bending moment is zero at the supported ends in all cases.
    """
    s = Shell(model=model)
    s.r = R
    s.a = L
    s.b = np.pi*R/n
    s.stack = [0.]
    s.plyt = t
    s.laminaprop = (SCHWEIZERHOF_E, SCHWEIZERHOF_NU)
    s.m = m
    s.n = nt
    for e in ('x1', 'x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, e + f, 1.)
            setattr(s, e + f + 'r', 1.)
    s.y1v = s.y2v = 0.
    s.y1wr = s.y2wr = 0.
    if bc == 'hinged':
        s.x1v = s.x1w = s.x2v = s.x2w = 0.
    elif bc == 'B1':
        s.x1u = s.x1v = s.x1w = 0.
    elif bc == 'B2':
        s.x1v = s.x1w = 0.
    else:
        raise ValueError(bc)
    s._rebuild()
    return s


def schweizerhof_k(s, follower):
    r"""Buckling factor ``k`` with the prestate of a linear static analysis"""
    s.clear_loads()
    s.add_pressure_load(-1., cte=False, follower=follower)
    k0 = s.calc_kC()
    if s.x1u == 1.:
        # the axial rigid-body motion, fixed at a node of cos(n y/r)
        k0 = k0 + s.calc_stiffness_point_constraint(0., s.b/2, u=True,
                v=False, w=False, kuvw=1e3*SCHWEIZERHOF_E*s.plyt)
    c0 = solve(k0, s.calc_fext(), silent=True)
    K = s.calc_kG(c=c0)
    if follower:
        K = K + s.calc_kCfollower()
    eigvals = lb(k0, K, silent=True, num_eigvalues=4)[0]
    # the lowest one is real, complex ones may come after it
    assert abs(np.imag(eigvals[0])) <= 1e-8*abs(eigvals[0])
    D = SCHWEIZERHOF_E*s.plyt**3/(12*(1 - SCHWEIZERHOF_NU**2))
    return np.real(eigvals[0])*s.r**3/D


@pytest.mark.parametrize('case', SCHWEIZERHOF1984)
def test_schweizerhof1984_cylinders(case):
    shell, LR, Rt, bc, n, k_c, k_f = case
    R = 100.
    s = schweizerhof_sector(LR*R, R, R/Rt, n, bc)
    if bc != 'hinged':
        s.add_pressure_load(-1., follower=True)
        K = s.calc_kCfollower().toarray()
        # the free upper edge makes the load stiffness unsymmetric
        assert np.linalg.norm(K - K.T) > 1e-3*np.linalg.norm(K)
    for k_ref, follower in ((k_c, False), (k_f, True)):
        if k_ref is None:
            continue
        k = schweizerhof_k(s, follower)
        # their 16-node shell elements against the CLPT with Sanders'
        # kinematics; 1.1% for k_c of the hinged shell 2, otherwise 0.4%
        assert np.isclose(k, k_ref, rtol=1.5e-2), (case, follower, k)


def han2004_table(models=('cylshell_fsdt_sanders', 'cylshell_tsdt_sanders'),
                  zp_factor=0.5):
    r"""Critical pressure of the sandwich rings against the elasticity
    solution of Han et al. (2004), with the pressure on ``zp = zp_factor*h``"""
    rows = []
    for face in FACES:
        for i, ratio in enumerate(RATIOS):
            elast = HAN2004[face][i][0]
            row = [face, ratio, elast]
            for model in models:
                s = sandwich_ring(face, ratio, model=model)
                row.append(ring_buckling_pressure(s, zp=zp_factor*H))
            rows.append(row)
    return rows


if __name__ == '__main__':
    print('Isotropic ring, r/h = 100, p r**3/D22 for n = 2')
    for model in ('cylshell_clpt_sanders', 'cylshell_clpt_donnell',
                  'cylshell_fsdt_sanders', 'cylshell_tsdt_sanders'):
        s = ring_shell(model, 1., [0.], [0.01], [ISO],
                       fsdt_shear_correction=5/6)
        out = []
        for load in ('follower', 'dead', 'central'):
            out.append(ring_buckling_pressure(s, load)/s.ABD[4, 4])
        print('    %-24s follower %.5f  dead %.5f  central %.5f' % (
            model, *out))
    print()
    print('Han et al. (2004), FSDT ring with the shear stiffness of the shell'
          ' formulas, zp = 0')
    for face in FACES:
        for i, ratio in enumerate(RATIOS):
            elast, core_only, gbar, s8r, c3d20 = HAN2004[face][i]
            vals = []
            for kind, ref in (('core', core_only), ('Gbar', gbar)):
                s = sandwich_ring(face, ratio,
                                  shear=han_shear_stiffness(face, kind))
                p = ring_buckling_pressure(s)
                vals += [p, 100*(p/ref - 1)]
            print('    %-8s R/h %3d  core %10.0f (%+.2f%%)  Gbar %10.0f '
                  '(%+.2f%%)' % (face, ratio, *vals))
    print()
    print('Han et al. (2004) elasticity vs panels, pressure on zp = h/2,'
          ' default shear correction of the FSDT')
    for face, ratio, elast, pf, pt in han2004_table():
        print('    %-8s R/h %3d  elast %8.0f  FSDT %8.0f (%+.1f%%)  TSDT '
              '%8.0f (%+.1f%%)' % (face, ratio, elast, pf,
                                   100*(pf/elast - 1), pt,
                                   100*(pt/elast - 1)))


def test_kinetic_criterion_cantilever():
    r"""Kinetic criterion with :func:`structsolve.freq`, the general solver of
    unsymmetric matrices: the cantilever B1 of Schweizerhof and Ramm (1984),
    shell 1, with the unsymmetric follower load stiffness, loses stability by
    divergence (a hybrid system, Section 6.4 of the paper), not by flutter:
    the lowest natural frequency squared stays real and vanishes at the
    critical load of the static criterion"""
    from structsolve import freq
    R = 100.
    s = schweizerhof_sector(6*R, R, 1., 3, 'B1')
    s.rho = 7.8e-9
    s._rebuild()
    s.add_pressure_load(-1., cte=False, follower=True)
    k0 = s.calc_kC()
    c0 = solve(k0, s.calc_fext(), silent=True)
    KL = s.calc_kG(c=c0) + s.calc_kCfollower()
    lam_cr = np.real(lb(k0, KL, silent=True, num_eigvalues=4)[0][0])
    kM = s.calc_kM()
    w2 = []
    for frac in (0., 0.5, 0.9, 1.):
        lambda2 = freq(k0 + frac*lam_cr*KL, kM, silent=True,
                       sparse_solver=False, sort=False)[0]
        finite = np.isfinite(lambda2)
        w2.append(-lambda2[finite][np.argmin(np.abs(lambda2[finite]))])
    w2 = np.array(w2)
    assert np.all(np.abs(w2.imag) <= 1e-8*np.abs(w2[0]))
    assert w2[0].real > w2[1].real > w2[2].real > 0
    assert abs(w2[3]) < 1e-6*abs(w2[0])
