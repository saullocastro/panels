r"""Cantilever cylinder under a follower pressure traced with the Riks method

Benchmark of Schweizerhof and Ramm (1984), Section 6.4, "Cylindrical
cantilever shell under uniform pressure", Fig. 12, Table 4 and Fig. 13:
E = 2.1e6, mu = 0.3, R = 100, shell 1 with L/R = 6 and R/t = 100, the upper
edge free and the lower edge hinged, either with the warping completely
restrained, B1, v = w = u = 0, or partially restrained, B2, v = w = 0 and u
fixed at one point only. As in the reference, a half wave of the
circumference, b = pi R/n with n = 3, is modelled, with symmetry conditions on the straight
edges, see ``test_follower_pressure_validation.schweizerhof_sector``, with
the CLPT and the kinematics of Sanders, ``cylshell_clpt_sanders``: Donnell's
kinematics are not accurate for n = 3 circumferential waves. The load is
p = K E/(12(1 - mu**2)) (t/R)**3, Eq. (60), the hydrostatic (follower)
pressure with its unsymmetric load stiffness, ``follower=True``, the free
edge making the load nonconservative.

The tests check the linear buckling factors of Table 4, the consistency of
the internal forces and of the tangent stiffness matrix of the imperfect
shell, the inertia of the tangent stiffness matrix along the fundamental
path of the perfect shell beyond its bifurcation, and, with 8 x 6 terms, the
load-deflection path of the imperfect shell B2 up to w/t = 0.3 traced with
the Riks method (the path of Fig. 13), with a geometric imperfection of 1 %
of t in the shape of the w part of the lowest linear buckling mode of the
follower load, added as in ``test_simo1990_cylinder_postbuckling.py``.

Notes
-----
The imperfection, of Ritz vector c0, is added exactly for the Sanders
kinematics, whose non-linear strains are quadratic forms of the rotations
phix = -w,x and phiy = -w,y + v/r, linear in c: the strains of the
imperfect shell are eps(c + c0) - eps(c0), whence::

    fint_imp(c) = fint(c + c0) - fint(c0) - kG(N0) c
    kT_imp(c) = kT(c + c0) - kG(N0)

with kG(N0) the geometric stiffness of the membrane resultants N0, linear
and non-linear parts, of the state c0, integrated at the Gauss points with
the rotations phix and phiy, see :func:`calc_kG_N`. The follower pressure
acts on the imperfect geometry, its force vector at c being Fp(c + c0) and
its load stiffness that of c + c0.

References
----------
.. [1] Schweizerhof, K., Ramm, E., Displacement dependent pressure loads in
   nonlinear finite element analyses, Computers & Structures, 18(6),
   1099-1114, 1984. https://doi.org/10.1016/0045-7949(84)90154-8

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from structsolve import Analysis, lb, solve
from structsolve.stopping import value_reached

from panels.bardell import calc_vec_f, calc_vec_fp
from test_follower_pressure_validation import (schweizerhof_sector,
        SCHWEIZERHOF_E as E, SCHWEIZERHOF_NU as NU)

R = 100.
#: shell 1 of [1]
L = 600.
t = 1.
N_WAVES = 3
D = E*t**3/(12*(1 - NU**2))
#: pressure of K = 1, Eq. (60)
P_K1 = D/R**3


def build_shell(bc='B1', m=14, nt=8, L=L, t=t, n=N_WAVES):
    r"""The sector of :func:`schweizerhof_sector` with the external follower
    pressure of ``K = 1``, incremented (``cte=False``), and, for B2, the
    penalty stiffness of the axial displacement fixed at one point of the
    lower edge, ``kP``, else ``None``"""
    s = schweizerhof_sector(L, R, t, n, bc, m=m, nt=nt)
    Dn = E*t**3/(12*(1 - NU**2))
    s.add_pressure_load(-Dn/R**3, cte=False, follower=True)
    kP = None
    if bc == 'B2':
        # the axial rigid-body motion, fixed at a node of cos(n y/r)
        kP = csr_matrix(s.calc_stiffness_point_constraint(0., s.b/2, u=True,
                v=False, w=False, kuvw=1e3*E*t))
    return s, kP


def buckling_K(s, kP=None, follower=True, num_eigvalues=4):
    r"""Lowest linear buckling factor ``K`` and its mode, with the prestate
    of a linear static analysis under the pressure of ``K = 1`` of
    :func:`build_shell`; ``follower=False`` for the constant directional
    (dead) pressure, the load stiffness then being left out"""
    k0 = s.calc_kC()
    if kP is not None:
        k0 = k0 + kP
    c0 = solve(k0, s.calc_fext(), silent=True)
    KG = s.calc_kG(c=c0)
    if follower:
        KG = KG + s.calc_kCfollower()
    eigvals, eigvecs = lb(k0, KG, silent=True, num_eigvalues=num_eigvalues)
    # the lowest one is real
    assert abs(np.imag(eigvals[0])) <= 1e-8*abs(eigvals[0])
    return np.real(eigvals[0]), np.real(eigvecs[:, 0])


def _rows_phi(s, xs, ys):
    r"""Rows of the rotations phix = -w,x and phiy = -w,y + v/r at the grid
    of points ``xs`` x ``ys``, shape ``(len(xs)*len(ys), size)``"""
    m, n, a, b, r = s.m, s.n, s.a, s.b, s.r
    def tab(func, xis, e1t, e1r, e2t, e2r, k):
        return np.array([func(xi, e1t, e1r, e2t, e2r)[:k] for xi in xis])
    xis = 2*np.asarray(xs)/a - 1
    etas = 2*np.asarray(ys)/b - 1
    Fw = tab(calc_vec_f, xis, s.x1w, s.x1wr, s.x2w, s.x2wr, m)
    Fwx = tab(calc_vec_fp, xis, s.x1w, s.x1wr, s.x2w, s.x2wr, m)*2/a
    Fv = tab(calc_vec_f, xis, s.x1v, s.x1vr, s.x2v, s.x2vr, m)
    Gw = tab(calc_vec_f, etas, s.y1w, s.y1wr, s.y2w, s.y2wr, n)
    Gwy = tab(calc_vec_fp, etas, s.y1w, s.y1wr, s.y2w, s.y2wr, n)*2/b
    Gv = tab(calc_vec_f, etas, s.y1v, s.y1vr, s.y2v, s.y2vr, n)
    npts = len(xs)*len(ys)
    # term j*m + i of the functions i along x and j along y
    prod = lambda F, G: np.einsum('pi,qj->pqji', F, G).reshape(npts, m*n)
    size = s.get_size()
    phix = np.zeros((npts, size))
    phiy = np.zeros((npts, size))
    phix[:, 2::3] = -prod(Fwx, Gw)
    phiy[:, 2::3] = -prod(Fw, Gwy)
    phiy[:, 1::3] = prod(Fv, Gv)/r
    return phix, phiy


def calc_kG_N(s, N):
    r"""Geometric stiffness matrix of the membrane resultants N = (Nxx, Nyy,
    Nxy) given at the integration points, int G^T N G dA, with the rows G of
    the rotations phix and phiy of the Sanders kinematics"""
    xs, ys, wxs, wys = s.integration_points()
    phix, phiy = _rows_phi(s, xs, ys)
    w = np.outer(wxs, wys).ravel()
    Nxx, Nyy, Nxy = [np.asarray(Ni).ravel()*w for Ni in N]
    kG = (phix.T @ (Nxx[:, None]*phix) + phiy.T @ (Nyy[:, None]*phiy)
          + phix.T @ (Nxy[:, None]*phiy) + phiy.T @ (Nxy[:, None]*phix))
    return csr_matrix(kG)


def membrane_stresses(s, c, NLgeom=True):
    xs, ys = s.integration_points()[:2]
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    fields = s.stress(c, xs=X, ys=Y, NLgeom=NLgeom)[1]
    return fields['Nxx'], fields['Nyy'], fields['Nxy']


def w_max(s, c, grid=21):
    r"""Maximum ``|w|`` over the sector"""
    return np.abs(s.uvw(c, gridx=grid, gridy=grid)[1]['w']).max()


def imperfection(s, mode, xi):
    r"""The w part of ``mode`` scaled to a maximum ``|w0|`` of ``xi*t``"""
    c0 = np.zeros_like(mode)
    c0[2::3] = mode[2::3]
    w = np.ravel(s.uvw(c0, gridx=51, gridy=51)[1]['w'])
    return c0*xi*s.plyt/w[np.argmax(np.abs(w))]


class ImperfectShell(object):
    r"""Callables of structsolve.Analysis for the imperfect shell under the
    follower pressure, with the load factor ``inc``; ``kP`` is a constant
    penalty stiffness, e.g. of a point constraint"""
    def __init__(self, s, c0, kP=None):
        self.s = s
        self.c0 = c0
        size = s.get_size()
        self.kP = csr_matrix((size, size)) if kP is None else kP
        # elastic internal forces of c0 and the follower force Fp(c0) - Fp(0)
        # of the load factor 1
        self.fint0 = s.calc_fint(c0, inc=0.)
        self.fp0 = s.calc_fext_follower(c0, inc=1., reference=False)
        self.kG0 = calc_kG_N(s, membrane_stresses(s, c0))
        self.kT0 = self.calc_kT(np.zeros_like(c0), inc=0.)

    def calc_fext(self, inc=1., silent=True, c=None):
        r"""Fp(c + c0) times ``inc``, Fp(c0) when ``c`` is None"""
        if c is None:
            return self.s.calc_fext(inc=inc, silent=silent) + inc*self.fp0
        return self.s.calc_fext(inc=inc, silent=silent, c=c + self.c0)

    def calc_fint(self, c, silent=True, inc=1.):
        r"""Elastic internal forces of eps(c + c0) - eps(c0) minus the
        follower forces Fp(c + c0) - Fp(c0), at the load factor ``inc``"""
        return (self.s.calc_fint(c + self.c0, inc=inc) - self.fint0
                + inc*self.fp0 - self.kG0 @ c + self.kP @ c)

    def calc_kT(self, c, inc=1.):
        return csr_matrix(self.s.calc_kT(c=c + self.c0, inc=inc) - self.kG0
                          + self.kP)

    def calc_kC(self, c=None, NLgeom=False, silent=True, inc=1.):
        if c is None or not NLgeom:
            return self.kT0
        return self.calc_kT(c, inc=inc)

    def calc_kG(self, c=None, NLgeom=False, silent=True, inc=1.):
        return csr_matrix(self.kT0.shape)


def riks(model, stop, inc, max_inc=None, num_eigvals=20):
    r"""Riks analysis of ``model``, the load factor being ``K``, with the
    inertia of the unsymmetric tangent stiffness matrix at every converged
    increment: the number of negative real eigenvalues among the
    ``num_eigvals`` of smallest magnitude, by ARPACK on the LU
    factorization, and the sign of the determinant"""
    an = Analysis(model.calc_fext, model.calc_fint, model.calc_kC,
                  model.calc_kG)
    an.NL_method = 'arc_length_riks'
    an.final_load_factor = None
    an.maxArcLength = 1e4
    an.initialInc = inc
    an.maxInc = inc if max_inc is None else max_inc
    an.inertia = True
    an.inertia_num_eigvals = num_eigvals
    an.stop_criterion = stop
    an.static(NLgeom=True, silent=True)
    Ks = np.asarray(an.increments)
    # every converged increment is in equilibrium
    fext = model.calc_fext()
    for K, c in zip(Ks, an.cs):
        assert (np.linalg.norm(model.calc_fint(c, inc=K) - K*fext)
                <= 1e-7*np.linalg.norm(fext))
    return an, Ks


@pytest.mark.slow
def test_imperfection_identities():
    s, kP = build_shell('B2', m=8, nt=6)
    size = s.get_size()
    rng = np.random.default_rng(1)
    # kG of the linear resultants equals Shell.calc_kG(c), which has no
    # resultant of the non-linear strains
    c = rng.standard_normal(size)*1e-2
    kG = s.calc_kG(c=c, NLgeom=True).toarray()
    kG_L = calc_kG_N(s, membrane_stresses(s, c, NLgeom=False)).toarray()
    kG_N = calc_kG_N(s, membrane_stresses(s, c, NLgeom=True)).toarray()
    assert np.abs(kG_L - kG).max() <= 1e-12*np.abs(kG).max()
    assert np.abs(kG_N - kG).max() >= 1e-5*np.abs(kG).max()
    K_f, mode = buckling_K(s, kP)
    # a large imperfection, such that the non-linear resultants of c0 count
    model = ImperfectShell(s, imperfection(s, mode, 0.5), kP)
    # the imperfect shell is unstressed at c = 0 for any load factor
    assert np.linalg.norm(model.calc_fint(np.zeros(size), inc=0.)) == 0
    fext = np.linalg.norm(model.calc_fext())
    for K in (1., K_f):
        assert (np.linalg.norm(model.calc_fint(np.zeros(size), inc=K))
                <= 1e-12*K*fext)
    # the tangent stiffness matrix, unsymmetric with the load stiffness, is
    # the derivative of the internal forces
    c = np.zeros(size)
    c[2::3] = rng.standard_normal(size//3)
    c *= 0.3*t/w_max(s, c)
    c[0::3] = rng.standard_normal(size//3)*1e-3
    c[1::3] = rng.standard_normal(size//3)*1e-3
    v = rng.standard_normal(size)
    v[np.asarray(abs(model.kT0).sum(axis=1)).ravel() == 0] = 0
    h = 1e-5
    K = 0.8*K_f
    dfd = (model.calc_fint(c + h*v, inc=K)
           - model.calc_fint(c - h*v, inc=K))/(2*h)
    kT = model.calc_kT(c, inc=K)
    # its skew part is that of the load stiffness
    kF = s.calc_kCfollower()
    skew = (kT - kT.T).toarray()
    assert np.abs(skew).max() > 0
    assert np.abs(skew - K*(kF - kF.T)).max() <= 1e-6*np.abs(skew).max()
    assert np.linalg.norm(dfd - kT @ v) <= 1e-8*np.linalg.norm(kT @ v)
    # calc_fext(c=c) is minus the derivative of the residual with respect to
    # the load factor
    dK = 1e-3
    dfK = (model.calc_fint(c, inc=K + dK) - model.calc_fint(c, inc=K - dK))/(2*dK)
    fext = model.calc_fext(inc=1., c=c)
    assert (np.linalg.norm(dfK + fext - model.calc_fext(inc=1.))
            <= 1e-8*np.linalg.norm(fext))
    # the elastic internal forces are the gradient of the strain energy of
    # the strains eps(c + c0) - eps(c0)
    xs, ys, wxs, wys = s.integration_points()
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    names = ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy')
    def energy(c):
        # the fields of Shell.strain are overwritten at each call
        e1 = s.strain(c + model.c0, xs=X, ys=Y)[1]
        e1 = np.array([e1[k] for k in names])
        e0 = s.strain(model.c0, xs=X, ys=Y)[1]
        e0 = np.array([e0[k] for k in names])
        e = (e1 - e0).reshape(6, -1)
        return (0.5*np.einsum('ip,ij,jp,p->', e, s.ABD, e,
                              np.outer(wxs, wys).ravel())
                + 0.5*c @ (kP @ c))
    dU = (energy(c + h*v) - energy(c - h*v))/(2*h)
    fv = model.calc_fint(c, inc=0.) @ v
    assert abs(dU - fv) <= 1e-7*abs(fv)


@pytest.mark.slow
def test_linear_buckling_shell1():
    # Table 4 of [1], shell 1, k_c and k_f (Eq. (63)) of B1 and B2: the
    # sector of 14 x 8 terms is within 0.1 % of that of 22 x 8 terms, and
    # the 24 bicubic elements of [1] within 0.3 % of the latter; the
    # unsymmetric eigenvalue problem gives the k_f of the three
    # symmetrizations of [1], 8.09 to 8.12 for B2, to that precision
    for bc, k_c, k_f in (('B1', 10.89, 9.70), ('B2', 9.11, 8.10)):
        s, kP = build_shell(bc)
        K = s.calc_kCfollower().toarray()
        # the free upper edge makes the load stiffness unsymmetric
        assert np.linalg.norm(K - K.T) > 1e-3*np.linalg.norm(K)
        Kc = buckling_K(s, kP, follower=False)[0]
        Kf = buckling_K(s, kP)[0]
        assert np.isclose(Kc, k_c, rtol=5e-3), (bc, Kc)
        assert np.isclose(Kf, k_f, rtol=5e-3), (bc, Kf)
        # the follower load lowers the critical pressure by 12.3 % (B1) and
        # 12.5 % (B2) in [1]
        assert abs((Kc - Kf)/Kf - (k_c - k_f)/k_f) <= 5e-3


@pytest.mark.slow
def test_riks_perfect_shell_inertia():
    # the fundamental path of the perfect shell B1 beyond its bifurcation,
    # at K_f: one negative eigenvalue after it. structsolve counts the
    # eigenvalues of S K_T S, S = |diag(K0)|^(-1/2), with K0 the linear
    # stiffness: in the unknowns of the shell the negative eigenvalue is far
    # from the smallest ones by magnitude (about the 130th), those of the
    # Ritz terms of smallest norm being closer to zero
    s, kP = build_shell('B1')
    K_f = buckling_K(s, kP)[0]
    model = ImperfectShell(s, np.zeros(s.get_size()), kP)
    an, Ks = riks(model, lambda an: an.increments[-1] >= 10.5, inc=2.)
    nneg = np.asarray(an.num_negative_eigvals)
    det = np.asarray(an.det_signs)
    assert Ks[0] < K_f < Ks[-1]
    assert np.all(nneg[Ks < K_f] == 0) and np.all(det[Ks < K_f] == 1)
    assert np.all(nneg[Ks > K_f] == 1) and np.all(det[Ks > K_f] == -1)


@pytest.mark.slow
def test_riks_imperfect_B2_coarse():
    # B2, 8 x 6 terms, imperfection of 1 % of t, Riks up to w/t = 0.3: no
    # limit point, the load increases monotonically towards the bifurcation
    # load k_f of the same model and the tangent stiffness matrix keeps no
    # negative eigenvalue; k_f and the last load are values of the code
    # with these terms
    s, kP = build_shell('B2', m=8, nt=6)
    K_f, mode = buckling_K(s, kP)
    assert np.isclose(K_f, 8.19280, rtol=1e-4)
    model = ImperfectShell(s, imperfection(s, mode, 0.01), kP)
    an, Ks = riks(model, value_reached(lambda c: w_max(s, c)/t, 0.3),
                  inc=1., max_inc=4.)
    ws = np.array([w_max(s, c)/t for c in an.cs])
    assert ws[-1] >= 0.3
    assert np.all(np.diff(Ks) > 0) and Ks[-1] < K_f
    assert all(n == 0 for n in an.num_negative_eigvals)
    assert all(d == 1 for d in an.det_signs)
    assert np.isclose(Ks[-1], 7.9562, rtol=1e-3)
