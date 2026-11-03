r"""Post-buckling of an axially compressed cylinder traced with the Riks method

Benchmark of Simo, Fox and Rifai (1990), Section 6.3.4, "Post-buckling
analysis of a simply-supported cylinder; effect of initial imperfections",
pp. 54-57, Fig. 6.3.4 on p. 56: R = 36, L = 100, t = 0.125, E = 1e7,
nu = 0.3, axial compression. As in the reference, an L/10 x 15 deg section
is modelled, with symmetry conditions on the interior edges and a simple
support on the loaded edge, such that the lowest mode is the diamond mode of
5 axial half-waves and 12 circumferential waves. The load factor is the load
normalized by the linear buckling load and the abscissa the maximum axial
deflection of the loaded edge, as in Fig. 6.3.4.

The tests check the consistency of the internal forces and of the tangent
stiffness matrix of the imperfect shell, and the limit loads, snap-back,
bifurcation and post-buckling loads of Fig. 6.3.4 (a) and (d), for
imperfections of 50 % and 10 % of the thickness in the shape of the w part
of the lowest linear buckling mode, and for the perfect cylinder.

The values of Fig. 6.3.4 were read on the page rendered at 300 dpi with a
grid over the axes, precision about +-0.0003 in the deflection and +-0.01
in the load.

Notes
-----
The initial imperfection, the w part of the lowest linear buckling mode
scaled to max|w0| = xi t, of Ritz vector c0, is added exactly for the
Donnell kinematics, whose strains of the imperfect shell are eps(c + c0) -
eps(c0)::

    fint_imp(c) = fint(c + c0) - fint(c0) - kG(N0) c
    kT_imp(c) = kT(c + c0) - kG(N0)

with kG(N0) the geometric stiffness of the membrane resultants N0, linear
and non-linear parts, of the state c0. Shell.calc_kG(c=c0) has the linear
part only, and kG(N0) is integrated here at the Gauss points.

References
----------
.. [1] Simo, J.C., Fox, D.D., Rifai, M.S., On a stress resultant
   geometrically exact shell model. Part III: Computational aspects of the
   nonlinear theory, Computer Methods in Applied Mechanics and Engineering,
   79(1), 21-70, 1990. https://doi.org/10.1016/0045-7825(90)90094-3

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix, diags
from structsolve import Analysis, lb, solve
from structsolve.stopping import value_reached

from panels.shell import Shell
from panels.bardell import calc_vec_f, calc_vec_fp

R = 36.
L = 100.
t = 0.125
E = 1.e7
nu = 0.3
N_CL = E*t**2/(R*np.sqrt(3*(1 - nu**2)))


def build_shell(m=8, n=8):
    """The section, loaded edge at x = 0, reference load the linear
    buckling load"""
    s = Shell(a=L/10, b=R*np.pi/12, r=R, stack=[0.], plyt=t,
              laminaprop=(E, nu), m=m, n=n, model='cylshell_clpt_donnell')
    flags = dict(
        x1u=1, x1ur=1, x1v=0, x1vr=1, x1w=0, x1wr=1, # simple support
        x2u=0, x2ur=1, x2v=1, x2vr=0, x2w=1, x2wr=0, # symmetry
        y1u=1, y1ur=0, y1v=0, y1vr=1, y1w=1, y1wr=0, # symmetry
        y2u=1, y2ur=0, y2v=0, y2vr=1, y2w=1, y2wr=0, # symmetry
        )
    for k, v in flags.items():
        setattr(s, k, float(v))
    s.add_distr_load_fixed_x(0., funcx=lambda y: N_CL, cte=False)
    K = s.calc_kC()
    clin = solve(K, s.calc_fext(), silent=True)
    eigvals, eigvecs = lb(K, s.calc_kG(c=clin), num_eigvalues=3, silent=True)
    lbd_cr = eigvals[0]
    s.distr_loads_inc = []
    s.add_distr_load_fixed_x(0., funcx=lambda y: lbd_cr*N_CL, cte=False)
    return s, lbd_cr, np.real(eigvecs[:, 0])


def calc_kG_N(s, N):
    """Geometric stiffness matrix of the membrane resultants N = (Nxx, Nyy,
    Nxy) given at the integration points"""
    xs, ys, wxs, wys = s.integration_points()
    m, n, a, b = s.m, s.n, s.a, s.b
    F = np.array([calc_vec_f(2*x/a - 1, s.x1w, s.x1wr, s.x2w, s.x2wr)[:m] for x in xs])
    Fp = np.array([calc_vec_fp(2*x/a - 1, s.x1w, s.x1wr, s.x2w, s.x2wr)[:m] for x in xs])*2/a
    G = np.array([calc_vec_f(2*y/b - 1, s.y1w, s.y1wr, s.y2w, s.y2wr)[:n] for y in ys])
    Gp = np.array([calc_vec_fp(2*y/b - 1, s.y1w, s.y1wr, s.y2w, s.y2wr)[:n] for y in ys])*2/b
    Bx = np.einsum('pi,qj->pqji', Fp, G).reshape(len(xs)*len(ys), m*n)
    By = np.einsum('pi,qj->pqji', F, Gp).reshape(len(xs)*len(ys), m*n)
    w = np.outer(wxs, wys).ravel()
    Nxx, Nyy, Nxy = [np.asarray(Ni).ravel()*w for Ni in N]
    kww = (Bx.T @ (Nxx[:, None]*Bx) + By.T @ (Nyy[:, None]*By)
           + Bx.T @ (Nxy[:, None]*By) + By.T @ (Nxy[:, None]*Bx))
    idx = 3*np.arange(m*n) + 2
    I, J = np.meshgrid(idx, idx, indexing='ij')
    size = s.get_size()
    return coo_matrix((kww.ravel(), (I.ravel(), J.ravel())),
                      shape=(size, size)).tocsr()


def membrane_stresses(s, c, NLgeom=True):
    xs, ys = s.integration_points()[:2]
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    fields = s.stress(c, xs=X, ys=Y, NLgeom=NLgeom)[1]
    return fields['Nxx'], fields['Nyy'], fields['Nxy']


def imperfection(s, mode, xi):
    c0 = np.zeros_like(mode)
    c0[2::3] = mode[2::3]
    w = np.ravel(s.uvw(c0, gridx=51, gridy=51)[1]['w'])
    return c0*xi*t/w[np.argmax(np.abs(w))]


class ImperfectShell(object):
    """Callables of structsolve.Analysis for the imperfect shell"""
    def __init__(self, s, c0):
        self.s = s
        self.c0 = c0
        self.fint0 = s.calc_fint(c0)
        self.kG0 = calc_kG_N(s, membrane_stresses(s, c0))
        self.kT0 = self.calc_kT(np.zeros_like(c0))

    def calc_fext(self, inc=1., silent=True):
        return self.s.calc_fext(inc=inc, silent=silent)

    def calc_fint(self, c, silent=True):
        return self.s.calc_fint(c + self.c0) - self.fint0 - self.kG0 @ c

    def calc_kT(self, c):
        return csr_matrix(self.s.calc_kT(c=c + self.c0) - self.kG0)

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        if c is None or not NLgeom:
            return self.kT0
        return self.calc_kT(c)

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        return csr_matrix(self.kT0.shape)


class ScaledModel(object):
    """The callables in the unknowns c = S cs, S = diag(1/sqrt(K_ii)), such
    that the critical eigenvector of the branch switching is that of the
    bifurcation rather than of weakly stiff in-plane Ritz terms"""
    def __init__(self, s):
        self.s = s
        d = s.calc_kC().diagonal()
        self.S = np.ones_like(d)
        self.S[d > 0] = 1/np.sqrt(d[d > 0])
        self.Sm = diags(self.S)

    def calc_fext(self, inc=1., silent=True):
        return self.S*self.s.calc_fext(inc=inc, silent=silent)

    def calc_fint(self, c, silent=True):
        return self.S*self.s.calc_fint(self.S*c, silent=silent)

    def calc_kC(self, c=None, NLgeom=False, silent=True):
        c = None if c is None else self.S*c
        return csr_matrix(self.Sm @ self.s.calc_kC(c=c, NLgeom=NLgeom) @ self.Sm)

    def calc_kG(self, c=None, NLgeom=False, silent=True):
        c = None if c is None else self.S*c
        return csr_matrix(self.Sm @ self.s.calc_kG(c=c, NLgeom=NLgeom) @ self.Sm)


def u_max(s, c):
    """Maximum axial deflection of the loaded edge x = 0"""
    ys = np.linspace(0, s.b, 31)
    return np.max(s.uvw(c, xs=np.zeros_like(ys), ys=ys)[1]['u'])


def riks(an, s, umax, unscale=lambda c: c):
    an.NL_method = 'arc_length_riks'
    an.final_load_factor = None
    an.maxArcLength = 1e4
    an.stop_criterion = value_reached(lambda c: u_max(s, unscale(c)), umax)
    an.static(NLgeom=True, silent=True)
    lbds = np.asarray(an.increments)
    us = np.array([u_max(s, unscale(c)) for c in an.cs])
    # every converged increment is in equilibrium
    fext = an.calc_fext()
    for lbd, c in zip(lbds, an.cs):
        assert np.linalg.norm(an.calc_fint(c) - lbd*fext) <= 1e-8*np.linalg.norm(fext)
    return lbds, us


def test_imperfection_identities():
    s, lbd_cr, mode = build_shell(6, 6)
    size = s.get_size()
    rng = np.random.default_rng(1)
    # kG of the linear resultants equals Shell.calc_kG(c), which has no
    # resultant of the non-linear strains
    c = rng.standard_normal(size)*1e-2
    kG = s.calc_kG(c=c, NLgeom=True).toarray()
    kG_L = calc_kG_N(s, membrane_stresses(s, c, NLgeom=False)).toarray()
    kG_N = calc_kG_N(s, membrane_stresses(s, c, NLgeom=True)).toarray()
    assert np.abs(kG_L - kG).max() <= 1e-12*np.abs(kG).max()
    assert np.abs(kG_N - kG).max() >= 1e-4*np.abs(kG).max()
    model = ImperfectShell(s, imperfection(s, mode, 0.5))
    # the imperfect shell is unstressed at c = 0
    assert np.linalg.norm(model.calc_fint(np.zeros(size))) == 0
    # the tangent stiffness matrix is the derivative of the internal forces
    c = np.zeros(size)
    c[2::3] = rng.standard_normal(size//3)
    c *= 0.3*t/np.abs(s.uvw(c, gridx=21, gridy=21)[1]['w']).max()
    c[0::3] = rng.standard_normal(size//3)*1e-3
    c[1::3] = rng.standard_normal(size//3)*1e-3
    v = rng.standard_normal(size)
    v[np.asarray(abs(model.kT0).sum(axis=1)).ravel() == 0] = 0
    h = 1e-5
    dfd = (model.calc_fint(c + h*v) - model.calc_fint(c - h*v))/(2*h)
    kTv = model.calc_kT(c) @ v
    assert np.linalg.norm(dfd - kTv) <= 1e-9*np.linalg.norm(kTv)
    # the internal forces are the gradient of the strain energy of the
    # strains eps(c + c0) - eps(c0)
    xs, ys, wxs, wys = s.integration_points()
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    names = ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy')
    def energy(c):
        e1 = s.strain(c + model.c0, xs=X, ys=Y)[1]
        e1 = np.array([e1[k] for k in names])
        e0 = s.strain(model.c0, xs=X, ys=Y)[1]
        e0 = np.array([e0[k] for k in names])
        e = (e1 - e0).reshape(6, -1)
        return 0.5*np.einsum('ip,ij,jp,p->', e, s.ABD, e, np.outer(wxs, wys).ravel())
    dU = (energy(c + h*v) - energy(c - h*v))/(2*h)
    fv = model.calc_fint(c) @ v
    # (with Shell.calc_kG(c=c0) in place of kG(N0) the relative difference
    # is 1e-4 or larger)
    assert abs(dU - fv) <= 1e-7*abs(fv)


def test_riks_imperfect_cylinder_50pct():
    # Fig. 6.3.4 (a), imperfection of 50 % of the thickness
    s, lbd_cr, mode = build_shell(8, 8)
    # linear buckling load of the diamond mode, 0.42 % above the classical
    # one with m = n = 10 to 14 terms
    assert np.isclose(lbd_cr, 1.0047, atol=1e-4)
    model = ImperfectShell(s, imperfection(s, mode, 0.5))
    an = Analysis(model.calc_fext, model.calc_fint, model.calc_kC, model.calc_kG)
    an.initialInc = an.maxInc = 0.1
    lbds, us = riks(an, s, 0.021)
    ilim = np.flatnonzero(np.diff(lbds) < 0)[0]
    # limit load: 0.411 at 0.0108 read for [1], 0.402 at 0.0106 for Stanley
    assert abs(lbds[ilim] - 0.411) <= 0.025
    assert abs(lbds[ilim] - 0.402) <= 0.015
    assert abs(us[ilim] - 0.0108) <= 0.0005
    # post-buckling: the load at u_max = 0.02, 0.315 read for [1]
    imin = ilim + np.argmin(lbds[ilim:])
    assert lbds[imin] < 0.85*lbds[ilim]
    p02 = np.interp(0.02, us[imin:], lbds[imin:])
    assert abs(p02 - 0.315) <= 0.015


@pytest.mark.slow
def test_riks_imperfect_cylinder_10pct():
    # Fig. 6.3.4 (a), imperfection of 10 % of the thickness, with the
    # snap-back after the limit point
    s, lbd_cr, mode = build_shell(8, 8)
    model = ImperfectShell(s, imperfection(s, mode, 0.1))
    an = Analysis(model.calc_fext, model.calc_fint, model.calc_kC, model.calc_kG)
    an.initialInc = an.maxInc = 0.1
    lbds, us = riks(an, s, 0.021)
    ilim = np.flatnonzero(np.diff(lbds) < 0)[0]
    imin = ilim + np.argmin(lbds[ilim:])
    # limit load: 0.715 at 0.0168 read for [1], 0.717 at 0.0159-0.0166 for
    # Stanley
    assert abs(lbds[ilim] - 0.715) <= 0.015
    assert abs(us[ilim] - 0.0166) <= 0.0005
    # snap-back: the deflection decreases to 0.0142 (read for [1]) before
    # increasing again
    assert abs(us[ilim:imin + 1].min() - 0.0142) <= 0.0005
    # post-buckling: the load at u_max = 0.02, 0.330 read for [1]
    p02 = np.interp(0.02, us[imin:], lbds[imin:])
    assert abs(p02 - 0.330) <= 0.01


@pytest.mark.slow
def test_riks_perfect_cylinder_branch_switching():
    # Fig. 6.3.4 (d), the perfect cylinder, with the switch at the first
    # bifurcation along the critical eigenvector
    s, lbd_cr, mode = build_shell(8, 8)
    model = ScaledModel(s)
    an = Analysis(model.calc_fext, model.calc_fint, model.calc_kC, model.calc_kG)
    an.initialInc = an.maxInc = 0.03
    an.inertia = True
    an.branch_switching = 1
    # the count of negative eigenvalues of PARDISO is late here, the sign
    # of the determinant of SuperLU is not
    an.linear_solver = 'splu'
    lbds, us = riks(an, s, 0.021, unscale=lambda c: model.S*c)
    assert len(an.branch_points) == 1
    ib = an.branch_points[0]
    # bifurcation: 0.833 at 0.0183 read for [1]
    assert abs(lbds[ib] - 0.833) <= 0.015
    assert abs(us[ib] - 0.0183) <= 0.0005
    # the bifurcated branch: back to 0.0149 at the load 0.457, read for [1]
    imin = ib + np.argmin(lbds[ib:])
    i = ib + np.argmin(us[ib:imin + 1])
    assert abs(us[i] - 0.0149) <= 0.0006
    assert abs(lbds[i] - 0.457) <= 0.015
    # post-buckling: the load at u_max = 0.02, 0.334 read for [1]
    p02 = np.interp(0.02, us[imin:], lbds[imin:])
    assert abs(p02 - 0.334) <= 0.015
