r"""Utilities of the notebook simo1990_cylinder_postbuckling.ipynb

Post-buckling of the axially compressed cylinder of Simo, Fox and Rifai
(1990), Section 6.3.4, Fig. 6.3.4, with an L/10 x 15 deg section of the
Donnell model ``cylshell_clpt_donnell``, the initial imperfection added
exactly through ``ImperfectShell`` and the Riks method of structsolve. The
same model is tested in ``tests/tests_shell/test_simo1990_cylinder_postbuckling.py``.

"""
import time

import numpy as np
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


def edge_deflection(s, c, ny=61):
    """Average and maximum axial deflection of the loaded edge x = 0"""
    ys = np.linspace(0, s.b, ny)
    u = np.ravel(s.uvw(c, xs=np.zeros_like(ys), ys=ys)[1]['u'])
    return np.trapezoid(u, ys)/s.b, u.max()


# Values read from Fig. 6.3.4 of Simo, Fox and Rifai (1990), p. 56, on the
# page rendered at 300 dpi with a grid of 0.001 x 0.05 drawn over the axes;
# reading precision about +-0.0003 in the axial deflection and +-0.01 in the
# load normalized by the linear buckling load. (axial deflection, load)
#: Fig. (a), triangles, 9-ANS of Stanley (1985), 10 % imperfection
FIG_A_ANS_10 = [
    (0.0071, 0.331), (0.0095, 0.445), (0.0125, 0.570), (0.0148, 0.666),
    (0.0159, 0.717), (0.0166, 0.713), (0.0163, 0.693), (0.0157, 0.653),
    (0.0152, 0.619), (0.0148, 0.580), (0.0144, 0.532), (0.0141, 0.487),
    (0.0140, 0.430), (0.0143, 0.382), (0.0157, 0.346), (0.0184, 0.319),
    (0.0218, 0.324), (0.0296, 0.349), (0.0353, 0.395)]
#: Fig. (a), diamonds, Stanley (1985), 50 % imperfection
FIG_A_ANS_50 = [
    (0.0055, 0.229), (0.0066, 0.273), (0.0077, 0.310), (0.0084, 0.342),
    (0.0089, 0.366), (0.0094, 0.383), (0.0098, 0.397), (0.0106, 0.402),
    (0.0109, 0.385), (0.0112, 0.367), (0.0118, 0.354), (0.0122, 0.332),
    (0.0132, 0.312), (0.0151, 0.300), (0.0175, 0.297), (0.0204, 0.300),
    (0.0237, 0.318), (0.0265, 0.349), (0.0363, 0.420)]
#: Fig. (a), solid line, Simo et al., 10 % imperfection
FIG_A_SIMO_10 = [
    (0.0125, 0.575), (0.0150, 0.675), (0.0168, 0.715), (0.0158, 0.642),
    (0.0151, 0.580), (0.0146, 0.512), (0.0142, 0.445), (0.0143, 0.399),
    (0.0148, 0.366), (0.0158, 0.343), (0.0184, 0.326), (0.0200, 0.330),
    (0.0250, 0.344), (0.0300, 0.372), (0.0350, 0.406), (0.0400, 0.459)]
#: Fig. (a), solid line, Simo et al., 50 % imperfection
FIG_A_SIMO_50 = [
    (0.0108, 0.411), (0.0116, 0.366), (0.0126, 0.335), (0.0141, 0.319),
    (0.0150, 0.315), (0.0200, 0.315), (0.0250, 0.344), (0.0300, 0.370),
    (0.0350, 0.401), (0.0400, 0.449)]
#: Fig. (d), solid line, Simo et al., perfect cylinder, branch switching
FIG_D_SIMO = [
    (0.0183, 0.833), (0.0182, 0.742), (0.0171, 0.695), (0.0160, 0.599),
    (0.0151, 0.530), (0.0149, 0.457), (0.0160, 0.360), (0.0190, 0.335),
    (0.0200, 0.334), (0.0250, 0.355), (0.0300, 0.397), (0.0350, 0.436),
    (0.0400, 0.480)]
#: slope of the straight line of the figures: axial deflection at load 1
FIG_LINEAR_U1 = 0.0205


def riks(xi, m=8, n=8, umax=0.04):
    """Riks analysis up to the maximum axial deflection ``umax``

    ``xi > 0``: imperfection of ``xi*t`` in the shape of the lowest linear
    buckling mode; ``xi = 0``: the perfect shell, with the switch at the
    first bifurcation, in the unknowns of ``ScaledModel`` and with the sign
    of the determinant of SuperLU.

    Returns a dict with the load factors ``lbds``, the average and maximum
    axial deflections ``u_avg``, ``u_max``, the maximum ``|w|`` ``w_max``,
    ``branch_points``, ``lbd_cr`` and the run time.

    """
    s, lbd_cr, mode = build_shell(m, n)
    if xi > 0:
        model = ImperfectShell(s, imperfection(s, mode, xi))
        unscale = lambda c: c
    else:
        model = ScaledModel(s)
        unscale = lambda c: model.S*c
    an = Analysis(model.calc_fext, model.calc_fint, model.calc_kC, model.calc_kG)
    an.NL_method = 'arc_length_riks'
    an.final_load_factor = None
    an.maxArcLength = 1e4
    an.initialInc = an.maxInc = 0.1
    if xi == 0:
        an.initialInc = an.maxInc = 0.03
        an.inertia = True
        an.branch_switching = 1
        an.linear_solver = 'splu'
    an.stop_criterion = value_reached(
        lambda c: edge_deflection(s, unscale(c))[1], umax)
    t0 = time.perf_counter()
    an.static(NLgeom=True, silent=True)
    elapsed = time.perf_counter() - t0
    fext = an.calc_fext()
    res = max(np.linalg.norm(an.calc_fint(c) - lbd*fext)/np.linalg.norm(fext)
              for lbd, c in zip(an.increments, an.cs))
    cs = [unscale(c) for c in an.cs]
    u = np.array([[0., 0.]] + [edge_deflection(s, c) for c in cs])
    w = [0.] + [np.abs(s.uvw(c, gridx=21, gridy=21)[1]['w']).max() for c in cs]
    return dict(lbds=np.r_[0., an.increments], u_avg=u[:, 0], u_max=u[:, 1],
                w_max=np.array(w), branch_points=[i + 1 for i in an.branch_points],
                lbd_cr=lbd_cr, time=elapsed, max_residual=res, shell=s, cs=cs)


def path_values(r):
    """First limit (or bifurcation) load, deflections at it, minimum load
    after it, smallest u_max between them and the loads at u_max = 0.02,
    0.03 and 0.04"""
    l, u = r['lbds'], r['u_max']
    ilim = np.flatnonzero(np.diff(l) < 0)[0]
    imin = ilim + np.argmin(l[ilim:])
    iback = ilim + np.argmin(u[ilim:imin + 1])
    out = dict(P_lim=l[ilim], u_lim=u[ilim], u_avg_lim=r['u_avg'][ilim],
               u_back=u[iback], P_back=l[iback], P_min=l[imin], u_min=u[imin])
    for x in (0.02, 0.03, 0.04):
        out['P(%g)' % x] = np.interp(x, u[imin:], l[imin:])
    return out
