r"""Imperfect axially compressed cylinder of Simo, Fox and Rifai (1990)

Benchmark of Simo, Fox and Rifai (1990), Section 6.3.4, "Post-buckling
analysis of a simply-supported cylinder; effect of initial imperfections",
pp. 54-57, Fig. 6.3.4 on p. 56: R = 36, L = 100, t = 0.125, E = 1e7,
nu = 0.3, axial compression. As in the reference, an L/10 x 15 deg section
is modelled, with symmetry conditions on the interior edges and a simple
support on the loaded edge, such that the lowest mode is the diamond mode of
5 axial half-waves and 12 circumferential waves.

The test checks the consistency of the internal forces and of the tangent
stiffness matrix of the imperfect shell, with an imperfection of 50 % of
the thickness in the shape of the w part of the lowest linear buckling
mode.

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
from scipy.sparse import coo_matrix, csr_matrix
from structsolve import lb, solve

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


