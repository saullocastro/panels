r"""Exact 3D elasticity solution of Pagano (1970)

Pagano, N. J., "Exact solutions for rectangular bidirectional composites and
sandwich plates", J. Compos. Mater., 4, 20-34, 1970.

Simply supported ("pinned") rectangular laminate of orthotropic layers with
their axes of symmetry along `x`, `y` and `z`, under the normal traction
`\sigma_z(x, y, h/2) = q_0 \sin(p x) \sin(q y)` on the top face, the bottom face
being traction-free, with `p = \pi/a` and `q = \pi/b`. The edge conditions of
Pagano's Eq. (5), `\sigma_x = v = w = 0` at `x = 0, a` and `\sigma_y = u = w =
0` at `y = 0, b`, are satisfied identically by the displacement field of his
Eq. (6):

.. math::

    u = U(z) \cos px \sin qy \qquad
    v = V(z) \sin px \cos qy \qquad
    w = W(z) \sin px \sin qy

Instead of the closed-form roots of Pagano's Eqs. (10)-(22), the equivalent
first-order system `X' = A X` for the amplitudes `X = \{U, V, W, T_{xz},
T_{yz}, S_z\}` of the displacements and of the transverse stresses is
integrated exactly with the matrix exponential of each layer, which enforces
the interface continuity of Pagano's paper and also handles isotropic layers,
for which his Eqs. (32)-(34) apply. The solution is verified against the
Tables 1-3 of the paper in ``test_pagano1970.py``.
"""
import numpy as np
from scipy.linalg import expm


def orthotropic_C(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23):
    r"""3D stiffness `[C_{ij}]` of an orthotropic material in its axes

    Voigt order `\{\sigma_1, \sigma_2, \sigma_3, \tau_{23}, \tau_{13},
    \tau_{12}\}`, with `\nu_{ij}` the strain in `j` under a stress in `i`.
    """
    S = np.zeros((6, 6))
    S[0, 0] = 1/E1
    S[1, 1] = 1/E2
    S[2, 2] = 1/E3
    S[0, 1] = S[1, 0] = -nu12/E1
    S[0, 2] = S[2, 0] = -nu13/E1
    S[1, 2] = S[2, 1] = -nu23/E2
    S[3, 3] = 1/G23
    S[4, 4] = 1/G13
    S[5, 5] = 1/G12
    return np.linalg.inv(S)


def rotate_90(C):
    r"""Stiffness of the material rotated by 90 degrees about `z`"""
    perm = [1, 0, 2, 4, 3, 5]
    return C[np.ix_(perm, perm)]


def layer_matrix(C, p, q):
    r"""`A` such that `X' = A X`, with `X = \{U, V, W, T_{xz}, T_{yz}, S_z\}`"""
    C11, C12, C13 = C[0, 0], C[0, 1], C[0, 2]
    C22, C23, C33 = C[1, 1], C[1, 2], C[2, 2]
    C44, C55, C66 = C[3, 3], C[4, 4], C[5, 5]
    A = np.zeros((6, 6))
    # U' = T_xz/C55 - p W,  V' = T_yz/C44 - q W
    A[0, 3] = 1/C55
    A[0, 2] = -p
    A[1, 4] = 1/C44
    A[1, 2] = -q
    # W' = (S_z + C13 p U + C23 q V)/C33
    Wp = np.array([C13*p, C23*q, 0, 0, 0, 1])/C33
    A[2] = Wp
    # sigma_x = Sx sin sin, Sx = -C11 p U - C12 q V + C13 W'
    Sx = np.array([-C11*p, -C12*q, 0, 0, 0, 0]) + C13*Wp
    Sy = np.array([-C12*p, -C22*q, 0, 0, 0, 0]) + C23*Wp
    # tau_xy = C66 (q U + p V) cos cos
    Txy = C66*np.array([q, p, 0, 0, 0, 0])
    # equilibrium along x, y and z
    A[3] = -p*Sx + q*Txy
    A[4] = p*Txy - q*Sy
    A[5] = np.array([0, 0, 0, p, q, 0])
    return A


class Pagano1970:
    r"""Exact solution for a laminate of orthotropic layers

    Parameters
    ----------
    a, b : float
        Plate dimensions.
    Cs : list of (6, 6) arrays
        3D stiffness of each layer in the plate axes, from bottom to top.
    hs : list of float
        Thickness of each layer, from bottom to top.
    q0 : float
        Amplitude of the normal traction `\sigma_z` on the top face.
    """
    def __init__(self, a, b, Cs, hs, q0=1.):
        self.a = a
        self.b = b
        self.Cs = [np.asarray(C, dtype=float) for C in Cs]
        self.hs = np.asarray(hs, dtype=float)
        self.h = self.hs.sum()
        self.zi = -self.h/2 + np.concatenate(([0.], np.cumsum(self.hs)))
        self.q0 = q0
        self.p = np.pi/a
        self.q = np.pi/b
        self.As = [layer_matrix(C, self.p, self.q) for C in self.Cs]
        # X_top = T X_bottom, with X_bottom = {U0, V0, W0, 0, 0, 0}
        T = np.eye(6)
        for A, hk in zip(self.As, self.hs):
            T = expm(A*hk) @ T
        X0 = np.zeros(6)
        X0[:3] = np.linalg.solve(T[3:, :3], np.array([0., 0., q0]))
        self.X0 = X0

    def state(self, z):
        r"""Amplitudes `\{U, V, W, T_{xz}, T_{yz}, S_z\}` at the height `z`"""
        X = self.X0.copy()
        for k, (A, hk) in enumerate(zip(self.As, self.hs)):
            za, zb = self.zi[k], self.zi[k+1]
            if z <= zb or k == len(self.hs) - 1:
                return expm(A*(z - za)) @ X, k
            X = expm(A*hk) @ X
        raise ValueError(z)

    def w(self, x, y, z=0.):
        X, _ = self.state(z)
        return X[2]*np.sin(self.p*x)*np.sin(self.q*y)

    def stresses(self, x, y, z):
        r"""`\sigma_x, \sigma_y, \sigma_z, \tau_{yz}, \tau_{xz}, \tau_{xy}`"""
        X, k = self.state(z)
        C = self.Cs[k]
        A = self.As[k]
        p, q = self.p, self.q
        U, V, W, Txz, Tyz, Sz = X
        Wp = A[2] @ X
        sp, cp = np.sin(p*x), np.cos(p*x)
        sq, cq = np.sin(q*y), np.cos(q*y)
        sx = (-C[0, 0]*p*U - C[0, 1]*q*V + C[0, 2]*Wp)*sp*sq
        sy = (-C[0, 1]*p*U - C[1, 1]*q*V + C[1, 2]*Wp)*sp*sq
        sz = Sz*sp*sq
        tyz = Tyz*sp*cq
        txz = Txz*cp*sq
        txy = C[5, 5]*(q*U + p*V)*cp*cq
        return sx, sy, sz, tyz, txz, txy


# Material of Pagano's Eq. (38), and core of his Eq. (40), in psi. Pagano's
# nu_ij is the strain in j under a stress in i, e.g. nu_LT "measuring strain in
# the T-direction under uniaxial normal stress in the L-direction", such that
# his nu_zx = nu_zy = 0.25 of the core give nu13 = nu23 = 0.25*Exx/Ezz = 0.02
# in the convention of orthotropic_C
PAGANO_PLY = dict(E1=25e6, E2=1e6, E3=1e6, nu12=0.25, nu13=0.25, nu23=0.25,
                  G12=0.5e6, G13=0.5e6, G23=0.2e6)
PAGANO_CORE = dict(E1=0.04e6, E2=0.04e6, E3=0.5e6, nu12=0.25,
                   nu13=0.25*0.04/0.5, nu23=0.25*0.04/0.5,
                   G12=0.016e6, G13=0.06e6, G23=0.06e6)
E_T = 1e6


def ply_C(thetadeg):
    C = orthotropic_C(**PAGANO_PLY)
    if thetadeg == 0:
        return C
    if thetadeg == 90:
        return rotate_90(C)
    raise ValueError('only 0 and 90 degree plies')


def normalized(sol, S):
    r"""Normalizations of Pagano's Eq. (39), with `\sigma = q_0`"""
    sig = sol.q0
    h = sol.h
    def wbar(x, y, z=0.):
        return 100*E_T*sol.w(x, y, z)/(sig*h*S**4)
    def stress(x, y, zbar):
        sx, sy, sz, tyz, txz, txy = sol.stresses(x, y, zbar*h)
        return (sx/(sig*S**2), sy/(sig*S**2), txz/(sig*S), tyz/(sig*S),
                txy/(sig*S**2))
    return wbar, stress
