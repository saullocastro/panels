#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Kernels of the base-flange connection ``'BFxcte'``

Connection along `y` between the edge `x_1 =` ``xcte1`` of the stiffener's
base, or skin, ``p1``, and the edge `x_2 =` ``xcte2`` of the flange ``p2``,
normal to ``p1``. With the local axes of the flange rotated by 90 degrees
about the common axis `y`, the penalty energy is:

.. math::

    U = \frac{1}{2} \int_y k_t \left[ (u_1 - w_2)^2 + (v_1 - v_2)^2 +
        (w_1 + u_2)^2 \right] + k_r \left( \omega_1 - \omega_2 \right)^2 dy

where `\omega_i` is the rotation of the normal of each panel about `y`,
positive with the right-hand rule in its own local axes. For the models
based on the classical laminated plate theory (CLPT), with 3 DOFs `u, v, w`
per term, `\omega_i = -w_{i,x}`, also with the kinematics of Sanders-Koiter,
whose rotation about `y` has no term of the curvature, see
:func:`.fkCBFxcte11`, :func:`.fkCBFxcte12` and :func:`.fkCBFxcte22`.

For the models based on shear deformation theories, ``'plate_fsdt_donnell'``
and ``'plate_tsdt_donnell'``, with 5 DOFs `u, v, w, \phi_x, \phi_y` per
term, `\omega_i = \phi_{x,i}` and the rotation penalty is `k_r (\phi_{x,1} -
\phi_{x,2})^2`, see :func:`.fkCBFxcte11_sdt`, :func:`.fkCBFxcte12_sdt` and
:func:`.fkCBFxcte22_sdt`. For the TSDT, the derivative `w_{,x}` that enters
the displacement field is not penalized: at a T-joint only the rotation of
the normals is common to the two panels. The rotation `\phi_y` of each panel
is a drilling rotation of the other and is not penalized either.

The terms are derived in ``theory/multidomain_penalization/connections.py``.

"""
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE


cdef int DOF = 3

cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil


def fkCBFxcte11(double kt, double kr, object p1, double xcte1,
          int size, int row0, int col0):
    r"""
    Penalty approach calculation to base-flange xcte panel 1 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Shell
        Shell() object
    xcte1 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the MultiDomain.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCBFxcte11 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to xcte of panel p1 position.

    """
    cdef int i1, j1, k1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double x1u1, x1ur1, x2u1, x2ur1
    cdef double x1v1, x1vr1, x2v1, x2vr1
    cdef double x1w1, x1wr1, x2w1, x2wr1
    cdef double y1u1, y1ur1, y2u1, y2ur1
    cdef double y1v1, y1vr1, y2v1, y2vr1
    cdef double y1w1, y1wr1, y2w1, y2wr1

    cdef long [:] kCBFxcte11r, kCBFxcte11c
    cdef double [:] kCBFxcte11v

    cdef double xicte1
    cdef double g1Auf1Bu, g1Avf1Bv, g1Awf1Bw
    cdef double f1Au, f1Av, f1Aw, f1Awxi, f1Bu, f1Bv, f1Bw, f1Bwxi

    a1 = p1.a
    b1 = p1.b
    m1 = p1.m
    n1 = p1.n
    x1u1 = p1.x1u ; x1ur1 = p1.x1ur ; x2u1 = p1.x2u ; x2ur1 = p1.x2ur
    x1v1 = p1.x1v ; x1vr1 = p1.x1vr ; x2v1 = p1.x2v ; x2vr1 = p1.x2vr
    x1w1 = p1.x1w ; x1wr1 = p1.x1wr ; x2w1 = p1.x2w ; x2wr1 = p1.x2wr
    y1u1 = p1.y1u ; y1ur1 = p1.y1ur ; y2u1 = p1.y2u ; y2ur1 = p1.y2ur
    y1v1 = p1.y1v ; y1vr1 = p1.y1vr ; y2v1 = p1.y2v ; y2vr1 = p1.y2vr
    y1w1 = p1.y1w ; y1wr1 = p1.y1wr ; y2w1 = p1.y2w ; y2wr1 = p1.y2wr

    xicte1 = 2*xcte1/a1 - 1.

    fdim = 3*m1*n1*m1*n1

    kCBFxcte11r = np.zeros((fdim,), dtype=INT)
    kCBFxcte11c = np.zeros((fdim,), dtype=INT)
    kCBFxcte11v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for j1 in range(n1):
            for l1 in range(n1):
                g1Auf1Bu = integral_ff(j1, l1, y1u1, y1ur1, y2u1, y2ur1, y1u1, y1ur1, y2u1, y2ur1)
                g1Avf1Bv = integral_ff(j1, l1, y1v1, y1vr1, y2v1, y2vr1, y1v1, y1vr1, y2v1, y2vr1)
                g1Awf1Bw = integral_ff(j1, l1, y1w1, y1wr1, y2w1, y2wr1, y1w1, y1wr1, y2w1, y2wr1)

                for i1 in range(m1):
                    f1Au = f(i1, xicte1, x1u1, x1ur1, x2u1, x2ur1)
                    f1Av = f(i1, xicte1, x1v1, x1vr1, x2v1, x2vr1)
                    f1Aw = f(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)
                    f1Awxi = fp(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)

                    for k1 in range(m1):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        f1Bu = f(k1, xicte1, x1u1, x1ur1, x2u1, x2ur1)
                        f1Bv = f(k1, xicte1, x1v1, x1vr1, x2v1, x2vr1)
                        f1Bw = f(k1, xicte1, x1w1, x1wr1, x2w1, x2wr1)
                        f1Bwxi = fp(k1, xicte1, x1w1, x1wr1, x2w1, x2wr1)

                        c += 1
                        kCBFxcte11r[c] = row+0
                        kCBFxcte11c[c] = col+0
                        kCBFxcte11v[c] += 0.5*b1*f1Au*f1Bu*g1Auf1Bu*kt
                        c += 1
                        kCBFxcte11r[c] = row+1
                        kCBFxcte11c[c] = col+1
                        kCBFxcte11v[c] += 0.5*b1*f1Av*f1Bv*g1Avf1Bv*kt
                        c += 1
                        kCBFxcte11r[c] = row+2
                        kCBFxcte11c[c] = col+2
                        kCBFxcte11v[c] += 0.5*b1*kt*(f1Aw*f1Bw*g1Awf1Bw + 4*f1Awxi*f1Bwxi*g1Awf1Bw*kr/((a1*a1)*kt))

    kCBFxcte11 = coo_matrix((kCBFxcte11v, (kCBFxcte11r, kCBFxcte11c)), shape=(size, size))

    return kCBFxcte11


def fkCBFxcte12(double kt, double kr, object p1, object p2,
          double xcte1, double xcte2,
          int size, int row0, int col0):
    r"""
    Penalty approach calculation to base-flange xcte panel 1 and panel 2 coupling position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Shell
        First Shell object
    p2 : Shell
        Second Shell object
    xcte1 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    xcte2 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the MultiDomain.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCBFxcte12 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to xcte of panel 1 and panel 2 coupling position.

    """
    cdef int i1, j1, k2, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, a2, b1, b2
    cdef double x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCBFxcte12r, kCBFxcte12c
    cdef double [:] kCBFxcte12v

    cdef double xicte1, xicte2
    cdef double g1Auf2Bw, g1Avf2Bv, g1Awf2Bu, g1Awf2Bw
    cdef double f1Au, f1Av, f1Aw, f1Awxi, f2Bu, f2Bv, f2Bw, f2Bwxi

    a1 = p1.a
    a2 = p2.a
    b1 = p1.b
    b2 = p2.b
    m1 = p1.m
    n1 = p1.n
    m2 = p2.m
    n2 = p2.n
    x1u1 = p1.x1u ; x1ur1 = p1.x1ur ; x2u1 = p1.x2u ; x2ur1 = p1.x2ur
    x1v1 = p1.x1v ; x1vr1 = p1.x1vr ; x2v1 = p1.x2v ; x2vr1 = p1.x2vr
    x1w1 = p1.x1w ; x1wr1 = p1.x1wr ; x2w1 = p1.x2w ; x2wr1 = p1.x2wr
    y1u1 = p1.y1u ; y1ur1 = p1.y1ur ; y2u1 = p1.y2u ; y2ur1 = p1.y2ur
    y1v1 = p1.y1v ; y1vr1 = p1.y1vr ; y2v1 = p1.y2v ; y2vr1 = p1.y2vr
    y1w1 = p1.y1w ; y1wr1 = p1.y1wr ; y2w1 = p1.y2w ; y2wr1 = p1.y2wr

    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    xicte1 = 2*xcte1/a1 - 1.
    xicte2 = 2*xcte2/a2 - 1.

    fdim = 4*m1*n1*m2*n2

    kCBFxcte12r = np.zeros((fdim,), dtype=INT)
    kCBFxcte12c = np.zeros((fdim,), dtype=INT)
    kCBFxcte12v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for j1 in range(n1):
            for l2 in range(n2):
                g1Auf2Bw = integral_ff(j1, l2, y1u1, y1ur1, y2u1, y2ur1, y1w2, y1wr2, y2w2, y2wr2)
                g1Avf2Bv = integral_ff(j1, l2, y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2)
                g1Awf2Bu = integral_ff(j1, l2, y1w1, y1wr1, y2w1, y2wr1, y1u2, y1ur2, y2u2, y2ur2)
                g1Awf2Bw = integral_ff(j1, l2, y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2)

                for i1 in range(m1):
                    f1Au = f(i1, xicte1, x1u1, x1ur1, x2u1, x2ur1)
                    f1Av = f(i1, xicte1, x1v1, x1vr1, x2v1, x2vr1)
                    f1Aw = f(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)
                    f1Awxi = fp(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)

                    for k2 in range(m2):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        f2Bu = f(k2, xicte2, x1u2, x1ur2, x2u2, x2ur2)
                        f2Bv = f(k2, xicte2, x1v2, x1vr2, x2v2, x2vr2)
                        f2Bw = f(k2, xicte2, x1w2, x1wr2, x2w2, x2wr2)
                        f2Bwxi = fp(k2, xicte2, x1w2, x1wr2, x2w2, x2wr2)

                        c += 1
                        kCBFxcte12r[c] = row+0
                        kCBFxcte12c[c] = col+2
                        kCBFxcte12v[c] += -0.5*b1*f1Au*f2Bw*g1Auf2Bw*kt
                        c += 1
                        kCBFxcte12r[c] = row+1
                        kCBFxcte12c[c] = col+1
                        kCBFxcte12v[c] += -0.5*b1*f1Av*f2Bv*g1Avf2Bv*kt
                        c += 1
                        kCBFxcte12r[c] = row+2
                        kCBFxcte12c[c] = col+0
                        kCBFxcte12v[c] += 0.5*b1*f1Aw*f2Bu*g1Awf2Bu*kt
                        c += 1
                        kCBFxcte12r[c] = row+2
                        kCBFxcte12c[c] = col+2
                        kCBFxcte12v[c] += -2*b1*f1Awxi*f2Bwxi*g1Awf2Bw*kr/(a1*a2)

    kCBFxcte12 = coo_matrix((kCBFxcte12v, (kCBFxcte12r, kCBFxcte12c)), shape=(size, size))

    return kCBFxcte12


def fkCBFxcte22(double kt, double kr, object p1, object p2, double xcte2,
          int size, int row0, int col0):
    r"""
    Penalty approach calculation to base-flange xcte panel 2 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Shell
        First Shell object
    p2 : Shell
        Second Shell object
    xcte2 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the MultiDomain.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCBFxcte22 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to xcte of panel p2 position.

    """
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double b1, a2
    cdef double x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCBFxcte22r, kCBFxcte22c
    cdef double [:] kCBFxcte22v

    cdef double xicte2
    cdef double g2Auf2Bu, g2Avf2Bv, g2Awf2Bw
    cdef double f2Au, f2Bu, f2Av, f2Bv, f2Aw, f2Bw, f2Awxi, f2Bwxi
    b1 = p1.b
    a2 = p2.a
    m2 = p2.m
    n2 = p2.n
    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    xicte2 = 2*xcte2/a2 - 1.

    fdim = 3*m2*n2*m2*n2

    kCBFxcte22r = np.zeros((fdim,), dtype=INT)
    kCBFxcte22c = np.zeros((fdim,), dtype=INT)
    kCBFxcte22v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for j2 in range(n2):
            for l2 in range(n2):
                g2Auf2Bu = integral_ff(j2, l2, y1u2, y1ur2, y2u2, y2ur2, y1u2, y1ur2, y2u2, y2ur2)
                g2Avf2Bv = integral_ff(j2, l2, y1v2, y1vr2, y2v2, y2vr2, y1v2, y1vr2, y2v2, y2vr2)
                g2Awf2Bw = integral_ff(j2, l2, y1w2, y1wr2, y2w2, y2wr2, y1w2, y1wr2, y2w2, y2wr2)

                for i2 in range(m2):
                    f2Au = f(i2, xicte2, x1u2, x1ur2, x2u2, x2ur2)
                    f2Av = f(i2, xicte2, x1v2, x1vr2, x2v2, x2vr2)
                    f2Aw = f(i2, xicte2, x1w2, x1wr2, x2w2, x2wr2)
                    f2Awxi = fp(i2, xicte2, x1w2, x1wr2, x2w2, x2wr2)

                    for k2 in range(m2):
                        row = row0 + DOF*(j2*m2 + i2)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        f2Bu = f(k2, xicte2, x1u2, x1ur2, x2u2, x2ur2)
                        f2Bv = f(k2, xicte2, x1v2, x1vr2, x2v2, x2vr2)
                        f2Bw = f(k2, xicte2, x1w2, x1wr2, x2w2, x2wr2)
                        f2Bwxi = fp(k2, xicte2, x1w2, x1wr2, x2w2, x2wr2)

                        c += 1
                        kCBFxcte22r[c] = row+0
                        kCBFxcte22c[c] = col+0
                        kCBFxcte22v[c] += 0.5*b1*f2Au*f2Bu*g2Auf2Bu*kt
                        c += 1
                        kCBFxcte22r[c] = row+1
                        kCBFxcte22c[c] = col+1
                        kCBFxcte22v[c] += 0.5*b1*f2Av*f2Bv*g2Avf2Bv*kt
                        c += 1
                        kCBFxcte22r[c] = row+2
                        kCBFxcte22c[c] = col+2
                        kCBFxcte22v[c] += 0.5*b1*kt*(f2Aw*f2Bw*g2Awf2Bw + 4*f2Awxi*f2Bwxi*g2Awf2Bw*kr/((a2*a2)*kt))

    kCBFxcte22 = coo_matrix((kCBFxcte22v, (kCBFxcte22r, kCBFxcte22c)), shape=(size, size))

    return kCBFxcte22


# ----------------------------------------------------------------------------
# Models based on shear deformation theories (FSDT and TSDT), 5 DOFs per term
# ----------------------------------------------------------------------------

cdef int DOF_SDT = 5
U, V, W, PHIX, PHIY = range(5)
FIELDS_SDT = ('u', 'v', 'w', 'phix', 'phiy')


def _check_sdt(*panels):
    for p in panels:
        if p.model not in ('plate_fsdt_donnell', 'plate_tsdt_donnell'):
            raise ValueError("Expected a model based on shear deformation "
                             "theories, got model '{0}'".format(p.model))


def _flags_sdt(object p, fields, str direction):
    r"""Boundary flags ``(1t, 1r, 2t, 2r)`` along ``direction`` of each field
    of ``fields``"""
    out = np.zeros((len(fields), 4), dtype=DOUBLE)
    for t, field in enumerate(fields):
        name = FIELDS_SDT[field]
        out[t, 0] = getattr(p, direction + '1' + name)
        out[t, 1] = getattr(p, direction + '1' + name + 'r')
        out[t, 2] = getattr(p, direction + '2' + name)
        out[t, 3] = getattr(p, direction + '2' + name + 'r')
    return out


cdef void _terms_sdt(int nt, long [::1] dofA, long [::1] dofB,
        double [::1] coeff, double [:, ::1] alA, double [:, ::1] alB,
        double [:, ::1] acA, double [:, ::1] acB, int nalA, int nalB,
        int nacA, int nacB, double cteA, double cteB, double jac,
        int along_x, int upper, int row0, int col0,
        long [::1] r, long [::1] c, double [::1] v) noexcept nogil:
    r"""Penalty terms `coeff \int q_A q_B` along the connection

    ``al*`` are the flags of the functions along the connection, integrated
    with ``integral_ff``, ``ac*`` the flags of the functions across it,
    evaluated at the natural coordinates ``cteA`` and ``cteB``. ``nal*`` and
    ``nac*`` are the numbers of terms along and across the connection. The
    term `(i, j)`, with `i` along `x`, has the Ritz constants at
    ``DOF_SDT*(j*m + i)``. With ``upper = 1`` only the upper triangle is
    kept.

    """
    cdef int t, iA, iB, jA, jB, row, col, pos, mA, mB
    cdef double I, gA, gB
    pos = 0
    mA = nalA if along_x else nacA
    mB = nalB if along_x else nacB
    for t in range(nt):
        for iA in range(nalA):
            for iB in range(nalB):
                I = jac*integral_ff(iA, iB, alA[t, 0], alA[t, 1], alA[t, 2],
                        alA[t, 3], alB[t, 0], alB[t, 1], alB[t, 2], alB[t, 3])
                for jA in range(nacA):
                    gA = f(jA, cteA, acA[t, 0], acA[t, 1], acA[t, 2], acA[t, 3])
                    for jB in range(nacB):
                        gB = f(jB, cteB, acB[t, 0], acB[t, 1], acB[t, 2], acB[t, 3])
                        if along_x:
                            row = row0 + DOF_SDT*(jA*mA + iA) + dofA[t]
                            col = col0 + DOF_SDT*(jB*mB + iB) + dofB[t]
                        else:
                            row = row0 + DOF_SDT*(iA*mA + jA) + dofA[t]
                            col = col0 + DOF_SDT*(iB*mB + jB) + dofB[t]
                        if upper and row > col:
                            v[pos] = 0.
                        else:
                            v[pos] = coeff[t]*I*gA*gB
                        r[pos] = row
                        c[pos] = col
                        pos += 1


def _block_sdt(terms, object pA, object pB, double cteA, double cteB,
        str along, int upper, int size, int row0, int col0):
    r"""Block of the connection matrix between ``pA`` and ``pB`` from the
    list ``terms`` of ``(dofA, dofB, coeff)``, see :func:`._terms_sdt`"""
    cdef int nt, nalA, nalB, nacA, nacB, n
    cdef long [::1] dofA, dofB, r, c
    cdef double [::1] coeff, v
    cdef double [:, ::1] alA, alB, acA, acB
    cdef double jac
    cdef int along_x = along == 'x'
    across = 'y' if along_x else 'x'
    nt = len(terms)
    dofA = np.array([t[0] for t in terms], dtype=INT)
    dofB = np.array([t[1] for t in terms], dtype=INT)
    coeff = np.array([t[2] for t in terms], dtype=DOUBLE)
    alA = _flags_sdt(pA, [t[0] for t in terms], along)
    alB = _flags_sdt(pB, [t[1] for t in terms], along)
    acA = _flags_sdt(pA, [t[0] for t in terms], across)
    acB = _flags_sdt(pB, [t[1] for t in terms], across)
    if along == 'x':
        nalA, nalB, nacA, nacB = pA.m, pB.m, pA.n, pB.n
        #NOTE the connection spans the length of pA
        jac = pA.a/2.
    else:
        nalA, nalB, nacA, nacB = pA.n, pB.n, pA.m, pB.m
        jac = pA.b/2.
    n = nt*nalA*nalB*nacA*nacB
    r = np.zeros(n, dtype=INT)
    c = np.zeros(n, dtype=INT)
    v = np.zeros(n, dtype=DOUBLE)
    with nogil:
        _terms_sdt(nt, dofA, dofB, coeff, alA, alB, acA, acB, nalA, nalB,
                   nacA, nacB, cteA, cteB, jac, along_x, upper, row0,
                   col0, r, c, v)
    return coo_matrix((v, (r, c)), shape=(size, size))


def fkCBFxcte11_sdt(double kt, double kr, object p1, double xcte1,
          int size, int row0, int col0):
    r"""
    Base-flange xcte connection for the models based on shear deformation
    theories, block of the base

    The rotation penalty is on `\phi_x`, see the module docstring of
    :mod:`panels.multidomain.connections.kCBFxcte`.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Panel
        Base, or skin.
    xcte1 : float
        Coordinate `x_1` of the connection in ``p1``.
    size : int
        Size of the assembly.
    row0 : int
        Row position of the block in the assembly.
    col0 : int
        Column position of the block in the assembly.

    Returns
    -------
    kCBFxcte11_sdt : scipy.sparse.coo_matrix
        Upper triangle of the block of the base.

    """
    _check_sdt(p1)
    terms = [(U, U, kt), (V, V, kt), (W, W, kt), (PHIX, PHIX, kr)]
    cte = 2*xcte1/p1.a - 1.
    return _block_sdt(terms, p1, p1, cte, cte, 'y', 1, size, row0, col0)


def fkCBFxcte12_sdt(double kt, double kr, object p1, object p2,
          double xcte1, double xcte2,
          int size, int row0, int col0):
    r"""
    Base-flange xcte connection for the models based on shear deformation
    theories, block of the base and the flange

    The rotation penalty is on `\phi_x`, see the module docstring of
    :mod:`panels.multidomain.connections.kCBFxcte`.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Panel
        Base, or skin.
    p2 : Panel
        Flange.
    xcte1, xcte2 : float
        Coordinates `x_1` and `x_2` of the connection in ``p1`` and ``p2``.
    size : int
        Size of the assembly.
    row0 : int
        Row position of the block in the assembly.
    col0 : int
        Column position of the block in the assembly.

    Returns
    -------
    kCBFxcte12_sdt : scipy.sparse.coo_matrix
        Coupling block, rows of the base and columns of the flange.

    """
    _check_sdt(p1, p2)
    terms = [(U, W, -kt), (V, V, -kt), (W, U, kt), (PHIX, PHIX, -kr)]
    cte1 = 2*xcte1/p1.a - 1.
    cte2 = 2*xcte2/p2.a - 1.
    return _block_sdt(terms, p1, p2, cte1, cte2, 'y', 0, size, row0, col0)


def fkCBFxcte22_sdt(double kt, double kr, object p1, object p2,
          double xcte2,
          int size, int row0, int col0):
    r"""
    Base-flange xcte connection for the models based on shear deformation
    theories, block of the flange

    The rotation penalty is on `\phi_x`, see the module docstring of
    :mod:`panels.multidomain.connections.kCBFxcte`.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Panel
        Base, or skin.
    p2 : Panel
        Flange.
    xcte2 : float
        Coordinate `x_2` of the connection in ``p2``.
    size : int
        Size of the assembly.
    row0 : int
        Row position of the block in the assembly.
    col0 : int
        Column position of the block in the assembly.

    Returns
    -------
    kCBFxcte22_sdt : scipy.sparse.coo_matrix
        Upper triangle of the block of the flange.

    """
    _check_sdt(p1, p2)
    terms = [(U, U, kt), (V, V, kt), (W, W, kt), (PHIX, PHIX, kr)]
    cte = 2*xcte2/p2.a - 1.
    return _block_sdt(terms, p2, p2, cte, cte, 'y', 1, size, row0, col0)
