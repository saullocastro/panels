#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Kernels of the base-flange connection ``'BFycte'``

Connection along `x` between the edge `y_1 =` ``ycte1`` of the stiffener's
base, or skin, ``p1``, and the edge `y_2 =` ``ycte2`` of the flange ``p2``,
normal to ``p1``. With the local axes of the flange rotated by 90 degrees
about the common axis `x`, the penalty energy is:

.. math::

    U = \frac{1}{2} \int_x k_t \left[ (u_1 - u_2)^2 + (v_1 - w_2)^2 +
        (w_1 + v_2)^2 \right] + k_r \left( \omega_1 - \omega_2 \right)^2 dx

where `\omega_i` is the rotation of the normal of each panel about `x`,
positive with the right-hand rule in its own local axes. With the kinematics
of Donnell and for flat plates, `\omega_i = w_{i,y}`. With the kinematics of
Sanders-Koiter (``'cylshell_clpt_sanders'``), the rotation of the normal also
includes the rigid-body rotation of the cross-section, `\omega_i = w_{i,y} -
v_i/r_i`, consistent with `\phi_y = -w_{,y} + v/r` of the model. The
arguments ``rinv1`` and ``rinv2`` are `1/r_1` and `1/r_2` for panels with
the kinematics of Sanders-Koiter and zero otherwise, see
:meth:`.MultiDomain.get_kC_conn`.

Without the term `v_1/r_1`, as up to version 0.7.1, a rigid-body rotation of
a stiffened cylinder about its axis, `v_1 = r_1 \theta`, is penalized by the
rotation penalty, since the Sanders-Koiter skin rotates by `\theta` while the
flange must follow with `w_{2,y} = \theta`. With the Donnell kinematics the
term `v/r` is neglected in the rotations and in the curvatures of the shell
alike, and the kernels are unchanged.

For the models based on shear deformation theories, ``'plate_fsdt_donnell'``
and ``'plate_tsdt_donnell'``, with 5 DOFs `u, v, w, \phi_x, \phi_y` per
term, the rotation of the normal about `x` is `\omega_i = -\phi_{y,i}`, and
the rotation penalty is `k_r (\phi_{y,1} - \phi_{y,2})^2`, see
:func:`.fkCBFycte11_sdt`, :func:`.fkCBFycte12_sdt` and
:func:`.fkCBFycte22_sdt`. For the TSDT, the derivative `w_{,y}` that enters
the displacement field is not penalized: at a T-joint only the rotation of
the normals is common to the two panels. The rotation `\phi_x` of each panel
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


def fkCBFycte11(double kt, double kr, object p1, double ycte1,
          int size, int row0, int col0, double rinv1=0.):
    r"""
    Penalty approach calculation to base-flange ycte panel 1 position.

    Block of the base ``p1``, see the module docstring of
    :mod:`panels.multidomain.connections.kCBFycte`.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Panel
        Panel() object
    ycte1 : float
        Dimension value that determines the flag value eta.
        If ycte1 = 0 => eta = -1, if ycte1 = p1.b => eta = 1.
        Where eta=-1 stands for boundary 1 and eta=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.
    rinv1 : float, optional
        `1/r_1` when ``p1`` has the kinematics of Sanders-Koiter, zero
        otherwise.

    Returns
    -------
    kCBFycte11 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel p1 position.

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

    cdef long [:] kCBFycte11r, kCBFycte11c
    cdef double [:] kCBFycte11v

    cdef double etacte1
    cdef double f1Auf1Bu, f1Avf1Bv, f1Awf1Bw, f1Avf1Bw, f1Awf1Bv
    cdef double g1Au, g1Av, g1Aw, g1Aweta, g1Bu, g1Bv, g1Bw, g1Bweta

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

    etacte1 = 2*ycte1/b1 - 1.

    fdim = 5*m1*n1*m1*n1

    kCBFycte11r = np.zeros((fdim,), dtype=INT)
    kCBFycte11c = np.zeros((fdim,), dtype=INT)
    kCBFycte11v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):
                f1Auf1Bu = integral_ff(i1, k1, x1u1, x1ur1, x2u1, x2ur1, x1u1, x1ur1, x2u1, x2ur1)
                f1Avf1Bv = integral_ff(i1, k1, x1v1, x1vr1, x2v1, x2vr1, x1v1, x1vr1, x2v1, x2vr1)
                f1Awf1Bw = integral_ff(i1, k1, x1w1, x1wr1, x2w1, x2wr1, x1w1, x1wr1, x2w1, x2wr1)
                f1Avf1Bw = integral_ff(i1, k1, x1v1, x1vr1, x2v1, x2vr1, x1w1, x1wr1, x2w1, x2wr1)
                f1Awf1Bv = integral_ff(i1, k1, x1w1, x1wr1, x2w1, x2wr1, x1v1, x1vr1, x2v1, x2vr1)

                for j1 in range(n1):
                    g1Au = f(j1, etacte1, y1u1, y1ur1, y2u1, y2ur1)
                    g1Av = f(j1, etacte1, y1v1, y1vr1, y2v1, y2vr1)
                    g1Aw = f(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
                    g1Aweta = fp(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)

                    for l1 in range(n1):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g1Bu = f(l1, etacte1, y1u1, y1ur1, y2u1, y2ur1)
                        g1Bv = f(l1, etacte1, y1v1, y1vr1, y2v1, y2vr1)
                        g1Bw = f(l1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
                        g1Bweta = fp(l1, etacte1, y1w1, y1wr1, y2w1, y2wr1)

                        c += 1
                        kCBFycte11r[c] = row+0
                        kCBFycte11c[c] = col+0
                        kCBFycte11v[c] += 0.5*a1*f1Auf1Bu*g1Au*g1Bu*kt
                        c += 1
                        kCBFycte11r[c] = row+1
                        kCBFycte11c[c] = col+1
                        kCBFycte11v[c] += 0.5*a1*f1Avf1Bv*g1Av*g1Bv*kt
                        c += 1
                        kCBFycte11r[c] = row+2
                        kCBFycte11c[c] = col+2
                        kCBFycte11v[c] += 0.5*a1*kt*(f1Awf1Bw*g1Aw*g1Bw + 4*f1Awf1Bw*g1Aweta*g1Bweta*kr/((b1*b1)*kt))
                        if rinv1 != 0:
                            #NOTE Sanders-Koiter rotation w1,y - v1/r1
                            c += 1
                            kCBFycte11r[c] = row+1
                            kCBFycte11c[c] = col+1
                            kCBFycte11v[c] += 0.5*a1*kr*rinv1*rinv1*f1Avf1Bv*g1Av*g1Bv
                            c += 1
                            kCBFycte11r[c] = row+1
                            kCBFycte11c[c] = col+2
                            kCBFycte11v[c] += -a1*kr*rinv1*f1Avf1Bw*g1Av*g1Bweta/b1
                            c += 1
                            kCBFycte11r[c] = row+2
                            kCBFycte11c[c] = col+1
                            kCBFycte11v[c] += -a1*kr*rinv1*f1Awf1Bv*g1Aweta*g1Bv/b1

    kCBFycte11 = coo_matrix((kCBFycte11v, (kCBFycte11r, kCBFycte11c)), shape=(size, size))

    return kCBFycte11


def fkCBFycte12(double kt, double kr, object p1, object p2,
          double ycte1, double ycte2,
          int size, int row0, int col0, double rinv1=0., double rinv2=0.):
    r"""
    Penalty approach calculation to base-flange ycte panel 1 and panel 2 coupling position.

    Coupling block between the base ``p1`` and the flange ``p2``, see the
    module docstring of :mod:`panels.multidomain.connections.kCBFycte`.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Panel
        First Panel object
    p2 : Panel
        Second Panel object
    ycte1 : float
        Dimension value that determines the flag value eta.
        If ycte1 = 0 => eta = -1, if ycte1 = p1.b => eta = 1.
        Where eta=-1 stands for boundary 1 and eta=1 stands for boundary 2.
    ycte2 : float
        Dimension value that determines the flag value eta.
        If ycte1 = 0 => eta = -1, if ycte1 = p1.b => eta = 1.
        Where eta=-1 stands for boundary 1 and eta=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.
    rinv1, rinv2 : float, optional
        `1/r_1` and `1/r_2` when ``p1`` and ``p2`` have the kinematics of
        Sanders-Koiter, zero otherwise.

    Returns
    -------
    kCBFycte12 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel 1 and panel 2 coupling position.

    """
    cdef int i1, j1, k2, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, b1, b2
    cdef double x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCBFycte12r, kCBFycte12c
    cdef double [:] kCBFycte12v

    cdef double etacte1, etacte2
    cdef double f1Auf2Bu, f1Avf2Bw, f1Awf2Bv, f1Awf2Bw, f1Avf2Bv
    cdef double g1Au, g1Av, g1Aw, g1Aweta, g2Bu, g2Bv, g2Bw, g2Bweta

    a1 = p1.a
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

    etacte1 = 2*ycte1/b1 - 1.
    etacte2 = 2*ycte2/b2 - 1.

    fdim = 7*m1*n1*m2*n2

    kCBFycte12r = np.zeros((fdim,), dtype=INT)
    kCBFycte12c = np.zeros((fdim,), dtype=INT)
    kCBFycte12v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k2 in range(m2):
                f1Auf2Bu = integral_ff(i1, k2, x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2)
                f1Avf2Bw = integral_ff(i1, k2, x1v1, x1vr1, x2v1, x2vr1, x1w2, x1wr2, x2w2, x2wr2)
                f1Awf2Bv = integral_ff(i1, k2, x1w1, x1wr1, x2w1, x2wr1, x1v2, x1vr2, x2v2, x2vr2)
                f1Awf2Bw = integral_ff(i1, k2, x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2)
                f1Avf2Bv = integral_ff(i1, k2, x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2)

                for j1 in range(n1):
                    g1Au = f(j1, etacte1, y1u1, y1ur1, y2u1, y2ur1)
                    g1Av = f(j1, etacte1, y1v1, y1vr1, y2v1, y2vr1)
                    g1Aw = f(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
                    g1Aweta = fp(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)

                    for l2 in range(n2):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        g2Bu = f(l2, etacte2, y1u2, y1ur2, y2u2, y2ur2)
                        g2Bv = f(l2, etacte2, y1v2, y1vr2, y2v2, y2vr2)
                        g2Bw = f(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
                        g2Bweta = fp(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

                        c += 1
                        kCBFycte12r[c] = row+0
                        kCBFycte12c[c] = col+0
                        kCBFycte12v[c] += -0.5*a1*f1Auf2Bu*g1Au*g2Bu*kt
                        c += 1
                        kCBFycte12r[c] = row+1
                        kCBFycte12c[c] = col+2
                        kCBFycte12v[c] += -0.5*a1*f1Avf2Bw*g1Av*g2Bw*kt
                        c += 1
                        kCBFycte12r[c] = row+2
                        kCBFycte12c[c] = col+1
                        kCBFycte12v[c] += 0.5*a1*f1Awf2Bv*g1Aw*g2Bv*kt
                        c += 1
                        kCBFycte12r[c] = row+2
                        kCBFycte12c[c] = col+2
                        kCBFycte12v[c] += -2*a1*f1Awf2Bw*g1Aweta*g2Bweta*kr/(b1*b2)
                        if rinv1 != 0 or rinv2 != 0:
                            #NOTE Sanders-Koiter rotations w1,y - v1/r1 and
                            #     w2,y - v2/r2
                            c += 1
                            kCBFycte12r[c] = row+1
                            kCBFycte12c[c] = col+1
                            kCBFycte12v[c] += -0.5*a1*kr*rinv1*rinv2*f1Avf2Bv*g1Av*g2Bv
                            c += 1
                            kCBFycte12r[c] = row+1
                            kCBFycte12c[c] = col+2
                            kCBFycte12v[c] += a1*kr*rinv1*f1Avf2Bw*g1Av*g2Bweta/b2
                            c += 1
                            kCBFycte12r[c] = row+2
                            kCBFycte12c[c] = col+1
                            kCBFycte12v[c] += a1*kr*rinv2*f1Awf2Bv*g1Aweta*g2Bv/b1

    kCBFycte12 = coo_matrix((kCBFycte12v, (kCBFycte12r, kCBFycte12c)), shape=(size, size))

    return kCBFycte12


def fkCBFycte22(double kt, double kr, object p1, object p2,
          double ycte2,
          int size, int row0, int col0, double rinv2=0.):
    r"""
    Penalty approach calculation to base-flange ycte panel 2 position.

    Block of the flange ``p2``, see the module docstring of
    :mod:`panels.multidomain.connections.kCBFycte`.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Panel
        First Panel object
    p2 : Panel
        Second Panel object
    ycte2 : float
        Dimension value that determines the flag value eta.
        If ycte1 = 0 => eta = -1, if ycte1 = p1.b => eta = 1.
        Where eta=-1 stands for boundary 1 and eta=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.
    rinv2 : float, optional
        `1/r_2` when ``p2`` has the kinematics of Sanders-Koiter, zero
        otherwise.

    Returns
    -------
    kCBFycte22 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel p2 position.

    """
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double a1, b2
    cdef double x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCBFycte22r, kCBFycte22c
    cdef double [:] kCBFycte22v

    cdef double etacte2
    cdef double f2Auf2Bu, f2Avf2Bv, f2Awf2Bw, f2Avf2Bw, f2Awf2Bv
    cdef double g2Au, g2Bu, g2Av, g2Bv, g2Aw, g2Bw, g2Aweta, g2Bweta
    a1 = p1.a
    b2 = p2.b
    m2 = p2.m
    n2 = p2.n
    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    etacte2 = 2*ycte2/b2 - 1.

    fdim = 5*m2*n2*m2*n2

    kCBFycte22r = np.zeros((fdim,), dtype=INT)
    kCBFycte22c = np.zeros((fdim,), dtype=INT)
    kCBFycte22v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i2 in range(m2):
            for k2 in range(m2):
                f2Auf2Bu = integral_ff(i2, k2, x1u2, x1ur2, x2u2, x2ur2, x1u2, x1ur2, x2u2, x2ur2)
                f2Avf2Bv = integral_ff(i2, k2, x1v2, x1vr2, x2v2, x2vr2, x1v2, x1vr2, x2v2, x2vr2)
                f2Awf2Bw = integral_ff(i2, k2, x1w2, x1wr2, x2w2, x2wr2, x1w2, x1wr2, x2w2, x2wr2)
                f2Avf2Bw = integral_ff(i2, k2, x1v2, x1vr2, x2v2, x2vr2, x1w2, x1wr2, x2w2, x2wr2)
                f2Awf2Bv = integral_ff(i2, k2, x1w2, x1wr2, x2w2, x2wr2, x1v2, x1vr2, x2v2, x2vr2)

                for j2 in range(n2):
                    g2Au = f(j2, etacte2, y1u2, y1ur2, y2u2, y2ur2)
                    g2Av = f(j2, etacte2, y1v2, y1vr2, y2v2, y2vr2)
                    g2Aw = f(j2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
                    g2Aweta = fp(j2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

                    for l2 in range(n2):
                        row = row0 + DOF*(j2*m2 + i2)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g2Bu = f(l2, etacte2, y1u2, y1ur2, y2u2, y2ur2)
                        g2Bv = f(l2, etacte2, y1v2, y1vr2, y2v2, y2vr2)
                        g2Bw = f(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
                        g2Bweta = fp(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

                        c += 1
                        kCBFycte22r[c] = row+0
                        kCBFycte22c[c] = col+0
                        kCBFycte22v[c] += 0.5*a1*f2Auf2Bu*g2Au*g2Bu*kt
                        c += 1
                        kCBFycte22r[c] = row+1
                        kCBFycte22c[c] = col+1
                        kCBFycte22v[c] += 0.5*a1*f2Avf2Bv*g2Av*g2Bv*kt
                        c += 1
                        kCBFycte22r[c] = row+2
                        kCBFycte22c[c] = col+2
                        kCBFycte22v[c] += 0.5*a1*kt*(f2Awf2Bw*g2Aw*g2Bw + 4*f2Awf2Bw*g2Aweta*g2Bweta*kr/((b2*b2)*kt))
                        if rinv2 != 0:
                            #NOTE Sanders-Koiter rotation w2,y - v2/r2
                            c += 1
                            kCBFycte22r[c] = row+1
                            kCBFycte22c[c] = col+1
                            kCBFycte22v[c] += 0.5*a1*kr*rinv2*rinv2*f2Avf2Bv*g2Av*g2Bv
                            c += 1
                            kCBFycte22r[c] = row+1
                            kCBFycte22c[c] = col+2
                            kCBFycte22v[c] += -a1*kr*rinv2*f2Avf2Bw*g2Av*g2Bweta/b2
                            c += 1
                            kCBFycte22r[c] = row+2
                            kCBFycte22c[c] = col+1
                            kCBFycte22v[c] += -a1*kr*rinv2*f2Awf2Bv*g2Aweta*g2Bv/b2

    kCBFycte22 = coo_matrix((kCBFycte22v, (kCBFycte22r, kCBFycte22c)), shape=(size, size))

    return kCBFycte22


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


def fkCBFycte11_sdt(double kt, double kr, object p1, double ycte1,
          int size, int row0, int col0):
    r"""
    Base-flange ycte connection for the models based on shear deformation
    theories, block of the base

    The rotation penalty is on `\phi_y`, see the module docstring of
    :mod:`panels.multidomain.connections.kCBFycte`.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Panel
        Base, or skin.
    ycte1 : float
        Coordinate `y_1` of the connection in ``p1``.
    size : int
        Size of the assembly.
    row0 : int
        Row position of the block in the assembly.
    col0 : int
        Column position of the block in the assembly.

    Returns
    -------
    kCBFycte11_sdt : scipy.sparse.coo_matrix
        Upper triangle of the block of the base.

    """
    _check_sdt(p1)
    terms = [(U, U, kt), (V, V, kt), (W, W, kt), (PHIY, PHIY, kr)]
    cte = 2*ycte1/p1.b - 1.
    return _block_sdt(terms, p1, p1, cte, cte, 'x', 1, size, row0, col0)


def fkCBFycte12_sdt(double kt, double kr, object p1, object p2,
          double ycte1, double ycte2,
          int size, int row0, int col0):
    r"""
    Base-flange ycte connection for the models based on shear deformation
    theories, block of the base and the flange

    The rotation penalty is on `\phi_y`, see the module docstring of
    :mod:`panels.multidomain.connections.kCBFycte`.

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
    ycte1, ycte2 : float
        Coordinates `y_1` and `y_2` of the connection in ``p1`` and ``p2``.
    size : int
        Size of the assembly.
    row0 : int
        Row position of the block in the assembly.
    col0 : int
        Column position of the block in the assembly.

    Returns
    -------
    kCBFycte12_sdt : scipy.sparse.coo_matrix
        Coupling block, rows of the base and columns of the flange.

    """
    _check_sdt(p1, p2)
    terms = [(U, U, -kt), (V, W, -kt), (W, V, kt), (PHIY, PHIY, -kr)]
    cte1 = 2*ycte1/p1.b - 1.
    cte2 = 2*ycte2/p2.b - 1.
    return _block_sdt(terms, p1, p2, cte1, cte2, 'x', 0, size, row0, col0)


def fkCBFycte22_sdt(double kt, double kr, object p1, object p2,
          double ycte2,
          int size, int row0, int col0):
    r"""
    Base-flange ycte connection for the models based on shear deformation
    theories, block of the flange

    The rotation penalty is on `\phi_y`, see the module docstring of
    :mod:`panels.multidomain.connections.kCBFycte`.

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
    ycte2 : float
        Coordinate `y_2` of the connection in ``p2``.
    size : int
        Size of the assembly.
    row0 : int
        Row position of the block in the assembly.
    col0 : int
        Column position of the block in the assembly.

    Returns
    -------
    kCBFycte22_sdt : scipy.sparse.coo_matrix
        Upper triangle of the block of the flange.

    """
    _check_sdt(p1, p2)
    terms = [(U, U, kt), (V, V, kt), (W, W, kt), (PHIY, PHIY, kr)]
    cte = 2*ycte2/p2.b - 1.
    return _block_sdt(terms, p2, p2, cte, cte, 'x', 1, size, row0, col0)
