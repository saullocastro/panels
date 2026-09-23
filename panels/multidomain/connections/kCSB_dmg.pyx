#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Kernels of the damaged skin-base connection ``'SB_TSL'``

Stiffness matrix of the area connection between a top panel ``p1`` and a
bottom panel ``p2`` with a traction-separation law at the interface, see
:ref:`cohesive_zone`. The kernels evaluate, with a Gauss-Legendre rule of
``nr_x_gauss`` x ``nr_y_gauss`` points over the domain of ``p1``:

.. math::

    [K] = \int_A k^w_{CZ} [B_\Delta]^T [B_\Delta] \, dA, \qquad
    \{c\}^T [K] \{c\} = \int_A k^w_{CZ} \left( \Delta_u^2 + \Delta_v^2 +
    \Delta_w^2 \right) dA

where `k^w_{CZ} = k_o (1 - d)` is given at each integration point by
``kw_tsl``, the penalty energy of the connection is `U = \frac{1}{2} \{c\}^T
[K] \{c\}`, and `[B_\Delta]` gives, from the Ritz constants `\{c\}`, the
displacement jump between the two surfaces in contact, with the slope of each
panel:

.. math::

    \begin{aligned}
        \Delta_u &= u^t + d^t w^t_{,x} - u^b + d^b w^b_{,x} \\
        \Delta_v &= v^t + d^t w^t_{,y} - v^b + d^b w^b_{,y} \\
        \Delta_w &= w^t - w^b
    \end{aligned}

where `d^t` and `d^b` are the distances from the mid-planes of the top and
bottom panels to the interface, `u^t, v^t, w^t` and `u^b, v^b, w^b` their
mid-plane displacements. The compatibility of the thesis of D'Souza (2024)
[nathan2024MSc]_, Eqs. 4.57--4.62, assumes equal slopes of both panels
(Eq. 4.61) and writes `\Delta_u = u^t + (d^t + d^b) w^t_{,x} - u^b`
(Eq. 4.62), which holds for a perfect bond but gives a spurious tangential
separation inside the fracture process zone, where the two arms rotate in
opposite directions. The kernel of the undamaged connection ``'SB'``,
``kCSB.pyx``, keeps the compatibility of the thesis.

With `f`, `g` the approximation functions along `\xi` and `\eta`, and `w_{,x}
= (2/a) f_{,\xi} g`, `w_{,y} = (2/b) f g_{,\eta}`, the terms of each block are
listed in the docstrings of :func:`.fkCSB11_dmg` (top-top),
:func:`.fkCSB12_dmg` (top-bottom) and :func:`.fkCSB22_dmg` (bottom-bottom),
before the factor `w_\xi w_\eta \, ab/4` of the Gauss-Legendre rule, with `a`,
`b` the dimensions of ``p1``. The bottom panel must cover the same area as the
top one, its functions are evaluated at the natural coordinates of the top
panel, which :meth:`.MultiDomain.get_kC_conn` checks.

The kernels were verified by comparing `\{c\}^T [K] \{c\}` with the
Gauss-Legendre integral of `k^w_{CZ} (\Delta_u^2 + \Delta_v^2 + \Delta_w^2)`
evaluated from the displacement field, for random `\{c\}`, random boundary
flags, panels with different numbers of terms and thicknesses, and a random
`k^w_{CZ}` field: the relative difference is below `10^{-15}`, see
``tests/multidomain/test_sb_tsl.py``.

:meth:`.MultiDomain.get_kC_conn` uses these kernels only when the connection
has ``use_kernels=True``. Otherwise the same matrix, with the same integration
rule, is computed by matrix products in ``MultiDomain._kC_TSL``, which is much
faster.

"""
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE


cdef int DOF = 3

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffp(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfp(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil


def fkCSB11_dmg(double dt, object p1, int size, int row0, int col0, 
            int nr_x_gauss, int nr_y_gauss, double [:,::1] kw_tsl):
    r"""
    Penalty approach calculation to skin-base ycte panel 1 position.

    Block top-top of the damaged skin-base connection, see the module
    docstring of :mod:`panels.multidomain.connections.kCSB_dmg` and
    :ref:`cohesive_zone`. With the notation of the module docstring, the
    terms at each integration point, before the factor `w_\xi w_\eta \,
    ab/4 \, k^w_{CZ}`, are:

    - `(u, u)`: `f_u f_u g_u g_u`, and similarly for `(v, v)`
    - `(u, w)`: `\frac{2}{a} d^t f_u f_{w,\xi} g_u g_w`
    - `(v, w)`: `\frac{2}{b} d^t f_v f_w g_v g_{w,\eta}`
    - `(w, w)`: `f_w f_w g_w g_w + \frac{4 (d^t)^2}{a^2} f_{w,\xi} f_{w,\xi}
      g_w g_w + \frac{4 (d^t)^2}{b^2} f_w f_w g_{w,\eta} g_{w,\eta}`

    and the symmetric `(w, u)`, `(w, v)`. These are the terms of the thesis
    of D'Souza (2024) [nathan2024MSc]_ with `d^t` in place of `d^t + d^b`.
    Only the upper triangle is returned.

    Parameters
    ----------
    dt : float
        Distance from the mid-plane of the top panel ``p1`` to the
        interface, ``dt = sum(p1.plyts)/2.``. The tangential separation is
        evaluated with the slope of each panel, see the module docstring.
    p1 : Panel
        Top panel
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.
    nr_x_gauss, nr_y_gauss : int
        Number of integration points in x and y
    kw_tsl : numpy array
        Out of plane stiffness due to the TSL for each damage instance. This is a grid that is mapped to the
        integration points provided by nr_x_gauss and nr_y_gauss, with values for each of those points.
            [:,::1] reads with an increment of 1 in the column

    Returns
    -------
    kCSB11 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel p1 position.

    """
    cdef int i1, k1, j1, l1, c, row, col, ptx, pty
    cdef int m1, n1
    cdef double a1, b1, xi, eta, weight
    cdef double x1u1, x1ur1, x2u1, x2ur1
    cdef double x1v1, x1vr1, x2v1, x2vr1
    cdef double x1w1, x1wr1, x2w1, x2wr1
    cdef double y1u1, y1ur1, y2u1, y2ur1
    cdef double y1v1, y1vr1, y2v1, y2vr1
    cdef double y1w1, y1wr1, y2w1, y2wr1

    cdef long [:] kCSB11r, kCSB11c
    cdef double [:] kCSB11v
    cdef double [:] weights_xi, weights_eta, xis, etas

    cdef double f1Au, f1Av, f1Aw, f1Awxi, f1Bu, f1Bv, f1Bw, f1Bwxi
    cdef double g1Au, g1Av, g1Aw, g1Aweta, g1Bu, g1Bv, g1Bw, g1Bweta
    cdef double kt

    a1 = p1.a
    b1 = p1.b
    m1 = p1.m
    n1 = p1.n
    # Panel 1
    x1u1 = p1.x1u ; x1ur1 = p1.x1ur ; x2u1 = p1.x2u ; x2ur1 = p1.x2ur
    x1v1 = p1.x1v ; x1vr1 = p1.x1vr ; x2v1 = p1.x2v ; x2vr1 = p1.x2vr
    x1w1 = p1.x1w ; x1wr1 = p1.x1wr ; x2w1 = p1.x2w ; x2wr1 = p1.x2wr
    y1u1 = p1.y1u ; y1ur1 = p1.y1ur ; y2u1 = p1.y2u ; y2ur1 = p1.y2ur
    y1v1 = p1.y1v ; y1vr1 = p1.y1vr ; y2v1 = p1.y2v ; y2vr1 = p1.y2vr
    y1w1 = p1.y1w ; y1wr1 = p1.y1wr ; y2w1 = p1.y2w ; y2wr1 = p1.y2wr

    fdim = 7*m1*n1*m1*n1
    # 7 is depenedent on the number of terms in the for loop which populate the values of the stiff matrices 
    # i.e. in this case there are 7 instances of terms being added so it preallocates that amount of memory

    # Calc gauss points and weights
    xis, weights_xi = roots_legendre(nr_x_gauss)
    etas, weights_eta = roots_legendre(nr_y_gauss)
    
    kCSB11r = np.zeros((fdim,), dtype=INT)
    kCSB11c = np.zeros((fdim,), dtype=INT)
    kCSB11v = np.zeros((fdim,), dtype=DOUBLE)

    # print(f'        KCSB_11 -- kw_tsl {np.min(kw_tsl):.2e} {np.max(kw_tsl):.2e}')
    
    with nogil:
        # kCSB11
        
        for pty in range(nr_y_gauss):
            for ptx in range(nr_x_gauss):
                # Makes it more efficient when reading data (kw_tsl) from memory as memory is read along a row
                # So also accessing memory in the same way helps it out so its not deleting and reaccessing the
                # same memory everytime. 
                
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                # Extracting the correct kt
                    # Currently, the outer loop of y and inner of x, causes it to go through all x for a single y
                    # That is going through all col for a single row then onto the next row
                    # (as per x, y and results by calc_results)
                kt = kw_tsl[pty, ptx]
                
                c = -1
                for i1 in range(m1):
                    # NOTE: When any of these are uncommented, make sure they are defined earlier (cdef etc)
                    f1Au = f(i1, xi, x1u1, x1ur1, x2u1, x2ur1)
                    # f1Auxi = fp(i1, xi, x1u1, x1ur1, x2u1, x2ur1)
                    f1Av = f(i1, xi, x1v1, x1vr1, x2v1, x2vr1)
                    # f1Avxi = fp(i1, xi, x1v1, x1vr1, x2v1, x2vr1)
                    f1Aw = f(i1, xi, x1w1, x1wr1, x2w1, x2wr1)
                    f1Awxi = fp(i1, xi, x1w1, x1wr1, x2w1, x2wr1)
                    # f1Awxixi = fpp(i1, xi, x1w1, x1wr1, x2w1, x2wr1)
                    
                    for k1 in range(m1):
                        f1Bu = f(k1, xi, x1u1, x1ur1, x2u1, x2ur1)
                        # f1Buxi = fp(k1, xi, x1u1, x1ur1, x2u1, x2ur1)
                        f1Bv = f(k1, xi, x1v1, x1vr1, x2v1, x2vr1)
                        # f1Bvxi = fp(k1, xi, x1v1, x1vr1, x2v1, x2vr1)
                        f1Bw = f(k1, xi, x1w1, x1wr1, x2w1, x2wr1)
                        f1Bwxi = fp(k1, xi, x1w1, x1wr1, x2w1, x2wr1)
                        # f1Bwxixi = fpp(k1, xi, x1w1, x1wr1, x2w1, x2wr1)
                        
                        for j1 in range(n1): 
                            g1Au = f(j1, eta, y1u1, y1ur1, y2u1, y2ur1)
                            # g1Aueta = fp(j1, eta, y1u1, y1ur1, y2u1, y2ur1)
                            g1Av = f(j1, eta, y1v1, y1vr1, y2v1, y2vr1)
                            # g1Aveta = fp(j1, eta, y1v1, y1vr1, y2v1 y2vr1)
                            g1Aw = f(j1, eta, y1w1, y1wr1, y2w1, y2wr1)
                            g1Aweta = fp(j1, eta, y1w1, y1wr1, y2w1, y2wr1)
                            # g1Awetaeta = fpp(j1, eta, y1w1, y1wr1, y2w1, y2wr1)
                                    
                            for l1 in range(n1):
                                g1Bu = f(l1, eta, y1u1, y1ur1, y2u1, y2ur1)
                                # g1Bueta = fp(l1, eta, y1u1, y1ur1, y2u1, y2ur1)
                                g1Bv = f(l1, eta, y1v1, y1vr1, y2v1, y2vr1)
                                # g1Bveta = fp(l1, eta, y1v1, y1vr1, y2v1, y2vr1)
                                g1Bw = f(l1, eta, y1w1, y1wr1, y2w1, y2wr1)
                                g1Bweta = fp(l1, eta, y1w1, y1wr1, y2w1, y2wr1)
                                # g1Bwetaeta = fpp(l1, eta, y1w1, y1wr1, y2w1, y2wr1)
        
        
                                row = row0 + DOF*(j1*m1 + i1)
                                col = col0 + DOF*(l1*m1 + k1)
        
                                #NOTE symmetry - 11
                                if row > col:
                                    continue
        
                                c += 1
                                kCSB11r[c] = row+0
                                kCSB11c[c] = col+0
                                kCSB11v[c] += weight*0.25*a1*b1*f1Au*f1Bu*g1Au*g1Bu*kt
                                c += 1
                                kCSB11r[c] = row+0
                                kCSB11c[c] = col+2
                                kCSB11v[c] += weight*0.5*b1*dt*f1Au*f1Bwxi*g1Au*g1Bw*kt
                                c += 1
                                kCSB11r[c] = row+1
                                kCSB11c[c] = col+1
                                kCSB11v[c] += weight*0.25*a1*b1*f1Av*f1Bv*g1Av*g1Bv*kt
                                c += 1
                                kCSB11r[c] = row+1
                                kCSB11c[c] = col+2
                                kCSB11v[c] += weight*0.5*a1*dt*f1Av*f1Bw*g1Av*g1Bweta*kt
                                c += 1
                                kCSB11r[c] = row+2
                                kCSB11c[c] = col+0
                                kCSB11v[c] += weight*0.5*b1*dt*f1Awxi*f1Bu*g1Aw*g1Bu*kt
                                c += 1
                                kCSB11r[c] = row+2
                                kCSB11c[c] = col+1
                                kCSB11v[c] += weight*0.5*a1*dt*f1Aw*f1Bv*g1Aweta*g1Bv*kt
                                c += 1
                                kCSB11r[c] = row+2
                                kCSB11c[c] = col+2
                                kCSB11v[c] += weight*(0.25*a1*b1*kt*(f1Aw*f1Bw*g1Aw*g1Bw + 4*(dt*dt)*f1Aw*f1Bw*g1Aweta*g1Bweta/(b1*b1) + 4*(dt*dt)*f1Awxi*f1Bwxi*g1Aw*g1Bw/(a1*a1)))

    kCSB11 = coo_matrix((kCSB11v, (kCSB11r, kCSB11c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return kCSB11


def fkCSB12_dmg(double dt, double db, object p1, object p2, int size, int row0, int col0,
                int nr_x_gauss, int nr_y_gauss, double [:,::1] kw_tsl):
    r"""
    Penalty approach calculation to skin-base ycte panel 1 and panel 2 coupling position.

    Block top-bottom of the damaged skin-base connection, see the module
    docstring of :mod:`panels.multidomain.connections.kCSB_dmg` and
    :ref:`cohesive_zone`. With the superscripts `t` for the top panel ``p1``
    (rows) and `b` for the bottom panel ``p2`` (columns), the terms at each
    integration point, before the factor `w_\xi w_\eta \, ab/4 \, k^w_{CZ}`,
    are:

    - `(u^t, u^b)`: `- f^t_u f^b_u g^t_u g^b_u`, and similarly for `(v^t,
      v^b)`
    - `(u^t, w^b)`: `+ \frac{2}{a} d^b f^t_u f^b_{w,\xi} g^t_u g^b_w`
    - `(v^t, w^b)`: `+ \frac{2}{b} d^b f^t_v f^b_w g^t_v g^b_{w,\eta}`
    - `(w^t, u^b)`: `- \frac{2}{a} d^t f^t_{w,\xi} f^b_u g^t_w g^b_u`
    - `(w^t, v^b)`: `- \frac{2}{b} d^t f^t_w f^b_v g^t_{w,\eta} g^b_v`
    - `(w^t, w^b)`: `- f^t_w f^b_w g^t_w g^b_w + \frac{4 d^t d^b}{a^2}
      f^t_{w,\xi} f^b_{w,\xi} g^t_w g^b_w + \frac{4 d^t d^b}{b^2} f^t_w f^b_w
      g^t_{w,\eta} g^b_{w,\eta}`

    With respect to the kernel of the thesis of D'Souza (2024)
    [nathan2024MSc]_, the terms `(u^t, w^b)`, `(v^t, w^b)` and `(w^t, w^b)`
    are new or modified, and `(w^t, u^b)`, `(w^t, v^b)` use `d^t`. The kernel
    of the thesis also built the approximation functions in `\eta` of the top
    panel with the flags of the edge `y_1` in place of `y_2`, which had no
    effect on the DCB, where all `y` flags are 1.

    The block couples the rows of ``p1`` to the columns of ``p2``, it must be
    transposed when ``p1`` comes after ``p2`` in the assembly.

    Parameters
    ----------
    dt, db : float
        Distances from the mid-planes of the top panel ``p1`` and of the
        bottom panel ``p2`` to the interface.
    p1 : Panel
        Top panel
    p2 : Panel
        Bottom panel
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
    kw_tsl : numpy array
        Out of plane stiffness due to the TSL for each damage instance. This is a grid that is mapped to the
        integration points provided by nr_x_gauss and nr_y_gauss, with values for each of those points.
            [:,::1] reads with an increment of 1 in the column

    Returns
    -------
    kCBFycte12 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel 1 and panel 2 coupling position.

    """
    cdef int i1, j1, k2, l2, c, row, col, ptx, pty
    cdef int m1, n1, m2, n2
    cdef double a1, b1, xi, eta, weight
    cdef double x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCSB12r, kCSB12c
    cdef double [:] kCSB12v
    cdef double [:] weights_xi, weights_eta, xis, etas

    cdef double f1Au, f2Bu, f1Av, f2Bv, f1Aw, f2Bw, f1Awxi, f2Bwxi
    cdef double g1Au, g2Bu, g1Av, g2Bv, g1Aw, g2Bw, g1Aweta, g2Bweta
    cdef double kt

    a1 = p1.a
    b1 = p1.b
    m1 = p1.m
    n1 = p1.n
    m2 = p2.m
    n2 = p2.n
    # Panel 1 (ends in 1)
    x1u1 = p1.x1u ; x1ur1 = p1.x1ur ; x2u1 = p1.x2u ; x2ur1 = p1.x2ur
    x1v1 = p1.x1v ; x1vr1 = p1.x1vr ; x2v1 = p1.x2v ; x2vr1 = p1.x2vr
    x1w1 = p1.x1w ; x1wr1 = p1.x1wr ; x2w1 = p1.x2w ; x2wr1 = p1.x2wr
    y1u1 = p1.y1u ; y1ur1 = p1.y1ur ; y2u1 = p1.y2u ; y2ur1 = p1.y2ur
    y1v1 = p1.y1v ; y1vr1 = p1.y1vr ; y2v1 = p1.y2v ; y2vr1 = p1.y2vr
    y1w1 = p1.y1w ; y1wr1 = p1.y1wr ; y2w1 = p1.y2w ; y2wr1 = p1.y2wr

    # Panel 2 (ends in 2)
    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    fdim = 7*m1*n1*m2*n2

    # Calc gauss points and weights
    xis, weights_xi = roots_legendre(nr_x_gauss)
    etas, weights_eta = roots_legendre(nr_y_gauss)

    kCSB12r = np.zeros((fdim,), dtype=INT)
    kCSB12c = np.zeros((fdim,), dtype=INT)
    kCSB12v = np.zeros((fdim,), dtype=DOUBLE)
    
    # print(f'        KCSB_12 -- kw_tsl {np.min(kw_tsl):.2e} {np.max(kw_tsl):.2e}')

    with nogil:
        # kCSB12
        
        for ptx in range(nr_x_gauss):
            for pty in range(nr_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                # Extracting the correct kt
                    # Currently, the outer loop of x and inner of y, causes it to go through all y for a single x
                    # That is going through all rows for a single col then onto the next col
                    # (as per x, y and results by calc_results)
                kt = kw_tsl[pty, ptx]
                
                c = -1
                for i1 in range(m1):
                    f1Au = f(i1, xi, x1u1, x1ur1, x2u1, x2ur1)
                    # f1Auxi = fp(i1, xi, x1u1, x1ur1, x2u1, x2ur1)
                    f1Av = f(i1, xi, x1v1, x1vr1, x2v1, x2vr1)
                    # f1Avxi = fp(i1, xi, x1v1, x1vr1, x2v1, x2vr1)
                    f1Aw = f(i1, xi, x1w1, x1wr1, x2w1, x2wr1)
                    f1Awxi = fp(i1, xi, x1w1, x1wr1, x2w1, x2wr1)
                    # f1Awxixi = fpp(i1, xi, x1w1, x1wr1, x2w1, x2wr1)
                    
                    for k2 in range(m2):
                        f2Bu = f(k2, xi, x1u2, x1ur2, x2u2, x2ur2)
                        # f2Buxi = fp(k2, xi, x1u2, x1ur2, x2u2, x2ur2)
                        f2Bv = f(k2, xi, x1v2, x1vr2, x2v2, x2vr2)
                        # f2Bvxi = fp(k2, xi, x1v2, x1vr2, x2v2, x2vr2)
                        f2Bw = f(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        f2Bwxi = fp(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        # f2Bwxixi = fpp(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        
                        for j1 in range(n1):
                            g1Au = f(j1, eta, y1u1, y1ur1, y2u1, y2ur1)
                            # g1Aueta = fp(j1, eta, y1u1, y1ur1, y1u1, y1ur1)
                            g1Av = f(j1, eta, y1v1, y1vr1, y2v1, y2vr1)
                            # g1Aveta = fp(j1, eta, y1v1, y1vr1, y1v1, y1vr1)
                            g1Aw = f(j1, eta, y1w1, y1wr1, y2w1, y2wr1)
                            g1Aweta = fp(j1, eta, y1w1, y1wr1, y2w1, y2wr1)
                            # g1Awetaeta = fpp(j1, eta, y1w1, y1wr1, y1w1, y1wr1)
                                    
                            for l2 in range(n2):
                                g2Bu = f(l2, eta, y1u2, y1ur2, y2u2, y2ur2)
                                # g2Bueta = fp(l2, eta, y1u2, y1ur2, y2u2, y2ur2)
                                g2Bv = f(l2, eta, y1v2, y1vr2, y2v2, y2vr2)
                                # g2Bveta = fp(l2, eta, y1v2, y1vr2, y2v2, y2vr2)
                                g2Bw = f(l2, eta, y1w2, y1wr2, y2w2, y2wr2)
                                g2Bweta = fp(l2, eta, y1w2, y1wr2, y2w2, y2wr2)
                                # g2Bwetaeta = fpp(l2, eta, y1w2, y1wr2, y2w2, y2wr2)
        
        
                                row = row0 + DOF*(j1*m1 + i1)
                                col = col0 + DOF*(l2*m2 + k2)
        
                                #NO symmetry - 12
                                # if row > col:
                                #     continue
        
                                c += 1
                                kCSB12r[c] = row+0
                                kCSB12c[c] = col+0
                                kCSB12v[c] += -weight*0.25*a1*b1*f1Au*f2Bu*g1Au*g2Bu*kt
                                c += 1
                                kCSB12r[c] = row+0
                                kCSB12c[c] = col+2
                                kCSB12v[c] += weight*0.5*b1*db*f1Au*f2Bwxi*g1Au*g2Bw*kt
                                c += 1
                                kCSB12r[c] = row+1
                                kCSB12c[c] = col+1
                                kCSB12v[c] += -weight*0.25*a1*b1*f1Av*f2Bv*g1Av*g2Bv*kt
                                c += 1
                                kCSB12r[c] = row+1
                                kCSB12c[c] = col+2
                                kCSB12v[c] += weight*0.5*a1*db*f1Av*f2Bw*g1Av*g2Bweta*kt
                                c += 1
                                kCSB12r[c] = row+2
                                kCSB12c[c] = col+0
                                kCSB12v[c] += -weight*0.5*b1*dt*f1Awxi*f2Bu*g1Aw*g2Bu*kt
                                c += 1
                                kCSB12r[c] = row+2
                                kCSB12c[c] = col+1
                                kCSB12v[c] += -weight*0.5*a1*dt*f1Aw*f2Bv*g1Aweta*g2Bv*kt
                                c += 1
                                kCSB12r[c] = row+2
                                kCSB12c[c] = col+2
                                kCSB12v[c] += weight*(0.25*a1*b1*kt*(-f1Aw*f2Bw*g1Aw*g2Bw + 4*dt*db*f1Aw*f2Bw*g1Aweta*g2Bweta/(b1*b1) + 4*dt*db*f1Awxi*f2Bwxi*g1Aw*g2Bw/(a1*a1)))

    kCSB12 = coo_matrix((kCSB12v, (kCSB12r, kCSB12c)), shape=(size, size))

    return kCSB12


def fkCSB22_dmg(double db, object p1, object p2, int size, int row0, int col0,
                int nr_x_gauss, int nr_y_gauss, double [:,::1] kw_tsl):
    r"""
    Penalty approach calculation to skin-base ycte panel 2 position.

    Block bottom-bottom of the damaged skin-base connection, see the module
    docstring of :mod:`panels.multidomain.connections.kCSB_dmg` and
    :ref:`cohesive_zone`. With the functions of the bottom panel ``p2``, the
    terms at each integration point, before the factor `w_\xi w_\eta \, ab/4
    \, k^w_{CZ}` with `a`, `b` the dimensions of ``p1``, are:

    - `(u, u)`: `f_u f_u g_u g_u`, and similarly for `(v, v)`
    - `(u, w)`: `- \frac{2}{a} d^b f_u f_{w,\xi} g_u g_w`
    - `(v, w)`: `- \frac{2}{b} d^b f_v f_w g_v g_{w,\eta}`
    - `(w, w)`: `f_w f_w g_w g_w + \frac{4 (d^b)^2}{a^2} f_{w,\xi} f_{w,\xi}
      g_w g_w + \frac{4 (d^b)^2}{b^2} f_w f_w g_{w,\eta} g_{w,\eta}`

    and the symmetric `(w, u)`, `(w, v)`. With respect to the kernel of the
    thesis of D'Souza (2024) [nathan2024MSc]_, the terms `(u, w)`, `(v, w)`
    and the terms with `(d^b)^2` of `(w, w)` are new. Only the upper
    triangle is returned.

    Parameters
    ----------
    db : float
        Distance from the mid-plane of the bottom panel ``p2`` to the
        interface, ``db = sum(p2.plyts)/2.``.
    p1 : Panel
        Top panel, it defines the integration domain
    p2 : Panel
        Bottom panel
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

    Returns
    -------
    kCSB22 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel p2 position.

    """
    cdef int i2, k2, j2, l2, c, row, col, ptx, pty
    cdef int m2, n2
    cdef double a1, b1, xi, eta, weight
    cdef double x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCSB22r, kCSB22c
    cdef double [:] kCSB22v
    cdef double [:] weights_xi, weights_eta, xis, etas

    cdef double f2Au, f2Bu, f2Av, f2Bv, f2Aw, f2Bw, f2Awxi, f2Bwxi
    cdef double g2Au, g2Bu, g2Av, g2Bv, g2Aw, g2Bw, g2Aweta, g2Bweta
    cdef double kt

    a1 = p1.a
    b1 = p1.b
    m2 = p2.m
    n2 = p2.n
    # Panel 2
    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    fdim = 7*m2*n2*m2*n2
    
    # Calc gauss points and weights
    xis, weights_xi = roots_legendre(nr_x_gauss)
    etas, weights_eta = roots_legendre(nr_y_gauss)

    kCSB22r = np.zeros((fdim,), dtype=INT)
    kCSB22c = np.zeros((fdim,), dtype=INT)
    kCSB22v = np.zeros((fdim,), dtype=DOUBLE)
    
    # print(f'        KCSB_22 -- kw_tsl {np.min(kw_tsl):.2e} {np.max(kw_tsl):.2e}')

    with nogil:
        
        for ptx in range(nr_x_gauss):
            for pty in range(nr_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                # Extracting the correct kt
                    # Currently, the outer loop of x and inner of y, causes it to go through all y for a single x
                    # That is going through all rows for a single col then onto the next col
                    # (as per x, y and results by calc_results)
                kt = kw_tsl[pty, ptx]
                # kt = kw_tsl[ptx, pty]
                
                c = -1
                for i2 in range(m2):
                    f2Au = f(i2, xi, x1u2, x1ur2, x2u2, x2ur2)
                    # f2Auxi = fp(i2, xi, x1u2, x1ur2, x2u2, x2ur2)
                    f2Av = f(i2, xi, x1v2, x1vr2, x2v2, x2vr2)
                    # f2Avxi = fp(i2, xi, x1v2, x1vr2, x2v2, x2vr2)
                    f2Aw = f(i2, xi, x1w2, x1wr2, x2w2, x2wr2)
                    f2Awxi = fp(i2, xi, x1w2, x1wr2, x2w2, x2wr2)
                    # f2Awxixi = fpp(i2, xi, x1w2, x1wr2, x2w2, x2wr2)
                    
                    for k2 in range(m2):
                        f2Bu = f(k2, xi, x1u2, x1ur2, x2u2, x2ur2)
                        # f2Buxi = fp(k2, xi, x1u2, x1ur2, x2u2, x2ur2)
                        f2Bv = f(k2, xi, x1v2, x1vr2, x2v2, x2vr2)
                        # f2Bvxi = fp(k2, xi, x1v2, x1vr2, x2v2, x2vr2)
                        f2Bw = f(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        f2Bwxi = fp(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        # f2Bwxixi = fpp(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        
                        for j2 in range(n2):
                            g2Au = f(j2, eta, y1u2, y1ur2, y2u2, y2ur2)
                            # g2Aueta = fp(j2, eta, y1u2, y1ur2, y2u2, y2ur2)
                            g2Av = f(j2, eta, y1v2, y1vr2, y2v2, y2vr2)
                            # g2Aveta = fp(j2, eta, y1v2, y1vr2, y2v2, y2vr2)
                            g2Aw = f(j2, eta, y1w2, y1wr2, y2w2, y2wr2)
                            g2Aweta = fp(j2, eta, y1w2, y1wr2, y2w2, y2wr2)
                            # g2Awetaeta = fpp(j2, eta, y1w2, y1wr2, y2w2, y2wr2)
                                    
                            for l2 in range(n2):
                                g2Bu = f(l2, eta, y1u2, y1ur2, y2u2, y2ur2)
                                # g2Bueta = fp(l2, eta, y1u2, y1ur2, y2u2, y2ur2)
                                g2Bv = f(l2, eta, y1v2, y1vr2, y2v2, y2vr2)
                                # g2Bveta = fp(l2, eta, y1v2, y1vr2, y2v2, y2vr2)
                                g2Bw = f(l2, eta, y1w2, y1wr2, y2w2, y2wr2)
                                g2Bweta = fp(l2, eta, y1w2, y1wr2, y2w2, y2wr2)
                                # g2Bwetaeta = fpp(l2, eta, y1w2, y1wr2, y2w2, y2wr2)
                                
                                
                                row = row0 + DOF*(j2*m2 + i2)
                                col = col0 + DOF*(l2*m2 + k2)
        
                                #NOTE symmetry
                                if row > col:
                                    continue

                                c += 1
                                kCSB22r[c] = row+0
                                kCSB22c[c] = col+0
                                kCSB22v[c] += weight*0.25*a1*b1*f2Au*f2Bu*g2Au*g2Bu*kt
                                c += 1
                                kCSB22r[c] = row+0
                                kCSB22c[c] = col+2
                                kCSB22v[c] += -weight*0.5*b1*db*f2Au*f2Bwxi*g2Au*g2Bw*kt
                                c += 1
                                kCSB22r[c] = row+1
                                kCSB22c[c] = col+1
                                kCSB22v[c] += weight*0.25*a1*b1*f2Av*f2Bv*g2Av*g2Bv*kt
                                c += 1
                                kCSB22r[c] = row+1
                                kCSB22c[c] = col+2
                                kCSB22v[c] += -weight*0.5*a1*db*f2Av*f2Bw*g2Av*g2Bweta*kt
                                c += 1
                                kCSB22r[c] = row+2
                                kCSB22c[c] = col+0
                                kCSB22v[c] += -weight*0.5*b1*db*f2Awxi*f2Bu*g2Aw*g2Bu*kt
                                c += 1
                                kCSB22r[c] = row+2
                                kCSB22c[c] = col+1
                                kCSB22v[c] += -weight*0.5*a1*db*f2Aw*f2Bv*g2Aweta*g2Bv*kt
                                c += 1
                                kCSB22r[c] = row+2
                                kCSB22c[c] = col+2
                                kCSB22v[c] += weight*(0.25*a1*b1*kt*(f2Aw*f2Bw*g2Aw*g2Bw + 4*(db*db)*f2Aw*f2Bw*g2Aweta*g2Bweta/(b1*b1) + 4*(db*db)*f2Awxi*f2Bwxi*g2Aw*g2Bw/(a1*a1)))

    kCSB22 = coo_matrix((kCSB22v, (kCSB22r, kCSB22c)), shape=(size, size))

    return kCSB22

