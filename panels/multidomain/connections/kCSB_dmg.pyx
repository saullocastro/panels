#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np


INT = long
DOUBLE = np.float64
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

cdef extern from 'legendre_gauss_quadrature.hpp':
    void leggauss_quad(int n, double *points, double* weights) nogil


# TODO: explain dsb parameter
def fkCSB11_dmg(double dsb, object p1, int size, int row0, int col0, 
            int no_x_gauss, int no_y_gauss, double [:,::1] kw_tsl):
    r"""
    Penalty approach calculation to skin-base ycte panel 1 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    dsb : float
        dsb = sum(pA.plyts)/2. + sum(pB.plyts)/2.
    p1 : Panel
        Panel() object
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.
    no_x_gauss, no_y_gauss : int
        Number of integration points in x and y
    kw_tsl : numpy array
        Out of plane stiffness due to the TSL for each damage instance. This is a grid that is mapped to the
        integration points provided by no_x_gauss and no_y_gauss, with values for each of those points.
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

    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    kCSB11r = np.zeros((fdim,), dtype=INT)
    kCSB11c = np.zeros((fdim,), dtype=INT)
    kCSB11v = np.zeros((fdim,), dtype=DOUBLE)

    # print(f'        KCSB_11 -- kw_tsl {np.min(kw_tsl):.2e} {np.max(kw_tsl):.2e}')
    
    with nogil:
        # kCSB11
        
        for pty in range(no_y_gauss):
            for ptx in range(no_x_gauss):
                # Makes it more efficient when reading data from memory as memory is read along a row
                # So also accessing memory in the same way helps it out so its not deleting and reaccessing the
                # same memory everytime. 
                
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                # Extracting the correct kt
                    # Currently, the outer loop of x and inner of y, causes it to go through all y for a single x
                    # That is going through all rows for a single col then onto the next col
                    # (as per x, y and results by calc_results)
                kt = kw_tsl[pty, ptx]
                # kt = kw_tsl[ptx, pty] # Wrong
                
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
                                kCSB11v[c] += weight*0.5*b1*dsb*f1Au*f1Bwxi*g1Au*g1Bw*kt
                                c += 1
                                kCSB11r[c] = row+1
                                kCSB11c[c] = col+1
                                kCSB11v[c] += weight*0.25*a1*b1*f1Av*f1Bv*g1Av*g1Bv*kt
                                c += 1
                                kCSB11r[c] = row+1
                                kCSB11c[c] = col+2
                                kCSB11v[c] += weight*0.5*a1*dsb*f1Av*f1Bw*g1Av*g1Bweta*kt
                                c += 1
                                kCSB11r[c] = row+2
                                kCSB11c[c] = col+0
                                kCSB11v[c] += weight*0.5*b1*dsb*f1Awxi*f1Bu*g1Aw*g1Bu*kt
                                c += 1
                                kCSB11r[c] = row+2
                                kCSB11c[c] = col+1
                                kCSB11v[c] += weight*0.5*a1*dsb*f1Aw*f1Bv*g1Aweta*g1Bv*kt
                                c += 1
                                kCSB11r[c] = row+2
                                kCSB11c[c] = col+2
                                kCSB11v[c] += weight*(0.25*a1*b1*kt*(f1Aw*f1Bw*g1Aw*g1Bw + 4*(dsb*dsb)*f1Aw*f1Bw*g1Aweta*g1Bweta/(b1*b1) + 4*(dsb*dsb)*f1Awxi*f1Bwxi*g1Aw*g1Bw/(a1*a1)))

    kCSB11 = coo_matrix((kCSB11v, (kCSB11r, kCSB11c)), shape=(size, size))

    return kCSB11


# TODO: explain dsb parameter
def fkCSB12_dmg(double dsb, object p1, object p2, int size, int row0, int col0,
                int no_x_gauss, int no_y_gauss, double [:,::1] kw_tsl):
    r"""
    Penalty approach calculation to skin-base ycte panel 1 and panel 2 coupling position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    dsb : float
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

    cdef double f1Au, f2Bu, f1Av, f2Bv, f1Aw, f2Bw, f1Awxi
    cdef double g1Au, g2Bu, g1Av, g2Bv, g1Aw, g2Bw, g1Aweta
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

    fdim = 5*m1*n1*m2*n2

    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])

    kCSB12r = np.zeros((fdim,), dtype=INT)
    kCSB12c = np.zeros((fdim,), dtype=INT)
    kCSB12v = np.zeros((fdim,), dtype=DOUBLE)
    
    # print(f'        KCSB_12 -- kw_tsl {np.min(kw_tsl):.2e} {np.max(kw_tsl):.2e}')

    with nogil:
        # kCSB12
        
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
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
                        # f2Bwxi = fp(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        # f2Bwxixi = fpp(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        
                        for j1 in range(n1):
                            g1Au = f(j1, eta, y1u1, y1ur1, y1u1, y1ur1)
                            # g1Aueta = fp(j1, eta, y1u1, y1ur1, y1u1, y1ur1)
                            g1Av = f(j1, eta, y1v1, y1vr1, y1v1, y1vr1)
                            # g1Aveta = fp(j1, eta, y1v1, y1vr1, y1v1, y1vr1)
                            g1Aw = f(j1, eta, y1w1, y1wr1, y1w1, y1wr1)
                            g1Aweta = fp(j1, eta, y1w1, y1wr1, y1w1, y1wr1)
                            # g1Awetaeta = fpp(j1, eta, y1w1, y1wr1, y1w1, y1wr1)
                                    
                            for l2 in range(n2):
                                g2Bu = f(l2, eta, y1u2, y1ur2, y2u2, y2ur2)
                                # g2Bueta = fp(l2, eta, y1u2, y1ur2, y2u2, y2ur2)
                                g2Bv = f(l2, eta, y1v2, y1vr2, y2v2, y2vr2)
                                # g2Bveta = fp(l2, eta, y1v2, y1vr2, y2v2, y2vr2)
                                g2Bw = f(l2, eta, y1w2, y1wr2, y2w2, y2wr2)
                                # g2Bweta = fp(l2, eta, y1w2, y1wr2, y2w2, y2wr2)
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
                                kCSB12r[c] = row+1
                                kCSB12c[c] = col+1
                                kCSB12v[c] += -weight*0.25*a1*b1*f1Av*f2Bv*g1Av*g2Bv*kt
                                c += 1
                                kCSB12r[c] = row+2
                                kCSB12c[c] = col+0
                                kCSB12v[c] += -weight*0.5*b1*dsb*f1Awxi*f2Bu*g1Aw*g2Bu*kt
                                c += 1
                                kCSB12r[c] = row+2
                                kCSB12c[c] = col+1
                                kCSB12v[c] += -weight*0.5*a1*dsb*f1Aw*f2Bv*g1Aweta*g2Bv*kt
                                c += 1
                                kCSB12r[c] = row+2
                                kCSB12c[c] = col+2
                                kCSB12v[c] += -weight*0.25*a1*b1*f1Aw*f2Bw*g1Aw*g2Bw*kt

    kCSB12 = coo_matrix((kCSB12v, (kCSB12r, kCSB12c)), shape=(size, size))

    return kCSB12


def fkCSB22_dmg(object p1, object p2, int size, int row0, int col0,
                int no_x_gauss, int no_y_gauss, double [:,::1] kw_tsl):
    r"""
    Penalty approach calculation to skin-base ycte panel 2 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
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

    cdef double f2Au, f2Bu, f2Av, f2Bv, f2Aw, f2Bw
    cdef double g2Au, g2Bu, g2Av, g2Bv, g2Aw, g2Bw
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

    fdim = 3*m2*n2*m2*n2
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])

    kCSB22r = np.zeros((fdim,), dtype=INT)
    kCSB22c = np.zeros((fdim,), dtype=INT)
    kCSB22v = np.zeros((fdim,), dtype=DOUBLE)
    
    # print(f'        KCSB_22 -- kw_tsl {np.min(kw_tsl):.2e} {np.max(kw_tsl):.2e}')

    with nogil:
        
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
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
                    # f2Awxi = fp(i2, xi, x1w2, x1wr2, x2w2, x2wr2)
                    # f2Awxixi = fpp(i2, xi, x1w2, x1wr2, x2w2, x2wr2)
                    
                    for k2 in range(m2):
                        f2Bu = f(k2, xi, x1u2, x1ur2, x2u2, x2ur2)
                        # f2Buxi = fp(k2, xi, x1u2, x1ur2, x2u2, x2ur2)
                        f2Bv = f(k2, xi, x1v2, x1vr2, x2v2, x2vr2)
                        # f2Bvxi = fp(k2, xi, x1v2, x1vr2, x2v2, x2vr2)
                        f2Bw = f(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        # f2Bwxi = fp(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        # f2Bwxixi = fpp(k2, xi, x1w2, x1wr2, x2w2, x2wr2)
                        
                        for j2 in range(n2):
                            g2Au = f(j2, eta, y1u2, y1ur2, y2u2, y2ur2)
                            # g2Aueta = fp(j2, eta, y1u2, y1ur2, y2u2, y2ur2)
                            g2Av = f(j2, eta, y1v2, y1vr2, y2v2, y2vr2)
                            # g2Aveta = fp(j2, eta, y1v2, y1vr2, y2v2, y2vr2)
                            g2Aw = f(j2, eta, y1w2, y1wr2, y2w2, y2wr2)
                            # g2Aweta = fp(j2, eta, y1w2, y1wr2, y2w2, y2wr2)
                            # g2Awetaeta = fpp(j2, eta, y1w2, y1wr2, y2w2, y2wr2)
                                    
                            for l2 in range(n2):
                                g2Bu = f(l2, eta, y1u2, y1ur2, y2u2, y2ur2)
                                # g2Bueta = fp(l2, eta, y1u2, y1ur2, y2u2, y2ur2)
                                g2Bv = f(l2, eta, y1v2, y1vr2, y2v2, y2vr2)
                                # g2Bveta = fp(l2, eta, y1v2, y1vr2, y2v2, y2vr2)
                                g2Bw = f(l2, eta, y1w2, y1wr2, y2w2, y2wr2)
                                # g2Bweta = fp(l2, eta, y1w2, y1wr2, y2w2, y2wr2)
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
                                kCSB22r[c] = row+1
                                kCSB22c[c] = col+1
                                kCSB22v[c] += weight*0.25*a1*b1*f2Av*f2Bv*g2Av*g2Bv*kt
                                c += 1
                                kCSB22r[c] = row+2
                                kCSB22c[c] = col+2
                                kCSB22v[c] += weight*0.25*a1*b1*f2Aw*f2Bw*g2Aw*g2Bw*kt

    kCSB22 = coo_matrix((kCSB22v, (kCSB22r, kCSB22c)), shape=(size, size))

    return kCSB22
