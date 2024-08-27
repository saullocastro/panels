#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix, csr_matrix
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
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

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



def fcrack(object p_top, object p_bot, int size, double [:,::1] kw_tsl_i_1, double [:,::1] del_d_i_1, 
           double [:,::1] kw_tsl_i, int no_x_gauss, int no_y_gauss, double [:,::1] dmg_index):
    # double[:,::1] defines the variable as a 2D, C contiguous memoryview of doubles
    
    cdef int m_top, n_top, m_bot, n_bot
    cdef int i_top, j_top, i_bot, j_bot
    cdef int row_start_top, row_start_bot
    cdef double xi, eta, weight
    cdef int ptx, pty, row
    cdef double a_top, b_top, a_bot, b_bot
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top, x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top, x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top, y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top, y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    # Only single power of SF so only A. No B needed like for u*u
    cdef double fAw_top, fAw_bot 
    cdef double gAw_top, gAw_bot

    cdef double [::1] fcrack
    
    cdef double del_d_i_1_iter, kw_tsl_i_1_iter, kw_tsl_i_iter
    
    cdef double [:] weights_xi, weights_eta, xis, etas
    
    m_top = p_top.m
    n_top = p_top.n
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_top = p_top.a
    b_top = p_top.b
    a_bot = p_bot.a
    b_bot = p_bot.b
    
    row_start_top = p_top.row_start
    row_start_bot = p_bot.row_start
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr

    # Panel bot (ends in bot)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    fcrack = np.zeros(size, dtype=DOUBLE)
    
    with nogil:
        
        # TOP Panel
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]
    
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
    
                weight = weights_xi[ptx] * weights_eta[pty]
                
                # Extracting the correct values (later in fcrack's eqn) [pty, ptx]
                    # Currently, the outer loop of x and inner of y, causes it to go through all y for a single x
                    # That is going through all rows for a single col then onto the next col
                    # (as per x, y and results by calc_results)
                del_d_i_1_iter = del_d_i_1[pty, ptx]
                kw_tsl_i_1_iter = kw_tsl_i_1[pty, ptx]
                kw_tsl_i_iter = kw_tsl_i[pty, ptx]
                
                for j_top in range(n_top):
                    gAw_top = f(j_top, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
                    
                    for i_top in range(m_top):
                        fAw_top = f(i_top, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                        
                        row = row_start_top + DOF*(j_top*m_top + i_top)
                        
                        fcrack[row+2] += (weight*a_top*b_top/4) * 0.5 * (del_d_i_1_iter*(kw_tsl_i_1_iter - kw_tsl_i_iter)) * (fAw_top * gAw_top)
        
        # BOT Panel
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]
    
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
    
                weight = weights_xi[ptx] * weights_eta[pty]
                
                # Extracting the correct values (later in fcrack's eqn) [pty, ptx]
                    # Currently, the outer loop of x and inner of y, causes it to go through all y for a single x
                    # That is going through all rows for a single col then onto the next col
                    # (as per x, y and results by calc_results)
                del_d_i_1_iter = del_d_i_1[pty, ptx]
                kw_tsl_i_1_iter = kw_tsl_i_1[pty, ptx]
                kw_tsl_i_iter = kw_tsl_i[pty, ptx]
                
                
                for j_bot in range(n_bot):
                    gAw_bot = f(j_bot, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)
                    
                    for i_bot in range(m_bot):
                        fAw_bot = f(i_bot, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                        
                        row = row_start_bot + DOF*(j_bot*m_bot + i_bot)
                        
                        fcrack[row+2] += -(weight*a_bot*b_bot/4) * 0.5 * (del_d_i_1_iter*(kw_tsl_i_1_iter - kw_tsl_i_iter)) * (fAw_bot * gAw_bot)
    
    
    return fcrack
    
    
    
def k_crack11(object p_top, int size, int row0, int col0, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f, 
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index):

    cdef int m_top, n_top
    cdef int i1, k1, j1, l1
    cdef double xi, eta, weight
    cdef int ptx, pty, c, row, col
    cdef double a_top, b_top
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top
    
    cdef double fAw_top, fBw_top 
    cdef double gAw_top, gBw_top
    
    cdef long [:] k_crack11r, k_crack11c
    cdef double [:] k_crack11v
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    cdef double [:] weights_xi, weights_eta, xis, etas
    
    m_top = p_top.m
    n_top = p_top.n
    
    a_top = p_top.a
    b_top = p_top.b
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    # Dimension of the number of values that need to be stored
    fdim = 1*m_top*n_top*m_top*n_top    # 1 bec only 1 term is being added in the for loops
    
    k_crack11r = np.zeros((fdim,), dtype=INT)
    k_crack11c = np.zeros((fdim,), dtype=INT)
    k_crack11v = np.zeros((fdim,), dtype=DOUBLE)
    
    with nogil:
        
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                del_d_i_1_iter = del_d_i_1[pty, ptx]
                del_d_i_iter = del_d_i[pty, ptx]
                
                # CHANGE TO ABS < CRITERIA - For now its fine bec its being manually set to 0
                if del_d_i_iter == 0:
                    continue
                
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
                
            
                
                c = -1
                for i1 in range(m_top):
                    fAw_top = f(i1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                    
                    for k1 in range(m_top):
                        fBw_top = f(k1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                        
                        for j1 in range(n_top): 
                            gAw_top = f(j1, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
                                    
                            for l1 in range(n_top):
                                gBw_top = f(l1, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
        
                                row = row0 + DOF*(j1*m_top + i1)
                                col = col0 + DOF*(l1*m_top + k1)
        
                                #NOTE symmetry - 11
                                if row > col:
                                    continue
        
                                c += 1
                                k_crack11r[c] = row+2
                                k_crack11c[c] = col+2
                                # k_crack11v[c] += -a_top*b_top*del_o*fAw_top*fBw_top*gAw_top*gBw_top*k_o*weight/(4*(del_f - del_o)) - # OLD WRONG
                                k_crack11v[c] += a_top*b_top*del_f*del_d_i_1_iter*del_o*fAw_top*fBw_top*gAw_top*gBw_top*k_o*weight/(4*(del_d_i_iter*del_d_i_iter)*(del_f - del_o))

                                # with gil:
                                #     print(k_crack11v[c])


    k_crack11 = coo_matrix((k_crack11v, (k_crack11r, k_crack11c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return k_crack11


def k_crack12(object p_top, object p_bot, int size, int row0, int col0, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f, 
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index):

    cdef int m_top, n_top, m_bot, n_bot
    cdef int i1, k2, j1, l2
    cdef double xi, eta, weight
    cdef int ptx, pty, c, row, col
    cdef double a_top, b_top
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top, x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top, x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top, y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top, y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    cdef double fAw_top, gAw_top
    cdef double fBw_bot, gBw_bot
    
    cdef long [:] k_crack12r, k_crack12c
    cdef double [:] k_crack12v
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    cdef double [:] weights_xi, weights_eta, xis, etas
    
    m_top = p_top.m
    n_top = p_top.n
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_top = p_top.a
    b_top = p_top.b
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr

    # Panel bot (ends in bot)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    # Dimension of the number of values that need to be stored
    fdim = 1*m_bot*n_bot*m_top*n_top    # 1 bec only 1 term is being added in the for loops
    
    k_crack12r = np.zeros((fdim,), dtype=INT)
    k_crack12c = np.zeros((fdim,), dtype=INT)
    k_crack12v = np.zeros((fdim,), dtype=DOUBLE)
    
    with nogil:
        
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                del_d_i_1_iter = del_d_i_1[pty, ptx]
                del_d_i_iter = del_d_i[pty, ptx]
                
                if del_d_i_iter == 0:
                    continue
                
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
                
                c = -1
                for i1 in range(m_top):
                    fAw_top = f(i1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                    
                    for k2 in range(m_bot):
                        fBw_bot = f(k2, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                        
                        for j1 in range(n_top):
                            gAw_top = f(j1, eta, y1w_top, y1wr_top, y1w_top, y1wr_top)
                                    
                            for l2 in range(n_bot):
                                gBw_bot = f(l2, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)
        
                                row = row0 + DOF*(j1*m_top + i1)
                                col = col0 + DOF*(l2*m_bot + k2)
        
                                #NOTE No symmetry - 12
                                # if row > col:
                                #     continue
        
                                c += 1
                                k_crack12r[c] = row+2
                                k_crack12c[c] = col+2
                                # k_crack12v[c] += a_top*b_top*del_o*fAw_top*fBw_bot*gAw_top*gBw_bot*k_o*weight/(4*(del_f - del_o)) # OLD WRONG
                                k_crack12v[c] += -a_top*b_top*del_f*del_d_i_1_iter*del_o*fAw_top*fBw_bot*gAw_top*gBw_bot*k_o*weight/(4*(del_d_i_iter*del_d_i_iter)*(del_f - del_o))

    k_crack12 = coo_matrix((k_crack12v, (k_crack12r, k_crack12c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return k_crack12


def k_crack22(object p_bot, int size, int row0, int col0, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f,
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index):

    cdef int m_bot, n_bot
    cdef int i1, k1, j1, l1
    cdef double xi, eta, weight
    cdef int ptx, pty, c, row, col
    cdef double a_bot, b_bot
    
    cdef double x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    cdef double fAw_bot, fBw_bot 
    cdef double gAw_bot, gBw_bot
    
    cdef long [:] k_crack22r, k_crack22c
    cdef double [:] k_crack22v
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    cdef double [:] weights_xi, weights_eta, xis, etas
    
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_bot = p_bot.a
    b_bot = p_bot.b
    
    # Panel top (ends in top)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)

    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    # Dimension of the number of values that need to be stored
    fdim = 1*m_bot*n_bot*m_bot*n_bot    # 1 bec only 1 term is being added in the for loops
    
    k_crack22r = np.zeros((fdim,), dtype=INT)
    k_crack22c = np.zeros((fdim,), dtype=INT)
    k_crack22v = np.zeros((fdim,), dtype=DOUBLE)
    
    with nogil:
        
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]

                weight = weights_xi[ptx] * weights_eta[pty]
                
                del_d_i_1_iter = del_d_i_1[pty, ptx]
                del_d_i_iter = del_d_i[pty, ptx]
                
                if del_d_i_iter == 0:
                    continue
                
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
                
                c = -1
                for i1 in range(m_bot):
                    fAw_bot = f(i1, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                    
                    for k1 in range(m_bot):
                        fBw_bot = f(k1, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                        
                        for j1 in range(n_bot): 
                            gAw_bot = f(j1, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)
                                    
                            for l1 in range(n_bot):
                                gBw_bot = f(l1, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)
        
                                row = row0 + DOF*(j1*m_bot + i1)
                                col = col0 + DOF*(l1*m_bot + k1)
        
                                #NOTE symmetry - 22
                                if row > col:
                                    continue
        
                                c += 1
                                k_crack22r[c] = row+2
                                k_crack22c[c] = col+2
                                # k_crack22v[c] += -a_bot*b_bot*del_o*fAw_bot*fBw_bot*gAw_bot*gBw_bot*k_o*weight/(4*(del_f - del_o)) # OLD REMOVE
                                k_crack22v[c] += a_bot*b_bot*del_f*del_d_i_1_iter*del_o*fAw_bot*fBw_bot*gAw_bot*gBw_bot*k_o*weight/(4*(del_d_i_iter*del_d_i_iter)*(del_f - del_o))

    k_crack22 = coo_matrix((k_crack22v, (k_crack22r, k_crack22c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return k_crack22


# Evaluated at a specific point (xi, eta)
def k_crack_term2_partA_11(object p_top, int size, int row0, int col0, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f, 
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index, 
              double xi, double eta, int ptx, int pty):

    cdef int m_top, n_top
    cdef int i1, k1, j1, l1
    cdef int c, row, col
    cdef double a_top, b_top
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top
    
    cdef double fAw_top, fBw_top 
    cdef double gAw_top, gBw_top
    
    cdef long [:] k_crack_term2_partA_11r, k_crack_term2_partA_11c
    cdef double [:] k_crack_term2_partA_11v
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    m_top = p_top.m
    n_top = p_top.n
    
    a_top = p_top.a
    b_top = p_top.b
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr
    
    # Dimension of the number of values that need to be stored
    fdim = 1*m_top*n_top*m_top*n_top    # 1 bec only 1 term is being added in the for loops
    
    k_crack_term2_partA_11r = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_11c = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_11v = np.zeros((fdim,), dtype=DOUBLE)
    
    with nogil:
        
        del_d_i_1_iter = del_d_i_1[pty, ptx]
        del_d_i_iter = del_d_i[pty, ptx]
        
        # CHANGE TO ABS < CRITERIA - For now its fine bec its being manually set to 0
        # if del_d_i_iter == 0:
        #     continue
        
        # if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
        #     continue
        
        c = -1
        for i1 in range(m_top):
            fAw_top = f(i1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
            
            for k1 in range(m_top):
                fBw_top = f(k1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                
                for j1 in range(n_top): 
                    gAw_top = f(j1, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
                            
                    for l1 in range(n_top):
                        gBw_top = f(l1, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)

                        row = row0 + DOF*(j1*m_top + i1)
                        col = col0 + DOF*(l1*m_top + k1)

                        #NOTE symmetry - 11
                        if row > col:
                            continue

                        c += 1
                        k_crack_term2_partA_11r[c] = row+2
                        k_crack_term2_partA_11c[c] = col+2
                        # k_crack_term2_partA_11v[c] += -a_top*b_top*del_o*fAw_top*fBw_top*gAw_top*gBw_top*k_o*weight/(4*(del_f - del_o)) - # OLD WRONG
                        k_crack_term2_partA_11v[c] += del_f*del_d_i_1_iter*del_o*fAw_top*fBw_top*gAw_top*gBw_top*k_o/((del_d_i_iter*del_d_i_iter*del_d_i_iter)*(del_f - del_o))

                        # with gil:
                        #     print(k_crack_term2_partA_11v[c])


    k_crack_term2_partA_11 = coo_matrix((k_crack_term2_partA_11v, (k_crack_term2_partA_11r, k_crack_term2_partA_11c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return k_crack_term2_partA_11


def k_crack_term2_partA_12(object p_top, object p_bot, int size, int row0, int col0, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f, 
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index, 
              double xi, double eta, int ptx, int pty):

    cdef int m_top, n_top, m_bot, n_bot
    cdef int i1, k2, j1, l2
    cdef int c, row, col
    cdef double a_top, b_top
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top, x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top, x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top, y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top, y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    cdef double fAw_top, gAw_top
    cdef double fBw_bot, gBw_bot
    
    cdef long [:] k_crack_term2_partA_12r, k_crack_term2_partA_12c
    cdef double [:] k_crack_term2_partA_12v
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    m_top = p_top.m
    n_top = p_top.n
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_top = p_top.a
    b_top = p_top.b
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr

    # Panel bot (ends in bot)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    # Dimension of the number of values that need to be stored
    fdim = 1*m_bot*n_bot*m_top*n_top    # 1 bec only 1 term is being added in the for loops
    
    k_crack_term2_partA_12r = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_12c = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_12v = np.zeros((fdim,), dtype=DOUBLE)
    
    with nogil:
        
        del_d_i_1_iter = del_d_i_1[pty, ptx]
        del_d_i_iter = del_d_i[pty, ptx]
        
        # if del_d_i_iter == 0:
        #     continue
        
        # if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
        #     continue
        
        c = -1
        for i1 in range(m_top):
            fAw_top = f(i1, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
            
            for k2 in range(m_bot):
                fBw_bot = f(k2, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                
                for j1 in range(n_top):
                    gAw_top = f(j1, eta, y1w_top, y1wr_top, y1w_top, y1wr_top)
                            
                    for l2 in range(n_bot):
                        gBw_bot = f(l2, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)

                        row = row0 + DOF*(j1*m_top + i1)
                        col = col0 + DOF*(l2*m_bot + k2)

                        #NOTE No symmetry - 12
                        # if row > col:
                        #     continue

                        c += 1
                        k_crack_term2_partA_12r[c] = row+2
                        k_crack_term2_partA_12c[c] = col+2
                        # k_crack_term2_partA_12v[c] += a_top*b_top*del_o*fAw_top*fBw_bot*gAw_top*gBw_bot*k_o*weight/(4*(del_f - del_o)) # OLD WRONG
                        k_crack_term2_partA_12v[c] += -del_f*del_d_i_1_iter*del_o*fAw_top*fBw_bot*gAw_top*gBw_bot*k_o/((del_d_i_iter*del_d_i_iter*del_d_i_iter)*(del_f - del_o))

    k_crack_term2_partA_12 = coo_matrix((k_crack_term2_partA_12v, (k_crack_term2_partA_12r, k_crack_term2_partA_12c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return k_crack_term2_partA_12


def k_crack_term2_partA_22(object p_bot, int size, int row0, int col0, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f,
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index, 
              double xi, double eta, int ptx, int pty):

    cdef int m_bot, n_bot
    cdef int i1, k1, j1, l1
    cdef int c, row, col
    cdef double a_bot, b_bot
    
    cdef double x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    cdef double fAw_bot, fBw_bot 
    cdef double gAw_bot, gBw_bot
    
    cdef long [:] k_crack_term2_partA_22r, k_crack_term2_partA_22c
    cdef double [:] k_crack_term2_partA_22v
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_bot = p_bot.a
    b_bot = p_bot.b
    
    # Panel top (ends in top)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    # Dimension of the number of values that need to be stored
    fdim = 1*m_bot*n_bot*m_bot*n_bot    # 1 bec only 1 term is being added in the for loops
    
    k_crack_term2_partA_22r = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_22c = np.zeros((fdim,), dtype=INT)
    k_crack_term2_partA_22v = np.zeros((fdim,), dtype=DOUBLE)
    
    with nogil:
        
        del_d_i_1_iter = del_d_i_1[pty, ptx]
        del_d_i_iter = del_d_i[pty, ptx]
        
        # if del_d_i_iter == 0:
        #     continue
        
        # if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
        #     continue
        
        c = -1
        for i1 in range(m_bot):
            fAw_bot = f(i1, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
            
            for k1 in range(m_bot):
                fBw_bot = f(k1, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                
                for j1 in range(n_bot): 
                    gAw_bot = f(j1, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)
                            
                    for l1 in range(n_bot):
                        gBw_bot = f(l1, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)

                        row = row0 + DOF*(j1*m_bot + i1)
                        col = col0 + DOF*(l1*m_bot + k1)

                        #NOTE symmetry - 22
                        if row > col:
                            continue

                        c += 1
                        k_crack_term2_partA_22r[c] = row+2
                        k_crack_term2_partA_22c[c] = col+2
                        # k_crack22v[c] += -a_bot*b_bot*del_o*fAw_bot*fBw_bot*gAw_bot*gBw_bot*k_o*weight/(4*(del_f - del_o)) # OLD REMOVE
                        k_crack_term2_partA_22v[c] += del_f*del_d_i_1_iter*del_o*fAw_bot*fBw_bot*gAw_bot*gBw_bot*k_o/((del_d_i_iter*del_d_i_iter*del_d_i_iter)*(del_f - del_o))

    k_crack_term2_partA_22 = coo_matrix((k_crack_term2_partA_22v, (k_crack_term2_partA_22r, k_crack_term2_partA_22c)), shape=(size, size))
    # Builds a matrix of size = size x size (so complete size of global MD) and populates it with the data in ..v 
        # where the rows and cols where that data should go are specified by ..r, ...c
    # This way its at the correct positions in the global MD matrix as row and col are the starting indices of the 
        # submatrices in the global MD matrix

    return k_crack_term2_partA_22


# Evaluated at a specific point (xi, eta)
def calc_k_crack_term2_partB(object p_top, object p_bot, int size, int no_x_gauss, int no_y_gauss, double [:,::1] dmg_index, 
            double [:] c_i, double xi, double eta, int ptx, int pty):
    # double[:,::1] defines the variable as a 2D, C contiguous memoryview of doubles
    
    cdef int m_top, n_top, m_bot, n_bot
    cdef int i_top, j_top, i_bot, j_bot
    cdef int i_outer, j_outer
    cdef int row_start_top, row_start_bot
    cdef int row
    cdef double a_top, b_top, a_bot, b_bot
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top, x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top, x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top, y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top, y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    # Only single power of SF so only A. No B needed like for u*u
    cdef double fAw_top, fAw_bot 
    cdef double gAw_top, gAw_bot

    cdef double [::1] s_delta
    cdef double [:,::1] k_crack_term2_partB
    
    m_top = p_top.m
    n_top = p_top.n
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_top = p_top.a
    b_top = p_top.b
    a_bot = p_bot.a
    b_bot = p_bot.b
    
    row_start_top = p_top.row_start
    row_start_bot = p_bot.row_start
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr

    # Panel bot (ends in bot)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    s_delta = np.zeros(size, dtype=DOUBLE)
    k_crack_term2_partB = np.zeros((size, size), dtype=DOUBLE)
    
    with nogil:
        
        # TOP PANEL
        # if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
        #     continue

        # Extracting the correct values (later in fcrack's eqn) [pty, ptx]
            # Currently, the outer loop of x and inner of y, causes it to go through all y for a single x
            # That is going through all rows for a single col then onto the next col
            # (as per x, y and results by calc_results)

        for j_top in range(n_top):
            gAw_top = f(j_top, eta, y1w_top, y1wr_top, y2w_top, y2wr_top)
            
            for i_top in range(m_top):
                fAw_top = f(i_top, xi, x1w_top, x1wr_top, x2w_top, x2wr_top)
                
                row = row_start_top + DOF*(j_top*m_top + i_top)
                
                s_delta[row+2] += (fAw_top * gAw_top)
        
        # BOT Panel
        # if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
        #     continue

        # Extracting the correct values (later in fcrack's eqn) [pty, ptx]
            # Currently, the outer loop of x and inner of y, causes it to go through all y for a single x
            # That is going through all rows for a single col then onto the next col
            # (as per x, y and results by calc_results)
        
        for j_bot in range(n_bot):
            gAw_bot = f(j_bot, eta, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot)
            
            for i_bot in range(m_bot):
                fAw_bot = f(i_bot, xi, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot)
                
                row = row_start_bot + DOF*(j_bot*m_bot + i_bot)
                
                s_delta[row+2] += - (fAw_bot * gAw_bot)
                        
        for i_outer in range(size):
            for j_outer in range(size):
                k_crack_term2_partB[i_outer, j_outer] = c_i[i_outer]*s_delta[j_outer]
    
    return k_crack_term2_partB


def k_crack_term2(object p_top, object p_bot, int size, int no_x_gauss, int no_y_gauss,
              double k_o, double del_o, double del_f, 
              double [:,::1] del_d_i_1, double [:,::1] del_d_i, double [:,::1] dmg_index, double [:] c_i):

    cdef int m_top, n_top, m_bot, n_bot
    cdef int i1, k2, j1, l2
    cdef double xi, eta, weight
    cdef int ptx, pty, c, row, col
    cdef double a_top, b_top
    
    cdef double x1u_top, x1ur_top, x2u_top, x2ur_top, x1u_bot, x1ur_bot, x2u_bot, x2ur_bot
    cdef double x1v_top, x1vr_top, x2v_top, x2vr_top, x1v_bot, x1vr_bot, x2v_bot, x2vr_bot
    cdef double x1w_top, x1wr_top, x2w_top, x2wr_top, x1w_bot, x1wr_bot, x2w_bot, x2wr_bot
    cdef double y1u_top, y1ur_top, y2u_top, y2ur_top, y1u_bot, y1ur_bot, y2u_bot, y2ur_bot
    cdef double y1v_top, y1vr_top, y2v_top, y2vr_top, y1v_bot, y1vr_bot, y2v_bot, y2vr_bot
    cdef double y1w_top, y1wr_top, y2w_top, y2wr_top, y1w_bot, y1wr_bot, y2w_bot, y2wr_bot
    
    cdef double fAw_top, gAw_top
    cdef double fBw_bot, gBw_bot
    
    cdef double [:,::1] k_crack_term2
    cdef double [:,::1] k_crack_term2_partA, k_crack_term2_partB
    
    cdef double del_d_i_1_iter, del_d_i_iter
    
    cdef double [:] weights_xi, weights_eta, xis, etas
    cdef int iter_1, iter_2, iter_3
    
    m_top = p_top.m
    n_top = p_top.n
    m_bot = p_bot.m
    n_bot = p_bot.n
    
    a_top = p_top.a
    b_top = p_top.b
    
    # Panel top (ends in top)
    x1u_top = p_top.x1u ; x1ur_top = p_top.x1ur ; x2u_top = p_top.x2u ; x2ur_top = p_top.x2ur
    x1v_top = p_top.x1v ; x1vr_top = p_top.x1vr ; x2v_top = p_top.x2v ; x2vr_top = p_top.x2vr
    x1w_top = p_top.x1w ; x1wr_top = p_top.x1wr ; x2w_top = p_top.x2w ; x2wr_top = p_top.x2wr
    y1u_top = p_top.y1u ; y1ur_top = p_top.y1ur ; y2u_top = p_top.y2u ; y2ur_top = p_top.y2ur
    y1v_top = p_top.y1v ; y1vr_top = p_top.y1vr ; y2v_top = p_top.y2v ; y2vr_top = p_top.y2vr
    y1w_top = p_top.y1w ; y1wr_top = p_top.y1wr ; y2w_top = p_top.y2w ; y2wr_top = p_top.y2wr
    
    # Panel bot (ends in bot)
    x1u_bot = p_bot.x1u ; x1ur_bot = p_bot.x1ur ; x2u_bot = p_bot.x2u ; x2ur_bot = p_bot.x2ur
    x1v_bot = p_bot.x1v ; x1vr_bot = p_bot.x1vr ; x2v_bot = p_bot.x2v ; x2vr_bot = p_bot.x2vr
    x1w_bot = p_bot.x1w ; x1wr_bot = p_bot.x1wr ; x2w_bot = p_bot.x2w ; x2wr_bot = p_bot.x2wr
    y1u_bot = p_bot.y1u ; y1ur_bot = p_bot.y1ur ; y2u_bot = p_bot.y2u ; y2ur_bot = p_bot.y2ur
    y1v_bot = p_bot.y1v ; y1vr_bot = p_bot.y1vr ; y2v_bot = p_bot.y2v ; y2vr_bot = p_bot.y2vr
    y1w_bot = p_bot.y1w ; y1wr_bot = p_bot.y1wr ; y2w_bot = p_bot.y2w ; y2wr_bot = p_bot.y2wr
    
    # Initializing gauss points, weights
    xis = np.zeros(no_x_gauss, dtype=DOUBLE)
    weights_xi = np.zeros(no_x_gauss, dtype=DOUBLE)
    etas = np.zeros(no_y_gauss, dtype=DOUBLE)
    weights_eta = np.zeros(no_y_gauss, dtype=DOUBLE)
    
    # Calc gauss points and weights
    leggauss_quad(no_x_gauss, &xis[0], &weights_xi[0])
    leggauss_quad(no_y_gauss, &etas[0], &weights_eta[0])
    
    k_crack_term2 = np.zeros((size, size), dtype=DOUBLE)
    k_crack_term2_partA = np.zeros((size, size), dtype=DOUBLE)
    # k_crack_term2_partA = csr_matrix((size, size), dtype=DOUBLE)

    with nogil:
        
        for ptx in range(no_x_gauss):
            for pty in range(no_y_gauss):
                # Takes the correct index instead of the location
                xi = xis[ptx]
                eta = etas[pty]
                
                if dmg_index[pty, ptx] == 0 or dmg_index[pty, ptx] == 1:
                    continue
                
                del_d_i_1_iter = del_d_i_1[pty, ptx]
                del_d_i_iter = del_d_i[pty, ptx]
                
                if del_d_i_iter == 0:
                    continue
    
                weight = weights_xi[ptx] * weights_eta[pty]
                
                for iter_1 in range(size):
                    for iter_2 in range(size):
                        k_crack_term2_partA[iter_1, iter_2] = 0 # initlizing it to zero for a new point
                
                with gil:
                    # k_crack_term2_partA = csr_matrix(k_crack_term2_partA)
                    k_crack_term2_partA += k_crack_term2_partA_11(p_top=p_top, size=size, row0=p_top.row_start,
                                  col0=p_top.col_start, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss,
                                  k_o=k_o, del_o=del_o, del_f=del_f, del_d_i_1=del_d_i_1, 
                                  del_d_i=del_d_i, dmg_index=dmg_index, xi=xi, eta=eta, ptx=ptx, pty=pty)
                    
                    k_crack_term2_partA += k_crack_term2_partA_12(p_top=p_top, p_bot=p_bot, size=size, 
                                  row0=p_top.row_start, col0=p_bot.col_start, no_x_gauss=no_x_gauss, 
                                  no_y_gauss=no_y_gauss, k_o=k_o, del_o=del_o, del_f=del_f, del_d_i_1=del_d_i_1, 
                                  del_d_i=del_d_i, dmg_index=dmg_index, xi=xi, eta=eta, ptx=ptx, pty=pty)
                    
                    k_crack_term2_partA += k_crack_term2_partA_22(p_bot=p_bot, size=size, 
                                  row0=p_bot.row_start, col0=p_bot.col_start, no_x_gauss=no_x_gauss, 
                                  no_y_gauss=no_y_gauss, k_o=k_o, del_o=del_o, del_f=del_f, del_d_i_1=del_d_i_1, 
                                  del_d_i=del_d_i, dmg_index=dmg_index, xi=xi, eta=eta, ptx=ptx, pty=pty)
                    
                    k_crack_term2_partB = calc_k_crack_term2_partB(p_top=p_top, p_bot=p_bot, size=size, 
                                   no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, dmg_index=dmg_index, 
                                   c_i=c_i, xi=xi, eta=eta, ptx=ptx, pty=pty)
                
                for iter_1 in range(size): # since theyre square matrices
                    for iter_2 in range(size):
                        for iter_3 in range(size):
                            k_crack_term2[iter_1, iter_2] += (a_top*b_top/4)*weight*(k_crack_term2_partA[iter_1, iter_3] * k_crack_term2_partB[iter_3, iter_2])
    
    return np.asarray(k_crack_term2)