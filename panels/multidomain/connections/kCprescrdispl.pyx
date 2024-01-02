# Calculates the matrices (numerical value) based on the expressions of each element generated from the symbolic code

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

cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil


def fkCpd(double kw, object p, double xp, double yp, int size, int row0, int col0):
    r"""
    Penalty for prescribed displacement at a point.

    Parameters
    ----------
    kw : float
        Translation penalty stiffness.
    p : Panel
        Panel() object
    xp, yp : float
        Positions x and y of the prescribed displacement. This is in physical (x,y) coordinates
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Collumn position of constitutive matrix being calculated.

    Returns
    -------
    kCpd : scipy.sparse.coo_matrix
        A sparse matrix with the penalty stiffness for prescribed point displacement.

    """
    cdef int i, k, j, l, c, row, col
    cdef int m, n
    cdef double a, b, xip, etap
    
    # BCs
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    # x1, x2 are the switch cases for the BCs

    cdef long [:] kCpdr, kCpdc # Contains the positions (rows and cols) of the sparse matrix whose values are in kCpdv
    cdef double [:] kCpdv

    cdef double fAw, fBw, gAw, gBw

    a = p.a
    b = p.b
    m = p.m
    n = p.n
    x1u = p.x1u ; x1ur = p.x1ur ; x2u = p.x2u ; x2ur = p.x2ur
    x1v = p.x1v ; x1vr = p.x1vr ; x2v = p.x2v ; x2vr = p.x2vr
    x1w = p.x1w ; x1wr = p.x1wr ; x2w = p.x2w ; x2wr = p.x2wr
    y1u = p.y1u ; y1ur = p.y1ur ; y2u = p.y2u ; y2ur = p.y2ur
    y1v = p.y1v ; y1vr = p.y1vr ; y2v = p.y2v ; y2vr = p.y2vr
    y1w = p.y1w ; y1wr = p.y1wr ; y2w = p.y2w ; y2wr = p.y2wr

    fdim = 1*m*n*m*n # 1 bec the matrix has only 1 sparse term. Might be better to make it a var and then change it

    kCpdr = np.zeros((fdim,), dtype=INT)
    kCpdc = np.zeros((fdim,), dtype=INT)
    kCpdv = np.zeros((fdim,), dtype=DOUBLE)

    # A and B are the left and right terms when you have multiple terms
    # For eg: S1p*S2p, then the first term is A and 2nd is B so that shape functions are now independent


    # i,j controls f and g for A (so starting from the left term)
    # l,m ...  for B (... right)
    # i, j are for the rows (see the line where the row is being updated) - WHY THOUHG?? CHECK 
    # k, l are for the cols
    
    with nogil:
        xip = 2*xp/a - 1.
        etap = 2*yp/b - 1.
        
        c = -1
        for j in range(n):
            gAw = f(j, etap, y1w, y1wr, y2w, y2wr) # the point location should be in natural coords
    
            for l in range(n):
                gBw = f(l, etap, y1w, y1wr, y2w, y2wr)
        
                for i in range(m):
                    fAw = f(i, xip, x1w, x1wr, x2w, x2wr)
            
                    for k in range(m):
                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        fBw = f(k, xip, x1w, x1wr, x2w, x2wr)
            
                        c += 1
                        kCpdr[c] = row+2
                        kCpdc[c] = col+2
                        kCpdv[c] += fAw*fBw*gAw*gBw


    kCpd = coo_matrix((kCpdv, (kCpdr, kCpdc)), shape=(size, size))

    return kCpd


def fkCld_xcte(double kw, object p, double xp, int size, int row0, int col0):
    r"""
    Penalty for prescribed displacement at a point.

    Parameters
    ----------
    kw : float
        Translation penalty stiffness.
    p : Panel
        Panel() object
    xp : float
        Positions x of the prescribed line displacement. This is in physical (x,y) coordinates
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Collumn position of constitutive matrix being calculated.

    Returns
    -------
    kCld_xctepd : scipy.sparse.coo_matrix
        A sparse matrix with the penalty stiffness for prescribed line displacement with constant x (so integral over y).

    """
    cdef int i, k, j, l, c, row, col
    cdef int m, n
    cdef double a, b, xip
    
    # BCs
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    # x1, x2 are the limits of x

    cdef long [:] kCld_xcter, kCld_xctec # Contains the positions (rows and cols) of the sparse matrix whose values are in kCpdv
    cdef double [:] kCld_xctev
    
    cdef double fAw, fBw, gAwgBw

    a = p.a
    b = p.b
    m = p.m
    n = p.n
    x1u = p.x1u ; x1ur = p.x1ur ; x2u = p.x2u ; x2ur = p.x2ur
    x1v = p.x1v ; x1vr = p.x1vr ; x2v = p.x2v ; x2vr = p.x2vr
    x1w = p.x1w ; x1wr = p.x1wr ; x2w = p.x2w ; x2wr = p.x2wr
    y1u = p.y1u ; y1ur = p.y1ur ; y2u = p.y2u ; y2ur = p.y2ur
    y1v = p.y1v ; y1vr = p.y1vr ; y2v = p.y2v ; y2vr = p.y2vr
    y1w = p.y1w ; y1wr = p.y1wr ; y2w = p.y2w ; y2wr = p.y2wr

    fdim = 1*m*n*m*n # 1 bec the matrix has only 1 sparse term. Might be better to make it a var and then change it

    kCld_xcter = np.zeros((fdim,), dtype=INT)
    kCld_xctec = np.zeros((fdim,), dtype=INT)
    kCld_xctev = np.zeros((fdim,), dtype=DOUBLE)

    # A and B are the left and right terms when you have multiple terms
    # For eg: S1p*S2p, then the first term is A and 2nd is B so that shape functions are now independent


    # i,j controls f and g for A (so starting from the left term)
    # l,m ...  for B (... right)
    # i, j are for the rows (see the line where the row is being updated) - WHY THOUHG?? CHECK 
    # k, l are for the cols
    
    with nogil:
        xip = 2*xp/a - 1.
        
        c = -1
        for j in range(n):    
            for l in range(n):
                # j and l are for y coord i.e. g - since thats being integrated, loop it outside
                gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                for i in range(m):
                    fAw = f(i, xip, x1w, x1wr, x2w, x2wr)
            
                    for k in range(m):
                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        fBw = f(k, xip, x1w, x1wr, x2w, x2wr)
            
                        c += 1
                        kCld_xcter[c] = row+2
                        kCld_xctec[c] = col+2
                        kCld_xctev[c] += b*fAw*fBw*gAwgBw/2.

    kCld_xctepd = coo_matrix((kCld_xctev, (kCld_xcter, kCld_xctec)), shape=(size, size))

    return kCld_xctepd

