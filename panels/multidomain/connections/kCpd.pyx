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


def fkCpd(double ku, double kv, double kw, object p, double xp, double yp, 
          int size, int row0, int col0):
    r"""
    Penalty for prescribed displacement at a point.

    Parameters
    ----------
    ku, kv, kw : float
        Translation penalty stiffnesses.
        Use this to specify which disp is being provided. The corresponding stiffness is non zero - rest are zero
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
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCpd : scipy.sparse.coo_matrix
        A sparse matrix with the penalty stiffness for prescribed point displacement.
        f here = function i.e. ftn for kCpd

    """
    cdef int i, k, j, l, c, row, col
    cdef int m, n # no of terms in x,y
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

    cdef double fAu, fBu, gAu, gBu
    cdef double fAv, fBv, gAv, gBv
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

    fdim = 3*m*n*m*n # 3 bec the matrix (from the sym expressions) has only 1 sparse term. Might be better to make it a var and then change it

    kCpdr = np.zeros((fdim,), dtype=INT)
    kCpdc = np.zeros((fdim,), dtype=INT)
    kCpdv = np.zeros((fdim,), dtype=DOUBLE)

    # A and B are the left and right terms when you have multiple terms
    # For eg: S1p*S2p, then the first term is A and 2nd is B so that shape functions are now independent


    # i,j controls f and g for A (so starting from the left term)
    # k,l ...  for B (... right)
    
    # i, j are for the rows (see the line where the row is being updated) - check var_descr.py CHECK 
    # k, l are for the cols
    
    with nogil:
        xip = 2*xp/a - 1.
        etap = 2*yp/b - 1.
        
        c = -1
        for j in range(n):
            gAu = f(j, etap, y1u, y1ur, y2u, y2ur) # the point location should be in natural coords
            gAv = f(j, etap, y1v, y1vr, y2v, y2vr)
            gAw = f(j, etap, y1w, y1wr, y2w, y2wr)
            # Generates a single shape function, based on the bardell function number j 
                # So its actually gAw_j
    
            for l in range(n):
                gBu = f(l, etap, y1u, y1ur, y2u, y2ur)
                gBv = f(l, etap, y1v, y1vr, y2v, y2vr)
                gBw = f(l, etap, y1w, y1wr, y2w, y2wr)
        
                for i in range(m):
                    fAu = f(i, xip, x1u, x1ur, x2u, x2ur)
                    fAv = f(i, xip, x1v, x1vr, x2v, x2vr)
                    fAw = f(i, xip, x1w, x1wr, x2w, x2wr)
            
                    for k in range(m):
                        row = row0 + DOF*(j*m + i) # row0 = starting row of this submatrix in the global matrix
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        fBu = f(k, xip, x1u, x1ur, x2u, x2ur)
                        fBv = f(k, xip, x1v, x1vr, x2v, x2vr)
                        fBw = f(k, xip, x1w, x1wr, x2w, x2wr)
            
                        c += 1
                        kCpdr[c] = row+0
                        kCpdc[c] = col+0
                        kCpdv[c] += ku*fAu*fBu*gAu*gBu
                        c += 1
                        kCpdr[c] = row+1
                        kCpdc[c] = col+1
                        kCpdv[c] += kv*fAv*fBv*gAv*gBv
                        c += 1
                        kCpdr[c] = row+2
                        kCpdc[c] = col+2
                        kCpdv[c] += kw*fAw*fBw*gAw*gBw


    kCpd = coo_matrix((kCpdv, (kCpdr, kCpdc)), shape=(size, size))

    return kCpd


def fkCld_xcte(double ku, double kv, double kw, object p, double xp,
               int size, int row0, int col0):
    r"""
    Penalty for prescribed line displacement with a fixed x coordinate.

    Parameters
    ----------
    ku, kv, kw : float
        Translation penalty stiffnesses.
    p : Panel
        Panel() object
    xp : float
        Position x of the prescribed line displacement. This is in physical (x,y) coordinates
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCld_xcte : scipy.sparse.coo_matrix
        A sparse matrix with the penalty stiffness for prescribed line
        displacement with constant x (so integral over y).

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
    
    cdef double fAu, fBu, gAugBu
    cdef double fAv, fBv, gAvgBv
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

    fdim = 3*m*n*m*n # 1 bec the matrix has only 1 sparse term. Might be better to make it a var and then change it

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
                
                # this already integrates it so extra integration needed
                gAugBu = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                gAvgBv = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                for i in range(m):
                    fAu = f(i, xip, x1u, x1ur, x2u, x2ur)
                    fAv = f(i, xip, x1v, x1vr, x2v, x2vr)
                    fAw = f(i, xip, x1w, x1wr, x2w, x2wr)
            
                    for k in range(m):
                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        fBu = f(k, xip, x1u, x1ur, x2u, x2ur)
                        fBv = f(k, xip, x1v, x1vr, x2v, x2vr)
                        fBw = f(k, xip, x1w, x1wr, x2w, x2wr)
            
                        c += 1
                        kCld_xcter[c] = row+2
                        kCld_xctec[c] = col+2
                        kCld_xctev[c] += ku*b*fAu*fBu*gAugBu/2.
                        c += 1
                        kCld_xcter[c] = row+2
                        kCld_xctec[c] = col+2
                        kCld_xctev[c] += kv*b*fAv*fBv*gAvgBv/2.
                        c += 1
                        kCld_xcter[c] = row+2
                        kCld_xctec[c] = col+2
                        kCld_xctev[c] += kw*b*fAw*fBw*gAwgBw/2.

    kCld_xcte = coo_matrix((kCld_xctev, (kCld_xcter, kCld_xctec)), shape=(size, size))
    # Creates a sparse matrix by putting ..v and the corresponding row and col given by ..r ..c

    return kCld_xcte


def fkCld_ycte(double ku, double kv, double kw, object p, double yp,
               int size, int row0, int col0):
    r"""
    Penalty for prescribed displacement at a point.

    Parameters
    ----------
    ku, kv, kw : float
        Translation penalty stiffnesses.
    p : Panel
        Panel() object
    yp : float
        Position y of the prescribed line displacement. This is in physical (x,y) coordinates
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCld_ycte : scipy.sparse.coo_matrix
        A sparse matrix with the penalty stiffness for prescribed line
        displacement with constant y (so integral over x).

    """
    cdef int i, k, j, l, c, row, col
    cdef int m, n
    cdef double a, b, etap
    
    # BCs
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    # x1, x2 are the limits of x

    cdef long [:] kCld_ycter, kCld_yctec # Contains the positions (rows and cols) of the sparse matrix whose values are in kCpdv
    cdef double [:] kCld_yctev
    
    cdef double fAufBu, gAu, gBu
    cdef double fAvfBv, gAv, gBv
    cdef double fAwfBw, gAw, gBw

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

    fdim = 3*m*n*m*n # 1 bec the matrix has only 1 sparse term. Might be better to make it a var and then change it

    kCld_ycter = np.zeros((fdim,), dtype=INT)
    kCld_yctec = np.zeros((fdim,), dtype=INT)
    kCld_yctev = np.zeros((fdim,), dtype=DOUBLE)

    # A and B are the left and right terms when you have multiple terms
    # For eg: S1p*S2p, then the first term is A and 2nd is B so that shape functions are now independent


    # i,j controls f and g for A (so starting from the left term)
    # l,m ...  for B (... right)
    # i, j are for the rows (see the line where the row is being updated) - WHY THOUHG?? CHECK 
    # k, l are for the cols
    
    with nogil:
        etap = 2*yp/b - 1.
        
        c = -1


        for i in range(m):
            for k in range(m):
                # i and k are for x coord i.e. f - since thats being integrated, loop it outside
                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):    
                    gAu = f(j, etap, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, etap, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, etap, y1w, y1wr, y2w, y2wr)

                    for l in range(n):
            
                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = f(l, etap, y1u, y1ur, y2u, y2ur)
                        gBv = f(l, etap, y1v, y1vr, y2v, y2vr)
                        gBw = f(l, etap, y1w, y1wr, y2w, y2wr)
            
                        c += 1
                        kCld_ycter[c] = row+2
                        kCld_yctec[c] = col+2
                        kCld_yctev[c] += ku*a*fAufBu*gAu*gBu/2.
                        c += 1
                        kCld_ycter[c] = row+2
                        kCld_yctec[c] = col+2
                        kCld_yctev[c] += kv*a*fAvfBv*gAv*gBv/2.
                        c += 1
                        kCld_ycter[c] = row+2
                        kCld_yctec[c] = col+2
                        kCld_yctev[c] += kw*a*fAwfBw*gAw*gBw/2.

    kCld_ycte = coo_matrix((kCld_yctev, (kCld_ycter, kCld_yctec)), shape=(size, size))

    return kCld_ycte
