#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange

# \panels\panels\core\src
cdef extern from 'bardell_functions.hpp':
    double vec_f(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double vec_fp(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double vec_fpp(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil

DOUBLE = np.float64

cdef int NMAX = 30
cdef int DOF = 3


def fuvw(double [::1] c, object s, double [::1] xs, double [::1] ys,
        int num_cores=4):
    '''
        Calculates the displacement field at all points in the provided grid
    '''
    cdef double a, b
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    a = s.a
    b = s.b
    m = s.m
    n = s.n
    x1u = s.x1u; x1ur = s.x1ur; x2u = s.x2u; x2ur = s.x2ur
    x1v = s.x1v; x1vr = s.x1vr; x2v = s.x2v; x2vr = s.x2vr
    x1w = s.x1w; x1wr = s.x1wr; x2w = s.x2w; x2wr = s.x2wr
    y1u = s.y1u; y1ur = s.y1ur; y2u = s.y2u; y2ur = s.y2ur
    y1v = s.y1v; y1vr = s.y1vr; y2v = s.y2v; y2vr = s.y2vr
    y1w = s.y1w; y1wr = s.y1wr; y2w = s.y2w ; y2wr = s.y2wr

    cdef int size_core, pti, i, j
    cdef double [:, ::1] us, vs, ws, phixs, phiys
    cdef double [:, ::1] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size == num_cores:
        add_size = 0
    new_size = size + add_size

    if (size % num_cores) != 0:
        xs_core = np.ascontiguousarray(np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
    else:
        xs_core = np.ascontiguousarray(np.reshape(xs, (num_cores, -1)), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.reshape(ys, (num_cores, -1)), dtype=DOUBLE)

    size_core = xs_core.shape[1]

    us = np.zeros((num_cores, size_core), dtype=DOUBLE)
    vs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    ws = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phixs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    phiys = np.zeros((num_cores, size_core), dtype=DOUBLE)

    for pti in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfuvw(&c[0], m, n, a, b, &xs_core[pti,0],
              &ys_core[pti,0], size_core, &us[pti,0], &vs[pti,0], &ws[pti,0],
              x1u, x1ur, x2u, x2ur,
              x1v, x1vr, x2v, x2vr,
              x1w, x1wr, x2w, x2wr,
              y1u, y1ur, y2u, y2ur,
              y1v, y1vr, y2v, y2vr,
              y1w, y1wr, y2w, y2wr)

        cfwx(&c[0], m, n, a, b, &xs_core[pti,0], &ys_core[pti,0],
             size_core, &phixs[pti,0],
             x1w, x1wr, x2w, x2wr,
             y1w, y1wr, y2w, y2wr)

        cfwy(&c[0], m, n, a, b, &xs_core[pti,0], &ys_core[pti,0],
             size_core, &phiys[pti,0],
             x1w, x1wr, x2w, x2wr,
             y1w, y1wr, y2w, y2wr)

    for i in range(num_cores):
        for j in range(size_core):
            phixs[i, j] *= -1.
            phiys[i, j] *= -1.
    return (np.ravel(us)[:size], np.ravel(vs)[:size], np.ravel(ws)[:size],
            np.ravel(phixs)[:size], np.ravel(phiys)[:size])


def fstrain(double [::1] c, object s, double [::1] xs, double [::1] ys, int
            num_cores=4, int NLgeom=0):
    #  [::1] means a contigious array
    
    '''
    Calculates the strain field at all points in the provided grid 
    
        c : ritz coeffients of the whole system i.e. size = n_rows of K_global
        
        s : Shell obj
        
        xs and ys : contigious arrays of the grid divisions in the x and y direction
    '''
    
    cdef double a, b, r
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    a = s.a
    b = s.b
    r = s.r
    m = s.m
    n = s.n
    x1u = s.x1u; x1ur = s.x1ur; x2u = s.x2u; x2ur = s.x2ur
    x1v = s.x1v; x1vr = s.x1vr; x2v = s.x2v; x2vr = s.x2vr
    x1w = s.x1w; x1wr = s.x1wr; x2w = s.x2w; x2wr = s.x2wr
    y1u = s.y1u; y1ur = s.y1ur; y2u = s.y2u; y2ur = s.y2ur
    y1v = s.y1v; y1vr = s.y1vr; y2v = s.y2v; y2vr = s.y2vr
    y1w = s.y1w; y1wr = s.y1wr; y2w = s.y2w; y2wr = s.y2wr

    cdef int size_core, pti
    cdef double [:, ::1] exxs, eyys, gxys, kxxs, kyys, kxys
    cdef double [:, ::1] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size == num_cores:
        add_size = 0
    new_size = size + add_size

# ???????????????????????????
    if (size % num_cores) != 0:
        xs_core = np.ascontiguousarray(np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
    else:
        xs_core = np.ascontiguousarray(np.reshape(xs, (num_cores, -1)), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.reshape(ys, (num_cores, -1)), dtype=DOUBLE)

    size_core = xs_core.shape[1]

    exxs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    eyys = np.zeros((num_cores, size_core), dtype=DOUBLE)
    gxys = np.zeros((num_cores, size_core), dtype=DOUBLE)
    kxxs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    kyys = np.zeros((num_cores, size_core), dtype=DOUBLE)
    kxys = np.zeros((num_cores, size_core), dtype=DOUBLE)

    for pti in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfstrain(&c[0], m, n, a, b, r,
              &xs_core[pti,0], &ys_core[pti,0], size_core,
              &exxs[pti,0], &eyys[pti,0], &gxys[pti,0],
              &kxxs[pti,0], &kyys[pti,0], &kxys[pti,0],
              x1u, x1ur, x2u, x2ur,
              x1v, x1vr, x2v, x2vr,
              x1w, x1wr, x2w, x2wr,
              y1u, y1ur, y2u, y2ur,
              y1v, y1vr, y2v, y2vr,
              y1w, y1wr, y2w, y2wr, NLgeom)

    return (np.ravel(exxs)[:size], np.ravel(eyys)[:size], np.ravel(gxys)[:size],
            np.ravel(kxxs)[:size], np.ravel(kyys)[:size], np.ravel(kxys)[:size])
# slices the first 'size' elems from the flattened array

cdef void cfuvw(double *c, int m, int n, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws,
        double x1u, double x1ur, double x2u, double x2ur,
        double x1v, double x1vr, double x2v, double x2vr,
        double x1w, double x1wr, double x2w, double x2wr,
        double y1u, double y1ur, double y2u, double y2ur,
        double y1v, double y1vr, double y2v, double y2vr,
        double y1w, double y1wr, double y2w, double y2wr) nogil:
    cdef int i, j, col, pti
    cdef double x, y, u, v, w, xi, eta
    cdef double *fu
    cdef double *fv
    cdef double *fw
    cdef double *gu
    cdef double *gv
    cdef double *gw

    fu = <double *>malloc(NMAX * sizeof(double *))
    gu = <double *>malloc(NMAX * sizeof(double *))
    fv = <double *>malloc(NMAX * sizeof(double *))
    gv = <double *>malloc(NMAX * sizeof(double *))
    fw = <double *>malloc(NMAX * sizeof(double *))
    gw = <double *>malloc(NMAX * sizeof(double *))

    for pti in range(size):
        x = xs[pti]
        y = ys[pti]

        xi = 2*x/a - 1.
        eta = 2*y/b - 1.

        vec_f(fu, xi, x1u, x1ur, x2u, x2ur)
        vec_f(gu, eta, y1u, y1ur, y2u, y2ur)
        vec_f(fv, xi, x1v, x1vr, x2v, x2vr)
        vec_f(gv, eta, y1v, y1vr, y2v, y2vr)
        vec_f(fw, xi, x1w, x1wr, x2w, x2wr)
        vec_f(gw, eta, y1w, y1wr, y2w, y2wr)

        u = 0
        v = 0
        w = 0

        for j in range(n):
            for i in range(m):
                col = DOF*(j*m + i)
                u += c[col+0]*fu[i]*gu[j]
                v += c[col+1]*fv[i]*gv[j]
                w += c[col+2]*fw[i]*gw[j]

        us[pti] = u
        vs[pti] = v
        ws[pti] = w

    free(fu)
    free(gu)
    free(fv)
    free(gv)
    free(fw)
    free(gw)


cdef void cfwx(double *c, int m, int n, double a, double b, double *xs,
        double *ys, int size, double *wxs,
        double x1w, double x1wr, double x2w, double x2wr,
        double y1w, double y1wr, double y2w, double y2wr) nogil:
    cdef int i, j, col, pti
    cdef double x, y, wx, xi, eta
    cdef double *fwxi
    cdef double *gw

    fwxi = <double *>malloc(NMAX * sizeof(double *))
    gw = <double *>malloc(NMAX * sizeof(double *))

    for pti in range(size):
        x = xs[pti]
        y = ys[pti]

        xi = 2*x/a - 1.
        eta = 2*y/b - 1.

        vec_fp(fwxi, xi, x1w, x1wr, x2w, x2wr)
        vec_f(gw, eta, y1w, y1wr, y2w, y2wr)

        wx = 0

        for j in range(n):
            for i in range(m):
                col = DOF*(j*m + i)
                wx += (2/a)*c[col+2]*fwxi[i]*gw[j]

        wxs[pti] = wx

    free(fwxi)
    free(gw)


cdef void cfwy(double *c, int m, int n, double a, double b, double *xs,
        double *ys, int size, double *wys,
        double x1w, double x1wr, double x2w, double x2wr,
        double y1w, double y1wr, double y2w, double y2wr) nogil:
    cdef int i, j, col, pti
    cdef double x, y, wy, xi, eta
    cdef double *fw
    cdef double *gweta

    fw = <double *>malloc(NMAX * sizeof(double *))
    gweta = <double *>malloc(NMAX * sizeof(double *))

    for pti in range(size):
        x = xs[pti]
        y = ys[pti]

        xi = 2*x/a - 1.
        eta = 2*y/b - 1.

        vec_f(fw, xi, x1w, x1wr, x2w, x2wr)
        vec_fp(gweta, eta, y1w, y1wr, y2w, y2wr)

        wy = 0

        for j in range(n):
            for i in range(m):
                col = DOF*(j*m + i)
                wy += (2/b)*c[col+2]*fw[i]*gweta[j]

        wys[pti] = wy

    free(fw)
    free(gweta)


def fg(double[:, ::1] g, double x, double y, object s):
    # Accessing the contiguous memory layout for C 
    # ::1 at the 2nd position means that the elements in this 2nd dimension will be 1 element apart in memory.
    
    if s.__class__.__name__ != 'Shell':
        raise ValueError('A Shell object must be passed')
    a = s.a
    b = s.b
    m = s.m
    n = s.n
    cfg(g, m, n, x, y, a, b,
        s.x1u, s.x1ur, s.x2u, s.x2ur,
        s.x1v, s.x1vr, s.x2v, s.x2vr,
        s.x1w, s.x1wr, s.x2w, s.x2wr,
        s.y1u, s.y1ur, s.y2u, s.y2ur,
        s.y1v, s.y1vr, s.y2v, s.y2vr,
        s.y1w, s.y1wr, s.y2w, s.y2wr)


cdef void cfg(double[:,::1] g, int m, int n,
              double x, double y, double a, double b,
              double x1u, double x1ur, double x2u, double x2ur,
              double x1v, double x1vr, double x2v, double x2vr,
              double x1w, double x1wr, double x2w, double x2wr,
              double y1u, double y1ur, double y2u, double y2ur,
              double y1v, double y1vr, double y2v, double y2vr,
              double y1w, double y1wr, double y2w, double y2wr) nogil:
    cdef int i, j, col
    cdef double xi, eta
    cdef double *fu
    cdef double *fv
    cdef double *fw
    cdef double *fw_xi
    cdef double *gu
    cdef double *gv
    cdef double *gw
    cdef double *gw_eta

    fu = <double *>malloc(NMAX * sizeof(double *))
    # NMAX defined earlier
    # <double *> = explicitly casting the result of malloc to a pointer to double. 
    #  result of malloc = initially a generic pointer (void*). By casting it to a double*, you inform
    # the compiler that you intend to treat the allocated memory as if it stores a double 
    gu = <double *>malloc(NMAX * sizeof(double *))
    fv = <double *>malloc(NMAX * sizeof(double *))
    gv = <double *>malloc(NMAX * sizeof(double *))
    fw = <double *>malloc(NMAX * sizeof(double *))
    fw_xi = <double *>malloc(NMAX * sizeof(double *))
    gw = <double *>malloc(NMAX * sizeof(double *))
    gw_eta = <double *>malloc(NMAX * sizeof(double *))

    xi = 2*x/a - 1.
    eta = 2*y/b - 1.

    # Returns the bardel functions
    vec_f(fu, xi, x1u, x1ur, x2u, x2ur)
    vec_f(gu, eta, y1u, y1ur, y2u, y2ur)
    vec_f(fv, xi, x1v, x1vr, x2v, x2vr)
    vec_f(gv, eta, y1v, y1vr, y2v, y2vr)
    vec_f(fw, xi, x1w, x1wr, x2w, x2wr)
    vec_fp(fw_xi, xi, x1w, x1wr, x2w, x2wr)
    vec_f(gw, eta, y1w, y1wr, y2w, y2wr)
    vec_fp(gw_eta, eta, y1w, y1wr, y2w, y2wr)

    for j in range(n):
        for i in range(m):
            col = DOF*(j*m + i)
            g[0, col+0] = fu[i]*gu[j]
            g[1, col+1] = fv[i]*gv[j]
            g[2, col+2] = fw[i]*gw[j]
            g[3, col+2] = -(2/a)*fw_xi[i]*gw[j]
            g[4, col+2] = -(2/b)*fw[i]*gw_eta[j]

    free(fu)
    free(gu)
    free(fv)
    free(gv)
    free(fw)
    free(fw_xi)
    free(gw)
    free(gw_eta)

# *c passed as pointer - modifies it wo needing to pass it back
'''
    Calculates the strain at every point in the provided grid
'''
cdef void cfstrain(double *c, int m, int n, double a, double b,
        double r,
        double *xs, double *ys, int size,
        double *exxs, double *eyys, double *gxys,
        double *kxxs, double *kyys, double *kxys,
        double x1u, double x1ur, double x2u, double x2ur,
        double x1v, double x1vr, double x2v, double x2vr,
        double x1w, double x1wr, double x2w, double x2wr,
        double y1u, double y1ur, double y2u, double y2ur,
        double y1v, double y1vr, double y2v, double y2vr,
        double y1w, double y1wr, double y2w, double y2wr, int NLgeom) nogil:
    cdef int i, j, col, pti
    cdef double x, y, xi, eta
    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef int flagcyl

    cdef double *fu
    cdef double *fuxi
    cdef double *fv
    cdef double *fvxi
    cdef double *fw
    cdef double *fwxi
    cdef double *fwxixi

    cdef double *gu
    cdef double *gueta
    cdef double *gv
    cdef double *gveta
    cdef double *gw
    cdef double *gweta
    cdef double *gwetaeta

    cdef double wxi, weta

    fu = <double *>malloc(NMAX * sizeof(double *))
    fuxi = <double *>malloc(NMAX * sizeof(double *))
    gu = <double *>malloc(NMAX * sizeof(double *))
    gueta = <double *>malloc(NMAX * sizeof(double *))
    fv = <double *>malloc(NMAX * sizeof(double *))
    fvxi = <double *>malloc(NMAX * sizeof(double *))
    gv = <double *>malloc(NMAX * sizeof(double *))
    gveta = <double *>malloc(NMAX * sizeof(double *))
    fw = <double *>malloc(NMAX * sizeof(double *))
    fwxi = <double *>malloc(NMAX * sizeof(double *))
    fwxixi = <double *>malloc(NMAX * sizeof(double *))
    gw = <double *>malloc(NMAX * sizeof(double *))
    gweta = <double *>malloc(NMAX * sizeof(double *))
    gwetaeta = <double *>malloc(NMAX * sizeof(double *))

    if r == 0:
        flagcyl = 0
    else:
        flagcyl = 1

    # Goes through the passed x and y (which are already all the points on the grid)
    for pti in range(size):
        x = xs[pti]
        y = ys[pti]

        xi = 2*x/a - 1.
        eta = 2*y/b - 1.

        # u functions at x and y (loop gets it through all points)
        vec_f(fu, xi, x1u, x1ur, x2u, x2ur)
        vec_fp(fuxi, xi, x1u, x1ur, x2u, x2ur)
        vec_f(gu, eta, y1u, y1ur, y2u, y2ur)
        vec_fp(gueta, eta, y1u, y1ur, y2u, y2ur)
        # v functions 
        vec_f(fv, xi, x1v, x1vr, x2v, x2vr)
        vec_fp(fvxi, xi, x1v, x1vr, x2v, x2vr)
        vec_f(gv, eta, y1v, y1vr, y2v, y2vr)
        vec_fp(gveta, eta, y1v, y1vr, y2v, y2vr)
        # w functions 
        vec_f(fw, xi, x1w, x1wr, x2w, x2wr)
        vec_fp(fwxi, xi, x1w, x1wr, x2w, x2wr)
        vec_fpp(fwxixi, xi, x1w, x1wr, x2w, x2wr)
        vec_f(gw, eta, y1w, y1wr, y2w, y2wr)
        vec_fp(gweta, eta, y1w, y1wr, y2w, y2wr)
        vec_fpp(gwetaeta, eta, y1w, y1wr, y2w, y2wr)

        wxi = 0
        weta = 0

        # Sum through all m*n terms
        for j in range(n):
            for i in range(m):
                col = DOF*(j*m + i)
                wxi += c[col+2]*fwxi[i]*gw[j]
                weta += c[col+2]*fw[i]*gweta[j]
                # +2 to get w (+0 is u, +1 is v)

        exx = 0
        eyy = 0
        gxy = 0
        kxx = 0
        kyy = 0
        kxy = 0

        for j in range(n):
            for i in range(m):
                col = DOF*(j*m + i)
                exx += c[col+0]*fuxi[i]*gu[j]*(2/a) + NLgeom*2/(a*a)*c[col+2]*fwxi[i]*gw[j]*wxi # same as wxi^2
                if flagcyl == 1:
                    eyy += c[col+1]*fv[i]*gveta[j]*(2/b) + 1/r*c[col+2]*fw[i]*gw[j] + NLgeom*2/(b*b)*c[col+2]*fw[i]*gweta[j]*weta
                else:
                    eyy += c[col+1]*fv[i]*gveta[j]*(2/b) + NLgeom*2/(b*b)*c[col+2]*fw[i]*gweta[j]*weta
                gxy += c[col+0]*fu[i]*gueta[j]*(2/b) + c[col+1]*fvxi[i]*gv[j]*(2/a) + NLgeom*4/(a*b)*(
                                        c[col+2]*fwxi[i]*gw[j]*weta +
                                        wxi*c[col+2]*fw[i]*gweta[j] )
                kxx += -c[col+2]*fwxixi[i]*gw[j]*4/(a*a)
                kyy += -c[col+2]*fw[i]*gwetaeta[j]*4/(b*b)
                kxy += -2*c[col+2]*fwxi[i]*gweta[j]*4/(a*b)

        # Strains and curvatures at each point in the grid 
        exxs[pti] = exx
        eyys[pti] = eyy
        gxys[pti] = gxy
        kxxs[pti] = kxx
        kyys[pti] = kyy
        kxys[pti] = kxy

    # Frees up the allocated mem
    free(fu)
    free(fuxi)
    free(gu)
    free(gueta)
    free(fv)
    free(fvxi)
    free(gv)
    free(gveta)
    free(fw)
    free(fwxi)
    free(fwxixi)
    free(gw)
    free(gweta)
    free(gwetaeta)

