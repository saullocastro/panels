#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from cython.parallel import prange

cdef extern from 'bardell_functions.hpp':
    double calc_vec_f(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_vec_fx(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_vec_fxx(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil

DOUBLE = np.float64
ctypedef np.double_t cDOUBLE

cdef int NMAX = 30
cdef int NUM = 3


def fuvw(np.ndarray[cDOUBLE, ndim=1] c, object s,
        np.ndarray[cDOUBLE, ndim=1] xs, np.ndarray[cDOUBLE, ndim=1] ys,
        int num_cores=4):
    cdef double a, b
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry
    a = s.a
    b = s.b
    m = s.m
    n = s.n
    u1tx = s.u1tx ; u1rx = s.u1rx ; u2tx = s.u2tx ; u2rx = s.u2rx
    v1tx = s.v1tx ; v1rx = s.v1rx ; v2tx = s.v2tx ; v2rx = s.v2rx
    w1tx = s.w1tx ; w1rx = s.w1rx ; w2tx = s.w2tx ; w2rx = s.w2rx
    u1ty = s.u1ty ; u1ry = s.u1ry ; u2ty = s.u2ty ; u2ry = s.u2ry
    v1ty = s.v1ty ; v1ry = s.v1ry ; v2ty = s.v2ty ; v2ry = s.v2ry
    w1ty = s.w1ty ; w1ry = s.w1ry ; w2ty = s.w2ty ; w2ry = s.w2ry

    cdef int size_core, pti
    cdef np.ndarray[cDOUBLE, ndim=2] us, vs, ws, phixs, phiys
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size == num_cores:
        add_size = 0
    new_size = size + add_size

    if (size % num_cores) != 0:
        xs_core = np.ascontiguousarray(np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
    else:
        xs_core = np.ascontiguousarray(xs.reshape(num_cores, -1), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(ys.reshape(num_cores, -1), dtype=DOUBLE)

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
              u1tx, u1rx, u2tx, u2rx,
              v1tx, v1rx, v2tx, v2rx,
              w1tx, w1rx, w2tx, w2rx,
              u1ty, u1ry, u2ty, u2ry,
              v1ty, v1ry, v2ty, v2ry,
              w1ty, w1ry, w2ty, w2ry)

        cfwx(&c[0], m, n, a, b, &xs_core[pti,0], &ys_core[pti,0],
             size_core, &phixs[pti,0],
             w1tx, w1rx, w2tx, w2rx,
             w1ty, w1ry, w2ty, w2ry)

        cfwy(&c[0], m, n, a, b, &xs_core[pti,0], &ys_core[pti,0],
             size_core, &phiys[pti,0],
             w1tx, w1rx, w2tx, w2rx,
             w1ty, w1ry, w2ty, w2ry)

    phixs *= -1.
    phiys *= -1.
    return (us.ravel()[:size], vs.ravel()[:size], ws.ravel()[:size],
            phixs.ravel()[:size], phiys.ravel()[:size])


def fstrain(np.ndarray[cDOUBLE, ndim=1] c, object s,
        np.ndarray[cDOUBLE, ndim=1] xs, np.ndarray[cDOUBLE, ndim=1] ys,
        int num_cores=4, int NLgeom=0):
    cdef double a, b, r, alpharad
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry
    a = s.a
    b = s.b
    r = s.r
    alpharad = s.alpharad
    m = s.m
    n = s.n
    u1tx = s.u1tx ; u1rx = s.u1rx ; u2tx = s.u2tx ; u2rx = s.u2rx
    v1tx = s.v1tx ; v1rx = s.v1rx ; v2tx = s.v2tx ; v2rx = s.v2rx
    w1tx = s.w1tx ; w1rx = s.w1rx ; w2tx = s.w2tx ; w2rx = s.w2rx
    u1ty = s.u1ty ; u1ry = s.u1ry ; u2ty = s.u2ty ; u2ry = s.u2ry
    v1ty = s.v1ty ; v1ry = s.v1ry ; v2ty = s.v2ty ; v2ry = s.v2ry
    w1ty = s.w1ty ; w1ry = s.w1ry ; w2ty = s.w2ty ; w2ry = s.w2ry

    cdef int size_core, pti
    cdef np.ndarray[cDOUBLE, ndim=2] exxs, eyys, gxys, kxxs, kyys, kxys
    cdef np.ndarray[cDOUBLE, ndim=2] xs_core, ys_core

    size = xs.shape[0]
    add_size = num_cores - (size % num_cores)
    if add_size == num_cores:
        add_size = 0
    new_size = size + add_size

    if (size % num_cores) != 0:
        xs_core = np.ascontiguousarray(np.hstack((xs, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(np.hstack((ys, np.zeros(add_size))).reshape(num_cores, -1), dtype=DOUBLE)
    else:
        xs_core = np.ascontiguousarray(xs.reshape(num_cores, -1), dtype=DOUBLE)
        ys_core = np.ascontiguousarray(ys.reshape(num_cores, -1), dtype=DOUBLE)

    size_core = xs_core.shape[1]

    exxs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    eyys = np.zeros((num_cores, size_core), dtype=DOUBLE)
    gxys = np.zeros((num_cores, size_core), dtype=DOUBLE)
    kxxs = np.zeros((num_cores, size_core), dtype=DOUBLE)
    kyys = np.zeros((num_cores, size_core), dtype=DOUBLE)
    kxys = np.zeros((num_cores, size_core), dtype=DOUBLE)

    if alpharad != 0:
        raise NotImplementedError('Conical shells not suported')

    for pti in prange(num_cores, nogil=True, chunksize=1, num_threads=num_cores,
                    schedule='static'):
        cfstrain(&c[0], m, n, a, b, r, alpharad,
              &xs_core[pti,0], &ys_core[pti,0], size_core,
              &exxs[pti,0], &eyys[pti,0], &gxys[pti,0],
              &kxxs[pti,0], &kyys[pti,0], &kxys[pti,0],
              u1tx, u1rx, u2tx, u2rx,
              v1tx, v1rx, v2tx, v2rx,
              w1tx, w1rx, w2tx, w2rx,
              u1ty, u1ry, u2ty, u2ry,
              v1ty, v1ry, v2ty, v2ry,
              w1ty, w1ry, w2ty, w2ry, NLgeom)

    return (exxs.ravel()[:size], eyys.ravel()[:size], gxys.ravel()[:size],
            kxxs.ravel()[:size], kyys.ravel()[:size], kxys.ravel()[:size])


cdef void cfuvw(double *c, int m, int n, double a, double b, double *xs,
        double *ys, int size, double *us, double *vs, double *ws,
        double u1tx, double u1rx, double u2tx, double u2rx,
        double v1tx, double v1rx, double v2tx, double v2rx,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double u1ty, double u1ry, double u2ty, double u2ry,
        double v1ty, double v1ry, double v2ty, double v2ry,
        double w1ty, double w1ry, double w2ty, double w2ry) nogil:
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

        calc_vec_f(fu, xi, u1tx, u1rx, u2tx, u2rx)
        calc_vec_f(gu, eta, u1ty, u1ry, u2ty, u2ry)
        calc_vec_f(fv, xi, v1tx, v1rx, v2tx, v2rx)
        calc_vec_f(gv, eta, v1ty, v1ry, v2ty, v2ry)
        calc_vec_f(fw, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_f(gw, eta, w1ty, w1ry, w2ty, w2ry)

        u = 0
        v = 0
        w = 0

        for j in range(n):
            for i in range(m):
                col = NUM*(j*m + i)
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
        double w1tx, double w1rx, double w2tx, double w2rx,
        double w1ty, double w1ry, double w2ty, double w2ry) nogil:
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

        calc_vec_fx(fwxi, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_f(gw, eta, w1ty, w1ry, w2ty, w2ry)

        wx = 0

        for j in range(n):
            for i in range(m):
                col = NUM*(j*m + i)
                wx += (2/a)*c[col+2]*fwxi[i]*gw[j]

        wxs[pti] = wx

    free(fwxi)
    free(gw)


cdef void cfwy(double *c, int m, int n, double a, double b, double *xs,
        double *ys, int size, double *wys,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double w1ty, double w1ry, double w2ty, double w2ry) nogil:
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

        calc_vec_f(fw, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_fx(gweta, eta, w1ty, w1ry, w2ty, w2ry)

        wy = 0

        for j in range(n):
            for i in range(m):
                col = NUM*(j*m + i)
                wy += (2/b)*c[col+2]*fw[i]*gweta[j]

        wys[pti] = wy

    free(fw)
    free(gweta)


def fg(double[:,::1] g, double x, double y, object s):
    cdef int i, j, col
    cdef double xi, eta
    cdef double *fu
    cdef double *fv
    cdef double *fw
    cdef double *gu
    cdef double *gv
    cdef double *gw

    if s.__class__.__name__ != 'Shell':
        raise ValueError('A Shell object must be passed')

    fu = <double *>malloc(NMAX * sizeof(double *))
    gu = <double *>malloc(NMAX * sizeof(double *))
    fv = <double *>malloc(NMAX * sizeof(double *))
    gv = <double *>malloc(NMAX * sizeof(double *))
    fw = <double *>malloc(NMAX * sizeof(double *))
    gw = <double *>malloc(NMAX * sizeof(double *))

    xi = 2*x/s.a - 1.
    eta = 2*y/s.b - 1.

    calc_vec_f(fu, xi, s.u1tx, s.u1rx, s.u2tx, s.u2rx)
    calc_vec_f(gu, eta, s.u1ty, s.u1ry, s.u2ty, s.u2ry)
    calc_vec_f(fv, xi, s.v1tx, s.v1rx, s.v2tx, s.v2rx)
    calc_vec_f(gv, eta, s.v1ty, s.v1ry, s.v2ty, s.v2ry)
    calc_vec_f(fw, xi, s.w1tx, s.w1rx, s.w2tx, s.w2rx)
    calc_vec_f(gw, eta, s.w1ty, s.w1ry, s.w2ty, s.w2ry)

    for j in range(s.n):
        for i in range(s.m):
            col = NUM*(j*s.m + i)
            g[0, col+0] = fu[i]*gu[j]
            g[1, col+1] = fv[i]*gv[j]
            g[2, col+2] = fw[i]*gw[j]

    free(fu)
    free(gu)
    free(fv)
    free(gv)
    free(fw)
    free(gw)


cdef void cfstrain(double *c, int m, int n, double a, double b,
        double r, double alpharad,
        double *xs, double *ys, int size,
        double *exxs, double *eyys, double *gxys,
        double *kxxs, double *kyys, double *kxys,
        double u1tx, double u1rx, double u2tx, double u2rx,
        double v1tx, double v1rx, double v2tx, double v2rx,
        double w1tx, double w1rx, double w2tx, double w2rx,
        double u1ty, double u1ry, double u2ty, double u2ry,
        double v1ty, double v1ry, double v2ty, double v2ry,
        double w1ty, double w1ry, double w2ty, double w2ry, int NLgeom) nogil:
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

    for pti in range(size):
        x = xs[pti]
        y = ys[pti]

        xi = 2*x/a - 1.
        eta = 2*y/b - 1.

        calc_vec_f(fu, xi, u1tx, u1rx, u2tx, u2rx)
        calc_vec_fx(fuxi, xi, u1tx, u1rx, u2tx, u2rx)
        calc_vec_f(gu, eta, u1ty, u1ry, u2ty, u2ry)
        calc_vec_fx(gueta, eta, u1ty, u1ry, u2ty, u2ry)
        calc_vec_f(fv, xi, v1tx, v1rx, v2tx, v2rx)
        calc_vec_fx(fvxi, xi, v1tx, v1rx, v2tx, v2rx)
        calc_vec_f(gv, eta, v1ty, v1ry, v2ty, v2ry)
        calc_vec_fx(gveta, eta, v1ty, v1ry, v2ty, v2ry)
        calc_vec_f(fw, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_fx(fwxi, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_fxx(fwxixi, xi, w1tx, w1rx, w2tx, w2rx)
        calc_vec_f(gw, eta, w1ty, w1ry, w2ty, w2ry)
        calc_vec_fx(gweta, eta, w1ty, w1ry, w2ty, w2ry)
        calc_vec_fxx(gwetaeta, eta, w1ty, w1ry, w2ty, w2ry)

        wxi = 0
        weta = 0

        for j in range(n):
            for i in range(m):
                col = NUM*(j*m + i)
                wxi += c[col+2]*fwxi[i]*gw[j]
                weta += c[col+2]*fw[i]*gweta[j]

        exx = 0
        eyy = 0
        gxy = 0
        kxx = 0
        kyy = 0
        kxy = 0

        for j in range(n):
            for i in range(m):
                col = NUM*(j*m + i)
                exx += c[col+0]*fuxi[i]*gu[j]*(2/a) + NLgeom*2/(a*a)*c[col+2]*fwxi[i]*gw[j]*wxi
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

        exxs[pti] = exx
        eyys[pti] = eyy
        gxys[pti] = gxy
        kxxs[pti] = kxx
        kyys[pti] = kyy
        kxys[pti] = kxy

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
