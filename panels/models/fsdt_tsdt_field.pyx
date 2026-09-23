#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Field variables of the plate and cylindrical shell models using the
first-order (FSDT) and Reddy's third-order (TSDT) shear deformation theories

The degrees of freedom of each term of the approximation are ``u, v, w, phix,
phiy``. See the kinematic equations in ``theory/shells/fsdt_tsdt/fsdt_tsdt.py``.
For the models with the Sanders-Koiter kinematics, the rotation of the
normal about `x` is `\phi_y + v/r`, which is the field ``phiy`` returned by
:func:`.fuvw` and the approximation of ``phiy`` of :func:`.fg`.

"""
import numpy as np
from libc.stdlib cimport malloc, free

cdef extern from 'bardell_functions.hpp':
    double vec_f(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) noexcept nogil
    double vec_fp(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) noexcept nogil
    double vec_fpp(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) noexcept nogil

DOUBLE = np.float64

cdef int NMAX = 30
cdef int DOF = 5
cdef int NFIELDS = 5

FIELDS = ('u', 'v', 'w', 'phix', 'phiy')


def is_tsdt(object s):
    return 1 if 'tsdt' in s.model else 0


def is_sanders(object s):
    return 1 if 'sanders' in s.model else 0


def rinv(object s):
    r'''`1/r` for the cylinders and zero for the plates, which may have
    ``r = None``'''
    return 1./s.r if 'cylshell' in s.model else 0.


def strain_names(object s):
    r'''Names of the generalized strains returned by :func:`.fstrain`

    They are in the order of the rows and columns of the constitutive matrix
    ``Shell.ABD``: the mid-surface strains, the first-order changes of
    curvature and the mid-surface transverse shear strains, plus, for the
    TSDT, the third-order terms ``kxx3, kyy3, kxy3`` of the in-plane strains
    and the second-order terms ``gyz2, gxz2`` of the transverse shear strains.

    '''
    if is_tsdt(s) == 1:
        return ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'kxx3', 'kyy3',
                'kxy3', 'gyz', 'gxz', 'gyz2', 'gxz2')
    return ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz')


def stress_names(object s):
    r'''Names of the generalized stresses, conjugate to :func:`.strain_names`
    '''
    if is_tsdt(s) == 1:
        return ('Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Pxx', 'Pyy', 'Pxy',
                'Qy', 'Qx', 'Ry', 'Rx')
    return ('Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qy', 'Qx')


cdef void read_bcs(object s, double *bcs):
    # bcs[field*8 + 0..3] along x, bcs[field*8 + 4..7] along y
    cdef int k
    for k, field in enumerate(FIELDS):
        bcs[k*8 + 0] = getattr(s, 'x1' + field)
        bcs[k*8 + 1] = getattr(s, 'x1' + field + 'r')
        bcs[k*8 + 2] = getattr(s, 'x2' + field)
        bcs[k*8 + 3] = getattr(s, 'x2' + field + 'r')
        bcs[k*8 + 4] = getattr(s, 'y1' + field)
        bcs[k*8 + 5] = getattr(s, 'y1' + field + 'r')
        bcs[k*8 + 6] = getattr(s, 'y2' + field)
        bcs[k*8 + 7] = getattr(s, 'y2' + field + 'r')


cdef void basis(double *bcs, double xi, double eta, double *f, double *fxi,
        double *fxixi, double *g, double *geta, double *getaeta) noexcept nogil:
    # values and derivatives of the approximation functions of all the fields
    # at (xi, eta), stored as f[field*NMAX + i] and g[field*NMAX + j]
    cdef int k
    cdef double *bc
    for k in range(NFIELDS):
        bc = &bcs[k*8]
        vec_f(&f[k*NMAX], xi, bc[0], bc[1], bc[2], bc[3])
        vec_fp(&fxi[k*NMAX], xi, bc[0], bc[1], bc[2], bc[3])
        vec_fpp(&fxixi[k*NMAX], xi, bc[0], bc[1], bc[2], bc[3])
        vec_f(&g[k*NMAX], eta, bc[4], bc[5], bc[6], bc[7])
        vec_fp(&geta[k*NMAX], eta, bc[4], bc[5], bc[6], bc[7])
        vec_fpp(&getaeta[k*NMAX], eta, bc[4], bc[5], bc[6], bc[7])


cdef void derivatives(double *c, int m, int n, double a, double b,
        double *f, double *fxi, double *fxixi, double *g, double *geta,
        double *getaeta, double *val, double *dx, double *dy, double *dxx,
        double *dyy, double *dxy) noexcept nogil:
    # values and physical derivatives of each field at the point
    cdef int i, j, k, col
    cdef double ck
    for k in range(NFIELDS):
        val[k] = 0; dx[k] = 0; dy[k] = 0; dxx[k] = 0; dyy[k] = 0; dxy[k] = 0
    for j in range(n):
        for i in range(m):
            col = DOF*(j*m + i)
            for k in range(NFIELDS):
                ck = c[col+k]
                val[k] += ck*f[k*NMAX+i]*g[k*NMAX+j]
                dx[k] += ck*(2/a)*fxi[k*NMAX+i]*g[k*NMAX+j]
                dy[k] += ck*(2/b)*f[k*NMAX+i]*geta[k*NMAX+j]
                dxx[k] += ck*(2/a)*(2/a)*fxixi[k*NMAX+i]*g[k*NMAX+j]
                dyy[k] += ck*(2/b)*(2/b)*f[k*NMAX+i]*getaeta[k*NMAX+j]
                dxy[k] += ck*(2/a)*(2/b)*fxi[k*NMAX+i]*geta[k*NMAX+j]


def fuvw(double [::1] c, object s, double [::1] xs, double [::1] ys,
        int num_cores=4):
    '''
        Calculates the displacement field at all points in the provided grid

        Returns ``u, v, w, phix, phiy``, where ``phix`` and ``phiy`` are the
        rotations of the normal, `\phi_x` and `\phi_y`, and `\phi_y + v/r`
        for the Sanders-Koiter kinematics.
    '''
    cdef double a, b, sr
    cdef int m, n, pti, k, size
    cdef double bcs[40]
    cdef double [:, ::1] out
    cdef double *f
    cdef double *fxi
    cdef double *fxixi
    cdef double *g
    cdef double *geta
    cdef double *getaeta
    cdef double val[5]
    cdef double dx[5]
    cdef double dy[5]
    cdef double dxx[5]
    cdef double dyy[5]
    cdef double dxy[5]
    a = s.a
    b = s.b
    m = s.m
    n = s.n
    #NOTE r is only read for the cylinders, a plate may have r = None
    sr = rinv(s)*is_sanders(s)
    read_bcs(s, bcs)
    size = xs.shape[0]
    out = np.zeros((NFIELDS, size), dtype=DOUBLE)
    f = <double *>malloc(6*NFIELDS*NMAX*sizeof(double))
    fxi = &f[NFIELDS*NMAX]
    fxixi = &f[2*NFIELDS*NMAX]
    g = &f[3*NFIELDS*NMAX]
    geta = &f[4*NFIELDS*NMAX]
    getaeta = &f[5*NFIELDS*NMAX]
    with nogil:
        for pti in range(size):
            basis(bcs, 2*xs[pti]/a - 1., 2*ys[pti]/b - 1., f, fxi, fxixi, g,
                  geta, getaeta)
            derivatives(&c[0], m, n, a, b, f, fxi, fxixi, g, geta, getaeta,
                        val, dx, dy, dxx, dyy, dxy)
            for k in range(NFIELDS):
                out[k, pti] = val[k]
            out[4, pti] += sr*val[1]
    free(f)
    return tuple(np.asarray(out[k]) for k in range(NFIELDS))


def fstrain(double [::1] c, object s, double [::1] xs, double [::1] ys,
        int num_cores=4, int NLgeom=0):
    '''
        Calculates the generalized strains at all points in the provided grid

        The strains are returned in the order given by :func:`.strain_names`.
    '''
    cdef double a, b, h, c1, ri, sr, bx, by
    cdef int m, n, pti, size, tsdt
    cdef double bcs[40]
    cdef double [:, ::1] out
    cdef double *f
    cdef double *fxi
    cdef double *fxixi
    cdef double *g
    cdef double *geta
    cdef double *getaeta
    cdef double val[5]
    cdef double dx[5]
    cdef double dy[5]
    cdef double dxx[5]
    cdef double dyy[5]
    cdef double dxy[5]
    cdef double gyz, gxz
    a = s.a
    b = s.b
    m = s.m
    n = s.n
    tsdt = is_tsdt(s)
    c1 = 0.
    if tsdt == 1:
        h = sum(s.plyts)
        c1 = 4./(3.*h*h)
    #NOTE r is only read for the cylinders, a plate may have r = None
    ri = rinv(s)
    sr = ri*is_sanders(s)
    read_bcs(s, bcs)
    size = xs.shape[0]
    out = np.zeros((13 if tsdt == 1 else 8, size), dtype=DOUBLE)
    f = <double *>malloc(6*NFIELDS*NMAX*sizeof(double))
    fxi = &f[NFIELDS*NMAX]
    fxixi = &f[2*NFIELDS*NMAX]
    g = &f[3*NFIELDS*NMAX]
    geta = &f[4*NFIELDS*NMAX]
    getaeta = &f[5*NFIELDS*NMAX]
    with nogil:
        for pti in range(size):
            basis(bcs, 2*xs[pti]/a - 1., 2*ys[pti]/b - 1., f, fxi, fxixi, g,
                  geta, getaeta)
            derivatives(&c[0], m, n, a, b, f, fxi, fxixi, g, geta, getaeta,
                        val, dx, dy, dxx, dyy, dxy)
            # u, v, w, phix, phiy = 0, 1, 2, 3, 4
            # rotations of the von Karman terms, beta_y = w,y - v/r (Sanders)
            bx = dx[2]
            by = dy[2] - sr*val[1]
            out[0, pti] = dx[0] + NLgeom*0.5*bx*bx
            out[1, pti] = dy[1] + ri*val[2] + NLgeom*0.5*by*by
            out[2, pti] = dy[0] + dx[1] + NLgeom*bx*by
            out[3, pti] = dx[3]
            out[4, pti] = dy[4] + sr*dy[1]
            out[5, pti] = dy[3] + dx[4] + sr*(1.5*dx[1] - 0.5*dy[0])
            gyz = val[4] + dy[2]
            gxz = val[3] + dx[2]
            if tsdt == 1:
                out[6, pti] = -c1*(dx[3] + dxx[2])
                out[7, pti] = -c1*(dy[4] + dyy[2])
                out[8, pti] = -c1*(dy[3] + dx[4] + 2*dxy[2])
                out[9, pti] = gyz
                out[10, pti] = gxz
                out[11, pti] = -3*c1*gyz
                out[12, pti] = -3*c1*gxz
            else:
                out[6, pti] = gyz
                out[7, pti] = gxz
    free(f)
    return tuple(np.asarray(out[k]) for k in range(out.shape[0]))


def fg(double[:, ::1] g, double x, double y, object s):
    '''
        Approximation functions of ``u, v, w, phix, phiy`` at ``(x, y)``,
        stored in the rows of ``g``, which must have shape ``(5, size)``. For
        the Sanders-Koiter kinematics the row of ``phiy`` gives the rotation
        of the normal `\phi_y + v/r`
    '''
    cdef int m, n, i, j, k, col
    cdef double a, b, sr
    cdef double bcs[40]
    cdef double *f
    cdef double *fxi
    cdef double *fxixi
    cdef double *gg
    cdef double *geta
    cdef double *getaeta
    if s.__class__.__name__ != 'Shell':
        raise ValueError('A Shell object must be passed')
    a = s.a
    b = s.b
    m = s.m
    n = s.n
    sr = rinv(s)*is_sanders(s)
    read_bcs(s, bcs)
    f = <double *>malloc(6*NFIELDS*NMAX*sizeof(double))
    fxi = &f[NFIELDS*NMAX]
    fxixi = &f[2*NFIELDS*NMAX]
    gg = &f[3*NFIELDS*NMAX]
    geta = &f[4*NFIELDS*NMAX]
    getaeta = &f[5*NFIELDS*NMAX]
    basis(bcs, 2*x/a - 1., 2*y/b - 1., f, fxi, fxixi, gg, geta, getaeta)
    for j in range(n):
        for i in range(m):
            col = DOF*(j*m + i)
            for k in range(NFIELDS):
                g[k, col+k] = f[k*NMAX+i]*gg[k*NMAX+j]
            g[4, col+1] = sr*f[1*NMAX+i]*gg[1*NMAX+j]
    free(f)
