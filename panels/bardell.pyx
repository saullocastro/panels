#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
import numpy as np

NUM = 30 # maximum number of terms currently implemented
DOUBLE = np.float64


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double x1t, double x1r, double x2t, double x2r) nogil
    double fp(int i, double xi, double x1t, double x1r, double x2t, double x2r) nogil
    double fpp(int i, double xi, double x1t, double x1r, double x2t, double x2r) nogil
    void vec_f(double *f, double xi, double x1t, double x1r, double x2t, double x2r) nogil
    void vec_fp(double *f, double xi, double x1t, double x1r, double x2t, double x2r) nogil
    void vec_fpp(double *f, double xi, double x1t, double x1r, double x2t, double x2r) nogil


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fppfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil


def calc_f(int i, double xi, double x1t, double x1r, double x2t, double x2r):
    return f(i, xi, x1t, x1r, x2t, x2r)
def calc_fp(int i, double xi, double x1t, double x1r, double x2t, double x2r):
    return fp(i, xi, x1t, x1r, x2t, x2r)
def calc_fpp(int i, double xi, double x1t, double x1r, double x2t, double x2r):
    return fpp(i, xi, x1t, x1r, x2t, x2r)
def calc_vec_f(double xi, double x1t, double x1r, double x2t, double x2r):
    cdef double [::1] out = np.zeros(NUM, dtype=DOUBLE)
    vec_f(&out[0], xi, x1t, x1r, x2t, x2r)
    return np.asarray(out)
def calc_vec_fp(double xi, double x1t, double x1r, double x2t, double x2r):
    cdef double [::1] out = np.zeros(NUM, dtype=DOUBLE)
    vec_fp(&out[0], xi, x1t, x1r, x2t, x2r)
    return np.asarray(out)
def calc_vec_fpp(double xi, double x1t, double x1r, double x2t, double x2r):
    cdef double [::1] out = np.zeros(NUM, dtype=DOUBLE)
    vec_fpp(&out[0], xi, x1t, x1r, x2t, x2r)
    return np.asarray(out)


def calc_integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                         double y1t, double y1r, double y2t, double y2r):
    return integral_ff(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_ffp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                          double y1t, double y1r, double y2t, double y2r):
    return integral_ffp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_ffpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                           double y1t, double y1r, double y2t, double y2r):
    return integral_ffpp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_fpfp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                           double y1t, double y1r, double y2t, double y2r):
    return integral_fpfp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_fpfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                            double y1t, double y1r, double y2t, double y2r):
    return integral_fpfpp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_fppfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                             double y1t, double y1r, double y2t, double y2r):
    return integral_fppfpp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)

