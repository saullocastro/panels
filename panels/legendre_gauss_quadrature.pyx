#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
cdef extern from 'legendre_gauss_quadrature.hpp':
    double leggauss_quad(int n, double *points, double *weights) nogil
    double leggauss_quad_304(int n, double *points, double *weights) nogil


def get_points_weights(int n,
                       double [::1] points,
                       double [::1] weights):
    leggauss_quad(n, &points[0], &weights[0])

def get_points_weights_304(int n,
                       double [::1] points,
                       double [::1] weights):
    leggauss_quad_304(n, &points[0], &weights[0])

