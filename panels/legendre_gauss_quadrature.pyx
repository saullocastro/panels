#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from __future__ import division

import numpy as np
cimport numpy as np

cdef extern from 'legendre_gauss_quadrature.hpp':
    double leggauss_quad(int n, double *points, double *weights) nogil

ctypedef np.double_t cDOUBLE

def get_points_weights(int n,
        np.ndarray[cDOUBLE, ndim=1] points,
        np.ndarray[cDOUBLE, ndim=1] weights):
    leggauss_quad(n, &points[0], &weights[0])

