from __future__ import division

cimport cython
from libc.math cimport exp

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.boundscheck(False)
def build_isotropic_kernel(double amp, double factor,
                           np.ndarray[DTYPE_t, ndim=2] x1,
                           np.ndarray[DTYPE_t, ndim=2] x2):
    cdef double r2, d
    cdef unsigned int i, j, k
    cdef unsigned int ndim = x1.shape[1]
    cdef unsigned int n1 = x1.shape[0]
    cdef unsigned int n2 = x2.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] K = np.empty((n1, n2), dtype=DTYPE)

    for i in range(n1):
        for j in range(n2):
            r2 = 0.0
            for k in range(ndim):
                d = x1[i, k] - x2[j, k]
                r2 += d*d
            K[i, j] = amp * exp(r2 * factor)

    return K
