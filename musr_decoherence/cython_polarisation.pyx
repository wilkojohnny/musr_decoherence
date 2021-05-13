import numpy as np
cimport cython
from cython.parallel import prange

@cython.boundscheck(False)
@cython.wraparound(False)
def minus_half(double complex [:, :] R):
    # code which multiplies the top half of a matrix R by -1, like two of the pauli spin matrices
    cdef Py_ssize_t n_total = R.shape[0]

    cdef Py_ssize_t n_half = int(n_total / 2)
    cdef Py_ssize_t i, j

    for i in prange(0, n_half, nogil=True):
        for j in range(n_total):
            R[i, j] = (-1+0j)*R[i, j]

    return R

cdef extern from "<complex.h>" nogil:
    double creal(double complex z)
    double cabs( double complex z)

cdef extern from "<math.h>" nogil:
    double cos(double z)

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_amplitudes_angavg(double complex [:, ::1] sx, double complex [:, ::1] sy, double complex [:, ::1] sz,
                           int size):
    """
    calculate the angular average from si = Rinv*s_i*R for x y z
    """
    cdef Py_ssize_t dim = sx.shape[0]

    result = np.zeros((dim, dim), dtype=np.float64)
    cdef double [:,:] result_view = result

    cdef Py_ssize_t i, j

    for i in prange(dim, nogil=True):
        for j in range(dim):
            result_view[i, j] = creal(cabs(sx[i, j]) * cabs(sx[i, j]) + cabs(sy[i, j]) * cabs(sy[i, j])
                                 + cabs(sz[i, j]) * cabs(sz[i, j])  ) / (3 * size / 2)
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_amplitudes_initpol(double complex [:, ::1] sx, double complex [:, ::1] sy, double complex [:, ::1] sz,
                            double wx, double wy, double wz, int size):
    """
    Calculate the amplitudes when NOT doing an angular average
    """
    cdef Py_ssize_t dim = sx.shape[0]

    result = np.zeros((dim, dim), dtype=np.float64)
    cdef double [:,:] result_view = result

    cdef Py_ssize_t i, j

    for i in prange(dim, nogil=True):
        for j in range(dim):
            result_view[i, j] = 1 / (size*1.0 / 2 ) * creal(
                                    cabs(sx[i, j]) * cabs(sx[i, j]) * wx * wx
                                    + cabs(sy[i, j]) * cabs(sy[i, j]) * wy * wy
                                    + cabs(sz[i, j]) * cabs(sz[i, j]) * wz * wz
                                    + wx * wy * (
                                        (cabs(sx[i,j] + 1j * sy[i, j]) * cabs(sx[i,j] + 1j * sy[i, j]))
                                        - (cabs(sx[i,j]) * cabs(sx[i,j])) - (cabs(sy[i,j]) * cabs(sy[i,j]))
                                    ) \
                                    + wy * wz * (
                                            (cabs(sz[i,j] + 1j * sy[i, j]) * cabs(sz[i,j] + 1j * sy[i, j]))
                                            - (cabs(sz[i,j]) * cabs(sz[i,j])) - (cabs(sy[i,j]) * cabs(sy[i,j]))
                                    ) \
                                    + wx * wz * (
                                            (cabs(sx[i,j] + sz[i, j]) * cabs(sx[i,j] + sz[i, j]))
                                            - (cabs(sx[i,j]) * cabs(sx[i,j])) - (cabs(sz[i,j]) * cabs(sz[i,j]))
                                    ))
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_oscillation(double [:, ::1] amplitudes, double [:, ::1] Ediff, double [:] t):
    """
    calculate the oscillatory terms, using a matrix of amplitudes and Ediff[i,j]=E[i]-E[j]
    """

    cdef Py_ssize_t n_times = t.shape[0]
    cdef Py_ssize_t amp_dim = amplitudes.shape[0]

    P = np.zeros((n_times), dtype=np.float64)
    cdef double [:] P_view = P

    # i does matrix rows, j matrix columns, k time
    cdef Py_ssize_t i, j, k

    # for each time (different times are done in parallel)
    for k in prange(n_times, nogil=True):
        for i in range(amp_dim):
            for j in range(0, i+1):
                if i == j:
                    P_view[k] += amplitudes[i, j] * 0.5
                else:
                    P_view[k] += amplitudes[i, j] * cos(Ediff[i, j] * t[k])
    return P