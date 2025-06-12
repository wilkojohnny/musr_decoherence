import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange

# Tell Cython which NumPy types we’ll be using
ctypedef np.float64_t float64_t
ctypedef np.complex128_t complex128_t

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

@cython.boundscheck(False)
@cython.wraparound(False)
def minus_half_vector(double complex[:] R):
    cdef Py_ssize_t n_half = len(R) // 2
    cdef Py_ssize_t i
    for i in range(n_half):
        R[i] *= -1

cdef extern from "<complex.h>" nogil:
    double creal(double complex z)
    double cabs( double complex z)
    double complex conj(double complex z)
    double complex cexp(double complex z)

cdef extern from "<math.h>" nogil:
    double cos(double z)
    double fabs(double z)
    double exp(double z)


cpdef void dot(double complex[:] A, double complex[:, :] B, double complex[:, :] out):
    '''matrix multiply vector A (1 x m) and B (m x r)

    Parameters
    ----------
    A : memoryview (numpy array)
        1 x m vector 
    B : memoryview (numpy array)
        m x r right matrix
    out : memoryview (numpy array)
        n x r output matrix
    '''
    cdef Py_ssize_t j, k
    cdef double complex s
    cdef Py_ssize_t m = A.shape[0]
    cdef Py_ssize_t l = B.shape[0], r = B.shape[1]

    for j in range(r):
        s = 0
        for k in range(m):
            s += A[k]*B[k, j]

        out[0, j] = s

cpdef double complex vdot(double complex[:] A, double complex[:] B):
    '''dot product between two vectors

    Parameters
    ----------
    A : memoryview (numpy array)
        1 x m vector 
    B : memoryview (numpy array)
        m x 1 right vector
    out : memoryview (numpy array)
        n x r output matrix
    '''
    cdef Py_ssize_t j, k
    cdef double complex s
    cdef Py_ssize_t m = A.shape[0]

    s = 0
    for k in range(m):
        s += A[k]*B[k]

    return s

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

def compress_fourier(fourier_result: list, tol : double, del_tol =1e-7):
    """
    compress the fourier series
    """
    cdef int i = 0

    while i < len(fourier_result) - 1:
        # test for degeneracy
        if fabs(fourier_result[i][1] - fourier_result[i + 1][1]) < tol:
            # degenerate eigenvalue: add the amplitudes, keep frequency the same
            fourier_result[i] = (fourier_result[i][0] + fourier_result[i + 1][0], fourier_result[i][1])
            # remove the i+1th (degenerate) eigenvalue
            del fourier_result[i + 1]
        else:
            i = i + 1

    i = 0
    # now remove any amplitudes which are less than del_tol (don't need loads of zeros...)
    while i < len(fourier_result):
        if fabs(fourier_result[i][0]) < del_tol:
            # remove the entry
            del fourier_result[i]
        else:
            i = i + 1

    return fourier_result

def calculate_second_order(double complex[:, :] R, double complex[:, :] Rinv, double[:] E,
                           double B_var, double tau_c, double[:] t) -> complex:
    """
    Calculate the second order correction, assuming a fluctuating background field
    :param R: matrix of eigenvectors
    :param Rinv: eigenvector matrix conjugate transpose
    :param E: eigenvalues
    :param B_var: variance of the B-field
    :param tau_c: correlation time
    :param t: muon time
    """
    # reminder: pauli_x: roll, pauli_z = minus_half, pauli_y = roll/minus_half * 1j

    cdef Py_ssize_t hilbert_dim = R.shape[0]
    C_terms_lookup = np.zeros(shape=[hilbert_dim, hilbert_dim, hilbert_dim, 3, 3], dtype=np.complex128)
    C_filled = np.zeros_like(C_terms_lookup, dtype=bool)

    cdef double complex[:,:,:,:,:] C_lookup_view = C_terms_lookup

    def C_abg(Py_ssize_t alpha, Py_ssize_t beta, Py_ssize_t gamma, Py_ssize_t sigma, Py_ssize_t sigma_prime)\
            -> complex:
        """
        Calculate the C_{alpha, beta, gamma}^{sigma, sigma_prime} terms which effectively
        become the 'selection rules' of the interaction. All parameters are integers.
        """
        if C_filled[alpha, beta, gamma, sigma, sigma_prime]:
            # give lookup value
            return C_lookup_view[alpha, beta, gamma, sigma, sigma_prime]

        # create memory views
        cdef double complex[:] gamma_bra_view = Rinv[gamma, :]
        cdef double complex[:] alpha_ket_view = R[:, alpha]
        cdef double complex[:] sigma_alpha_ket_view = np.zeros(hilbert_dim, dtype=np.complex128)
        cdef double complex[:] alpha_bra_view = Rinv[alpha, :]
        cdef double complex[:] beta_ket_view = R[:, beta]
        cdef double complex[:] sigma_prime_beta_ket_view = np.zeros(hilbert_dim, dtype=np.complex128)

        cdef double complex gamma_sigma_alpha = 0.j
        cdef double complex alpha_sigma_prime_beta = 0.j

        # do sigma|alpha>
        if sigma == 0:
            # x
            for i in range(hilbert_dim):
                sigma_alpha_ket_view[i] = alpha_ket_view[(i - hilbert_dim // 2) % hilbert_dim]
        elif sigma == 1:
            # y
            for i in range(hilbert_dim):
                sigma_alpha_ket_view[i] = alpha_ket_view[(i - hilbert_dim // 2) % hilbert_dim]
            minus_half_vector(sigma_alpha_ket_view)
        else:
            # z
            sigma_alpha_ket_view[:] = alpha_ket_view[:]
            minus_half_vector(sigma_alpha_ket_view)

        # compute <gamma|sigma|alpha>
        gamma_sigma_alpha = vdot(gamma_bra_view, sigma_alpha_ket_view)

        if sigma == 1:
            gamma_sigma_alpha *= 1.0j

        if cabs(gamma_sigma_alpha) > 0:
            # do alpha_sigma'_beta
            if beta == gamma and sigma == sigma_prime:
                this_C = gamma_sigma_alpha * conj(gamma_sigma_alpha)
            else:
                # calculate <alpha|sigma'|beta>
                # do sigma|beta>
                if sigma_prime == 0:
                    # x
                    for i in range(hilbert_dim):
                        sigma_prime_beta_ket_view[i] = beta_ket_view[(i - hilbert_dim // 2) % hilbert_dim]
                elif sigma_prime == 1:
                    # y
                    for i in range(hilbert_dim):
                        sigma_prime_beta_ket_view[i] = beta_ket_view[(i - hilbert_dim // 2) % hilbert_dim]
                    minus_half_vector(sigma_prime_beta_ket_view)
                else:
                    # z
                    sigma_prime_beta_ket_view[:] = beta_ket_view[:]
                    minus_half_vector(sigma_prime_beta_ket_view)

                # compute <alpha|sigma'|beta>
                alpha_sigma_prime_beta = vdot(alpha_bra_view, sigma_prime_beta_ket_view)

                if sigma_prime == 1:
                    alpha_sigma_prime_beta *= 1.0j

                this_C = alpha_sigma_prime_beta * gamma_sigma_alpha
        else:
            this_C = 0+0j

        C_lookup_view[alpha, beta, gamma, sigma, sigma_prime] = this_C
        C_lookup_view[alpha, gamma, beta, sigma_prime, sigma] = conj(this_C)

        C_filled[alpha, beta, gamma, sigma, sigma_prime] = True
        C_filled[alpha, gamma, beta, sigma_prime, sigma] = True

        return this_C

    cdef Py_ssize_t i, k, n = hilbert_dim
    cdef double tau_c_inv = 1.0 / tau_c

    cdef double[:, :] E_diff = np.zeros((hilbert_dim, hilbert_dim), dtype=np.float64)
    for i in range(hilbert_dim):
        for k in range(hilbert_dim):
            E_diff[i, k] = E[i] - E[k]

    # now do F_ab
    cdef double complex[:, :] F_ab = np.zeros((hilbert_dim, hilbert_dim), dtype=np.complex128)
    for i in range(hilbert_dim):
        for k in range(hilbert_dim):
            F_ab[i, k] = 1.0 / (-1j * (E_diff[i,k]) + tau_c_inv)


    cdef Py_ssize_t sigma, sigma_prime, alpha, beta, gamma, delta, nt = t.shape[0]
    cdef double[:] e_tau_decay = np.zeros(nt)

    cdef double[:] sigma_sum = np.zeros(nt, dtype=np.float64)
    cdef double[:] sigma_prime_sum = np.zeros(nt, dtype=np.float64)

    cdef double complex c0, c1, c2
    cdef double complex c0_coeff
    cdef double complex[:] c1_term = np.zeros(nt, dtype=np.complex128)
    cdef double complex[:] c1_re_coeff = np.zeros(nt, dtype=np.complex128)
    cdef double complex[:] c1_im_coeff = np.zeros(nt, dtype=np.complex128)
    cdef double complex[:] c2_term = np.zeros(nt, dtype=np.complex128)
    cdef double complex[:] c2_re_coeff = np.zeros(nt, dtype=np.complex128)
    cdef double complex[:] c2_im_coeff = np.zeros(nt, dtype=np.complex128)
    cdef double denom

    cdef double complex[:] e_bgt = np.empty(nt, dtype=np.complex128)
    cdef double complex[:] e_gdt = np.empty(nt, dtype=np.complex128)

    for i in range(nt):
        e_tau_decay[i] = exp(-t[i]/tau_c)

    for sigma in range(3):
        for i in range(nt):
            sigma_prime_sum[i] = 0.
        for sigma_prime in range(3):
            for alpha in range(hilbert_dim):
                for beta in range(hilbert_dim):
                    for gamma in range(hilbert_dim):
                        c0 = C_abg(alpha, beta, gamma, sigma, sigma_prime)
                        if c0 == 0:
                            continue
                        c0_coeff = c0 * F_ab[alpha, beta]

                        # precalculate e_bgt
                        for i in range(nt):
                            e_bgt[i] = cexp(1j * t[i] * E_diff[beta, gamma])

                        denom = E_diff[alpha, gamma]
                        if denom != 0:
                            for i in range(nt):
                                c2_im_coeff[i] = 1j * (cexp(1j * t[i] * denom) - 1) / denom
                        else:
                            for i in range(nt):
                                c2_im_coeff[i] = -t[i]

                        for i in range(nt):
                            c2_re_coeff[i] = -F_ab[beta, gamma] * (e_bgt[i] * e_tau_decay[i] - 1)

                        for delta in range(hilbert_dim):
                            c1 = C_abg(gamma, delta, beta, sigma, sigma_prime)
                            c2 = C_abg(gamma, delta, beta, sigma_prime, sigma)
                            if c1 == 0 and c2 == 0:
                                continue

                            for i in range(nt):
                                # re-initalize c1_term and c2_term
                                c1_term[i] = 0
                                c2_term[i] = 0

                                e_gdt[i] = cexp(1j * t[i] * E_diff[gamma, delta])


                            if c1 != 0:
                                denom = E_diff[gamma, delta] + E_diff[alpha, beta]
                                if denom != 0:
                                    for i in range(nt):
                                        c1_im_coeff[i] = -1j * (cexp(1j * t[i] * denom) - 1) / denom
                                else:
                                    for i in range(nt):
                                        c1_im_coeff[i] = t[i]

                                for i in range(nt):
                                    c1_re_coeff[i] = F_ab[gamma, delta] * (e_gdt[i] * e_tau_decay[i] - 1)

                                for i in range(nt):
                                    c1_term[i] = c1 * e_bgt[i] * (c1_re_coeff[i] + c1_im_coeff[i])

                            if c2 != 0:
                                for i in range(nt):
                                    c2_term[i] = c2 * e_gdt[i] * (c2_re_coeff[i] + c2_im_coeff[i])

                            for i in range(nt):
                                sigma_prime_sum[i] += (c0_coeff * (c1_term[i] + c2_term[i])).real
            for i in range(nt):
                sigma_prime_sum[i] *= B_var
        for i in range(nt):
            sigma_sum[i] += sigma_prime_sum[i] / 3.0

    for i in range(nt):
        sigma_sum[i] *= 1.0 / hilbert_dim

    return sigma_sum

