"""
TimeDependence.py -- tools for calculating the time dependence of MuSR simulations
Used primarily by DipolarPolarisation.py
Created by John Wilkinson, 21st July 2020
"""

import numpy as np
import math


# calculate polarisation function for one specific time (used in the integration routine)
# this used to be in Hamiltonians.py
def calc_p_average_t(t, const, amplitude, E):
    # calculate the oscillating term
    osc_term = 0
    for isotope_combination in range(0, len(amplitude)):
        for i in range(0, len(E[isotope_combination])):
            for j in range(i + 1, len(E[isotope_combination])):
                osc_term = osc_term + amplitude[isotope_combination][i][j] * np.cos((E[isotope_combination][i]
                                                                                     - E[isotope_combination][j]) * t)

    # add on the constant, and return, and divide by the size of the space
    return const + osc_term


def calc_outer_differences_gpu(E):
    """
    Given a list E (on the CPU), calculate all the possible differences between the elements on the GPU
    :param E: list which contains the numbers which need subtracting from each other
    :return Edev, a matrix where Edev[i,j]=E[i]-E[j]
    """
    size = len(E)
    import cupy as cp
    # put E on the GPU
    E_dev = cp.array(E, dtype='float32')

    # make empty result matrix
    E_diff = cp.zeros((size, size), dtype='float32')

    # calculate the number of blocks etc
    if size < 16:
        threads_per_block = 4
    else:
        threads_per_block = 16
    blocks = math.ceil(size / threads_per_block)

    diff_kernel = cp.RawKernel(r'''
    extern "C"__global__
    void diff_kernel(float *devA, float *devC, int N) {
       __shared__ float shrA [''' + str(threads_per_block) + '''];
       __shared__ float shrB [''' + str(threads_per_block) + '''];

       int y, x, i, j;

       // Determine thread row y and column x within thread block.
       y = threadIdx.y;
       x = threadIdx.x;

       // Determine matrix element row i and column j.
       i = blockIdx.y*blockDim.y + y;
       j = blockIdx.x*blockDim.x + x;

       // Threads in first column of thread block copy chunk of vector A from
       // device global memory to thread block shared memory.
       if (x == 0) shrA[y] = devA[i];
       __syncthreads();

       // Threads in first row of thread block copy chunk of vector B from
       // device global memory to thread block shared memory.
       if (y == 0) shrB[x] = devA[j];
       __syncthreads();

       if (i<j) {
           // Each thread computes its own matrix element.
           devC[i*N + j] = shrA[y]-shrB[x];
       }
    }
    ''', 'diff_kernel')

    diff_kernel((blocks, blocks), (threads_per_block, threads_per_block), (E_dev, E_diff, size))

    return E_diff


def calc_oscillating_term_gpu(E_diff_gpu, A_gpu, size, t):
    """
    Calculate the oscillating terms sum_ij(A[i][j]*cos((E[i]-E[j])t))
    :param E_gpu: matrix where E[i,j] = E[i]-E[j]
    :param A_gpu: matrix of the amplitudes, where A[i,j] is the amplitude of E_diff[i,j]
    :param size: size of E originally (sqrt(n elements in A))
    :param time: time this is evaluated at
    :return float -- the oscillating term at time t
    """

    import cupy as cp

    # calculate the number of blocks etc
    if size < 16:
        threads_per_block = 4
    else:
        threads_per_block = 16
    blocks = math.ceil(size / threads_per_block)

    p = cp.zeros((size, size), dtype='float32')

    osc_kernel = cp.RawKernel(r'''
    extern "C"__global__
    void osc_kernel(float *Amp, float *EDiff, int N, double t, float *p) {
       __shared__ float shrA [''' + str(threads_per_block) + '''];
       __shared__ float shrB [''' + str(threads_per_block) + '''];

       int y, x, i, j;

       // Determine thread row y and column x within thread block.
       y = threadIdx.y;
       x = threadIdx.x;
       // Determine matrix element row i and column j.
       i = blockIdx.y*blockDim.y + y;
       j = blockIdx.x*blockDim.x + x;
       
       if (i<j) {
            p[i*N + j] = Amp[i*N+j]*cosf(EDiff[i*N+j]*t);
       }
    }
    ''', 'osc_kernel')

    osc_kernel((blocks, blocks), (threads_per_block, threads_per_block), (A_gpu, E_diff_gpu, size, t, p))

    return cp.sum(cp.sum(p))
