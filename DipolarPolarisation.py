# DipolePolarisation.py -- calculates the muon polarisation function (in terms of time or frequency) for lots of
# spins (in the style of F--\mu--F)
# John Wilkinson 15/11/19

import functools
print = functools.partial(print, flush=True)

import subprocess  # gets git version
from datetime import datetime  # allows one to print out date and time
import Hamiltonians # allows one to calculate Hamiltonians
import TimeDependence
from MDecoherenceAtom import TDecoherenceAtom as atom  # for atoms
import time as human_time
import TCoord3D as coord  # coordinate utilities
import numpy.linalg as linalg  # matrix stuff
import numpy as np  # for numpy arrays
import math
no_plot = False
try:
    import matplotlib.pyplot as pyplot  # plotting
except ModuleNotFoundError:
    no_plot = True
import os  #


# do decoherence file preamble
def decoherence_file_preamble(file, nn_atoms, muon, fourier, starttime=None, endtime=None, timestep=None,
                              fourier_2d=None, tol=None):
    # program name, date and time completed
    file.writelines('! Decoherence Calculator Output - ' + datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + '\n!\n')

    # get the git version
    script_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        version_label = subprocess.check_output(["git", "describe", "--always"], cwd=script_dir).strip()
    except subprocess.CalledProcessError:
        version_label = '(version not available)'

    file.writelines('! Using version ' + str(version_label) + '\n!\n')

    # type of calculation
    if not fourier:
        file.writelines('! time calculation completed between t=' + str(starttime) + ' and ' + str(endtime) +
                        ' with a timestep of ' + str(timestep) + ' microseconds' + '\n!\n')
    else:
        if fourier_2d:
            file.writelines('! 2D fourier calculation, showing the amplitude between each transition pair. \n')
        else:
            file.writelines('! 1D fourier calculation, showing the amplitude of each E_i-E_j combination \n')
        file.writelines('! absolute tolerance between eigenvalues to treat them as equivalent was ' + str(tol)
                        + '\n!\n')

    file.writelines('! Muon position: ' + str(muon.position) + '\n')

    for atom in nn_atoms:
        lines_to_write = atom.verbose_description(gle_friendly=True)
        for line in lines_to_write:
            file.writelines(line)
    file.writelines('!\n')

    file.writelines('! start of data: \n')


# batch write data to file
def write_to_file(file, t, P):
    for i in range(0, len(t) - 1):
        file.writelines(str(t[i]) + ' ' + str(P[i]) + '\n')


# increment isotope id
def inc_isotope_id(basis, oldids=None):
    # if no ids supplied, just give a load of 0s
    if oldids is None:
        return [0 for xx in basis]
    else:
        # try to increase the first isotopeid by 1, if greater, then increment the next, etc
        for i in range(0, len(basis)):
            oldids[i] = oldids[i] + 1
            if oldids[i] < basis[i]:
                break
            else:
                oldids[i] = 0
        # if we've got this far and we're still at [0,0,...], make the first term negative
        if sum(oldids) == 0:  # sum just sees if its all 0, since we should never get anything negative
            oldids[0] = -1
        return oldids


def calc_dipolar_polarisation(all_spins: list, muon: atom, muon_sample_polarisation: coord = None,
                              times: np.ndarray = np.arange(0, 10, 0.1), do_quadrupoles=False, just_muon_interactions=False,
                              # other arguments
                              fourier: bool = False, fourier_2d: bool = False, outfile_location: str = None, tol: float = 1e-10,
                              plot: bool = False, shutup: bool = False, gpu: bool = False):
    '''
    :param all_spins: array of the spins
    :param muon:
    :param muon_sample_polarisation:
    :param times:
    :param fourier:
    :param fourier_2d:
    :param outfile_location:
    :param tol:
    :param plot:
    :param shutup:
    :param gpu: use GPU (requires cupy)
    :return:
    '''

    if not shutup:
        for atom in all_spins:
            print(atom)

    old_gpu = True

    if gpu:
        try:
            import cupy as cp
            if int(cp.cuda.device.Device(0).compute_capability) >= 30:
                print('💰 Rich user detected. Will use the GPU to its full potential.')
                old_gpu = False
            else:
                print('👴🏻 Grandpa-GPU mode activated. Will only utilise basic GPU optimisation.')
                old_gpu = True
        except ImportError:
            print('Can\'t find CuPy module. Have you set up CUDA?')
            gpu = False

    # type of calculation - can't do fourier2d if not fourier
    fourier_2d = fourier_2d and fourier

    # count number of spins
    N_spins = len(all_spins) - 1

    # count the number of combinations of isotopes
    isotope_combinations = 1
    for atoms in all_spins:
        isotope_combinations = isotope_combinations * len(atoms)
    if not shutup:
        print(str(isotope_combinations) + ' isotope combination(s) found')

    # put all these number of isotopes into an array
    number_isotopes = [len(atom) for atom in all_spins]

    current_isotope_ids = inc_isotope_id(basis=number_isotopes)

    # create frequency and amplitude arrays
    E = list()
    amplitude = list()
    const = 0
    while current_isotope_ids[0] != -1:  # the end signal is emitted by making the id of 0 = -1
        # put this combination of isotopes into an array (Spins), and calculate probability of this state
        probability = 1.
        Spins = []
        for atomid in range(0, len(all_spins)):
            Spins.append(all_spins[atomid][current_isotope_ids[atomid]])
            probability = probability * all_spins[atomid][current_isotope_ids[atomid]].abundance

        hilbert_dim = 1
        for spin in Spins:
            hilbert_dim *= spin.II + 1

        if not gpu:
            # create measurement operators for the muon's spin
            muon_spin_x = 2*Hamiltonians.measure_ith_spin(Spins, 0, Spins[0].pauli_x)
            muon_spin_y = 2*Hamiltonians.measure_ith_spin(Spins, 0, Spins[0].pauli_y)
            muon_spin_z = 2*Hamiltonians.measure_ith_spin(Spins, 0, Spins[0].pauli_z)

        start_time = human_time.time()

        # calculate hamiltonian
        hamiltonian = Hamiltonians.calc_dipolar_hamiltonian(Spins, just_muon_interactions=just_muon_interactions)

        if do_quadrupoles:
            hamiltonian += Hamiltonians.calc_quadrupolar_hamiltonian(Spins)

        del Spins

        # find eigenvalues and eigenvectors of hamiltonian
        if not shutup:
            print("Finding eigenvalues...")
        dense_hamiltonian = hamiltonian.todense()
        if gpu:
            dense_hamiltonian = cp.array(dense_hamiltonian, dtype='csingle')
            this_E, R = cp.linalg.eigh(dense_hamiltonian)
            R = R.transpose()
            Rinv = cp.conj(R)
        else:
            this_E, R = linalg.eigh(dense_hamiltonian)
            Rinv = R.H
        if not shutup:
            print("Found eigenvalues:")
            print(this_E)

        # now upload the muon spin matrices to the GPU (making them sparse if possible)
        # if gpu and not old_gpu:
        #     muon_spin_x = cp.sparse.csr_matrix(muon_spin_x, dtype='csingle')
        #     muon_spin_y = cp.sparse.csr_matrix(muon_spin_y, dtype='csingle')
        #     muon_spin_z = cp.sparse.csr_matrix(muon_spin_z, dtype='csingle')
        # elif gpu and old_gpu:
            # muon_spin_x = cp.array(muon_spin_x.todense(), dtype='csingle')
            # muon_spin_y = cp.array(muon_spin_y.todense(), dtype='csingle')
            # muon_spin_z = cp.array(muon_spin_z.todense(), dtype='csingle')

        # weights -- if single crystal, use that; otherwise use 1/sqrt(3) for each
        wx, wy, wz = (1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3))
        if muon_sample_polarisation is not None:
            wx, wy, wz = muon_sample_polarisation.totuple()

        # Calculate constant (lab book 1 page 105)
        thisconst = 0
        this_amplitude = np.zeros((len(R), len(R)))
        for i in range(0, len(R)):
            if gpu:
                # sx = cp.matmul(Rinv[i], muon_spin_x.dot(R[:, i]))
                # sy = cp.matmul(Rinv[i], muon_spin_y.dot(R[:, i]))
                # sz = cp.matmul(Rinv[i], muon_spin_z.dot(R[:, i]))
                R_swap = cp.concatenate((R[i][int(hilbert_dim/2):hilbert_dim], R[i][0:int(hilbert_dim/2)]))
                (sx, sy, sz) = calc_amplitudes_gpu(R[i], Rinv[i], R_swap, size=hilbert_dim)
            else:
                Rx = Rinv[i] * muon_spin_x
                Ry = Rinv[i] * muon_spin_y
                Rz = Rinv[i] * muon_spin_z
                sx = Rx * R[:, i]
                sy = Ry * R[:, i]
                sz = Rz * R[:, i]
            # angular average mode
            thisconst = thisconst + pow(abs(sx)*wx, 2) + pow(abs(sy)*wy, 2) + pow(abs(sz)*wz, 2)

            if not shutup:
                print(str(100 * i / len(R)) + '% complete...')
            if fourier_2d:
                jmin = 0
            else:
                jmin = i + 1
            for j in range(jmin, len(R)):
                if gpu:
                    # sx = cp.matmul(Rinv[j], Rx)
                    # sy = cp.matmul(Rinv[j], Ry)
                    # sz = cp.matmul(Rinv[j], Rz)
                    (sx, sy, sz) = calc_amplitudes_gpu(R[i], Rinv[j], R_swap, hilbert_dim)
                    # sx = cp.matmul(Rx, R[:, j])
                    # sy = cp.matmul(Ry, R[:, j])
                    # sz = cp.matmul(Rz, R[:, j])
                else:
                    sx = Rx * R[:, j]
                    sy = Ry * R[:, j]
                    sz = Rz * R[:, j]
                # do angular averaging
                this_amplitude[i][j] = (pow(abs(sx)*wx, 2) + pow(abs(sy)*wy, 2) + pow(abs(sz)*wz, 2)) \
                                        * probability / (hilbert_dim / 2)

        const = const + probability * thisconst / (2 * (hilbert_dim / 2))
        amplitude.append(this_amplitude.tolist())
        E.append(this_E.tolist())

        # increment the isotope ids
        current_isotope_ids = inc_isotope_id(basis=number_isotopes, oldids=current_isotope_ids)

    ## OUTPUT ##

    if fourier:

        # dump all into an array
        fourier_result = []

        # for each isotope
        for isotope_combination in range(0, len(amplitude)):
            # noinspection PyTypeChecker
            for i in range(0, len(E[isotope_combination])):
                if fourier_2d:
                    # noinspection PyTypeChecker
                    for j in range(0, len(E[isotope_combination])):
                        fourier_result.append((amplitude[isotope_combination][i][j], E[isotope_combination][i],
                                               E[isotope_combination][j]))
                else:
                    # noinspection PyTypeChecker
                    for j in range(i + 1, len(E[isotope_combination])):
                        fourier_result.append((amplitude[isotope_combination][i][j],
                                               abs(E[isotope_combination][i] - E[isotope_combination][j])))

        # go through the frequencies, if there's degenerate eigenvalues then add together the amplitudes
        if fourier_2d:
            fourier_result = sorted(fourier_result, key=lambda frequency: (frequency[1], frequency[2]))
            i = 0
            while i < len(fourier_result) - 1:
                # test for degeneracy (up to a tolerance for machine precision)
                if (abs((fourier_result[i][1]) - (fourier_result[i + 1][1])) < tol) \
                        and (abs(fourier_result[i][2] - fourier_result[i + 1][2]) < tol):
                    # degenerate eigenvalue: add the amplitudes, keep frequency the same
                    fourier_result[i] = (fourier_result[i][0] + fourier_result[i + 1][0],
                                         fourier_result[i][1], fourier_result[i][2])
                    # remove the i+1th (degenerate) eigenvalue
                    del fourier_result[i + 1]
                else:
                    i = i + 1
            # and sort and dedegenerate again...
            fourier_result = sorted(fourier_result, key=lambda frequency: (frequency[2], frequency[1]))
            i = 0
            while i < len(fourier_result) - 1:
                # test for degeneracy (up to a tolerance for machine precision)
                if (abs(fourier_result[i][1] - fourier_result[i + 1][1]) < tol) \
                        and (abs(fourier_result[i][2] - fourier_result[i + 1][2]) < tol):
                    # degenerate eigenvalue: add the amplitudes, keep frequency the same
                    fourier_result[i] = (fourier_result[i][0] + fourier_result[i + 1][0],
                                         fourier_result[i][1], fourier_result[i][2])
                    # remove the i+1th (degenerate) eigenvalue
                    del fourier_result[i + 1]
                else:
                    i = i + 1
        else:
            fourier_result = sorted(fourier_result, key=lambda frequency: frequency[1])
            i = 0
            while i < len(fourier_result) - 1:
                # test for degeneracy (up to a tolerance for machine precision)
                if abs((fourier_result[i][1]) - (fourier_result[i + 1][1])) < tol:
                    # degenerate eigenvalue: add the amplitudes, keep frequency the same
                    fourier_result[i] = (fourier_result[i][0] + fourier_result[i + 1][0], fourier_result[i][1])
                    # remove the i+1th (degenerate) eigenvalue
                    del fourier_result[i + 1]
                else:
                    i = i + 1

            i = 0
            # now remove any amplitudes which are less than 1e-15
            while i < len(fourier_result) - 1:
                if abs(fourier_result[i][0]) < 1e-7:
                    # remove the entry
                    del fourier_result[i]
                else:
                    i = i + 1

        # dump into file if requested
        if outfile_location is not None:
            outfile = open(outfile_location, "w")
            # do preamble
            decoherence_file_preamble(file=outfile, nn_atoms=all_spins, muon=muon, fourier=fourier,
                                      fourier_2d=fourier_2d, tol=tol)
            if fourier_2d:
                outfile.writelines('! frequency1 frequency2 amplitude \n')
                outfile.writelines([str(fourier_entry[1]) + ' ' + str(fourier_entry[2]) + ' ' + str(fourier_entry[0])
                                    + '\n' for fourier_entry in fourier_result])
            else:
                outfile.writelines('! frequency amplitude \n')
                outfile.writelines('0 ' + str(const[0, 0]) + '\n')
                outfile.writelines([str(fourier_entry[1]) + ' ' + str(fourier_entry[0]) + '\n' for fourier_entry
                                    in fourier_result])
            outfile.close()

        return np.array(fourier_result)
    else:

        P_average = []

        # calculate each time separately
        if not gpu:
            for time in np.nditer(times):
                if not shutup:
                    print("t=" + str(time))
                P_average.append(TimeDependence.calc_p_average_t(time, const, amplitude, E).max())
        else:
            # calculate the differences E[i]-E[j] and put into matrix on the GPU
            E_diff_device = TimeDependence.calc_outer_differences_gpu(E[0])

            # now upload the amplitudes onto the device
            amplitude_device = cp.asarray(amplitude[0], dtype='float32')

            for time in times:
                if not shutup:
                    print("t=" + str(time))
                P_average.append(TimeDependence.calc_oscillating_term_gpu(E_diff_device,
                                                                          amplitude_device,
                                                                          len(E[0]),
                                                                          time) + const)

        if not shutup:
            print("elapsed time: " + str(human_time.time() - start_time))

        if outfile_location is not None:
            # dump results in a file if requested
            outfile = open(outfile_location, "w")
            # do preamble
            decoherence_file_preamble(file=outfile, nn_atoms=all_spins, muon=muon, fourier=fourier,
                                      fourier_2d=fourier_2d, tol=tol, starttime=times[0], endtime=times[-1],
                                      timestep=times[1] - times[0])
            outfile.writelines('! t P_average \n')
            write_to_file(outfile, times, P_average)
            outfile.close()

        # plot the angular averaged muon polarisation
        if plot:
            pyplot.plot(times, P_average)
            pyplot.title('Muon Polarisation')
            pyplot.xlabel('time (microseconds)')
            pyplot.ylabel('Muon Polarisation')
            pyplot.show()

        return np.array(P_average)


def calc_amplitudes_gpu(R_i, Rinv_i, R_swap, size):
    """
    calculate the amplitudes of the interactions between eigenstates
    :param R: eigenvectors of the Hamiltonian
    :param Rinv: conj eigenvectors of the Hamiltonian
    :param weights: weights wx, wy, wz corresponding to whether a polycrystaline average or not
    :param size: size of the Hilbert space
    :return 2D list where a[i][j] is the amplitude of state E[i]-E[j]
    """

    import cupy as cp

    # calculate the number of blocks etc
    if size < 16:
        threads_per_block = 4
    else:
        threads_per_block = 16
    blocks = math.ceil(size / threads_per_block)

    a = cp.zeros((size), dtype='csingle')
    sz_kernel = cp.RawKernel(r'''
        #include <cupy/complex.cuh>
        extern "C"__global__
        void sz_kernel(const complex<float> *R, const complex<float> *Rinv, int N,
         complex<float> *a) {
        
           int x, i;

           // Determine thread row x within thread block.
           x = threadIdx.x;
           i = blockIdx.x*blockDim.x + x;

           if (i<N) {
            if (i<N/2) {
                a[i] = -Rinv[i]*R[i];
            } else {
                a[i] = Rinv[i]*R[i];
            }
           }
        }
        ''', 'sz_kernel')

    sz_kernel((blocks, ), (threads_per_block, ), (R_i, Rinv_i, size, a))
    sz = cp.sum(a)

    a = cp.zeros((size), dtype='csingle')
    sx_kernel = cp.RawKernel(r'''
            #include <cupy/complex.cuh>
            extern "C"__global__
            void sx_kernel(const complex<float> *R, const complex<float> *Rinv, int N,
             complex<float> *a) {

               int x, i;

               // Determine thread row x within thread block.
               x = threadIdx.x;
               i = blockIdx.x*blockDim.x + x;

               if (i<N) {
                    a[i] = Rinv[i]*R[i];
               }
            }
            ''', 'sx_kernel')
    sx_kernel((blocks, ), (threads_per_block, ), (R_swap, Rinv_i, size, a))
    sx = cp.sum(a)

    a = cp.zeros((size), dtype='csingle')
    sy_kernel = cp.RawKernel(r'''
            #include <cupy/complex.cuh>
            extern "C"__global__
            void sy_kernel(const complex<float> *R, const complex<float> *Rinv, int N,
             complex<float> *a) {
                
               // define 1j
               complex<float> jj(0.,1.);
                
               int x, i;

               // Determine thread row x within thread block.
               x = threadIdx.x;
               i = blockIdx.x*blockDim.x + x;

               if (i<N) {
                    if (i<N/2) {
                       a[i] = Rinv[i]*R[i];
                    } else {
                        a[i] = -Rinv[i]*R[i];
                    }
                    a[i] = jj*a[i];
               }
            }
            ''', 'sy_kernel')
    sy_kernel((blocks, ), (threads_per_block, ), (R_swap, Rinv_i, size, a))
    sy = cp.sum(a)


    return (sx, sy, sz)


