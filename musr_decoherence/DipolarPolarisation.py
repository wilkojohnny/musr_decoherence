# DipolePolarisation.py -- calculates the muon polarisation function (in terms of time or frequency) for lots of
# spins (in the style of F--\mu--F)
# John Wilkinson 15/11/19

import functools

print = functools.partial(print, flush=True)

import subprocess  # gets git version
from datetime import datetime  # allows one to print out date and time
from . import Hamiltonians  # allows one to calculate Hamiltonians
from . import TimeDependence
from .MDecoherenceAtom import TDecoherenceAtom as atom  # for atoms
import time as human_time
from .TCoord3D import TCoord3D as coord  # coordinate utilities
import scipy.linalg as linalg  # matrix stuff
import numpy as np  # for numpy arrays
import math
from tqdm import tqdm

no_plot = False
try:
    import matplotlib.pyplot as pyplot  # plotting
except ModuleNotFoundError:
    no_plot = True
import os
from enum import Enum
from . import cython_polarisation


class musr_type(Enum):
    ZF = 0
    zero_field = 0
    LF = 1
    longitudinal_field = 1
    TF = 2
    transverse_field = 2


# do decoherence file preamble
def decoherence_file_preamble(file, nn_atoms, muon, fourier, musr_type, field=0, starttime=None, endtime=None,
                              timestep=None, fourier_2d=None, tol=None):
    # program name, date and time completed
    file.writelines('! Decoherence Calculator Output - ' + datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + '\n!\n')

    # get the git version
    script_dir = os.path.dirname(os.path.realpath(__file__))
    try:
        version_label = subprocess.check_output(["git", "describe", "--always"], cwd=script_dir).strip()
    except subprocess.CalledProcessError:
        version_label = '(version not available)'
    except FileNotFoundError:
        version_label = '(version not available)'

    file.writelines('! Using version ' + str(version_label) + '\n!\n')

    file.writelines('! Calculated for ' + str(musr_type) + '-MUSR, with a field of ' + str(field) + 'Gauss\n!\n')

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
                              times: np.ndarray = np.arange(0, 10, 0.1), musr_type: musr_type = musr_type.zero_field,
                              field=0, do_quadrupoles=False, just_muon_interactions=False,
                              # other arguments
                              fourier: bool = False, fourier_2d: bool = False, outfile_location: str = None,
                              tol: float = 1e-10,
                              plot: bool = False, shutup: bool = False, gpu: bool = False,
                              include_first_order_dynamics=False, include_second_order_dynamics=False,
                              tau_c=None, B_var=np.array((0, 0, 0))
                              ):
    '''
    :param all_spins: array of the spins
    :param muon:
    :param muon_sample_polarisation:
    :param times:
    :param musr_type:
    :param do_quadrupoles:
    :param just_muon_interactions:
    :param field:
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

    polycrystalline = muon_sample_polarisation is None

    if gpu:
        try:
            import cupy as cp
        except ImportError:
            print('Can\'t find CuPy module. Have you set up CUDA? Try running pip install ./[gpu]')
            gpu = False

    # type of calculation - can't do fourier2d if not fourier
    fourier_2d = fourier_2d and fourier

    # covert the field from Gauss to Tesla
    field_tesla = field * 1e-4

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
    P_average = np.zeros(shape=times.shape)
    while current_isotope_ids[0] != -1:  # the end signal is emitted by making the id of 0 = -1
        # put this combination of isotopes into an array (Spins), and calculate probability of this state
        probability = 1.
        Spins = []
        this_E = None
        this_amplitude = None

        # if we are doing more than one isotope combination, then change the probability
        if isotope_combinations > 1:
            for atomid in range(0, len(all_spins)):
                Spins.append(all_spins[atomid][current_isotope_ids[atomid]])
                probability = probability * all_spins[atomid][current_isotope_ids[atomid]].abundance
        else:
            Spins = all_spins

        hilbert_dim = 1
        for spin in Spins:
            hilbert_dim *= spin.II + 1

        if not gpu:
            # create measurement operators for the muon's spin
            muon_spin_x = 2 * Hamiltonians.measure_ith_spin(Spins, 0, Spins[0].pauli_x)
            muon_spin_y = 2 * Hamiltonians.measure_ith_spin(Spins, 0, Spins[0].pauli_y)
            muon_spin_z = 2 * Hamiltonians.measure_ith_spin(Spins, 0, Spins[0].pauli_z)
        else:
            # we don't use the pauli_mu to calculate this when we use GPUs...
            muon_spin_x, muon_spin_y, muon_spin_z = None, None, None

        start_time = human_time.time()

        # calculate dipolar hamiltonian
        hamiltonian = Hamiltonians.calc_dipolar_hamiltonian(Spins, just_muon_interactions=just_muon_interactions)

        # add on the quadrupoles if wanted
        if do_quadrupoles:
            hamiltonian += Hamiltonians.calc_quadrupolar_hamiltonian(Spins)

        # if any of the atoms have an INTERNAL magnetic field, then add the Zeeman on for that
        # (external dealt with later)
        for i_spin, this_spin in enumerate(Spins):
            if this_spin.field is not None:
                hamiltonian += Hamiltonians.calc_zeeman_hamiltonian_term(Spins, this_spin.field * 1e-4, i_spin)

        # weights -- if single crystal, use that; otherwise use 1/sqrt(3) for each
        if not polycrystalline:
            # single crystal sample
            wx, wy, wz = muon_sample_polarisation.totuple()
            if musr_type == musr_type.LF:
                hamiltonian += Hamiltonians.calc_zeeman_hamiltonian(Spins, coord(wx, wy, wz) * field_tesla)
            elif musr_type == musr_type.TF:
                field_direction = coord(wx, wy, wz).get_perpendicular_vector(normalise=True)
                hamiltonian += Hamiltonians.calc_zeeman_hamiltonian(Spins, field_direction * field_tesla)

            # now calculate the polarisation or fourier components
            this_pol, this_E, _, this_amplitude = calc_hamiltonian_polarisation(hamiltonian, times,
                                                                                weights=(wx, wy, wz),
                                                                                fourier=fourier, fourier_2d=fourier_2d,
                                                                                muon_spin_matrices=(muon_spin_x,
                                                                                                    muon_spin_y,
                                                                                                    muon_spin_z),
                                                                                const=const, probability=probability,
                                                                                hilbert_dim=hilbert_dim, gpu=gpu,
                                                                                shutup=shutup)
            if this_pol is not None:
                P_average += this_pol


        else:
            # polycrystalline sample
            if musr_type == musr_type.zero_field:
                # calculate the polarisation or fourier components
                this_pol, this_E, this_R, this_amplitude = calc_hamiltonian_polarisation(hamiltonian, times,
                                                                                         weights=(None, None, None),
                                                                                         fourier=fourier,
                                                                                         fourier_2d=fourier_2d,
                                                                                         muon_spin_matrices=(
                                                                                             muon_spin_x,
                                                                                             muon_spin_y,
                                                                                             muon_spin_z),
                                                                                         const=const,
                                                                                         probability=probability,
                                                                                         hilbert_dim=hilbert_dim,
                                                                                         gpu=gpu,
                                                                                         shutup=shutup)

                # if first order perturbation is on, calculate this
                if include_first_order_dynamics:
                    for i_t, t in enumerate(times):
                        pert_z = np.real(
                            calc_polarisation_with_field_perturbation_integrand(all_spins, this_E, this_R,
                                                                                np.arange(0, 20, 1e-3), t,
                                                                                np.array((0, 0, 1)))) / 3
                        pert_x = np.real(
                            calc_polarisation_with_field_perturbation_integrand(all_spins, this_E, this_R,
                                                                                np.arange(0, 20, 1e-3), t,
                                                                                np.array((1, 0, 0)))) / 3
                        pert_y = np.real(
                            calc_polarisation_with_field_perturbation_integrand(all_spins, this_E, this_R,
                                                                                np.arange(0, 20, 1e-3), t,
                                                                                np.array((0, 1, 0)))) / 3
                        # print(np.max(pert_z))
                        # print(np.max(pert_x))
                        # print(np.max(pert_y))

                        # this_pol[i_t] += pert_z
                        # this_pol[i_t] += pert_y
                        # this_pol[i_t] += pert_x

                # if second order perturbation is on, calculate it
                if include_second_order_dynamics:
                    this_pol += calc_polarisation_with_field_perturbation_2ndorder(all_spins, this_E, this_R,
                                                                                   B_var,
                                                                                   tau_c, times, None)

                if this_pol is not None:
                    P_average = P_average + this_pol

            else:
                d_theta = math.pi / 7
                d_phi = math.pi / 7
                N_theta = math.pi / d_theta
                N_phi = 2 * math.pi / d_phi
                normalisation_factor = N_phi * math.sin(N_theta * d_theta / 2) * \
                                       math.sin((N_theta - 1) * d_theta / 2) / math.sin(d_theta / 2)
                for theta in np.arange(d_theta, math.pi, d_theta):
                    for phi in np.arange(0, 2 * math.pi, d_phi):
                        if not shutup:
                            print('theta: ' + '{:4f}'.format(theta) + '\t phi: ' + '{:4f}'.format(phi))
                        wx, wy, wz = math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)
                        if musr_type == musr_type.LF:
                            field_direction = coord(wx, wy, wz)
                        elif musr_type == musr_type.TF:
                            field_direction = coord(math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi),
                                                    -math.sin(theta))
                        current_hamiltonian = hamiltonian + Hamiltonians.calc_zeeman_hamiltonian(Spins,
                                                                                                 field_direction
                                                                                                 * field_tesla)
                        # calculate the polarisation or fourier components
                        this_pol, this_E, _, this_amplitude_ang = calc_hamiltonian_polarisation(current_hamiltonian,
                                                                                                times,
                                                                                                weights=(wx, wy, wz),
                                                                                                fourier=fourier,
                                                                                                fourier_2d=fourier_2d,
                                                                                                muon_spin_matrices= \
                                                                                                    (muon_spin_x,
                                                                                                     muon_spin_y,
                                                                                                     muon_spin_z),
                                                                                                const=const,
                                                                                                probability=probability,
                                                                                                hilbert_dim=hilbert_dim,
                                                                                                gpu=gpu, shutup=True)

                        if fourier:
                            if this_amplitude is None:
                                this_amplitude = np.zeros(shape=this_amplitude_ang.shape)
                            this_amplitude += this_amplitude_ang * math.sin(theta) / normalisation_factor
                        else:
                            P_average += this_pol * math.sin(theta) / normalisation_factor

        if fourier:
            E.append(this_E)
            amplitude.append(this_amplitude)

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
            fourier_result = cython_polarisation.compress_fourier(fourier_result, tol, 1e-7)

        # dump into file if requested
        if outfile_location is not None:
            outfile = open(outfile_location, "w")
            # do preamble
            decoherence_file_preamble(file=outfile, nn_atoms=all_spins, muon=muon, fourier=fourier,
                                      fourier_2d=fourier_2d, tol=tol, musr_type=musr_type)
            if fourier_2d:
                outfile.writelines('! frequency1 frequency2 amplitude \n')
                outfile.writelines([str(fourier_entry[1]) + ' ' + str(fourier_entry[2]) + ' ' + str(fourier_entry[0])
                                    + '\n' for fourier_entry in fourier_result])
            else:
                outfile.writelines('! frequency amplitude \n')
                outfile.writelines('0 ' + str(1 - [sum(i) for i in zip(*fourier_result)][0]) + '\n')
                outfile.writelines([str(fourier_entry[1]) + ' ' + str(fourier_entry[0]) + '\n' for fourier_entry
                                    in fourier_result])
            outfile.close()

        return np.array(fourier_result)
    else:

        if not shutup:
            print("elapsed time: " + str(human_time.time() - start_time))

        if outfile_location is not None:
            # dump results in a file if requested
            outfile = open(outfile_location, "w")
            # do preamble
            decoherence_file_preamble(file=outfile, nn_atoms=all_spins, muon=muon, fourier=fourier,
                                      fourier_2d=fourier_2d, tol=tol, starttime=times[0], endtime=times[-1],
                                      timestep=times[1] - times[0], musr_type=musr_type, field=field)
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

        if gpu:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()

        return np.array(P_average)


def calc_hamiltonian_polarisation(hamiltonian, times, weights, fourier, fourier_2d, muon_spin_matrices, const,
                                  probability,
                                  hilbert_dim, gpu=False, shutup=False):
    """
    calculate the polarisation from a Hamiltonian
    :param hamiltonian: the Hamiltonian to use
    :param times: numpy array of the times to calculate for
    :param gpu: if True, use GPU-accelleration
    return: P_average, eigenvalues, eigenvectors, amplutides
    """

    if gpu:
        try:
            import cupy as cp
        except ImportError:
            print('Can\'t find CuPy module. Have you set up CUDA?')
            gpu = False

    muon_spin_x, muon_spin_y, muon_spin_z = muon_spin_matrices
    wx, wy, wz = weights

    # find eigenvalues and eigenvectors of hamiltonian
    if not shutup:
        print("Finding eigenvalues...")
    dense_hamiltonian = hamiltonian.todense()
    if gpu:
        dense_hamiltonian = cp.array(dense_hamiltonian, dtype='csingle')
        this_E, R = cp.linalg.eigh(dense_hamiltonian)
        del dense_hamiltonian
        Rinv = R.transpose().conj()
    else:
        this_E, R = linalg.eigh(dense_hamiltonian, overwrite_a=True)
        del dense_hamiltonian
        Rinv = R.transpose().conj()
        R = R.copy(order='C')
    if not shutup:
        print("Found eigenvalues:")
        print(this_E)

    if not gpu:
        R_roll = np.roll(R, int(hilbert_dim / 2), 0)

        if not shutup:
            print('Calculating amplitudes...')
        this_amplitude = calc_amplitudes_cpu(R, Rinv, R_roll, (wx, wy, wz), hilbert_dim)
        del R_roll, Rinv
        if not shutup:
            print('Calculated amplitudes')
    else:
        R_roll = cp.roll(R, int(hilbert_dim / 2), 0)

        this_amplitude = calc_amplitudes_gpu(R, Rinv, R_roll, (wx, wy, wz), hilbert_dim)

        del Rinv, R_roll

    if not fourier:
        P_average = np.zeros(shape=times.shape)

        if not gpu:
            # for i_time, time in np.ndenumerate(times):
            #     if not shutup:
            #         print("t=" + str(time))
            #     P_average[i_time] += TimeDependence.calc_p_average_t(time, const, this_amplitude, this_E).max() \
            #                          * probability

            # calculate the differences E[i]-E[j] and put into matrix
            Ediff = np.subtract.outer(this_E, this_E)
            P_average = cython_polarisation.calc_oscillation(this_amplitude, Ediff, times) * probability
            del Ediff, this_amplitude
        else:
            # calculate the differences E[i]-E[j] and put into matrix on the GPU
            E_diff_device = TimeDependence.calc_outer_differences_gpu(this_E)

            # now upload the amplitudes onto the device
            amplitude_device = cp.asarray(this_amplitude, dtype='float32')

            del this_amplitude

            for i_time, time in np.ndenumerate(times):
                if not shutup:
                    print("t=" + str(time))
                P_average[i_time] += TimeDependence.calc_oscillating_term_gpu(E_diff_device, amplitude_device,
                                                                              len(this_E), time) * probability

            del E_diff_device
            del amplitude_device

        return P_average, this_E, R, None
    else:
        return None, this_E, R, this_amplitude


def calc_amplitudes_cpu(R, Rinv, R_roll, weights, size):
    """
    calculate the amplitudes using the CPU + Cython
    :param R: eigenvectors of the Hamiltonian
    :param Rinv: conj eigenvectors of the Hamiltonian
    :param weights: weights wx, wy, wz corresponding to whether a polycrystaline average or not
    :param size: size of the Hilbert space
    :return 2D list where a[i][j] is the amplitude of state E[i]-E[j]
    """

    # R_x is just R_roll
    s_x = np.dot(Rinv, R_roll)
    s_z = np.dot(Rinv, cython_polarisation.minus_half(R))
    s_y = np.dot(Rinv, cython_polarisation.minus_half(R_roll))

    del R, R_roll, Rinv

    if weights == (None, None, None):
        amplitudes = cython_polarisation.calc_amplitudes_angavg(s_x, s_y, s_z, size)
        del s_x, s_y, s_z
    else:
        wx, wy, wz = weights
        amplitudes = cython_polarisation.calc_amplitudes_initpol(s_x, s_y, s_z, wx, wy, wz, size)
    return amplitudes


def calc_amplitudes_gpu(R, Rinv, R_roll, weights, size):
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

    R_z = cp.zeros((size, size), dtype='complex64', order='F')
    R_y = cp.zeros((size, size), dtype='complex64', order='F')

    minus_kernel = cp.RawKernel(r'''
                #include <cupy/complex.cuh>
                extern "C"__global__
                void minus_kernel(const complex<float> *R, complex<float> *Res,
                                int N) {

                   int i, j;

                   // Determine thread position i j within thread block.
                   i = blockIdx.x*blockDim.x + threadIdx.x;
                   j = blockIdx.y*blockDim.y + threadIdx.y;

                   if (i<N && j<N) {
                    if (i<N/2) {
                        Res[i + N*j] = -R[i+ N*j];
                    } else {
                        Res[i + N*j] = R[i+ N*j];
                    }
                   }
                }
                ''', 'minus_kernel')

    minus_kernel((blocks, blocks), (threads_per_block, threads_per_block), (R, R_z, size))
    minus_kernel((blocks, blocks), (threads_per_block, threads_per_block), (R_roll, R_y, size))
    R_x = R_roll

    mod_squared = cp.ElementwiseKernel(
        'complex64 x', 'complex64 z',
        'z = abs(x); z = z * z',
        'mod_squared')

    if weights[0] is None or weights[1] is None or weights[2] is None:
        # angular average
        a = 1 / (3 * size / 2) * (mod_squared(cp.matmul(Rinv, R_x)) +
                                  mod_squared(cp.matmul(Rinv, R_y)) +
                                  mod_squared(cp.matmul(Rinv, R_z)))
    else:
        # not an angular average
        wx, wy, wz = weights
        a = 1 / (size / 2) * (mod_squared(cp.matmul(Rinv, R_x) * wx) +
                              mod_squared(cp.matmul(Rinv, R_y) * wy) +
                              mod_squared(cp.matmul(Rinv, R_z) * wz) +
                              # xy
                              wx * wy * (mod_squared(cp.matmul(Rinv, R_x) + 1j * cp.matmul(Rinv, R_y))
                                         - mod_squared(cp.matmul(Rinv, R_x)) - mod_squared(cp.matmul(Rinv, R_y))) +
                              # yz
                              wy * wz * (mod_squared(1j * cp.matmul(Rinv, R_y) + cp.matmul(Rinv, R_z))
                                         - mod_squared(cp.matmul(Rinv, R_y)) - mod_squared(cp.matmul(Rinv, R_z))) +
                              # xz
                              wx * wz * (mod_squared(cp.matmul(Rinv, R_x) + cp.matmul(Rinv, R_z))
                                         - mod_squared(cp.matmul(Rinv, R_x)) - mod_squared(cp.matmul(Rinv, R_z)))
                              )

    del R, Rinv, R_x, R_y, R_z, R_roll

    return a


def calc_polarisation_with_field_perturbation_integrand(spins, E, R, tau, t, direction):
    """
    calculate the polarisation of the muon, by calculating the integrand of the first-order perturbation.
    :param: spins: vector of the spins, with spins[0] being the muon
    :param: E: eigenvalues of the zeroth order Hamiltonian
    :param: R: matrix of eigenvectors of the zeroth ordder Hamiltonian
    :param: tau: time in perturbation land (to be integrated over)
    :param: t: muon time
    :param: direction: direction of the muon spin vector
    :return: [pert_x, pert_y, pert_z], which needs to be dot producted with B and integrated to get the perturbation
    """

    current_sum = np.zeros(shape=[3, len(tau)], dtype=np.complex128)
    Rinv = R.conj().transpose()
    # R[:,i] is the eigenvector column; Rinv[i] is the transpose so Rinv[i].R[:,j] = delta_ij

    sig_mu_x = Hamiltonians.measure_ith_spin(spins, 0, spins[0].pauli_x).todense()
    sig_mu_y = Hamiltonians.measure_ith_spin(spins, 0, spins[0].pauli_y).todense()
    sig_mu_z = Hamiltonians.measure_ith_spin(spins, 0, spins[0].pauli_z).todense()

    sig_mu_d = direction[0] * sig_mu_x + direction[1] * sig_mu_y + direction[2] * sig_mu_z

    t = t * np.ones(shape=tau.shape)

    for i_alpha, alpha in enumerate(E):
        for i_beta, beta in enumerate(E):
            for i_gamma, gamma in enumerate(E):
                prod = np.dot(np.dot(Rinv[i_alpha], sig_mu_d), R[:, i_beta][:, None])
                prod *= np.dot(np.dot(Rinv[i_gamma], sig_mu_d), R[:, i_alpha][:, None])
                prod = np.exp(1j * (alpha * t + beta * tau)) * prod[0, 0]
                prod *= np.exp(1j * gamma * (t - tau)) - np.exp(1j * beta * (tau - t))

                prod *= 1j * spins[0].gyromag_ratio / 2

                current_sum[0] += np.array(np.dot(np.dot(Rinv[i_beta], sig_mu_x), R[:, i_gamma][:, None]) * prod)[0]
                current_sum[1] += np.array(np.dot(np.dot(Rinv[i_beta], sig_mu_y), R[:, i_gamma][:, None]) * prod)[0]
                current_sum[2] += np.array(np.dot(np.dot(Rinv[i_beta], sig_mu_z), R[:, i_gamma][:, None]) * prod)[0]

    print(current_sum)

    return current_sum


def calc_polarisation_with_field_perturbation_2ndorder(spins: list,
                                                       E: np.ndarray,
                                                       R: np.ndarray,
                                                       B_var: np.ndarray,
                                                       tau_c: np.ndarray,
                                                       t: np.ndarray,
                                                       direction: np.ndarray = None):
    """
    Calculate the second order correction to the polarisation of the F-mu-F state, taking into account
    the 2nd order correction due to the field-field fluctuation term being non-zero
        calculate the polarisation of the muon, by calculating the integrand of the first-order perturbation.
    :param: spins: list of the spins, with spins[0] being the muon
    :param: E: eigenvalues of the zeroth order Hamiltonian, in Mrad/us
    :param: R: matrix of eigenvectors of the zeroth order Hamiltonian
    :param: tau_c: vector of x y z correlation times, in us
    :param: t: experiment time, in us
    :param: direction: direction of the muon spin vector. if None, do standard angular averaging
    :return: perturbation of the spin-spin correlation on the muon's spin
    """

    # calculate the inverse of R too
    Rinv = R.conj().transpose()

    n_eigs = len(E)

    sig_mu_x = Hamiltonians.measure_ith_spin(spins, 0, spins[0].pauli_x).todense()
    sig_mu_y = Hamiltonians.measure_ith_spin(spins, 0, spins[0].pauli_y).todense()
    sig_mu_z = Hamiltonians.measure_ith_spin(spins, 0, spins[0].pauli_z).todense()

    sig_mu = [sig_mu_x, sig_mu_y, sig_mu_z]

    # first of all, calculate the c_abg^ss' terms, using a lookup table
    C_terms_lookup = np.full(shape=[n_eigs, n_eigs, n_eigs, 3, 3], fill_value=None)

    def C_abg(alpha: int, beta: int, gamma: int, sigma: int, sigma_prime: int):
        """
        Calculate the C_{alpha, beta, gamma}^{sigma, sigma_prime} terms which effectively
        become the 'selection rules' of the interaction. All parameters are integers.
        """
        if not (C_terms_lookup[alpha, beta, gamma, sigma, sigma_prime] is None):
            # give lookup value
            return C_terms_lookup[alpha, beta, gamma, sigma, sigma_prime]

        # not got the lookup value yet, so calculate it...
        # R[:,i] is the eigenvector column (ket); Rinv[i] is the transpose (bra) so Rinv[i].R[:,j] = delta_ij
        gamma_bra = Rinv[gamma]
        alpha_ket = R[:, alpha]

        # do the delta/alpha braket
        gamma_sigma_alpha = np.dot(gamma_bra, np.dot(sig_mu[sigma], alpha_ket))

        if gamma_sigma_alpha != 0:
            if beta == gamma and sigma == sigma_prime:
                this_C = np.abs(gamma_sigma_alpha) ** 2
            else:
                alpha_bra = alpha_ket.conj().transpose()
                beta_ket = Rinv[beta]
                alpha_sigmap_beta = np.dot(alpha_bra, np.dot(sig_mu[sigma_prime], beta_ket))
                this_C = alpha_sigmap_beta * gamma_sigma_alpha
        else:
            this_C = np.complex128(0.0*1j)

        # store in the lookup table
        C_terms_lookup[alpha, beta, gamma, sigma, sigma_prime] = this_C

        # also do the conjugate
        C_terms_lookup[alpha, gamma, beta, sigma_prime, sigma] = this_C.conj()

        return this_C

    F_ab = np.zeros((len(E), len(E)), dtype=np.complex128)
    tau_c_inv = 1/tau_c
    E_diff = np.subtract.outer(E, E)
    for i in range(len(E)):
        for k in range(len(E)):
            F_ab[i, k] = 1./(-1j*E_diff[i, k] + tau_c_inv)

    e_tau_decay = np.exp(- t * tau_c_inv)

    sigma_sum = np.zeros(shape=t.shape)
    for sigma in range(0, 3):
        sigma_prime_sum = np.zeros(shape=t.shape)
        for sigma_prime in range(0, 3):
            for alpha in tqdm(range(0, n_eigs)):
                for beta in range(0, n_eigs):
                    for gamma in range(0, n_eigs):
                        c0 = C_abg(alpha, beta, gamma, sigma, sigma_prime)
                        if c0 == 0:
                            continue
                        c0_coeff = F_ab[alpha, beta]

                        # precalculate some exponentials
                        e_bgt = np.exp(1j * t * E_diff[beta, gamma])

                        # precalculate c2 term, as its independent of delta
                        denom = E_diff[alpha, gamma]
                        if denom != 0:
                            # calculate the two c1 terms as usual
                            c2_imag_coeff = 1j * (np.exp(1j * t * denom) - 1) / denom
                        else:
                            c2_imag_coeff = - t
                        c2_re_coeff = - F_ab[beta, gamma] * (e_bgt * e_tau_decay - 1)

                        for delta in range(0, n_eigs):
                            # check that one of the c terms are not zero
                            c1 = C_abg(gamma, delta, beta, sigma, sigma_prime)
                            c2 = C_abg(gamma, delta, beta, sigma_prime, sigma)
                            c1_term = np.zeros(shape=t.shape, dtype=np.complex128)
                            c2_term = np.zeros(shape=t.shape, dtype=np.complex128)

                            e_gdt = np.exp(1j * t * E_diff[gamma, delta])

                            if c1 == 0 and c2 == 0:
                                continue
                            if c1 != 0:
                                denom = E_diff[gamma, delta] + E_diff[alpha, beta]
                                if denom != 0:
                                    # calculate the two c1 terms as usual
                                    c1_imag_coeff = -1j * (np.exp(1j * t * denom) - 1) / denom
                                else:
                                    c1_imag_coeff = t
                                c1_re_coeff = F_ab[gamma, delta] * (e_gdt * e_tau_decay - 1)

                                c1_term = c1 * e_bgt * (c1_re_coeff + c1_imag_coeff)

                            if c2 != 0:
                                c2_term = c2 * e_gdt * (c2_re_coeff + c2_imag_coeff)

                            sigma_prime_sum += np.real(c0_coeff * (c1_term + c2_term))
            sigma_prime_sum *= B_var[sigma_prime]
        # TODO: make this adjust for the direction of the initial muon spin (just does polycrystalline for now...)
        sigma_sum += 2 / 3 * sigma_prime_sum

    return sigma_sum
