"""
TrotterPolarisation.py -- calculate the dipolar polarisation of a muon in a large spin system.
See Lab book 3, pg 67 for first creation notes
Inspired by:
Celio, M. PRL 56 p2720 (1986)
Stoudernmire, E.M. and White, S.R. New Journal of Physics 12 055026 (2010)
Created 2/7/2020
"""

from .MDecoherenceAtom import TDecoherenceAtom as atom
from .TCoord3D import TCoord3D as coord
from . import DipolarPolarisation
import numpy as np
from scipy import linalg, sparse
import math
import matplotlib.pyplot as pyplot
import time as human_time  # to time the operations


SPARSE_FORMAT = 'csc'


def main():
    All_spins = [atom(position=coord(0, 0, 0), name='mu'),
                 atom(position=coord(0, 0, 1), name='F'),
                 atom(position=coord(0, 1, 0), name='Sc'),
                 atom(position=coord(1, 0, 0), name='Na'),
                 atom(position=coord(0, 0, -1), name='F'),
                 atom(position=coord(0, -1, 0), name='Sc'),
                 atom(position=coord(-1, 0, 0), name='Na'),
                 ]
    muon = All_spins[0]

    start_time = human_time.time()
    calc_trotter_dipolar_polarisation(muon, All_spins)
    print("elapsed time: " + str(human_time.time() - start_time))


def calc_trotter_dipolar_polarisation(muon: atom, All_spins: list, times: np.ndarray = np.arange(0, 10, 0.1),
                                      plot: bool = True):
    """
    calculate the dipolar polarisation evolution for a muon in a sample. Considers the system to be bipartite
    :param muon: MDecoherenceAtom instance of the muon
    :param All_spins: list if MDecoherenceAtoms
    :param times: numpy array of the times to calculate at
    :param plot: set to True to get a plot of the polarisation
    :return: np.ndarray of the muon polarisation
    """

    # check All_spins is bipartite
    assert check_bipartite(All_spins)

    # calculate the initial states
    initial_state_x = calc_initial_states(All_spins, coord(1, 0, 0)).transpose()
    initial_state_y = calc_initial_states(All_spins, coord(0, 1, 0)).transpose()
    initial_state_z = calc_initial_states(All_spins, coord(0, 0, 1)).transpose()

    # split the Hamiltonian into systems
    n_system = math.floor((len(All_spins) - 1) / 2)

    # get the average spacing in the times array for dt
    dt = (times[-1] - times[0]) / len(times)
    trotter_step = 2 ** 15

    expH_mu_system = expH_mu_system_trotter(All_spins, n_system, dt, trotter_step)
    swap_system = swap_system12(All_spins, n_system)

    d_trotter_matrix = swap_system*expH_mu_system
    d_trotter_matrix = d_trotter_matrix.todense()
    d_trotter_matrix = expH_mu_system*d_trotter_matrix
    d_trotter_matrix = d_trotter_matrix ** trotter_step

    dim = d_trotter_matrix.shape[0]

    s_mu_x = 2 * sparse.kron(muon.pauli_x, sparse.eye(dim / 2, format=SPARSE_FORMAT), format=SPARSE_FORMAT)
    s_mu_y = 2 * sparse.kron(muon.pauli_y, sparse.eye(dim / 2, format=SPARSE_FORMAT), format=SPARSE_FORMAT)
    s_mu_z = 2 * sparse.kron(muon.pauli_z, sparse.eye(dim / 2, format=SPARSE_FORMAT), format=SPARSE_FORMAT)


    polarization = np.zeros([len(times), 1])
    for iteration in range(1, 20001):
        print('Iteration ' + str(iteration))
        it_polarization = np.zeros([len(times), 1], dtype=np.complex128)
        lambda_m = np.exp(1j * 2 * np.pi * np.random.uniform(size=int(dim / 2)))

        current_trotter_matrix = np.eye(dim)

        for i_t in range(0, len(times)):
            current_trotter_matrix = current_trotter_matrix * d_trotter_matrix

            this_polarisation = 0

            # sum over the states using the lambda_m
            p_x = np.asmatrix(np.zeros((int(dim), 1), dtype=np.complex128))
            p_y = np.asmatrix(np.zeros((int(dim), 1), dtype=np.complex128))
            p_z = np.asmatrix(np.zeros((int(dim), 1), dtype=np.complex128))

            for i_state in range(0, initial_state_x.shape[1]):
                p_x += lambda_m[i_state] * np.asmatrix(initial_state_x[:, i_state]).transpose()
                p_y += lambda_m[i_state] * np.asmatrix(initial_state_y[:, i_state]).transpose()
                p_z += lambda_m[i_state] * np.asmatrix(initial_state_z[:, i_state]).transpose()

            p_x = current_trotter_matrix * p_x
            p_y = current_trotter_matrix * p_y
            p_z = current_trotter_matrix * p_z

            this_polarisation += (p_x.H * s_mu_x * p_x)[0, 0] * 2 / dim
            this_polarisation += (p_y.H * s_mu_y * p_y)[0, 0] * 2 / dim
            this_polarisation += (p_z.H * s_mu_z * p_z)[0, 0] * 2 / dim
            it_polarization[i_t] = this_polarisation / 3

        # calculate the difference between the polarization this time and the previous iteration's
        last_polarization = polarization
        polarization = (polarization * (iteration - 1) + it_polarization) / iteration

        avg_diff = np.average(np.abs(polarization.real - last_polarization.real))
        print('avg_diff = ' + str(avg_diff))
        if avg_diff < 1e-3 and iteration > 1:
            break

    pyplot.plot(times, polarization)
    pyplot.show()


def check_bipartite(All_spins):
    """
    given a list of MDecoherenceAtoms, check that it is symmetric under inversion with the muon at the centre.
    :param All_spins: list of MDecoherenceAtoms, with the muon at index 0
    :return: True of bipartite, False if not
    """

    # check for an odd number of spins (system1+system2): even + muon -> odd
    assert len(All_spins) % 2 == 1

    n_spins = len(All_spins) - 1
    n_system = math.floor(n_spins / 2)

    for i_spin in range(0, n_system):
        if All_spins[i_spin + 1].position != All_spins[i_spin + 1 + n_system].position*-1.0:
            return False
        if All_spins[i_spin + 1].II != All_spins[i_spin + 1 + n_system].II:
            return False
        if All_spins[i_spin + 1].gyromag_ratio != All_spins[i_spin + 1 + n_system].gyromag_ratio:
            return False

    return True


def expH_mu_system_trotter(All_spins: list, n_system: int, dt: float, trotter_step: float):
    """
    calculate exp(-1j*H_mu_s*dt/trotter_step) kronecker producted with the identity matrix for system 2
    :param All_spins: list of MDecoherenceAtom objects, including the muon
    :param n_system: number of spins in the system
    :param dt: time step
    :param trotter_step: step of trotterization (i.e k in exp(A+B) = (exp(A/k)*exp(B/k))^k)
    :return: exp(-1j*H_mu_s*dt/trotter_step) kroneker producted with the identity matrix for system 2
    """

    # calculate the Hamiltonian
    H_mu_s = DecoherenceCalculator.calc_dipolar_hamiltonian(All_spins[0:n_system + 1], sparse_format=SPARSE_FORMAT)

    # find the dimension of system 2 (should be the same as system 1, so just do that)
    system_2_dim = 1
    for spin in All_spins[1:n_system + 1]:
        system_2_dim *= spin.II + 1

    # calculate the exponential term
    return sparse.kron(linalg.expm(-1j * H_mu_s * dt / trotter_step), sparse.eye(system_2_dim))


def swap_system12(All_spins: list, n_system:int):
    """
    swaps everything in system 1 with system 2
    :param All_spins: list of MDecoherenceAtom objects, including the muon
    :param n_system: number of spins in the system
    :return: swap matrix to swap system 1 with system 2
    """
    swap_matrix = swap_strided(All_spins, 1, 1+n_system)
    for i in range(1, n_system):
        swap_matrix *= swap_strided(All_spins, i+1, i+1+n_system)

    return swap_matrix


def swap_strided(All_spins: list, i: int, j: int):
    """
    create a SWAP matrix for equivalent spins i and j in All_spins
    :param All_spins: list of MDecoherenceAtom objects, including the muon
    :param i: ID first spin to be swapped
    :param j: ID of spin to be swappped with j
    :return: matrix to swap spins i and j in All_spins
    """

    # make it so that i<j
    if i > j:
        i, j = j, i

    assert i != j

    # make the identity matrix for the spins with id<i which are not involved in this swap
    left_identity_dim = 1
    right_identity_dim = 1
    middle_identity_dim = 1
    for spin in All_spins[0:i]:
        left_identity_dim *= spin.II + 1
    left_identity = sparse.eye(left_identity_dim)
    # and do the same for the right hand side
    for spin in All_spins[j+1:]:
        right_identity_dim *= spin.II + 1
    right_identity = sparse.eye(right_identity_dim)
    # and finally do the same for the middle bit
    middle_identity_dim = 1
    for spin in All_spins[i+1: j]:
        middle_identity_dim *= spin.II + 1
    i_dim = All_spins[i].II + 1
    j_dim = All_spins[j].II + 1
    assert i_dim == j_dim

    omega = np.exp(1j * 2 * math.pi / i_dim)

    CZ_d_diag = [omega ** (math.floor(i/(j_dim*middle_identity_dim)) * ((i % (j_dim * middle_identity_dim)) % j_dim)) \
                 for i in range(0, i_dim*middle_identity_dim*j_dim)]

    CZ_d = sparse.kron(left_identity,
                       sparse.kron(sparse.diags(CZ_d_diag, format=SPARSE_FORMAT),
                                   right_identity, format=SPARSE_FORMAT)
                       , format=SPARSE_FORMAT)

    qft_i = np.zeros((i_dim, i_dim))
    for i_qft_i in range(0, len(qft_i)):
        for j_qft_i in range(0, len(qft_i)):
            qft_i[i_qft_i, j_qft_i] = omega ** (i_qft_i * j_qft_i)
    qft_i *= 1 / math.sqrt(i_dim)

    qft_i = sparse.kron(left_identity,
                        sparse.kron(qft_i,
                                    sparse.eye(right_identity_dim*middle_identity_dim*j_dim, format=SPARSE_FORMAT)
                                    , format=SPARSE_FORMAT)
                        , format=SPARSE_FORMAT)

    qft_j = np.zeros((j_dim, j_dim))
    for i_qft_j in range(0, len(qft_j)):
        for j_qft_j in range(0, len(qft_j)):
            qft_j[i_qft_j, j_qft_j] = omega ** (i_qft_j * j_qft_j)
    qft_j *= 1 / math.sqrt(j_dim)

    qft_j = sparse.kron(sparse.eye(left_identity_dim*i_dim*middle_identity_dim,
                                   format=SPARSE_FORMAT)
                        , sparse.kron(qft_j, right_identity, format=SPARSE_FORMAT)
                        , format=SPARSE_FORMAT)

    cx_ij = qft_j*CZ_d*qft_j
    cx_j1 = qft_i*CZ_d*qft_i

    return cx_ij*cx_j1*cx_ij


def calc_initial_states(All_spins: list, polarization_direction: coord):
    """
    Calculates a list of vectors with all the initial states, in x y or z
    :param All_spins: list of spins, with muon in position 0 (this gets polarized in the +direction direction
    :param polarization_direction: direction of the muon's initial spin polarization
    :return: list, where list[i] is one of the states
    """

    # for each of the spins, calculate and diagoanlise the spin matrix
    all_spin_vectors = list()
    muon_spin_index = None
    for spin in All_spins:
        this_spin_matrix = + polarization_direction.xhat()*spin.pauli_x + polarization_direction.yhat()*spin.pauli_y \
                           + polarization_direction.zhat()*spin.pauli_z
        spin_values, spin_vectors = linalg.eigh(this_spin_matrix.todense())
        # if this is the muon, find out which index the +ve spin is in
        if spin.name == 'mu':
            muon_spin_index = spin_values.argmax()
        # save the spin vectors
        all_spin_vectors.append(spin_vectors.transpose())

    # now make the initial states
    muon_state = all_spin_vectors[0][muon_spin_index]
    initial_states = np.asmatrix(muon_state)
    for i_spin in range(1, len(All_spins)):
        initial_states = linalg.kron(initial_states, all_spin_vectors[i_spin])

    return initial_states


if __name__=='__main__':
    main()