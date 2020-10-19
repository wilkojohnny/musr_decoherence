"""
Trotter.py -- Calculate the time evolution of the F--mu--F etc polarisation using Trotter's decomposition
and switch states
Ideas based on:
Celio, M. PRL 56 p2720 (1986)
Stoudernmire, E.M. and White, S.R. New Journal of Physics 12 055026 (2010)
https://arxiv.org/abs/1304.4923v1
https://arxiv.org/abs/1901.05824v3
Created by John Wilkinson, 3/4/20
"""

from .MDecoherenceAtom import TDecoherenceAtom as atom  # atom tools
from .TCoord3D import TCoord3D as coord  # for coordinates
from scipy import sparse, linalg  # matrix utils
import numpy as np  # numeric utils
import matplotlib.pyplot as pyplot  # plotting utils
import time as human_time  # to time the operations

sparse_format = 'coo'  # format for the sparse matrices


def main():

    All_spins = [atom(position=coord(0, 0, 0), name='mu'),
                 atom(position=coord(0, 0, 1), name='F'),
                 atom(position=coord(0, 0, -1), name='F'),
                 atom(position=coord(0, 1, 0), name='F'),
                 # atom(position=coord(0, -1, 0), name='F'),
                 # atom(position=coord(1, 0, 0), name='F'),
                 # atom(position=coord(-1, 0, 0), name='F'),
                 # atom(position=coord(2, 0, 0), name='F'),
                 # atom(position=coord(-2, 0, 0), name='F'),
                 # atom(position=coord(0, 2, 0), name='F'),
                 # atom(position=coord(0, -2, 0), name='F'),
                 # atom(position=coord(0, 0, 2), name='F'),
                 ]

    dim = 2 ** len(All_spins)

    muon = All_spins[0]

    start_time = human_time.time()

    initial_state_x = calc_initial_states(All_spins, coord(1, 0, 0)).transpose()
    initial_state_y = calc_initial_states(All_spins, coord(0, 1, 0)).transpose()
    initial_state_z = calc_initial_states(All_spins, coord(0, 0, 1)).transpose()

    trotter_step = 2048
    dt = 0.1

    d_trotter_matrix = np.asmatrix(np.eye(dim))
    total_switch = sparse.eye(dim, format=sparse_format)
    k_to_iham = [i for i in range(0, len(All_spins))]

    # do the first Hamiltonian calculating round, which is just H_{i, i+1}
    for i in range(0, len(All_spins) - 1):
        # here k==i as no switching has been done yet
        exp_h = calc_trotter_hamiltonian_exp(All_spins, i, i + 1, i, dt, trotter_step)
        print('H_' + str(k_to_iham[i]) + ',' + str(k_to_iham[i + 1]) + ' at k=' + str(i))
        d_trotter_matrix = exp_h * d_trotter_matrix

    for switch_step in range(0, len(All_spins) - 2):
        print('switch step: ' + str(switch_step))
        # make and apply the switch matrices
        this_switch = sparse.eye(dim, format=sparse_format)

        for k in range(switch_step % 2, len(All_spins) - 1, 2):
            this_switch = switch_matrix(All_spins, k)*this_switch
            k_to_iham[k], k_to_iham[k+1] = k_to_iham[k+1], k_to_iham[k]

        d_trotter_matrix = this_switch * d_trotter_matrix
        total_switch = total_switch * this_switch

        # make and apply the Hamiltonians
        for k in range((switch_step + 1) % 2, len(All_spins) - 1, 2):
            exp_h = calc_trotter_hamiltonian_exp(All_spins, k_to_iham[k], k_to_iham[k+1], k, dt, trotter_step)
            print('H_' + str(k_to_iham[k]) + ',' + str(k_to_iham[k + 1]) + ' at k=' + str(k))
            d_trotter_matrix = exp_h*d_trotter_matrix

    d_trotter_matrix = total_switch * d_trotter_matrix
    d_trotter_matrix = d_trotter_matrix ** trotter_step
    d_trotter_matrix = d_trotter_matrix

    s_mu_x = 2 * sparse.kron(muon.pauli_x, sparse.eye(2 ** (len(All_spins) - 1), format=sparse_format), format=sparse_format)
    s_mu_y = 2 * sparse.kron(muon.pauli_y, sparse.eye(2 ** (len(All_spins) - 1), format=sparse_format), format=sparse_format)
    s_mu_z = 2 * sparse.kron(muon.pauli_z, sparse.eye(2 ** (len(All_spins) - 1), format=sparse_format), format=sparse_format)

    time = np.arange(0.1, 10, 0.1)

    polarization = np.zeros([len(time), 1])
    for iteration in range(1, 20001):
        print('Iteration ' + str(iteration))
        it_polarization = np.zeros([len(time), 1], dtype=np.complex128)
        lambda_m = np.exp(1j * 2 * np.pi * np.random.uniform(size=int(dim/2)))

        current_trotter_matrix = np.eye(dim)

        for i_t in range(0, len(time)):
            current_trotter_matrix = current_trotter_matrix*d_trotter_matrix

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

            # for i_state in range(0, initial_state_x.shape[1]):
            #     ket_state_x = current_trotter_matrix * np.asmatrix(initial_state_x[:, i_state]).transpose()
            #     bra_state_x = ket_state_x.H
            #     p_x = bra_state_x * s_mu_x * ket_state_x
            #     this_polarisation += p_x[0, 0]
            #
            #     ket_state_y = current_trotter_matrix * np.asmatrix(initial_state_y[:, i_state]).transpose()
            #     bra_state_y = ket_state_y.H
            #     p_y = bra_state_y * s_mu_y * ket_state_y
            #     this_polarisation += p_y[0, 0]
            #
            #     ket_state_z = current_trotter_matrix * np.asmatrix(initial_state_z[:, i_state]).transpose()
            #     bra_state_z = ket_state_z.H
            #     p_z = bra_state_z * s_mu_z * ket_state_z
            #     this_polarisation += p_z[0, 0]

            it_polarization[i_t] = this_polarisation / 3

        # calculate the difference between the polarization this time and the previous iteration's
        last_polarization = polarization
        polarization = (polarization * (iteration - 1) + it_polarization) / iteration

        avg_diff = np.average(np.abs(polarization.real - last_polarization.real))
        print('avg_diff = ' + str(avg_diff))
        if avg_diff < 1e-3 and iteration > 1:
            break

    print("elapsed time: " + str(human_time.time() - start_time))

    polarization = np.insert(polarization, 0, 1)
    time = np.insert(time, 0, 0)

    for i in range(0, len(time)):
        print(str(time[i]) + '\t' + str(polarization[i].real))

    pyplot.plot(time, polarization.real)
    pyplot.show()

    return 0


def calc_trotter_hamiltonian_exp(All_spins, i, j, k, dt, trotter_step):
    """
    Calculate the Trotter Hamiltonian, exponentiated
    :param All_spins: Array of all the spins in the calculation
    :param i: index of the first spin in All_spins
    :param j: index of the second spin in All_spins
    :param k: (k, k+1) are the positions of the spins in the current switch step
    :param dt: time step the Trotter Hamiltonian corresponds to
    :param trotter_step: trotter step (k in Celio's paper)
    :return: sparse matrix of expm(-1j*H_{ij}*dt/trotter_step)
    """

    left_1_dim = pow(2, max(k, 0))
    right_1_dim = pow(2, max(len(All_spins) - 2 - k, 0))

    left_1 = sparse.eye(left_1_dim, format=sparse_format)
    right_1 = sparse.eye(right_1_dim, format=sparse_format)

    h_ij = calc_trotter_hamiltonian_term(All_spins, i, j).todense()

    return sparse.kron(left_1,
                       sparse.kron(linalg.expm(-1j * h_ij * dt / trotter_step), right_1, format=sparse_format),
                       format=sparse_format)


def calc_trotter_hamiltonian_term(All_spins, i, j):
    """
    Calculates the Trotter dipolar Hamiltonian term H_{i,i+1}, for spins i and j (forces them to be nn)
    :param All_spins: Array of all the spins in the entire calculation
    :param i: index of the first spin to deal with
    :param j: index of the second spin to deal with
    :return: dim(i)dim(j)xdim(i)dim(j) matrix with the Hamiltonian term H_{i,i+1}
    """
    # calculate A
    A = 1.05456e-5 * All_spins[i].gyromag_ratio * All_spins[j].gyromag_ratio

    r = All_spins[i].position - All_spins[j].position

    i_dim = All_spins[i].II + 1
    j_dim = All_spins[j].II + 1

    # get all the operators we need
    i_x = sparse.kron(All_spins[i].pauli_x, sparse.eye(j_dim, format=sparse_format))
    j_x = sparse.kron(sparse.eye(i_dim, format=sparse_format), All_spins[j].pauli_x)

    i_y = sparse.kron(All_spins[i].pauli_y, sparse.eye(j_dim, format=sparse_format))
    j_y = sparse.kron(sparse.eye(i_dim, format=sparse_format), All_spins[j].pauli_y)

    i_z = sparse.kron(All_spins[i].pauli_z, sparse.eye(j_dim, format=sparse_format))
    j_z = sparse.kron(sparse.eye(i_dim, format=sparse_format), All_spins[j].pauli_z)

    # Calculate the hamiltonian!
    return A / pow(abs(r.r()), 3) * (i_x * j_x + i_y * j_y + i_z * j_z
                                     - 3 * (i_x * r.xhat() + i_y * r.yhat() + i_z * r.zhat())
                                     * (j_x * r.xhat() + j_y * r.yhat() + j_z * r.zhat()))


def switch_matrix(All_spins: list, k: int) -> sparse.csc_matrix:
    """
    Make a switch matrix to switch index k with index k+1
    :param K: index to switch with K+1
    :return: coo matrix to switch i with i+1 in All_spins
    """

    assert k < len(All_spins) - 1

    switch_2by2 = sparse.block_diag(([1], [[0, 1], [1, 0]], [1]), format=sparse_format)

    left_1_dim = pow(2, max(k, 0))
    right_1_dim = pow(2, max(len(All_spins)-2-k, 0))

    return sparse.kron(sparse.eye(left_1_dim, format=sparse_format),
                       sparse.kron(switch_2by2,
                                   sparse.eye(right_1_dim, format=sparse_format),
                                   format=sparse_format),
                       format=sparse_format)


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