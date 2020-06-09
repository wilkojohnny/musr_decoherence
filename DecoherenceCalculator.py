# DecoherenceCalculator.py - Calculate decoherence of muon state in any lattice with any structure

import TCoord3D as coord  # 3D coordinates class
import numpy as np  # for matrices
import scipy.sparse as sparse  # for sparse matrices


# make measurement operator for this spin
def measure_ith_spin(Spins, i, pauli_matrix):
    # calculate the dimension of the identity matrix on the LHS ...
    lhs_dim = 1
    for i_spin in range(0, i):
        lhs_dim = lhs_dim * Spins[i_spin].pauli_dimension

    # ... and the RHS
    rhs_dim = 1
    for i_spin in range(i + 1, len(Spins)):
        rhs_dim = rhs_dim * Spins[i_spin].pauli_dimension

    return sparse.kron(sparse.kron(sparse.identity(lhs_dim), pauli_matrix), sparse.identity(rhs_dim))


# calculate the Hamiltonian for the i j pair
def calc_hamiltonian_term(spins, i, j):
    # calculate A
    A = 1.05456e-5 * spins[i].gyromag_ratio * spins[j].gyromag_ratio

    r = spins[i].position - spins[j].position

    # get all the operators we need
    i_x = measure_ith_spin(spins, i, spins[i].pauli_x)
    j_x = measure_ith_spin(spins, j, spins[j].pauli_x)

    i_y = measure_ith_spin(spins, i, spins[i].pauli_y)
    j_y = measure_ith_spin(spins, j, spins[j].pauli_y)

    i_z = measure_ith_spin(spins, i, spins[i].pauli_z)
    j_z = measure_ith_spin(spins, j, spins[j].pauli_z)

    # Calculate the hamiltonian!
    return A / pow(abs(r.r()), 3) * (i_x * j_x + i_y * j_y + i_z * j_z
                                     - 3 * (i_x * r.xhat() + i_y * r.yhat() + i_z * r.zhat())
                                     * (j_x * r.xhat() + j_y * r.yhat() + j_z * r.zhat()))


def calc_dipolar_hamiltonian(spins, just_muon_interactions=False):
    current_hamiltonian = 0

    # if just muon interaction, then only do interactions between i=0 and all j
    if just_muon_interactions:
        i_max = 1
    else:
        i_max = len(spins)

    # calculate hamiltonian for each pair and add onto sum
    for i in range(0, i_max):
        for j in range(i + 1, len(spins)):
            current_hamiltonian = current_hamiltonian + calc_hamiltonian_term(spins, i, j)
    return current_hamiltonian


def calc_quadrupolar_hamiltonian(spins):
    current_hamiltonian = 0
    for spin_i, spin in enumerate(spins):
        if spin.II > 1 and abs(spin.Q) > 0:
            i_x = measure_ith_spin(spins, spin_i, spin.pauli_x)
            i_y = measure_ith_spin(spins, spin_i, spin.pauli_y)
            i_z = measure_ith_spin(spins, spin_i, spin.pauli_z)

            v_xx = spin.efg[0, 0]
            v_yy = spin.efg[1, 1]
            v_zz = spin.efg[2, 2]
            v_xy = spin.efg[0, 1]
            v_yz = spin.efg[1, 2]
            v_xz = spin.efg[0, 2]

            quadrupole_term = 434.4557 * spin.Q * (1 + spin.anti_shielding) / (spin.II * (spin.II - 1)) * \
                              (v_xx*(i_x ** 2) + v_yy * (i_y ** 2) + v_zz * (i_z ** 2)
                               + v_xy*(i_x*i_y + i_y*i_x) + v_yz*(i_y*i_z + i_z*i_y) + v_xz*(i_x*i_z + i_z*i_x))

            current_hamiltonian = current_hamiltonian + quadrupole_term
    return current_hamiltonian


def calc_efg(spins: list, i_spin: int) -> [float, float, float]:
    """
    Estimates the EFG for the spin spins[i_spin] due to the others.
    !! WARNING -- does not check that V_xy=V_yz=V_zx=0 (yet...)
    :param spins: list of TDecoherenceAtoms which have charge, and Q defined
    :param i_spin: particular spin we are looking at.
    :return: [V_xx, V_yy, V_zz] in angstrom^-3
    """

    # check that i_spin supports quadrupolar interaction
    if spins[i_spin].II == 1:
        return [0, 0, 0]

    V_xx = 0
    V_yy = 0
    V_zz = 0

    # for each spin
    for j_spin, spin in enumerate(spins):
        if i_spin == j_spin:
            continue
        rel_position = spin.position - spins[i_spin].position
        q = spin.charge
        r_i5 = rel_position.r() ** 5
        V_xx += q / r_i5 * (3 * (rel_position.ortho_x ** 2) - rel_position.r() ** 2)
        V_yy += q / r_i5 * (3 * (rel_position.ortho_y ** 2) - rel_position.r() ** 2)
        V_zz += q / r_i5 * (3 * (rel_position.ortho_z ** 2) - rel_position.r() ** 2)

    spins[i_spin].efg = [V_xx, V_yy, V_zz]

    return [V_xx, V_yy, V_zz]


def calc_zeeman_hamiltonian(spins, field: coord.TCoord3D):
    current_hamiltonian = 0

    # for each atom
    for i in range(0, len(spins)):
        # calculate the Hamiltonian
        Sx = measure_ith_spin(spins, i, spins[i].pauli_x)
        Sy = measure_ith_spin(spins, i, spins[i].pauli_y)
        Sz = measure_ith_spin(spins, i, spins[i].pauli_z)
        current_hamiltonian = current_hamiltonian - spins[i].gyromag_ratio * (field.ortho_x * Sx
                                                                              + field.ortho_y * Sy
                                                                              + field.ortho_z * Sz)
    return current_hamiltonian


# calculate polarisation function for one specific time (used in the integration routine)
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
