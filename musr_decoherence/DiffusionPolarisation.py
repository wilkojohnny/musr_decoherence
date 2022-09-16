"""
DiffusionPolarisation.py -- calculate FmuF states with muon diffusion, initially using the one-hop method as
given by Celio, Hyperfile Interactions *31* 153 (1986)
Created by Johnny Wilkinson, 15/09/2022
"""
import scipy.integrate

from . import Hamiltonians
import numpy as np
from scipy import linalg
from . import DipolarPolarisation


def calc_diffusion_polarisation(all_spins_init: list, all_spins_final: list, times: np.ndarray = np.arange(0, 10, 0.1),
                                nu: float = 0):
    """
    Calculate the muon polarisation for a muon diffusing between two sites, including quantum correlations
    :param: all_spins_init: list of the initial spins, with muon at site 0
    :param: all_spins_final: list of the final spins, with muon at site 0
    :param: times: nparray of times to calculate the polarisation at
    :param: nu: hop rate, in µs^-1
    :return: muon polarisation as numpyarray
    """

    init_hilbert_dim = 1
    for spin in all_spins_init:
        init_hilbert_dim *= spin.II + 1

    final_hilbert_dim = 1
    for spin in all_spins_final:
        final_hilbert_dim *= spin.II + 1

    assert init_hilbert_dim == final_hilbert_dim

    # construct the Hamiltonians for both sets of spins
    initial_hamiltonian = Hamiltonians.calc_dipolar_hamiltonian(spins=all_spins_init, just_muon_interactions=False)
    final_hamiltonian = Hamiltonians.calc_dipolar_hamiltonian(spins=all_spins_final, just_muon_interactions=False)

    # diagonalise both
    initial_hamiltonian_dense = initial_hamiltonian.todense()
    final_hamiltonian_dense = final_hamiltonian.todense()

    E_init, R_init = linalg.eigh(initial_hamiltonian_dense, overwrite_a=True)
    del initial_hamiltonian_dense
    R_init_inv = R_init.transpose().conj()
    R_init = R_init.copy(order='C')

    E_final, R_final = linalg.eigh(final_hamiltonian_dense, overwrite_a=True)
    del final_hamiltonian_dense
    R_final_inv = R_final.transpose().conj()
    R_final = R_final.copy(order='C')

    # calculate the polarisation function for the initial state
    wx, wy, wz = 1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)

    polarisation_normal = DipolarPolarisation.calc_hamiltonian_polarisation(hamiltonian=initial_hamiltonian,
                                                                            times=times,
                                                                            weights=(wx, wy, wz),
                                                                            fourier=False,
                                                                            fourier_2d=False,
                                                                            probability=1,
                                                                            hilbert_dim=init_hilbert_dim,
                                                                            gpu=False,
                                                                            shutup=False)[0]

    polarisation = polarisation_normal * np.exp(-nu * times)

    # do some general operations now...
    mu_spin_x = Hamiltonians.measure_ith_spin(all_spins_init, 0, all_spins_init[0].pauli_x).todense()
    mu_spin_y = Hamiltonians.measure_ith_spin(all_spins_init, 0, all_spins_init[0].pauli_y).todense()
    mu_spin_z = Hamiltonians.measure_ith_spin(all_spins_init, 0, all_spins_init[0].pauli_z).todense()

    B_2_x = np.dot(R_final_inv, mu_spin_x)
    B_2_y = np.dot(R_final_inv, mu_spin_y)
    B_2_z = np.dot(R_final_inv, mu_spin_z)

    def diffusion_integral(t_hop, t, nu):
        """
        calculate the diffusion integral
        """

        A_1 = np.dot(np.dot(R_init_inv, np.diag(np.exp(1j * E_init * t_hop))), R_init)
        A_2 = np.dot(np.dot(R_final_inv, np.diag(np.exp(1j * E_final * (t - t_hop)))), R_final)

        A_1_c = A_1.conj().transpose()

        this_sum_z = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(B_2_z, A_2), A_1), mu_spin_z), A_1_c),
                                np.diag(np.exp(-1j * E_final * (t - t_hop)))), R_final)
        this_sum_x = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(B_2_x, A_2), A_1), mu_spin_x), A_1_c),
                                  np.diag(np.exp(-1j * E_final * (t - t_hop)))), R_final)
        this_sum_y = np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(B_2_y, A_2), A_1), mu_spin_y), A_1_c),
                                  np.diag(np.exp(-1j * E_final * (t - t_hop)))), R_final)

        summand = (this_sum_x + this_sum_y + this_sum_z)/3

        return np.exp(-nu * t) * np.sum(summand, axis=None)

    # for each time in times:
    for i_t, t in enumerate(times):
        # calculate the integral
        integral = scipy.integrate.quad(diffusion_integral, 0, t, args=(t, nu))
        # append this onto the polarisaton
        polarisation[i_t] += nu / init_hilbert_dim * integral[0]

    import matplotlib.pyplot as plot
    plot.plot(times, polarisation)


    # for now, also compare to SCM dynamics...
    from . import Dynamiciser
    scm_polarisation = Dynamiciser.dynamicise(polarisation, t=times, dt=times[1]-times[0], nu=nu)
    plot.plot(times, scm_polarisation)

    plot.show()

    return polarisation
