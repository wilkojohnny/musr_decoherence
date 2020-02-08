# DipolePolarisation.py -- calculates sum_{nnnness} [n atoms in nnnness]/r_{nnnness}^6, until convergence
# John Wilkinson 16/12/19
import AtomObtainer  # for nnnfinder
from MDecoherenceAtom import TDecoherenceAtom as atom  # for atom object
import TCoord3D as coord  # for 3D coordinates


# calc_required_perturbation -- calculates the perturbation of the pert_nnnness nuclei (radially towards (-) or away (+)
# from original positions).
def calc_required_perturbation(muon_position, squish_radius=None, pert_nnnness=2, end_nnnness=-1, end_tol=1e-6,
                               max_nn_search_radius=20,
                               # arguments for manual input of lattice
                               lattice_type=None, lattice_parameter=None, lattice_angles=None,
                               input_coord_units=AtomObtainer.position_units.ALAT, atomic_basis=None,
                               perturbed_distances=None):

    # calculate the variance
    B_variance = calc_variance(muon_position=muon_position, squish_radius=squish_radius, start_nnnness=pert_nnnness,
                               end_nnnness=end_nnnness, end_tol=end_tol, max_nn_search_radius=max_nn_search_radius,
                               lattice_type=lattice_type, lattice_parameter=lattice_parameter,
                               lattice_angles=lattice_angles, input_coord_units=input_coord_units,
                               atomic_basis=atomic_basis, perturbed_distances=perturbed_distances)

    # get the start_nnnness term
    muon, All_Spins, _ = AtomObtainer.get_spins(muon_position=muon_position, squish_radius=squish_radius,
                                                lattice_type=lattice_type, lattice_parameter=lattice_parameter,
                                                lattice_angles=lattice_angles, input_coord_units=input_coord_units,
                                                atomic_basis=atomic_basis, perturbed_distances=perturbed_distances,
                                                max_nn_search_radius=max_nn_search_radius, nnnness=pert_nnnness,
                                                exclusive_nnnness=True)

    # calculate the required perturbation
    # calculate the
    gyro_ii = 0
    for i in range(1, len(All_Spins)):
        gyro_ii += All_Spins[i].II / 2 * (All_Spins[i].II / 2 + 1) * pow(All_Spins[i].gyromag_ratio, 2)

    print(gyro_ii)
    print(All_Spins[1].position.r())
    required_perturbation = pow(B_variance/gyro_ii, -1/6) - All_Spins[1].position.r()

    print('required perturbation = ' + str(required_perturbation))

    return required_perturbation


def calc_variance(muon_position, squish_radius=None, start_nnnness=2, end_nnnness=-1, end_tol=1e-6,
                  max_nn_search_radius=20,
                  # arguments for manual input of lattice
                  lattice_type=None, lattice_parameter=None, lattice_angles=None,
                  input_coord_units=AtomObtainer.position_units.ALAT, atomic_basis=None, perturbed_distances=None):

    current_sum = 0

    # header
    print('nnn_ness\tradius\tno\tterm\tsum\trel_diff')

    # for each nnnness until the new/old - 1 > end_tol
    new_term = 100
    nnnness = start_nnnness - 1
    r = 0
    while new_term > end_tol:
        # increment nnnness
        nnnness += 1

        # if nnnness is the same as end_nnnness, exit the loop and account for the others with the integral
        if nnnness == end_nnnness:
            # calculate the average multiplier to sort out diferent atoms
            average_gyromag_i = 0
            for basis_atom in atomic_basis:
                average_gyromag_i += pow(basis_atom.gyromag_ratio, 2) * basis_atom.II / 2 * (basis_atom.II / 2 + 1) \
                                     / len(atomic_basis)
            # estimate the rest fof the sum with an integral (assuming number of nns is 4*pi*r^2
            integral_extra = 4*3.1415926/(3*pow(r, 3))*average_gyromag_i
            print('Adding on an extra ' + str(integral_extra) + ' with the integral')
            current_sum = current_sum + integral_extra
            print('Total variance of B-field is ' + str(current_sum))
            break
        old_sum = current_sum

        # get the atoms which are nnn away from the muon
        muon, All_Spins, _ = AtomObtainer.get_spins(muon_position, squish_radius, lattice_type, lattice_parameter,
                                                    lattice_angles, input_coord_units, atomic_basis,
                                                    perturbed_distances, max_nn_search_radius=max_nn_search_radius,
                                                    nnnness=nnnness, exclusive_nnnness=True, shutup=False)

        # calculate r
        r = (muon.position - All_Spins[1].position).r()

        # calculate this term in the sum
        new_term = 0
        for i in range(1, len(All_Spins)):
            new_term += All_Spins[i].II/2*(All_Spins[i].II/2 + 1)*pow(All_Spins[i].gyromag_ratio, 2)
        new_term *= 1/pow(r, 6)

        # update sum
        current_sum = current_sum + new_term

        # calculate extent of convergence
        if old_sum != 0:
            rel_diff = current_sum/old_sum - 1
        else:
            rel_diff = 999

        # print out findings
        print(str(nnnness) + '\t' + str(r) + '\t' + str(len(All_Spins) - 1) + '\t' + str(new_term) + '\t' +
              str(current_sum) + '\t' + str(rel_diff))

    return current_sum