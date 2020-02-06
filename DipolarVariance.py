# DipolePolarisation.py -- calculates sum_{nnnness} [n atoms in nnnness]/r_{nnnness}^6, until convergence
# John Wilkinson 16/12/19
import AtomObtainer  # for nnnfinder
from MDecoherenceAtom import TDecoherenceAtom as atom  # for atom object
import TCoord3D as coord  # for 3D coordinates


# calc_required_perturbation -- calculates the perturbation of the pert_nnnness nuclei (radially towards (-) or away (+)
# from original positions).
def calc_required_perturbation(muon_position, squish_radius=None, pert_nnnness=2, end_tol=1e-6, max_nn_search_radius=20,
                               # arguments for manual input of lattice
                               lattice_type=None, lattice_parameter=None, lattice_angles=None,
                               input_coord_units=AtomObtainer.position_units.ALAT, atomic_basis=None,
                               perturbed_distances=None):

    # calculate the variance
    B_variance = calc_variance(muon_position=muon_position, squish_radius=squish_radius, start_nnnness=pert_nnnness,
                              end_tol=end_tol, max_nn_search_radius=max_nn_search_radius, lattice_type=lattice_type,
                              lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
                              input_coord_units=input_coord_units, atomic_basis=atomic_basis,
                              perturbed_distances=perturbed_distances)

    # get the start_nnnness term
    muon, All_Spins, _ = AtomObtainer.get_spins(muon_position, squish_radius, lattice_type, lattice_parameter,
                                                lattice_angles, input_coord_units, atomic_basis,
                                                perturbed_distances, max_nn_search_radius=max_nn_search_radius,
                                                nnnness=pert_nnnness, exclusive_nnnness=True, shutup=True)

    # calculate the required perturbation
    required_perturbation = pow(B_variance/(len(All_Spins) - 1), -1/6) - All_Spins[1].position.r()

    print('required perturbation = ' + str(required_perturbation))

    return required_perturbation


def calc_variance(muon_position, squish_radius=None, start_nnnness=2, end_tol=1e-6, max_nn_search_radius=20,
                 # arguments for manual input of lattice
                 lattice_type=None, lattice_parameter=None, lattice_angles=None,
                 input_coord_units=AtomObtainer.position_units.ALAT, atomic_basis=None, perturbed_distances=None):

    current_sum = 0

    # header
    print('nnn_ness\tradius\tno\tterm\tsum\trel_diff')

    # for each nnnness until the new/old - 1 > end_tol
    new_term = 100
    nnnness = start_nnnness - 1
    while new_term > end_tol:
        # increment nnnness
        nnnness += 1
        old_sum = current_sum

        # get the atoms which are nnn away from the muon
        muon, All_Spins, _ = AtomObtainer.get_spins(muon_position, squish_radius, lattice_type, lattice_parameter,
                                                    lattice_angles, input_coord_units, atomic_basis,
                                                    perturbed_distances, max_nn_search_radius=max_nn_search_radius,
                                                    nnnness=nnnness, exclusive_nnnness=True, shutup=True)

        # calculate r
        r = (muon.position - All_Spins[1].position).r()

        # calculate this term in the sum
        new_term = (len(All_Spins) - 1)/pow(r, 6)

        # update sum
        current_sum = current_sum + new_term

        # calculate extent of convergence
        if old_sum != 0:
            rel_diff = current_sum/old_sum - 1
        else:
            rel_diff = 999

        # print out findings
        print(str(nnnness) + '\t' + str(r) + '\t' + str(len(All_Spins) - 1) + '\t' + str(new_term) + '\t' + str(current_sum)
              + '\t' + str(rel_diff))

    return current_sum