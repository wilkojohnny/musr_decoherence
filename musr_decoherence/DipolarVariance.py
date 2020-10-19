# DipolePolarisation.py -- calculates sum_{nnnness} [n atoms in nnnness]/r_{nnnness}^6, until convergence
# John Wilkinson 16/12/19
from . import AtomObtainer  # for nnnfinder
from ase import Atoms, build
from . import MDecoherenceAtom
import numpy as np
import copy


# calc_required_perturbation -- calculates the perturbation of the pert_nnnness nuclei (radially towards (-) or away (+)
# from original positions).
def calc_required_perturbation_old(muon_position, squish_radius=None, pert_nnnness=2, end_nnnness=-1, end_tol=1e-6,
                                   max_nn_search_radius=20,
                                   # arguments for manual input of lattice
                                   lattice_type=None, lattice_parameter=None, lattice_angles=None,
                                   input_coord_units=AtomObtainer.position_units.ALAT, atomic_basis=None,
                                   perturbed_distances=None):
    # calculate the variance
    b_variance = calc_variance_old(muon_position=muon_position, squish_radius=squish_radius, start_nnnness=pert_nnnness,
                                   end_nnnness=end_nnnness, end_tol=end_tol, max_nn_search_radius=max_nn_search_radius,
                                   lattice_type=lattice_type, lattice_parameter=lattice_parameter,
                                   lattice_angles=lattice_angles, input_coord_units=input_coord_units,
                                   atomic_basis=atomic_basis, perturbed_distances=perturbed_distances)

    # get the start_nnnness term
    muon, all_spins, _ = AtomObtainer.get_spins(muon_position=muon_position, squish_radius=squish_radius,
                                                lattice_type=lattice_type, lattice_parameter=lattice_parameter,
                                                lattice_angles=lattice_angles, input_coord_units=input_coord_units,
                                                atomic_basis=atomic_basis, perturbed_distances=perturbed_distances,
                                                max_nn_search_radius=max_nn_search_radius, nnnness=pert_nnnness,
                                                exclusive_nnnness=True)

    # calculate the required perturbation
    # calculate the
    gyro_ii = 0
    for i in range(1, len(all_spins)):
        gyro_ii += all_spins[i].II / 2 * (all_spins[i].II / 2 + 1) * pow(all_spins[i].gyromag_ratio, 2)

    print(gyro_ii)
    print(all_spins[1].position.r())
    required_perturbation = pow(b_variance / gyro_ii, -1 / 6) - all_spins[1].position.r()

    print('required perturbation = ' + str(required_perturbation))

    return required_perturbation


def calc_variance_old(muon_position, squish_radius=None, start_nnnness=2, end_nnnness=-1, end_tol=1e-6,
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
            integral_extra = 4 * 3.1415926 / (3 * pow(r, 3)) * average_gyromag_i
            print('Adding on an extra ' + str(integral_extra) + ' with the integral')
            current_sum = current_sum + integral_extra
            print('Total variance of B-field is ' + str(current_sum))
            break
        old_sum = current_sum

        # get the atoms which are nnn away from the muon
        muon, all_spins, _ = AtomObtainer.get_spins(muon_position, squish_radius, lattice_type, lattice_parameter,
                                                    lattice_angles, input_coord_units, atomic_basis,
                                                    perturbed_distances, max_nn_search_radius=max_nn_search_radius,
                                                    nnnness=nnnness, exclusive_nnnness=True, shutup=False)

        # calculate r
        r = (muon.position - all_spins[1].position).r()

        # calculate this term in the sum
        new_term = 0
        for i in range(1, len(all_spins)):
            new_term += all_spins[i].II / 2 * (all_spins[i].II / 2 + 1) * pow(all_spins[i].gyromag_ratio, 2)
        new_term *= 1 / pow(r, 6)

        # update sum
        current_sum = current_sum + new_term

        # calculate extent of convergence
        if old_sum != 0:
            rel_diff = current_sum / old_sum - 1
        else:
            rel_diff = 999

        # print out findings
        print(str(nnnness) + '\t' + str(r) + '\t' + str(len(all_spins) - 1) + '\t' + str(new_term) + '\t' +
              str(current_sum) + '\t' + str(rel_diff))

    return current_sum


def calc_lambda_squish(unit_cell: Atoms, squish_nnnness: list, max_exact_distance: float = 20,
                       included_atoms: list = None, nnnness_tol=1e-3):
    """
    Calculates the required lambda_squish to account for the atoms beyond those in squish_nnnness.
    :param unit_cell: ASE atoms, including muon
    :param squish_nnnness: list of nnnnesses which are to be squished, e.g to squish nnn only do [3], or for nnn+nnnn do [3, 4]
    :param max_exact_distance: maximum distance between muon--atom for the exact calculation; beyond this an integral
                                is used to calculate the contribution to the NMR linewidth
    :param included_atoms: list of strings of atoms to include in the calculation (e.g ['F'], for just F, ['F', 'Na']
                            for Na and F)
    :param nnnness_tol: maximum distance between two nuclei for them to both be considered as nearest-neighbours.
    :return: value of lambda_squish
    """

    squish_variance_contribution, non_squish_variance_contribution, _ = \
        calc_dipolar_second_moment_sums(unit_cell, squish_nnnness, max_exact_distance, included_atoms, nnnness_tol)

    # find lambda_squish
    return (squish_variance_contribution / (non_squish_variance_contribution + squish_variance_contribution)) ** (1 / 6)


def calc_unsquished_distance(r_squish: float, unit_cell: Atoms, squish_nnnness: list, max_exact_distance: float = 20,
                             included_atoms: list = None, nnnness_tol=1e-3):
    """
    Calculates the value of the muon-squish_nnnness distance BEFORE squishing. Used to turn a fitted value of mu-x
    distance into the 'actual' distance.
    :param r_squish: value of mu-x distance after squishing (i.e what the fitting algorithm gives)
    :param unit_cell: ASE atoms, including muon
    :param squish_nnnness: list of nnnnesses which are to be squished, e.g to squish nnn only do [3], or for nnn+nnnn do [3, 4]
    :param max_exact_distance: maximum distance between muon--atom for the exact calculation; beyond this an integral
                                is used to calculate the contribution to the NMR linewidth
    :param included_atoms: list of strings of atoms to include in the calculation (e.g ['F'], for just F, ['F', 'Na']
                            for Na and F)
    :param nnnness_tol: maximum distance between two nuclei for them to both be considered as nearest-neighbours.
    :return: value of lambda_squish
    :return: mu-squish_nnnesss distance before squishing
    """

    _, non_squish_variance_contribution, nuclear_magnetic_factor = \
        calc_dipolar_second_moment_sums(unit_cell, squish_nnnness, max_exact_distance, included_atoms, nnnness_tol)

    return (r_squish ** -6 - non_squish_variance_contribution / nuclear_magnetic_factor) ** (-1/6)


def calc_dipolar_second_moment_sums(unit_cell: Atoms, squish_nnnness: list, max_exact_distance: float = 20,
                                    included_atoms: list = None, nnnness_tol=1e-3) -> (float, float, float):
    """
    Calculates the sum of the second moment of the dipolar field
    :param unit_cell: ASE atoms, including muon
    :param squish_nnnness: list of nnnnesses which are to be squished, e.g to squish nnn only do [3], or for nnn+nnnn do [3, 4]
    :param max_exact_distance: maximum distance between muon--atom for the exact calculation; beyond this an integral
                                is used to calculate the contribution to the NMR linewidth
    :param included_atoms: list of strings of atoms to include in the calculation (e.g ['F'], for just F, ['F', 'Na']
                            for Na and F)
    :param nnnness_tol: maximum distance between two nuclei for them to both be considered as nearest-neighbours.
    :return: (float, float, float): contribution to the second moment due to the nuclei which will end up 'squished',
                                  contribution to the second moment due to the nuclei which will NOT be 'squished' (i.e
                                  will be represented by the 'squished' nuclei),
                                  magnetic factor (sum over I(I+1)*gyromag_ratio^2) for squish atoms.
    """
    unit_cell = copy.deepcopy(unit_cell)

    # find where the muon is in atoms object
    muon_unit_cell_id = None
    for atom_id, atom in enumerate(unit_cell):
        if atom.symbol == 'mu':
            muon_unit_cell_id = atom_id
        elif atom.symbol == 'H' and muon_unit_cell_id is None:
            muon_unit_cell_id = -1*atom_id

    assert muon_unit_cell_id is not None

    if muon_unit_cell_id < 0:
        # The muon's label is H
        muon_unit_cell_id *= -1

    # by calculating the minimum distance between the muon and the supercell edge, guess the supercellness required
    muon_scaled_position = unit_cell.get_scaled_positions()[muon_unit_cell_id]
    muon_scaled_cell_distances = np.abs(np.round(muon_scaled_position, 0) - muon_scaled_position)
    unit_cell_lengths = unit_cell.get_cell_lengths_and_angles()[0:3]
    mu_cell_dist = muon_scaled_cell_distances * unit_cell_lengths
    # calculate (roughly) how much of a supercell we need, using the max_exact_distance (and make it odd!)
    supercell_dim = 2 * max(np.ceil((max_exact_distance - mu_cell_dist) / unit_cell_lengths)) + 1

    # remove muon and build the supercell
    del unit_cell[muon_unit_cell_id]
    supercell = build.make_supercell(unit_cell, np.diag([supercell_dim, supercell_dim, supercell_dim]))

    # put the muon back in
    muon_pos_supercell_scaled = (muon_scaled_position + (supercell_dim - 1) / 2) / supercell_dim
    muon_supercell = Atoms(symbols=['mu'], scaled_positions=[muon_pos_supercell_scaled], cell=supercell.get_cell())
    muon_supercell_muon = muon_supercell[0]
    supercell.append(muon_supercell_muon)

    # delete the muon_supercell as this should never be used again
    del muon_supercell

    # order all the atoms by distance from the muon, if it is wanted...
    muon_distances = []
    for i_atom, atom in enumerate(supercell):
        if atom.symbol in included_atoms:
            muon_distance = supercell.get_distances(-1, [i_atom])
            muon_distances.append((atom.symbol, muon_distance))

    # sort the muon distances
    muon_distances = sorted(muon_distances, key=lambda element: element[1])

    # check our supercell is big enough -- if not, kick up (means I need to fix something!)
    assert muon_distances[-1][1] >= max_exact_distance

    # \sigma^2 for the atoms being squished by \lambda_squish, and those that are not
    squish_variance_contribution = 0
    non_squish_variance_contribution = 0
    nuclear_magnetic_factor = 0
    current_nnnness = 1
    current_mu_at_dist = 0
    start_nnnness = min(squish_nnnness)
    for symbol, distance_from_muon in muon_distances:
        # are we in a new nnnness shell?
        if distance_from_muon > current_mu_at_dist + nnnness_tol:
            # yes we are
            current_nnnness += 1
            current_mu_at_dist = distance_from_muon
            # if distance is too great, break
            if distance_from_muon > max_exact_distance:
                break
        # print nnnness
        if current_nnnness <= start_nnnness + 1:
            [print('n', end='') for _ in range(0, current_nnnness)]
            print(': ' + symbol + ' at distance '+ str(distance_from_muon), end='\n')
        # do we care about this nnnness?
        if current_nnnness >= start_nnnness:
            # we do -- so calculate the lambda contribution for these up to the distance required
            gyromagnetic_ratio = MDecoherenceAtom.nucleon_properties[symbol]["gyromag_ratio"]
            II = MDecoherenceAtom.nucleon_properties[symbol]["II"]
            I = II / 2
            # dont do isotopes for now
            assert MDecoherenceAtom.nucleon_properties[symbol]["abundance"] == 1

            this_nuclear_magnetic_factor = I * (I + 1) * gyromagnetic_ratio ** 2
            lambda_contribution = this_nuclear_magnetic_factor / (distance_from_muon ** 6)
            if current_nnnness in squish_nnnness:
                squish_variance_contribution += lambda_contribution
                nuclear_magnetic_factor += this_nuclear_magnetic_factor
            else:
                non_squish_variance_contribution += lambda_contribution


    # add on the integral to deal with everything up to infinity
    cell_density = supercell.get_number_of_atoms() / supercell.get_volume()
    integral_contribution = (4 * np.pi / 3) * 1 / (max_exact_distance ** 6) * cell_density
    non_squish_variance_contribution += integral_contribution

    return squish_variance_contribution, non_squish_variance_contribution, nuclear_magnetic_factor