# DecoherenceCalculator.py - Calculate decoherence of muon state in any lattice with any structure

from MDecoherenceAtom import TDecoherenceAtom as atom  # import class for decoherence atom
import TCoord3D as coord  # 3D coordinates class
import AtomObtainer  # to obtain atoms from pw output file
import numpy as np  # for matrices
import scipy.sparse as sparse  # for sparse matrices
import scipy.linalg as linalg  # for linear algebra
import matplotlib.pyplot as pyplot  # for plotting
from datetime import datetime  # for date and time printing in the output file
from enum import Enum  # for enumerations for the inputs - makes everything a lot easier to read!
import subprocess  # to get the current git version


# class to define the enumerations for the different types of lattices available - same as pw.x
class ibrav(Enum):
    OTHER = 0  # lattice given by a b c alpha beta gamma
    CUBIC_SC = 1
    CUBIC_FCC = 2
    CUBIC_BCC = 3
    CUBIC_BCC_EXTRA = -3
    HEX_TRIG_PRIMITIVE = 4
    TRIG_RHOM_3FC = 5
    TRIG_RHOM_3F111 = -5
    TETRAGONAL_ST = 6
    TETRAGONAL_BCT = 7
    ORTHORHOMBIC_SO = 8
    ORTHORHOMBIC_BCO = 9
    ORTHORHOMBIC_BCO_ALT = -9
    ORTHORHOMBIC_BCO_ATYPE = 91
    ORTHORHOMBIC_FCO = 10
    ORTHORHOMBIC_BODYCENTRED = 11
    MONOCLINIC_UC = 12
    MONOCLINIC_UB = -12
    MONOCLINIC_BC_UC = 13
    MONOCLINIC_BC_UB = -13
    TRICLINIC = 14
    # not all are tested, so be careful using them!


# units for the positions given in the inputs - same as pw.x
class position_units(Enum):
    ALAT = 1  # coordinates are in terms of A, the lattice parameter in cartesian coordinates
    ANGSTROM = 3  # coordinates are in angstroms, in cartesian coordinates
    CRYSTAL = 4  # coordinates are in terms of the primitive crystal vectors
    # do the rest when required


# nnn finder
def nnn_finder(basis, muon, lattice_translation, nn=2, exclusive_nnnness=False, perturbations=None, squish_radii=None):
    # function which returns an array of TCoord3D of the nn etc
    # nn parameter: =2 for nn, 3 for nnn, 4 for nnnn...
    # nn_pert_distance: distance of the nn bond, set to None if perturbing manually/not perturbing at all

    # sort out mutability
    if perturbations is None:
        perturbations = []

    # get over silly inputs
    if nn < 2:
        nn = 2

    # convert basis into muon-centred basis
    muon_basis = [muon.position - atom.position for atom in basis]

    # guess where the muon is (in terms of n m l) through solving equations
    lattice_vector_matrix = np.column_stack([vector.totuple() for vector in lattice_translation])

    # list of n m and l to check
    nml_list = [[], [], []]

    # for each type of atom...
    for muon_atom_position in muon_basis:
        # set up the vector for the equation
        muatpos = muon_atom_position.toarray()

        # solve the matrices, and find n m l, and add these to the list of all n m l to check
        exact_nml = np.linalg.solve(lattice_vector_matrix, muatpos)

        # for each term in exact_nml, look +-1 in each direction (floor+-(nn-1) will do for now)
        for i in range(0, len(exact_nml)):
            flr_nml = np.floor(exact_nml[i])
            for nm_or_l in range(int(flr_nml) - (nn - 1), int(flr_nml) + (nn + 1)):
                # is this in the list?
                if nm_or_l not in nml_list[i]:
                    nml_list[i].append(nm_or_l)

    nearestneighbours = []

    # for each n m l combination and basis, find all the nn, nnn, nnnn etc...
    for n in nml_list[0]:
        for m in nml_list[1]:
            for l in nml_list[2]:
                for atom_basis in basis:
                    # find the coordinates of the atom wrt muon
                    atom_position = lattice_translation[0] * n + lattice_translation[1] * m \
                                    + lattice_translation[2] * l + atom_basis.position
                    # see if this is in the list of perturbations, if so then perturb this particular atom
                    for perturbation in perturbations:
                        if atom_position == perturbation[0]:
                            atom_position = perturbation[1]
                    # find r (distance from muon)
                    r = (atom_position - muon.position).r()
                    # ...and save in a list
                    nearestneighbours.append([r, atom_position, atom_basis])

    # sort the list by radius
    nearestneighbours.sort(key=lambda atom: atom[0])

    # find the closest two F atoms, and perturb by means of squisification
    ## the below squishification code is crap - so rmed
    # closest_F_radius = 0
    # for atom in nearestneighbours:  # atom is [mu-atom distance, position, TDecoherenceAtom object]
    #     if atom[2].name == 'F':
    #         if closest_F_radius == 0:
    #             closest_F_radius = atom[0]  # if no Fs have been registered yet, use this radius as the reference
    #         elif (atom[0] - closest_F_radius) < 1e-3:
    #             # this atom is also a nn
    #             pass
    #         else:
    #             break
    #         # if we get here, it means this atom needs squishification (if desired)
    #         if squish_radius is not None:
    #             atom[0] = squish_radius
    #             atom[1].set_r(squish_radius, muon.position)
    #             atom[2].position = atom[1]

    # sort the list by radius (just the beginning atoms might've changed)
    # nearestneighbours.sort(key=lambda atom: atom[0]) ## .. probably wrote this after a wine and cheese!!

    # now see what the radii are, and only spit out what we need
    current_nn = 1
    current_radius = 0
    chopped_nn = []
    for atom in nearestneighbours:
        # if we're at a bigger r than previously (up to a tolerance for rounding errors)
        if atom[0] - current_radius > 1e-4:
            # iterate nn
            current_nn = current_nn + 1
            # if current nn too big, exit
            if current_nn > nn:
                break
            # reset current radius
            current_radius = atom[0]
        # if we're at the right nn, add this to the list
        if (current_nn <= nn and not exclusive_nnnness) or (current_nn == nn and exclusive_nnnness):
            # if this is due for squishification, then squish
            try:
                squish_radius = squish_radii[current_nn-2]
                if squish_radius is not None:
                    atom[0] = squish_radius
                    atom[1].set_r(squish_radius, muon.position)
                    atom[2].position = atom[1]
            except IndexError:
                pass
            chopped_nn.append(atom)

    # return the nn asked for
    return chopped_nn


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


# calculate entire hamiltonian
def calc_dipolar_hamiltonian(spins):
    current_hamiltonian = 0

    # calculate hamiltonian for each pair and add onto sum
    for i in range(0, len(spins)):
        for j in range(i + 1, len(spins)):
            current_hamiltonian = current_hamiltonian + calc_hamiltonian_term(spins, i, j)
    return current_hamiltonian


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


# do decoherence file preamble
def decoherence_file_preamble(file, muon_position, nn_atoms, fourier, starttime=None, endtime=None, timestep=None,
                              fourier_2d=None, tol=None, use_xtl_input=None, xtl_input_location=None,
                              use_pw_output=None, perturbed_distances=None, squish_radius=None, nnnness=None,
                              exclusive_nnnness=None, lattice_type=None, lattice_parameter=None):
    # program name, date and time completed
    file.writelines('! Decoherence Calculator Output - ' + datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + '\n!\n')

    # get the git version
    version_label = subprocess.check_output(["git", "describe", "--always"]).strip()
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

    atoms_file_preamble(file, muon_position, nn_atoms, use_xtl_input, xtl_input_location, use_pw_output,
                        perturbed_distances, squish_radius, nnnness, exclusive_nnnness, lattice_type, lattice_parameter)

    file.writelines('! start of data: \n')


def atoms_file_preamble(file, muon_position, nn_atoms, use_xtl_input=None, xtl_input_location=None, use_pw_output=None,
                        perturbed_distances=None, squish_radius=None, nnnness=None, exclusive_nnnness=None,
                        lattice_type=None, lattice_parameter=None):
    # data source, if used
    if use_xtl_input:
        file.writelines('! Atom positional data was obtained from XTL (fractional crystal coordinate) file: ' +
                        xtl_input_location + '\n')
    if use_pw_output:
        file.writelines('! Atom positional data was obtained from QE PWSCF file: ' + xtl_input_location + '\n')

    # Basis atoms, with gyromag ratios and I values
    for atom in nn_atoms:
        lines_to_write = atom.verbose_description(gle_friendly=True)
        for line in lines_to_write:
            file.writelines(line)
    file.writelines('!\n')

    # muon position
    file.writelines('! muon position: ' + str(muon_position) + ' \n! \n')

    # atom perturbations
    if len(perturbed_distances) > 0:
        file.writelines('! atom position perturbations: \n')
        for iperturbpair in range(0, len(perturbed_distances)):
            file.writelines('!\t ' + str(perturbed_distances[iperturbpair][0]) + ' to '
                            + str(perturbed_distances[iperturbpair][1]) + '\n')
        file.writelines('! \n')
    elif isinstance(squish_radius, float) and not use_pw_output:
        file.writelines('! nearest neighbour F-mu radius adjusted to be ' + str(squish_radius) + ' angstroms. \n!\n')

    # nnn ness
    file.writelines('! Calculated by looking at ')
    for i in range(0, nnnness):
        file.writelines('n')

    file.writelines(' interactions \n! \n')

    if exclusive_nnnness == True and not use_pw_output:
        file.writelines('! Effects of interactions of atoms spatially closer than ')
        for i in range(0, nnnness):
            file.writelines('n')
        file.writelines(' have been ignored. \n! \n')

    # lattice type and parameter
    if not use_xtl_input:
        file.writelines('! lattice type: ' + str(lattice_type) + ' (based on QE convention) \n')
        file.writelines('! lattice parameter: ' + str(lattice_parameter) + ' Angstroms \n! \n')


def breit_rabi_file_preamble(file, field_polarisation, fields, muon_position, nn_atoms, use_xtl_input=None,
                             xtl_input_location=None, use_pw_output=None, perturbed_distances=None, squish_radius=None,
                             nnnness=None, exclusive_nnnness=None, lattice_type=None, lattice_parameter=None):
    # program name, date and time completed
    file.writelines('! Decoherence Calculator Output - ' + datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + '\n!\n')

    # get the git version
    version_label = subprocess.check_output(["git", "describe", "--always"]).strip()
    file.writelines('! Using version ' + str(version_label) + '\n!\n')

    file.writelines('! Breit-Rabi output, from ' + str(fields[0]) + 'G to ' + str(fields[-1]) + 'G.\n')
    file.writelines('! Field was applied in the direction of ' + str(field_polarisation) + '\n!\n')

    atoms_file_preamble(file, muon_position, nn_atoms, use_xtl_input, xtl_input_location, use_pw_output,
                        perturbed_distances, squish_radius, nnnness, exclusive_nnnness, lattice_type, lattice_parameter)


# batch write data to file
def write_to_file(file, t, P):
    for i in range(0, len(t) - 1):
        file.writelines(str(t[i]) + ' ' + str(P[i]) + '\n')


def main():
    #### INPUT ####

    # ## IF WE'RE USING PW_OUTPUT
    # pw_output_file_location = 'CaF2.relax.mu.pwo'
    # no_atoms = 11  # includes muon

    ## IF WE'RE USING AN XTL (crystal fractional coordinates) FILE
    # xtl_input_location = 'CaF2_final_structure_reduced.xtl'
    # (don't forget to define nnnness!)

    squish_radii = [1.172211, None]  # radius of the nn F-mu bond after squishification (1.18 standard, None for no squishification)

    ## IF WE'RE NOT USING pw output:
    # nn, nnn, nnnn?
    # nnnness = 2  # 2 = nn, 3 = nnn etc
    # exclusive_nnnness - if TRUE, then only calculate nnnness's interactions (and ignore the 2<=i<nnnness interactions)
    #   exclusive_nnnness = False

    ## IF NOT PW NOR XTL:
    # lattice type: https://www.quantum-espresso.org/Doc/INPUT_PW.html#idm45922794628048
    lattice_type = ibrav.CUBIC_FCC  # # can only do fcc and monoclinic (unique axis b)
    # lattice parameters and angles, in angstroms
    lattice_parameter = [5.44542, 0, 0]  # [a, b, c]
    lattice_angles = [90, 0, 0]  # [alpha, beta, gamma] in **degrees**

    # are atomic coordinates provided in terms of alat or in terms of the primitive lattice vectors?
    input_coord_units = position_units.ALAT

    # atoms and unit cell: dump only the basis vectors in here, the rest is calculated
    atomic_basis = [
        # atom(coord.TCoord3D(0, 0, 0), gyromag_ratio=np.array([18.0038, 0]), II=np.array([7, 0]), name='Ca',
        #     abundance=np.array([0.00145, 0.99855])),
        atom(coord.TCoord3D(0.25, 0.25, 0.25), gyromag_ratio=251.713, II=1, name='F'),
        atom(coord.TCoord3D(0.25, 0.25, 0.75), gyromag_ratio=251.713, II=1, name='F')
    ]

    # register the perturbed distances
    perturbed_distances = []

    # define muon position
    muon_position = coord.TCoord3D(.25, 0.25, 0.5)
    muon_polarisation = coord.TCoord3D(0, 0, 1)

    # calc_decoherence(muon_position=muon_position, squish_radius=squish_radii, lattice_type=lattice_type,
    #                  lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
    #                  input_coord_units=input_coord_units, atomic_basis=atomic_basis,
    #                  perturbed_distances=perturbed_distances, plot=True, nnnness=3, ask_each_atom=False,
    #                  fourier=False, fourier_2d=False, tol=1e-3, times=np.arange(0, 10, 0.1))

    # calc_entropy(muon_position=muon_position, squish_radius=squish_radii, lattice_type=lattice_type,
    #              lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
    #              muon_polarisation=muon_polarisation, input_coord_units=input_coord_units, atomic_basis=atomic_basis,
    #              perturbed_distances=perturbed_distances, nnnness=2, ask_each_atom=False)

    calc_breit_rabi(muon_position=muon_position, squish_radius=squish_radii, lattice_type=lattice_type,
                    lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
                    input_coord_units=input_coord_units, atomic_basis=atomic_basis,
                    perturbed_distances=perturbed_distances, nnnness=2, ask_each_atom=False,
                    fields=np.arange(0, 40, 0.05), field_polarisation=coord.TCoord3D(1,0,0),
                    outfile_location='/Users/johnny/Desktop/CaF2_xfield.dat', plot=True)


# from input data, generate a vector [..] of TDecoherenceAtoms which have positions in a muon-centred basis.
# return muon, All_spins, success
def get_spins(muon_position, squish_radius=None,
                     # arguments for manual input of lattice
                     lattice_type=None, lattice_parameter=None, lattice_angles=None,
                     input_coord_units=position_units.ALAT, atomic_basis=None, perturbed_distances=None,
                     # arguments for XTL
                     use_xtl_input=False, xtl_input_location=None,
                     # arguments for XTL or manual input
                     nnnness=2, exclusive_nnnness=False,
                     # arguments for pw.x output
                     use_pw_output=False, pw_output_file_location=None, no_atoms=0, ask_each_atom=False):

    # if told to use both pw and xtl, exit
    if use_pw_output and use_xtl_input:
        print('Cannot use pw and xtl inputs simultaneously. Aborting...')
        return None, None, False

    # check that everything is in order: if not, then leave
    if not use_pw_output and not use_xtl_input:
        # should have manually entered all the details in
        if lattice_type is None or lattice_parameter is None or lattice_angles is None or atomic_basis is None or \
                perturbed_distances is None:
            print('Not enough information given. Aborting...')
            return None, None, False
    elif use_pw_output:
        if pw_output_file_location is None or no_atoms <= 0:
            print('Not enough information given. Aborting...')
            return None, None, False
    else:
        if xtl_input_location is None:
            print('Not enough information given. Aborting...')
            return None, None, False

    if use_pw_output:
        # get the atoms from the Quantum Espresso pw.x output, and put into an array
        muon, nnn_atoms = AtomObtainer.get_atoms_from_pw_output(pw_output_file_location, no_atoms)
        All_Spins = [muon]
        for each_atom in nnn_atoms:
            All_Spins.append(each_atom)
        muon_position = muon.position
    else:
        if not use_xtl_input:
            # define a b, c, alpha, beta, gamma for clarity
            a = lattice_parameter[0]
            b = lattice_parameter[1]
            c = lattice_parameter[2]
            alpha = lattice_angles[0] * np.pi / 180.
            beta = lattice_angles[1] * np.pi / 180.
            gamma = lattice_angles[2] * np.pi / 180.

            # define primitive vectors a1, a2 and a3, from pw.x input description
            if lattice_type == ibrav.CUBIC_SC:
                # simple cubic
                a1 = coord.TCoord3D(a, 0, 0)
                a2 = coord.TCoord3D(0, a, 0)
                a3 = coord.TCoord3D(0, 0, a)
            elif lattice_type == ibrav.CUBIC_FCC:
                # fcc cubic
                a1 = coord.TCoord3D(-a * .5, 0, a * .5)
                a2 = coord.TCoord3D(0, a * .5, a * .5)
                a3 = coord.TCoord3D(-a * .5, a * .5, 0)
            elif lattice_type == ibrav.CUBIC_BCC:
                a1 = coord.TCoord3D(.5 * a, .5 * a, .5 * a)
                a2 = coord.TCoord3D(-.5 * a, .5 * a, .5 * a)
                a3 = coord.TCoord3D(-.5 * a, -.5 * a, .5 * a)
            elif lattice_type == ibrav.CUBIC_BCC_EXTRA:
                a1 = coord.TCoord3D(-.5 * a, .5 * a, .5 * a)
                a2 = coord.TCoord3D(.5 * a, -.5 * a, .5 * a)
                a3 = coord.TCoord3D(.5 * a, .5 * a, -.5 * a)
            elif lattice_type == ibrav.MONOCLINIC_UB:
                # monoclinic, unique axis b
                a1 = coord.TCoord3D(a, 0, 0)
                a2 = coord.TCoord3D(0, b, 0)
                a3 = coord.TCoord3D(c * np.cos(beta), 0, c * np.sin(beta))
            elif lattice_type == ibrav.OTHER:
                # other lattice type - a1 a2 a3 defined manually - so don't worry
                pass
            else:
                assert False

            primitive_lattice_vectors = [a1, a2, a3]

            # sort out the basis issues (this is just to avoid clogging the input area!)
            if input_coord_units == position_units.CRYSTAL:
                # CRYSTAL units: everything is in terms of crystal vectors
                # this means that the basis for the vectors is indeed the PLV - so change the position of the atoms to take
                # this into account in all coordinates

                # sort out the atoms:
                for basis_atom in atomic_basis:
                    basis_atom.position.set_basis(primitive_lattice_vectors)

                # sort out the perturbed pairs
                for perturbed_pair in perturbed_distances:
                    perturbed_pair[0].set_basis(primitive_lattice_vectors)
                    perturbed_pair[1].set_basis(primitive_lattice_vectors)

                # finally, do the muon
                muon_position.set_basis(primitive_lattice_vectors)
            elif input_coord_units == position_units.ALAT:
                # don't bother with basis if ALAT - just multiply all the coordinates by a!
                # sort out the atoms:
                for basis_atom in atomic_basis:
                    basis_atom.position = basis_atom.position * a

                # sort out the perturbed pairs
                for perturbed_pair in perturbed_distances:
                    perturbed_pair[0] = perturbed_pair[0] * a
                    perturbed_pair[1] = perturbed_pair[1] * a

                # finally, do the muon
                muon_position = muon_position * a
            else:
                # we're in cartesian-land with the distances given in angstroms - so do nothing!
                pass

            # create muon
            muon = atom(muon_position, gyromag_ratio=851.372, II=1, name='mu')
        else:
            # import the fractional coordinates from the XTL
            muon, atomic_basis, [a1, a2, a3] = AtomObtainer.get_atoms_from_xtl(xtl_file_location=xtl_input_location)
            lattice_type = ibrav.OTHER
            muon_position = muon.position

        # now what we want to do is calculate how many of these are nn, nnn, nnnn etc
        nnn_atoms = nnn_finder(atomic_basis, muon, [a1, a2, a3], nnnness, exclusive_nnnness,
                               perturbed_distances, squish_radius)

        # if ask_each_atom is True, ask the user if they want to include each individual atom
        if ask_each_atom:
            approved_nnn_atoms = []
            for this_nnn_atom in nnn_atoms:
                if AtomObtainer.query_yes_no('Include ' + str(this_nnn_atom) + '?'):
                    approved_nnn_atoms.append(this_nnn_atom)
            nnn_atoms = approved_nnn_atoms

        # as before, make a list of spins to calculate (including that of the muon)
        All_Spins = [muon]
        for i_atom in nnn_atoms:
            All_Spins.append(
                atom(i_atom[1], i_atom[2].gyromag_ratio, i_atom[2].II, i_atom[2].name, i_atom[2].abundance))

    # print the atoms in the list
    for i_atom in All_Spins:
        print(i_atom)

    # turn the spins into a muon-centred basis
    for spin in All_Spins:
        for isotopeid in range(0, len(spin)):
            spin[isotopeid].position = spin[isotopeid].position - muon_position

    return muon, All_Spins, True


def calc_decoherence(muon_position, squish_radius=None, times=np.arange(0, 10, 0.1),
                     # arguments for manual input of lattice
                     lattice_type=None, lattice_parameter=None, lattice_angles=None,
                     input_coord_units=position_units.ALAT, atomic_basis=None, perturbed_distances=None,
                     # arguments for XTL
                     use_xtl_input=False, xtl_input_location=None,
                     # arguments for XTL or manual input
                     nnnness=2, exclusive_nnnness=False,
                     # arguments for pw.x output
                     use_pw_output=False, pw_output_file_location=None, no_atoms=0,
                     # other arguments
                     fourier=False, fourier_2d=False, outfile_location=None, tol=1e-10, plot=False, shutup=False,
                     ask_each_atom=False):

    # type of calculation - can't do fourier2d if not fourier
    fourier_2d = fourier_2d and fourier

    # get the atoms and the muon
    muon, All_Spins, got_atoms = get_spins(muon_position, squish_radius, lattice_type, lattice_parameter, lattice_angles,
                                           input_coord_units, atomic_basis, perturbed_distances, use_xtl_input,
                                           xtl_input_location, nnnness, exclusive_nnnness, use_pw_output,
                                           pw_output_file_location, no_atoms, ask_each_atom)

    # count number of spins
    N_spins = len(All_Spins) - 1

    # count the number of combinations of isotopes
    isotope_combinations = 1
    for atoms in All_Spins:
        isotope_combinations = isotope_combinations * len(atoms)
    if not shutup:
        print(str(isotope_combinations) + ' isotope combination(s) found')

    # put all these number of isotopes into an array
    number_isotopes = [len(atom) for atom in All_Spins]

    current_isotope_ids = inc_isotope_id(basis=number_isotopes)

    # create frequency and amplitude arrays
    E = list()
    amplitude = list()
    const = 0
    while current_isotope_ids[0] != -1:  # the end signal is emitted by making the id of 0 = -1
        # put this combination of isotopes into an array (Spins), and calculate probability of this state
        probability = 1.
        Spins = []
        for atomid in range(0, len(All_Spins)):
            Spins.append(All_Spins[atomid][current_isotope_ids[atomid]])
            probability = probability * All_Spins[atomid][current_isotope_ids[atomid]].abundance

        # create measurement operators for the muon's spin
        muon_spin_x = 2*measure_ith_spin(Spins, 0, Spins[0].pauli_x)
        muon_spin_y = 2*measure_ith_spin(Spins, 0, Spins[0].pauli_y)
        muon_spin_z = 2*measure_ith_spin(Spins, 0, Spins[0].pauli_z)

        # calculate hamiltonian
        hamiltonian = calc_dipolar_hamiltonian(Spins)

        # find eigenvalues and eigenvectors of hamiltonian
        if not shutup:
            print("Finding eigenvalues...")
        dense_hamiltonian = hamiltonian.todense()
        this_E, R = linalg.eigh(dense_hamiltonian)
        Rinv = R.H
        if not shutup:
            print("Found eigenvalues:")
            print(this_E)

        # Calculate constant (lab book 1 page 105)
        thisconst = 0
        for i in range(0, len(R)):
            thisconst = thisconst + pow(abs(Rinv[i] * muon_spin_x * R[:, i]), 2) \
                        + pow(abs(Rinv[i] * muon_spin_y * R[:, i]), 2) \
                        + pow(abs(Rinv[i] * muon_spin_z * R[:, i]), 2)
        const = const + probability * thisconst / (6 * (muon_spin_x.shape[0] / 2))

        this_amplitude = np.zeros((len(R), len(R)))
        # now calculate oscillating term
        for i in range(0, len(R)):
            Rx = Rinv[i] * muon_spin_x
            Ry = Rinv[i] * muon_spin_y
            Rz = Rinv[i] * muon_spin_z
            if not shutup:
                print(str(100 * i / len(R)) + '% complete...')
            if fourier_2d:
                jmin = 0
            else:
                jmin = i + 1
            for j in range(jmin, len(R)):
                this_amplitude[i][j] = (pow(abs(Rx * R[:, j]), 2)
                                        + pow(abs(Ry * R[:, j]), 2)
                                        + pow(abs(Rz * R[:, j]), 2)) * probability / (3 * (muon_spin_x.shape[0] / 2))

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
            decoherence_file_preamble(file=outfile, muon_position=muon_position, nn_atoms=All_Spins, fourier=fourier,
                                      fourier_2d=fourier_2d, tol=tol, use_xtl_input=use_xtl_input,
                                      xtl_input_location=xtl_input_location, use_pw_output=use_pw_output,
                                      perturbed_distances=perturbed_distances, squish_radius=squish_radius, nnnness=nnnness,
                                      exclusive_nnnness=exclusive_nnnness, lattice_type=lattice_type,
                                      lattice_parameter=lattice_parameter)

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
        for time in np.nditer(times):
            if not shutup:
                print("t=" + str(time))
            P_average.append(calc_p_average_t(time, const, amplitude, E).max())
            # print(P_average[-1])

        if outfile_location is not None:
            # dump results in a file if requested
            outfile = open(outfile_location, "w")
            # do preamble
            decoherence_file_preamble(file=outfile, muon_position=muon_position, nn_atoms=All_Spins, fourier=fourier,
                                      fourier_2d=fourier_2d, tol=tol, use_xtl_input=use_xtl_input,
                                      xtl_input_location=xtl_input_location, use_pw_output=use_pw_output,
                                      perturbed_distances=perturbed_distances, squish_radius=squish_radius, nnnness=nnnness,
                                      exclusive_nnnness=exclusive_nnnness, lattice_type=lattice_type,
                                      lattice_parameter=lattice_parameter, starttime=times[0], endtime=times[-1],
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


def calc_entropy(muon_position, muon_polarisation: coord, squish_radius=None,
                     # arguments for manual input of lattice
                     lattice_type=None, lattice_parameter=None, lattice_angles=None,
                     input_coord_units=position_units.ALAT, atomic_basis=None, perturbed_distances=None,
                     # arguments for XTL
                     use_xtl_input=False, xtl_input_location=None,
                     # arguments for XTL or manual input
                     nnnness=2, exclusive_nnnness=False,
                     # arguments for pw.x output
                     use_pw_output=False, pw_output_file_location=None, no_atoms=0, ask_each_atom=False):

    muon, Spins, got_spins = get_spins(muon_position, squish_radius, lattice_type, lattice_parameter, lattice_angles,
                                         input_coord_units, atomic_basis, perturbed_distances, use_xtl_input,
                                         xtl_input_location, nnnness, exclusive_nnnness, use_pw_output,
                                         pw_output_file_location, no_atoms, ask_each_atom)

    # create measurement operators for the muon's spin
    muon_spin_x = 2*Spins[0].pauli_x
    muon_spin_y = 2*Spins[0].pauli_y
    muon_spin_z = 2*Spins[0].pauli_z

    # generate density matrix (=.5(1+muon_rhat) x 1^N)
    muon_spin_polariasation = muon_polarisation.ortho_x * muon_spin_x + muon_polarisation.ortho_y * muon_spin_y + \
                              muon_polarisation.ortho_z * muon_spin_z

    # calcualte the density matrix at time 0
    density_matrix_0 = sparse.kron(0.5*(sparse.identity(2) + muon_spin_polariasation),
                                   sparse.identity(pow(2, len(Spins)-1))) / pow(2, len(Spins)-1)

    # do a partial trace
    density_matrix_0 = trace(density_matrix_0.tocsr(), left_dim=2)

    # do x*log(x)
    rho_log_rho = xlogx(density_matrix_0)

    # do trace
    print(-1*trace(rho_log_rho))


def trace(matrix, left_dim: int = 0, right_dim: int = 0):
    # check matrix is square, and the input arguments are not silly
    assert matrix.shape[0] == matrix.shape[1]
    assert left_dim >= 0
    assert left_dim >= 0

    # if not doing partial trace, then
    if left_dim == 0 and right_dim == 0:
        current_trace = 0
        for i in range(0, matrix.shape[0]):
            current_trace += matrix[i][i]
    elif left_dim > 0:
        # trace over left dims
        # calculate what the final dimension should be (will always be an integer)
        res_dim = int(np.round(matrix.shape[0]/left_dim))
        # set up final matrix (and allow it to be complex)
        current_trace = np.zeros((res_dim, res_dim,), dtype=complex)
        # for each element in the final matrix (i,j)
        for i in range(0, res_dim):
            for j in range(0, res_dim):
                # sum over the indices
                for s in range(0, left_dim):
                    current_trace[i][j] = current_trace[i][j] + matrix[i + s*res_dim,j + s*res_dim]
        current_trace = sparse.coo_matrix(current_trace)
    elif right_dim > 0:
        # trace over right dims
        # calculate what the final dimension should be (will always be an integer)
        res_dim = int(np.round(matrix.shape[0] / right_dim))
        # set up final matrix (and allow it to be complex)
        current_trace = np.zeros((res_dim, res_dim,), dtype=complex)
        # for each element in the final matrix (i,j)
        for i in range(0, res_dim):
            for j in range(0, res_dim):
                # sum over the indices
                for s in range(0, right_dim):
                    current_trace[i][j] = current_trace[i][j] + matrix[i*right_dim + s, j*right_dim + s]
        current_trace = sparse.coo_matrix(current_trace)
    else:
        # something has gone horribly wrong!
        assert False

    return current_trace

# look up python compiler


# do x*log(x) of a matrix (takes into account lim(x->0)xlogx = 0)
def xlogx(x):
    # diagonalise x
    E, R, Rinv = linalg.eig(x.todense(), left=True, right=True)

    # log each element in E, unless it's 0, then make it 0
    for i in range(0, len(E)):
        if E[i] != 0:
            E[i] = E[i]*np.log(E[i])

    # make E a matrix
    E = sparse.diags(E)

    # return xlogx
    return R*E*Rinv


def calc_breit_rabi(muon_position, squish_radius=None, fields=np.arange(0, 1, 1e3),
                    field_polarisation=coord.TCoord3D(0, 0, 1),
                    # arguments for manual input of lattice
                    lattice_type=None, lattice_parameter=None, lattice_angles=None,
                    input_coord_units=position_units.ALAT, atomic_basis=None, perturbed_distances=None,
                    # arguments for XTL
                    use_xtl_input=False, xtl_input_location=None,
                    # arguments for XTL or manual input
                    nnnness=2, exclusive_nnnness=False,
                    # arguments for pw.x output
                    use_pw_output=False, pw_output_file_location=None, no_atoms=0,
                    # other arguments
                    outfile_location=None, plot=False,
                    ask_each_atom=False):

    # if no outfile nor plot is initiated, no point in continuing...
    assert not (outfile_location is None and plot is False)

    # normalise the magnetic field polarisation vector
    field_polarisation = field_polarisation / field_polarisation.r()

    # get the atoms and the muon
    muon, All_Spins, got_atoms = get_spins(muon_position, squish_radius, lattice_type, lattice_parameter,
                                           lattice_angles,
                                           input_coord_units, atomic_basis, perturbed_distances, use_xtl_input,
                                           xtl_input_location, nnnness, exclusive_nnnness, use_pw_output,
                                           pw_output_file_location, no_atoms, ask_each_atom)

    # work out how many energies we should have
    num_energies = 1
    for spin in All_Spins:
        num_energies *= spin.II + 1

    # open the output file
    output_file = None
    if outfile_location is not None:
        # set up the output file
        output_file = open(outfile_location, 'w+')
        breit_rabi_file_preamble(output_file, field_polarisation, fields, muon_position, All_Spins, use_xtl_input,
                                 xtl_input_location, use_pw_output, perturbed_distances, squish_radius, nnnness,
                                 exclusive_nnnness, lattice_type, lattice_parameter)
        output_file.write('! field (G) ')
        for i in range(0, num_energies):
            output_file.write('E' + str(i) + ' (MHz) ')
        output_file.write('\n')

    # calculate the dipolar Hamiltonian
    dipolar_hamiltonian = calc_dipolar_hamiltonian(All_Spins)

    # if plotting, set up the arrays to plot
    energies = None
    if plot:
        energies = np.zeros((num_energies, len(fields)))


    for i_field in range(0, len(fields)):
        # calculate the Zeeman terms
        field_v = field_polarisation*fields[i_field]*1e-4
        hamiltonian = dipolar_hamiltonian + calc_zeeman_hamiltonian(All_Spins, field_v)

        # print out the current field as a status update
        print('Field: ' + str(fields[i_field]) + 'G')

        # diagonalise the Hamiltonian
        E, R = linalg.eigh(hamiltonian.todense())

        # append the eigenvalues to the list if plotting (no reason to save them otherwises...)
        if plot:
            energies[:, i_field] = E

        # write to file
        if output_file is not None:
            output_file.write(str(fields[i_field]) + ' ')
            for energy in E:
                output_file.write(str(energy) + ' ')
            output_file.write('\n')

    if plot:
        for i in range(0, num_energies):
            pyplot.plot(fields, energies[i, :])
        pyplot.xlabel('Field (G)')
        pyplot.ylabel('E /MHz')
        pyplot.title('F-mu-F, Field in x-direction')
        pyplot.show()

    if output_file is not None:
        output_file.close()


if __name__ == '__main__':
    main()
