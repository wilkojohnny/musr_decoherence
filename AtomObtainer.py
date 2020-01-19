# AtomObtainer.py - obtain the atoms required for DecoherenceCalculator
# Only currently implemented for QE files, but may be extended to other files, etc in the future (we'll see...)
# 14/4/19 - Extended for xtl files (these can be exported from CIF files with VESTA.)

from MDecoherenceAtom import TDecoherenceAtom as atom  # import class for decoherence atom
import TCoord3D as coord  # 3D coordinates class
import sys  # for user input
import numpy as np  # for numpy arrays
from enum import Enum
import copy

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


def get_atoms_from_pw_output(pw_file_location: str, num_atoms=2):

    # open pw output file
    try:
        pw_out_file = open(pw_file_location, 'r')
    except IOError:
        print('File read error.')
        return

    # find the cell type
    current_line, pw_out_file = move_to_line(pw_out_file, 'bravais-lattice index')
    cell_type = get_numbers_from_str(current_line, True)

    # read the lattice parameter
    current_line, pw_out_file = move_to_line(pw_out_file, 'celldm')
    # now we're on the correct line
    # split the string based on spaces
    split_line = current_line.split()
    # get celldm(1)
    cell_dm_1 = float(split_line[split_line.index("celldm(1)=")+1])/1.8897  # 1.8897 converts bohr units to Angstroms
    # do the others as and when needed

    # read the crystal axes a1, a2 and a3, put into basis
    current_line, pw_out_file = move_to_line(pw_out_file, 'crystal axes:')

    # get the primitive crystal axes
    prim_vectors = []
    for i in range(0, 3):
        current_line = pw_out_file.readline()
        vector = get_numbers_from_str(current_line)
        prim_vectors.append(coord.TCoord3D(cell_dm_1*vector[0], cell_dm_1*vector[1], cell_dm_1*vector[2]))

    # go to the part of the file which lists final positions of the atoms
    current_line, pw_out_file = move_to_line(pw_out_file, 'Begin final coordinates')
    current_line, pw_out_file = move_to_line(pw_out_file, 'ATOMIC_POSITIONS')

    # for each atom...
    template_atoms = []
    current_atoms = []
    current_line = pw_out_file.readline()
    muon = None
    while 'End final coordinates' not in current_line:
        # split the line up
        atom_raw_data = current_line.split()
        atom_name = atom_raw_data[0]
        atom_position = coord.TCoord3D(float(atom_raw_data[1]), float(atom_raw_data[2]), float(atom_raw_data[3]))
                                       # , prim_vectors) removed - coordinates are in angstroms not primitive vectors.
        # have we seen this atom before?
        found_template_atom = False
        for template_atom in template_atoms:
            found_template_atom = template_atom == atom_name
            if found_template_atom:
                # this atom has a template - so duplicate, and give the new position
                current_atoms.append(template_atom.duplicate(atom_position))
                break

        # if this atom doesn't have a template, ask for the details
        if not found_template_atom:
            # work out (with the help of the user) what this types of atom is or and whether its a muon
            is_atom, new_atom = get_atom_information(atom_name, atom_position, muon is None)
            if is_atom:
                template_atoms.append(new_atom)
                current_atoms.append(new_atom)
            else:
                muon = new_atom

        # read next line in file
        current_line = pw_out_file.readline()

    # finished with file, so close it
    pw_out_file.close()

    # if no muon specified
    if muon is None:
        print("No muon found. Bye")
        exit()

    # calculate the distance between the atom and muon for each atom
    distance_atom_array = [[e_atom, (e_atom.position - muon.position).r()] for e_atom in current_atoms]

    # put into order based on radius from the muon
    distance_atom_array.sort(key=lambda a: a[1])

    # get the top n atoms, and return
    distance_atom_array = distance_atom_array[:-(len(distance_atom_array)-num_atoms)]

    final_atoms = [distance_atom_element[0] for distance_atom_element in distance_atom_array]

    return muon, final_atoms


def get_atoms_from_xtl(xtl_file_location: str):

    # open pw output file
    try:
        pw_out_file = open(xtl_file_location, 'r')
    except IOError:
        print('File read error.')
        return

    # define the cell type
    cell_type = 0

    # read the lattice parameters
    current_line, pw_out_file = move_to_line(pw_out_file, 'CELL')
    current_line = pw_out_file.readline()
    # now we're on the correct line
    # split the string based on spaces
    split_line = current_line.split()
    # get a b c alpha beta gamma (convert alpha beta gamma into radians)
    a = float(split_line[0])
    b = float(split_line[1])
    c = float(split_line[2])
    alpha = float(split_line[3])*np.pi/180
    beta = float(split_line[4])*np.pi/180
    gamma = float(split_line[5])*np.pi/180

    # define the primitive crystal axes
    a1 = coord.TCoord3D(a, 0, 0)
    a2 = coord.TCoord3D(b*np.cos(gamma), b*np.sin(gamma), 0)
    a3 = coord.TCoord3D(c*np.cos(beta), c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma),
                        c*np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma) -
                        np.power(np.cos(alpha), 2)-np.power(np.cos(beta), 2)-np.power(np.cos(gamma), 2))
                        / np.sin(gamma))
    prim_vectors = [a1, a2, a3]

    # go to the part of the file which lists positions of the atoms
    current_line, pw_out_file = move_to_line(pw_out_file, 'ATOMS')
    current_line, pw_out_file = move_to_line(pw_out_file, 'NAME')

    # for each atom...
    template_atoms = []
    current_atoms = []
    current_line = pw_out_file.readline()
    muon = None
    while 'EOF' not in current_line:
        # split the line up
        atom_raw_data = current_line.split()
        atom_name = atom_raw_data[0]
        atom_position = coord.TCoord3D(float(atom_raw_data[1]), float(atom_raw_data[2]), float(atom_raw_data[3])
                                       , prim_vectors)
        # have we seen this atom before?
        found_template_atom = False
        for template_atom in template_atoms:
            found_template_atom = template_atom == atom_name
            if found_template_atom:
                # this atom has a template - so duplicate, and give the new position
                current_atoms.append(template_atom.duplicate(atom_position))
                break

        # if this atom doesn't have a template, ask for the details
        if not found_template_atom:
            # work out (with the help of the user) what this types of atom is or and whether its a muon
            is_atom, new_atom = get_atom_information(atom_name, atom_position, False)
            template_atoms.append(new_atom)
            current_atoms.append(new_atom)

        # read next line in file
        current_line = pw_out_file.readline()

    # finished with file, so close it
    pw_out_file.close()

    # get the muon position, and create the muon posiiton
    muon_position = ask_atom_info('What is the position of the muon (in fractional coordinates)?', 3, isInt=False)
    muon_position = coord.TCoord3D(muon_position[0], muon_position[1], muon_position[2], prim_vectors)
    muon = atom(muon_position, gyromag_ratio=851.372, II=1, name='mu')

    return muon, current_atoms, prim_vectors


def move_to_line(file, line):
    # move to the line in the file which contains the string line
    current_line = ''
    while line not in current_line:
        try:
            current_line = file.readline()
        except EOFError:
            # if we reach the end of the file without finding the line, wrong file format
            print('File format invalid')
            return
    return current_line, file


def get_numbers_from_str(string: str, IntOnly=False):
    # extract the floats from the input string
    # split the string by spaces
    if IntOnly:
        converter = int
    else:
        converter = float

    split_string = string.split()
    output_nos = []
    for substring in split_string:
        # for each substring, see if it's a number - if it is, add to the array
        try:
            output_nos.append(converter(substring))
        except ValueError:
            pass
    return output_nos


def get_atom_information(atom_name, atom_position, ask_if_muon=False, is_muon=False):
    print("Found an atom, " + atom_name + " at position " + str(atom_position) + "\n")
    # ask if this is a muon
    if ask_if_muon:
        is_muon = query_yes_no("Is this a muon?", default="no")

    if is_muon:
        return False, atom(atom_position, gyromag_ratio=851.372, II=1, name='mu')

    # this is not a muon, so ask for the information
    atom_gyromag_ratio = ask_atom_info("Gyromagnetic ratio /MHz (*2pi)")
    if isinstance(atom_gyromag_ratio, np.ndarray):
        n_isotopes = len(atom_gyromag_ratio)
    else:
        n_isotopes = 1
    atom_II = ask_atom_info("2*I", n_isotopes, True)
    if n_isotopes > 1:
        atom_abundances = ask_atom_info("Abundances (decimal)", n_isotopes)
    else:
        atom_abundances = 1

    return True, atom(atom_position, atom_gyromag_ratio, atom_II, atom_name, atom_abundances)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def ask_atom_info(question, no_responses=None, isInt=False):
    while True:
        print(question + " (separate multiple with a space) > ", end="")
        response = input()

        if isInt:
            converter = int
        else:
            converter = float

        # try to split the response
        split_response = response.split()

        # if not enough given, complain
        if no_responses is not None and no_responses != len(split_response):
            print("Incorrect number of responses given")
        else:
            if len(split_response) == 1:
                try:
                    output = converter(response)
                    return output
                except ValueError:
                    print("Invalid value")
            elif len(split_response) == 0:
                return 0
            else:
                # if the length of responses is >1
                output = []
                satisfactory_output = True
                for each_response in split_response:
                    try:
                        output.append(converter(each_response))
                    except ValueError:
                        print("Invalid value")
                        satisfactory_output = False
                if satisfactory_output:
                    return np.array(output)

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

    atomic_basis = copy.deepcopy(atomic_basis)

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
        muon, nnn_atoms = get_atoms_from_pw_output(pw_output_file_location, no_atoms)
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
            muon, atomic_basis, [a1, a2, a3] = get_atoms_from_xtl(xtl_file_location=xtl_input_location)
            lattice_type = ibrav.OTHER
            muon_position = muon.position

        # now what we want to do is calculate how many of these are nn, nnn, nnnn etc
        nnn_atoms = nnn_finder(atomic_basis, muon, [a1, a2, a3], nnnness, exclusive_nnnness,
                               perturbed_distances, squish_radius)

        # if ask_each_atom is True, ask the user if they want to include each individual atom
        if ask_each_atom:
            approved_nnn_atoms = []
            for this_nnn_atom in nnn_atoms:
                if query_yes_no('Include ' + str(this_nnn_atom) + '?'):
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
            except IndexError:
                squish_radius = None
            if squish_radius is not None:
                atom[0] = squish_radius
                atom[1].set_r(squish_radius, muon.position)
                atom[2].position = atom[1]
            chopped_nn.append(atom)

    # return the nn asked for
    return chopped_nn


def atoms_file_preamble(file, muon_position, nn_atoms, use_xtl_input=None, xtl_input_location=None, use_pw_output=None,
                        pw_output_location=None, perturbed_distances=None, squish_radius=None, nnnness=None,
                        exclusive_nnnness=None, lattice_type=None, lattice_parameter=None):
    # data source, if used
    if use_xtl_input:
        file.writelines('! Atom positional data was obtained from XTL (fractional crystal coordinate) file: ' +
                        xtl_input_location + '\n')
    if use_pw_output:
        file.writelines('! Atom positional data was obtained from QE PWSCF file: ' + pw_output_location + '\n')

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