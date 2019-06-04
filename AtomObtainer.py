# AtomObtainer.py - obtain the atoms required for DecoherenceCalculator
# Only currently implemented for QE files, but may be extended to other files, etc in the future (we'll see...)
# 14/4/19 - Extended for xtl files (these can be exported from CIF files with VESTA.)

from MDecoherenceAtom import TDecoherenceAtom as atom  # import class for decoherence atom
import TCoord3D as coord  # 3D coordinates class
import sys  # for user input
import numpy as np  # for numpy arrays


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
