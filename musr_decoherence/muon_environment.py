"""
muon_environment.py -- for working out the atomic environment the muon is in.
Replaces AtomObtainer.py and AtomObtainerN.py
Created by John Wilkinson, 10/11/20 (the day after the vaccine announcement!)
"""

from ase import atoms, atom, build
import copy
import numpy as np
from .TCoord3D import TCoord3D as coord
from . import MDecoherenceAtom


def add_muon_to_aseatoms(ase_atoms: atoms, theta: float = 180, phi: float = 0, nn_indices: list = None,
                         muon_position: np.ndarray = None, plane_atom_index: int = None,
                         plane_atom_position: np.ndarray = None, midpoint=0.5) -> atoms:
    """
    adds a muon to ase_atoms
    :param ase_atoms: ASE atoms of the crystal the muon is going to be placed into
    :param theta: F--mu--F angle in degrees. Must also define nn_indices if this is not 180deg.
    :param phi: angle of the F--mu--F plane. Must also define theta and nn_indices to use this.
    :param nn_indices: ASE index of the nearest-neighbour atoms to place the muon in between. Do not define this AND
                       muon_position
    :param muon_position: position of the muon, in angstroms. Use EITHER this or nn_indices. Do not define this and
                          theta and phi
    :param plane_atom_index: index of atom which the muon moves away from to create the angle theta. Do not define this
                             and plane_atom_position.
    :param plane_atom_position: np array of position of the plane_atom (doesn't actually need to be a position of an
                                atom per se, but useful if it is. Do not define this and the index.
    :param midpoint: weighting of the midpoint to the two nnindices. 0 puts the muon on nn_indices[0], 1 puts it on
                     nn_indices[2]. 0.5 puts it in between the two.
    :return: ase_atoms with the muon
    """

    # check either muon_position xor nn_indices is None
    assert (muon_position is None) != (nn_indices is None)

    ase_atoms = copy.deepcopy(ase_atoms)

    # three possibilities:
    # 1) >2 nn_indices -> just find average, ignore angles
    # 2) 2 nn_indices or muon_position, theta not given (or 180) -> just find midpoint
    # 3) 2 nn_indices, theta and phi -- find muon_position using theta and phi

    # see how many nn_indices there are
    if nn_indices is not None:
        muon_position = np.zeros((3,))
        # possibility 1 (or 2)
        if len(nn_indices) > 2 or (len(nn_indices) == 2 and (theta == 180 or theta is None)):
            # if nn_indices are given, work out the average of them to get the muon position (there can be more than 2!)
            for nn_index in nn_indices:
                muon_position += ase_atoms[nn_index].position / len(nn_indices)
        elif len(nn_indices) == 2:
            # we need to calculate the muon position with theta and phi...

            # convert everything into TCoord3D objects (...easier to work with)
            nn_position_1 = ase_atoms[nn_indices[0]].position
            nn_position_1 = coord(nn_position_1[0], nn_position_1[1], nn_position_1[2])
            nn_position_2 = ase_atoms[nn_indices[1]].position
            nn_position_2 = coord(nn_position_2[0], nn_position_2[1], nn_position_2[2])

            # check the plane atom has been defined
            assert (plane_atom_index is None) != (plane_atom_position is None)

            if plane_atom_position is None:
                plane_atom_position = ase_atoms[plane_atom_index].position
            plane_atom_position_c = coord(plane_atom_position[0], plane_atom_position[1], plane_atom_position[2])

            muon_position = get_bent_muon_position(nn_position_1=nn_position_1, nn_position_2=nn_position_2,
                                                   plane_position=plane_atom_position_c, theta=theta, phi=phi,
                                                   midpoint=midpoint)

            muon_position = muon_position.tonumpyarray()

        else:
            print('Error with the muon position parameters.')
            assert False

    # now add the muon to the ASE atoms
    muon = atom.Atom('X', position=muon_position)
    ase_atoms.append(muon)

    return ase_atoms


def get_bent_muon_position(nn_position_1: coord, nn_position_2: coord, plane_position: coord, theta: float,
                           phi: float, midpoint: float) -> coord:
    """
    Get the position of the muon to create a bond of angle theta with nn_position_1 and nn_position_2, protruding
    out from plane_position. then rotates this bond by an angle theta wrt the axis nn_1-> nn_2.
    See pg 8 lab book 4 for details
    :param nn_position_1: TCoord3D of one of the nns
    :param nn_position_2: TCoord3D of the other nn
    :param plane_position: position of the atom (but doesn't have to be...) to define a plane with the nns, and to move
                           the muon away from
    :param theta: F--mu--F angle, in degrees
    :param phi: angle to rotate the F--mu--F bond around
    :param midpoint: the midpoint of nn_position_1 and nn_position_2 will be (nn_pos_1*midpoint+nn_pos_2*(1-midpoint))
    :return: TCoord3D of the position of the muon
    """

    # calcaulate mu_protrude (lambda in lab book; but I can't call it that because python...)
    m = (nn_position_2 - nn_position_1) * midpoint
    m_sq = m * m
    # import pdb; pdb.set_trace()
    p_m = (nn_position_1 + m - plane_position).rhat()

    c_theta = np.cos(theta * np.pi / 180)

    p_m_dot_m = p_m * m
    p_m_dot_m_sq = p_m_dot_m ** 2

    mu_protrude_sq = (m_sq - (2 * p_m_dot_m_sq - m_sq) * (c_theta ** 2)
                      + 2 * c_theta * np.sqrt((p_m_dot_m_sq - m_sq) * (p_m_dot_m_sq * (c_theta ** 2) - m_sq))) \
                     / (1 - (c_theta ** 2))
    mu_protrude = np.sqrt(mu_protrude_sq)
    if (theta % 360) > 180:
        mu_protrude *= -1

    # now do the phi-rotation:
    cos_phi = np.cos(phi * np.pi / 180)
    sin_phi = np.sin(phi * np.pi / 180)
    mx, my, mz = m.rhat().totuple()
    R = np.array([[cos_phi + mx * mx * (1 - cos_phi), mx * my * (1 - cos_phi) - mz * sin_phi,
                   mx * mz * (1 - cos_phi) + my * sin_phi],
                  [mx * my * (1 - cos_phi) + mz * sin_phi, cos_phi + my * my * (1 - cos_phi),
                   my * mz * (1 - cos_phi) - mx * sin_phi],
                  [mx * mz * (1 - cos_phi) - my * sin_phi, my * mz * (1 - cos_phi) + mx * sin_phi,
                   cos_phi + mz * mz * (1 - cos_phi)]])

    p_m = R.dot(p_m.tonumpyarray())
    p_m = coord(p_m[0], p_m[1], p_m[2])

    muon_position = nn_position_1 + m + p_m * mu_protrude

    return muon_position


def perturb_atoms(atoms_mu: atoms, perturbations: list, nn_supercell=3, nn_tol=1e-3) -> atoms:
    """
    perturb all the atoms in atoms_mu by the amount asked to by the perturbations in the list
    :param atoms_mu: ASE atoms including muon
    :param perturbations: either a list of [[ase_id, dist], [ase_id, dist],...] or [r_nn, r_nnn, ...].
    :param nn_supercell: supercell dimension to find nearest neighbours in
    :param nn_tol: tolerance for two nuclei to be considered to have the same nearest neighbour shell
    :return: ASE atoms with perturbations
    """

    # check the last atom in atoms_mu is a muon (label 'X') -- if not assert False
    assert atoms_mu[-1].symbol == 'X'

    if isinstance(perturbations[0], list):
        # we have a list of ase_id, dist
        for ase_id, dist in perturbations:
            atoms_mu.set_distance(ase_id, -1, dist, fix=1)
    else:
        # do nnnness
        # make a sc*sc*sc supercell
        nn_supercell += nn_supercell % 2
        atoms_mu_supercell = build.make_supercell(atoms_mu, np.diag([nn_supercell, nn_supercell, nn_supercell]))

        # get the most central muon
        muons = np.array([this_atom.position for this_atom in atoms_mu_supercell if this_atom.symbol == 'X'])
        muon_position = muons.mean(axis=0)
        for i_this_atom, this_atom in reversed(list(enumerate(atoms_mu_supercell))):
            if this_atom.symbol == 'X':
                atoms_mu_supercell.pop(i_this_atom)
        atoms_mu_supercell.append(atom.Atom(symbol='X', position=muon_position))

        # find all the distances to the most central muon, store in list with [id, distance]
        mu_distances = [[i for i in range(0, len(atoms_mu_supercell))],
                        atoms_mu_supercell.get_all_distances(mic=False)[-1]]
        mu_distances = list(map(list, zip(*mu_distances)))
        mu_distances = sorted(mu_distances, key=lambda l: l[1])

        # go up the nnness, perturbing as we go...
        nn_shell = -1
        nn_dist = 0
        for [atom_id, atom_distance] in mu_distances:
            if atom_distance == 0:
                continue
            if atom_distance > nn_dist + nn_tol:
                # new nnnness
                nn_shell += 1
                nn_dist = atom_distance
            try:
                atoms_mu_supercell.set_distance(atom_id, -1, distance=perturbations[nn_shell], fix=1)
            except IndexError:
                break
        atoms_mu = atoms_mu_supercell

    return atoms_mu


def make_supercell(atoms_mu:atoms, unperturbed_atoms: atoms = None, unperturbed_supercell=1,
                   small_output=False):
    """
    make a supercell with atoms_mu in the centre, and surrounded by unperturbed_supercell unperturbed_atoms
    :param atoms_mu: ASE atoms, maybe with distortions, including muon
    :param unperturbed_atoms: ASE atoms of unperturbed structure
    :param unperturbed_supercell: number of instances of unperturbed_atoms to bolt on to the end of atoms_mu
    :return: supercell of atoms_mu+unperturbed_supercell*unperturbed_atoms. If small_output==False, return ASE atoms,
             otherwise returns a list of [atom type, position]
    """

    # make the supercell structure
    atoms_mu = copy.deepcopy(atoms_mu)
    if unperturbed_atoms is None:
        unperturbed_atoms = copy.deepcopy(atoms_mu[:-1])
    else:
        unperturbed_atoms = copy.deepcopy(unperturbed_atoms)
    # if atoms_mu is already a supercell, then make unperturbed_atoms a supercell of the same size
    unperturbed_atoms = build.make_supercell(unperturbed_atoms, np.diag([1, 1, 1]) *
                                             atoms_mu.cell.lengths()[0] /
                                             unperturbed_atoms.cell.lengths()[0])
    muon = copy.deepcopy(atoms_mu[-1])

    output_list = []
    if small_output:
        output_list = [[this_atom.symbol, this_atom.position] for this_atom in atoms_mu]

    del atoms_mu[-1]

    for x_sign in range(-unperturbed_supercell, unperturbed_supercell + 1):
        for y_sign in range(-unperturbed_supercell, unperturbed_supercell + 1):
            for z_sign in range(-unperturbed_supercell, unperturbed_supercell + 1):
                if x_sign == y_sign == z_sign == 0:
                    continue
                translation_vector = np.sign(x_sign) * (abs(x_sign) - 1) * unperturbed_atoms.cell[0] + \
                                     np.sign(x_sign) * atoms_mu.cell[0]
                translation_vector += np.sign(y_sign) * (abs(y_sign) - 1) * unperturbed_atoms.cell[1] + \
                                      np.sign(y_sign) * atoms_mu.cell[1]
                translation_vector += np.sign(z_sign) * (abs(z_sign) - 1) * unperturbed_atoms.cell[2] + \
                                      np.sign(z_sign) * atoms_mu.cell[2]
                unperturbed_atoms.translate(translation_vector)
                for this_atom in unperturbed_atoms:
                    if small_output:
                        output_list.append([this_atom.symbol, copy.deepcopy(this_atom.position)])
                    else:
                        atoms_mu.append(this_atom)
                unperturbed_atoms.translate(-1 * translation_vector)

    atoms_mu.append(muon)

    if small_output:
        return output_list
    else:
        old_cell = atoms_mu.get_cell()
        atoms_mu.set_cell(old_cell*(2*unperturbed_supercell + 1), scale_atoms=False)
        atoms_mu.translate(unperturbed_supercell * old_cell[0])
        atoms_mu.translate(unperturbed_supercell * old_cell[1])
        atoms_mu.translate(unperturbed_supercell * old_cell[2])
        return atoms_mu


def get_dominant_nuclei(atoms_mu: atoms, nn_cutoff: int or None = None, hilbert_cutoff: int = 2048,
                        unperturbed_atoms: atoms = None, unperturbed_supercell=1) -> (atoms, list):
    """
    get the dominant nuclei in the interaction, until we hit either nn_degree or hilbert_cutoff.
    Works out dominance by mu_i/r^3 rather than by distance alone
    :param atoms_mu: ASE atoms including muon
    :param nn_cutoff: which shell of nuclei to include up to (and including). (1=F--mu--F, 2=FmuF+nnn, etc)
    :param hilbert_cutoff: hilbert space size cutoff. Warns if this breaks up a nearest-neighbour shell
    :param unperturbed_atoms: unperturbed structure to add to the supercell to make sure there are enough nns
    :param unperturbed_supercell: number of instances of unperturbed_atoms to bolt on to the end of atoms_mu
    :return: (ase atoms containing nuclei which have the dominant interaction, ase atoms of the supercell used to find
              these, list of the ids of the dominant nuclei in the supercell in argument 2)
    """

    atoms_mu = make_supercell(atoms_mu, unperturbed_atoms, unperturbed_supercell)

    # calculate all the distances between the muon and the nuclei
    mu_distances = [[i for i in range(0, len(atoms_mu))],
                    atoms_mu.get_all_distances(mic=False)[-1]]
    mu_distances = list(map(list, zip(*mu_distances)))
    mu_distances = mu_distances[:-1]

    mu_interactions = []
    for i_atom, mu_distance in mu_distances:
        # calculate the nuclear moment (or a quantity that's proportional to it...)
        symbol = atoms_mu[i_atom].symbol
        nuclear_properties = MDecoherenceAtom.nucleon_properties[symbol]
        hilbert_size = 1
        if isinstance(nuclear_properties['II'], np.ndarray):
            # calculate the average moment for all the isotopes
            moment = 0
            for i_isotope in range(0, len(nuclear_properties['II'])):
                moment += abs(nuclear_properties['gyromag_ratio'][i_isotope]) * nuclear_properties['II'][i_isotope] \
                          * nuclear_properties['abundance'][i_isotope]
                if nuclear_properties['II'] + 1 > hilbert_size:
                    hilbert_size = nuclear_properties['II'] + 1
        else:
            moment = abs(nuclear_properties['gyromag_ratio']) * nuclear_properties['II']
            hilbert_size = nuclear_properties['II'] + 1
        interaction_size = moment / (mu_distance ** 3)
        mu_interactions.append([i_atom, interaction_size, hilbert_size])

    mu_interactions = sorted(mu_interactions, key=lambda l: -l[1])

    current_interaction_size = np.inf
    current_interaction_shell = 0
    total_hilbert_size = 2
    surviving_atoms = atoms.Atoms()
    surviving_ids = []
    # go up in interactions, until we hit either the total hilbert size or the the nn_cutoff
    for [atom_id, interaction_size, hilbert_size] in mu_interactions:
        if interaction_size < current_interaction_size:
            # new interaction shell
            current_interaction_shell += 1
            current_interaction_size = interaction_size
            if nn_cutoff is not None and current_interaction_shell > nn_cutoff:
                break
        elif total_hilbert_size*hilbert_size > hilbert_cutoff:
            print('WARNING -- Breaking up a shell of nuclei with the hilbert cutoff. Maybe change the hilbert size?')
            break
        total_hilbert_size *= hilbert_size
        if total_hilbert_size > hilbert_cutoff:
            break
        surviving_atoms.append(atoms_mu[atom_id])
        surviving_ids.append(atom_id)

    # add the muon to the end
    surviving_atoms.append(atoms_mu[-1])
    surviving_ids.append(len(atoms_mu) - 1)

    return surviving_atoms, atoms_mu, surviving_ids


def calculate_quadrupoles(atoms_mu: atoms, dominant_indices: list, unperturbed_atoms: atoms = None, max_radius=50) \
    -> (list, list):
    """
    Calculate the EFGs of the atoms which have Q>0 in included_indices. Units of the EFGs are in Angstroms^-3.
    :param atoms_mu: perturbed atoms, including the muon at the end
    :param dominant_indices: indices of the dominant nuclei which will be used in the muon polarisation calculation.
                             Last output of get_dominant_nuclei.
    :param unperturbed_atoms: ASE atoms which have no perturbations, to use to create a big supercell to calculate the
                              EFGs
    :param max_radius: radius of ions to include to get the EFGs
    :return: list of EFG matrices, list of quadrupolar nuclei ids in atoms_mu
    """

    quadrupolar_nuclei = []
    nn_quad_indices = []

    # for each atom with a quadrupole moment
    for nn_index, sc_index in enumerate(dominant_indices):
        # if muon, can skip
        if atoms_mu[sc_index].symbol == 'X':
            continue
        try:
            Q = MDecoherenceAtom.nucleon_properties[atoms_mu[sc_index].symbol]['Q']
        except IndexError:
            continue
        if Q != 0:
            quadrupolar_nuclei.append(sc_index)
            nn_quad_indices.append(nn_index)

    if len(quadrupolar_nuclei) == 0:
        print('No Quadrupoles to calculate!')
        return [], []

    # if there are any quadrupolar nuclei, make the supercell
    no_supercells = int(np.ceil(max_radius / min(atoms_mu.cell.lengths())))
    atoms_supercell = make_supercell(atoms_mu, unperturbed_atoms, unperturbed_supercell=no_supercells,
                                     small_output=True)

    nuclear_symbols, nuclear_positions = tuple(map(list, zip(*atoms_supercell)))

    def efg_matrix(x, y, z):
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return np.array([[3 * x ** 2 - r ** 2, 3 * x * y, 3 * x * z],
                         [3 * x * y, 3 * y ** 2 - r ** 2, 3 * y * z],
                         [3 * x * z, 3 * y * z, 3 * z ** 2 - r ** 2]]) / (r ** 5)

    efgs = []

    # for each quadrupolarly active nucleus
    for quadrupolar_nuclear_id in quadrupolar_nuclei:
        quad_nuc_pos = atoms_mu[quadrupolar_nuclear_id].position
        distances_sq = np.array([(quad_nuc_pos[0] - nuc_pos[0]) ** 2 + (quad_nuc_pos[1] - nuc_pos[1]) ** 2
                                 + (quad_nuc_pos[2] - nuc_pos[2]) ** 2 for nuc_pos in nuclear_positions])
        distances = np.sqrt(distances_sq)

        current_charge = 0
        current_efg = np.zeros((3, 3))
        n = 0
        # for each nucleus (but not the one we're interested in!)
        for i_ion, ion in enumerate(atoms_supercell):
            if i_ion == quadrupolar_nuclear_id:
                continue

            this_symbol, this_position = atoms_supercell[i_ion]

            # replace 'X' with 'mu'
            if this_symbol == 'X':
                this_symbol = 'mu'

            current_distance = distances[i_ion]

            if current_distance > max_radius:
                continue

            # get the charge
            this_charge = MDecoherenceAtom.nucleon_properties[this_symbol]['charge']

            # find the total charge
            current_charge += this_charge

            # add on the EFG to the EFG matrix
            current_efg += this_charge * efg_matrix(quad_nuc_pos[0] - this_position[0],
                                                    quad_nuc_pos[1] - this_position[1],
                                                    quad_nuc_pos[2] - this_position[2])
            n += 1

        efgs.append(current_efg)

    return efgs, nn_quad_indices


def model_further_nuclei(nn_atoms_mu: atoms, nn_start: int = -1, draw_in_factor: float = None,
                         atoms_mu: atoms = None, nn_indices: list = None, unperturbed_atoms : atoms = None,
                         max_exact_distance = 150) -> atoms:
    """
    model the further nuclei by drawing in the furthest away nuclei by a factor draw_in_factor
    :param nn_atoms_mu: nn_atoms, including muon, which have been extracted for the calculation
    :param nn_start: nearest-neighbour degree to start drawing in from. -1 does the last shell,
                     -2 last and second-to-last, etc, or 2=nnn, 3=nnnn
    :param draw_in_factor: factor to draw atoms at nn_start and beyond in by. If not defined, tries to calculate it
                           automatically
    :param atoms_mu: ASE atoms, including muon, without the nnns extracted (only if not providing draw_in_factor)
    :param nn_indices: the indices of the nuclei in ase_atoms_mu to run the calculation on. The last index MUST
                       correspond to the muon. (only if not providing draw_in_factor)
    :param unperturbed_atoms: ASE atoms before muon perturbations were included (only if not providing draw_in_factor)
    :param max_exact_distance: largest distance to calculate the second moment at -- beyond this it just does an
                               integral (only if not providing draw_in_factor)
    :return: ASE atoms with the further nuclei modelled by drawing in the last nearest-neighbours
    """

    # calculate all the distances between the muon and the nuclei
    mu_distances = [[i for i in range(0, len(nn_atoms_mu))],
                    nn_atoms_mu.get_all_distances(mic=False)[-1]]
    mu_distances = list(map(list, zip(*mu_distances)))
    mu_distances = mu_distances[:-1]
    mu_distances = sorted(mu_distances, key=lambda l: l[1])

    # starting from the muon, organise the atoms by nearest-neighbours
    muon_index = len(nn_atoms_mu)-1
    assert nn_atoms_mu[muon_index].symbol == 'X'
    nn_ids = []
    current_distance = 0
    current_nn_ids = [muon_index]
    for i_atom, this_distance in mu_distances:
        # are we in a new nnn shell?
        if abs(current_distance - this_distance) > 1e-3:
            nn_ids.append(current_nn_ids)
            current_distance = this_distance
            current_nn_ids = []
        current_nn_ids.append(i_atom)
    nn_ids.append(current_nn_ids)

    if nn_start < 0:
        nn_start = len(nn_ids) + nn_start
        assert nn_start > 0

    if draw_in_factor is None:
        # convert the indices in nn_atoms_mu to those in atoms_mu
        atoms_mu_indices = [[nn_indices[i] for i in this_nn_shell] for this_nn_shell in nn_ids]
        draw_in_factor = calculate_draw_in_factor(atoms_mu=atoms_mu, nn_indices=nn_indices,
                                                  unperturbed_atoms=unperturbed_atoms,
                                                  draw_in_atoms=atoms_mu_indices[nn_start:],
                                                  max_exact_distance=max_exact_distance)

    print('draw_in_factor calculated as {:.4f} '.format(draw_in_factor))

    for nnnness in range(nn_start, len(nn_ids)):
        current_nnshell_atoms = nn_ids[nnnness]
        nnshell_distance = nn_atoms_mu.get_distance(-1, current_nnshell_atoms[0], mic=False)
        [nn_atoms_mu.set_distance(-1, current_nnshell_atoms[i], nnshell_distance*draw_in_factor, fix=0)
         for i, _ in enumerate(current_nnshell_atoms)]

    return nn_atoms_mu


def calculate_draw_in_factor(atoms_mu: atoms, nn_indices: list, unperturbed_atoms: atoms, draw_in_atoms: list,
                             max_exact_distance: float = 50) -> float:
    """
    calculate the drawing-in factor of the nuclei beyond a certain next-nearest-neighbour shell
    :param atoms_mu: ASE atoms, including muon, without the nnns extracted
    :param nn_indices: the indices of the nuclei in ase_atoms_mu to run the calculation on. The last index MUST
                       correspond to the muon.
    :param unperturbed_atoms: ASE atoms before muon perturbations were included
    :param draw_in_atoms: list of indexes to draw in (these must also be in nn_indices -- otherwise whats the point?!
    :param max_exact_distance: largest distance to calculate the second moment at -- beyond this it just does an
                               integral
    :return: the drawing-in factor to take into account the rest of the nuclei.
    """

    # turn draw_in_atoms into a 1D list if it is 2D
    if isinstance(draw_in_atoms[0], list):
        draw_in_atoms = [j for sub in draw_in_atoms for j in sub]

    # work out which atoms to ignore (i.e the ones in nn_indices but not in draw_in_atoms)
    ignored_atoms = set(nn_indices) - set(draw_in_atoms)

    # build a supercell
    no_supercells = int(np.ceil(max_exact_distance / min(atoms_mu.cell.lengths())))
    atoms_supercell = make_supercell(atoms_mu, unperturbed_atoms, unperturbed_supercell=no_supercells,
                                     small_output=True)

    nuclear_symbols, nuclear_positions = tuple(map(list, zip(*atoms_supercell)))

    # check nn_index[-1] is the muon index, and save this
    assert atoms_mu[nn_indices[-1]].symbol == 'X'
    muon_index = nn_indices[-1]
    muon_position = nuclear_positions[muon_index]

    mu_distances_sq = np.array([(muon_position[0] - nuc_pos[0]) ** 2 + (muon_position[1] - nuc_pos[1]) ** 2
                                + (muon_position[2] - nuc_pos[2]) ** 2 for nuc_pos in nuclear_positions])
    mu_distances = np.sqrt(mu_distances_sq)

    all_second_moment = 0
    squish_second_moment = 0

    # calculate the dipolar second moment for each nucleus in this supercell, closer than max_exact_distance
    for id, distance in enumerate(mu_distances):
        symbol = nuclear_symbols[id]
        if id in ignored_atoms:
            continue
        if distance > max_exact_distance:
            continue

        II = MDecoherenceAtom.nucleon_properties[symbol]['II']
        I = II / 2
        gyromag_ratio = MDecoherenceAtom.nucleon_properties[symbol]['gyromag_ratio']

        # dont do isotopes for now
        assert MDecoherenceAtom.nucleon_properties[symbol]["abundance"] == 1

        this_second_moment = I * (I + 1) * (gyromag_ratio ** 2) / (distance ** 6)

        all_second_moment += this_second_moment

        if id in nn_indices:
            squish_second_moment += this_second_moment

    # add on the integral to get the total second moment
    density = unperturbed_atoms.get_number_of_atoms() / unperturbed_atoms.get_volume()
    unit_cell_mag_factor = 0
    for this_atom in unperturbed_atoms:
        symbol = this_atom.symbol
        I = MDecoherenceAtom.nucleon_properties[symbol]['II'] / 2
        gyromag_ratio = MDecoherenceAtom.nucleon_properties[symbol]['gyromag_ratio']

        unit_cell_mag_factor += I * (I + 1) * (gyromag_ratio ** 2)
    unit_cell_mag_factor /= unperturbed_atoms.get_number_of_atoms()

    integral = 4 * np.pi / (3 * max_exact_distance) * density * unit_cell_mag_factor
    all_second_moment += integral

    # use this to calculate the drawing in factor
    draw_in_factor = (squish_second_moment / all_second_moment) ** (1/6)

    return draw_in_factor


def aseatoms_to_tdecoatoms(atoms_mu: atoms, muon_array_id: int = -1, muon_centred_coords: bool = True,
                           efgs = None, efg_ids = None) -> (MDecoherenceAtom.TDecoherenceAtom, list):
    """
    Converts ASE Atoms into an array of TDecoerenceAtom objects
    :param atoms_mu: ASE atoms, including muon
    :param muon_array_id: id of the muon location in atoms_mu
    :param muon_centred_coords: centre coordinates on the muon
    :param efgs: list of nparrays with the EFGs
    :param efg_ids: ids of the nuclei in atoms_mu which the efg in efgs correspond to (e.g efgs[i] is the EFG matrix 
                    corresponding to atoms_mu[efg_ids[i]]
    :return: muon, list of TDecoherenceAtoms (with muon in position 0)
    """

    if muon_centred_coords:
        muon_centred_coords = 1
    else:
        muon_centred_coords = 0

    if muon_array_id < 0:
        muon_array_id = len(atoms_mu) + muon_array_id

    # sort out muon
    muon = None
    muon_location = coord(0, 0, 0)
    if muon_array_id is not None:
        muon_location = coord(atoms_mu[muon_array_id].position[0], atoms_mu[muon_array_id].position[1],
                              atoms_mu[muon_array_id].position[2])
        muon = MDecoherenceAtom.TDecoherenceAtom(position=muon_location - muon_location * muon_centred_coords,
                                                 name='mu')

    All_spins = [muon]
    for i_atom in range(0, len(atoms_mu)):
        if i_atom != muon_array_id:
            atom_location = coord(atoms_mu[i_atom].position[0], atoms_mu[i_atom].position[1], atoms_mu[i_atom].position[2])
            # if there is no restrictions on the included nuclei, or the nucleus in question is marked for inclusion
            All_spins.append(MDecoherenceAtom.TDecoherenceAtom(position=atom_location - muon_location
                                                                        * muon_centred_coords,
                                                               name=atoms_mu[i_atom].symbol))
            if efg_ids is not None:
                if i_atom in efg_ids:
                    # get the index of the efg matrix
                    efg_index = efg_ids.index(i_atom)
                    All_spins[-1].efg = efgs[efg_index]

    return muon, All_spins
