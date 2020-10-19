"""
AtomObtainerN.py -- get atoms using the tools in ASE
Allows getting atoms from .cif or .pwo files, and for the user to define the muon location using a GUI (easier than
faffing with locations, squish factors, etc!)
Created 23/3/2020 by John Wilkinson (during COVID isolation!)
"""

import math

import copy

import numpy as np
from ase import Atoms, atom, io
from ase import build
from ase.gui.gui import GUI
from ase.gui.images import Images

from .MDecoherenceAtom import TDecoherenceAtom as Matom  # import class for decoherence atom
from .TCoord3D import TCoord3D as coord


def get_linear_fmuf_atoms(ase_atoms: Atoms, muon_position: np.ndarray, nnnness: int = 2, squish_radii: list = None,
                          lambda_squish: float = 1, enforce_nn_dist: float = 0, enforced_nn_indices: list = None,
                          included_nuclei: list = None) -> (Matom, list):
    """
    Get the TDecoAtoms muon, All_Spins in a *linear* F--mu--F state for these calculations
    :param ase_atoms: ASE atoms of the structure (without muon)
    :param muon_position: cartesian position of muon (in numpy array [x,y,z])
    :param nnnness: nnnness of the calculation
    :param squish_radii: list of perturbations to the structure for [nn, nnn, nnnn, ...]
    :param lambda_squish: (see https://arxiv.org/abs/2003.02762), the factor to perturb all the nns above what is
                           defined in squish_radii by. >1 is *not really* physical, but not impossible (as this
                           takes into account both physical nuclear movements *and* decoherence 'movements')
    :param enforce_nn_dist: perturbs the selected two atoms which define the muon location
    :param enforced_nn_indices: indices of the two atoms to perturb before nnnness
    :param included_nuclei: list of the symbols of nuclei to be converted. Useful if you want to ignore nuclei of a
                            certain type (e.g if they have small magnetic moments)
    :return: TDecoherenceAtom Muon, list[TDecoAtom] All_spins (including muon in pos 0)
    """

    # add the muon, doing the enforced distancing when doing so
    muon_aseatoms = add_muon_to_aseatoms(ase_atoms, muon_position, enforced_nn_indices, enforce_nn_dist)

    # find the nnnness atoms, and squishification
    nnn_aseatoms = ase_nnnfinder(atoms_mu=muon_aseatoms, nnnness=nnnness, squish_radii=squish_radii,
                                 lambda_squish=lambda_squish, supercell_size=5)

    # return muon, All_Spins
    return aseatoms_to_tdecoatoms(nnn_aseatoms, included_nuclei=included_nuclei)


def get_bent_fmuf_atoms(ase_atoms: Atoms, fluorines: list, plane_atom, fmuf_angle: float = 180, swing_angle: float = 0,
                        nnnness: int = 2, squish_radii: list = None, lambda_squish: float = 1,
                        included_nuclei: list = None, muon_centred_coords: bool = True,
                        nnnness_shells: bool = True) -> (Matom, list):
    """
    Get F--mu--F atom + nnnness environment for a bent F--mu--F bond (see pg 123 lab book 2)
    :param ase_atoms: ASE atoms of the structure, without the muon
    :param fluorines: The two fluorine nuclei which the muon primarily interacts with
    :param plane_atom: the atom, with which the 2F+plane_atom make up the plane which the F--mu--F bond is on,
                             or an angle swing_angle away from. (can be int of atom in ase_atoms, or coords)
    :param fmuf_angle: the F--mu--F bond angle
    :param swing_angle: the angle the planar F--mu--F molecule makes with the 2F+plane_atom plane
    :param nnnness: nnnness of the calculation
    :param squish_radii: list of perturbations to the structure for [nn, nnn, nnnn, ...], towards the muon.
    :param lambda_squish: (see https://arxiv.org/abs/2003.02762), the factor to perturb all the nns above what is
                           defined in squish_radii by. >1 is *not really* physical, but not impossible (as this
                           takes into account both physical nuclear movements *and* decoherence 'movements')
    :param included_nuclei: list of the symbols of nuclei to be converted. Useful if you want to ignore nuclei of a
                            certain type (e.g if they have small magnetic moments)
    :param muon_centred_coords: whether to translate coordinates such that the muon is at (0,0,0).
    :param nnnness_shells: whether to group the atoms in nnnness. If false, each atom has its own nnnness.
    :return: TDecoherenceAtom Muon, list[TDecoAtom] All_spins (including muon in pos 0)
    """

    mu_atoms = add_muon_to_aseatoms_bent(ase_atoms, fluorines, plane_atom, fmuf_angle, swing_angle)

    # mu_img = Images()
    # mu_img.initialize([mu_atoms])
    # mu_gui = GUI(mu_img)
    # mu_gui.run()

    mu_atoms_nnn = ase_nnnfinder(atoms_mu=mu_atoms, nnnness=nnnness, squish_radii=squish_radii,
                                 lambda_squish=lambda_squish, supercell_size=3, nnnness_shells=nnnness_shells)

    # return muon, All_Spins
    return aseatoms_to_tdecoatoms(mu_atoms_nnn, included_nuclei=included_nuclei, muon_centred_coords=muon_centred_coords)


def ase_nnnfinder(nnnness: int, pwo_file: str = None, atoms_mu: Atoms = None, squish_radii: list = None,
                  supercell_size: int = None, lambda_squish: float = 1, lambda_start_nnnness = None,
                  dft_correction: float = 0, nnnness_shells: bool = True) -> Atoms:
    """
    Find nearest-neighbours using ASE
    :param nnnness: nnnness to get
    :param pwo_file: location of pwo file to use. Do not define this and atoms_mu.
    :param atoms_mu: ASE atoms with muon. Do not define this and atoms_mu.
    :param squish_radii: list of perturbations to the structure for [nn, nnn, nnnn, ...]
    :param supercell_size: size of supercell to look for nearest neighbours in. Should be odd.
    :param lambda_squish: (see https://arxiv.org/abs/2003.02762), the factor to perturb all the nns above what is
                           defined in squish_radii by. >1 is *not really* physical, but not impossible (as this
                           takes into account both physical nuclear movements *and* decoherence 'movements')
    :param lambda_start_nnnness: nnnness where we start using lambda_squish. If None, it is applied at the end of
                                the squish_radii
    :param dft_correction: DFT correction factor to account for systematic DFT position deviations
    :param nnnness_shells: whether to group the atoms in nnnness. If false, each atom has its own nnnness.
    :return: ASE Atoms with just the nnnness asked for
    """

    # check only one of pwo_file or atoms_mu are set
    assert (pwo_file is None) ^ (atoms_mu is None)

    if lambda_start_nnnness is None and lambda_squish != 1:
        lambda_start_nnnness = len(squish_radii) + 2

    if not (0 <= dft_correction <= 1):
        print('WARNING - To be physical, DFT correction should really be between 0 and 1.')

    before_atoms_mu = None
    if pwo_file is not None:
        atoms_mu = io.read(pwo_file)
        before_atoms_mu = io.read(pwo_file, 0)

    # find out if we're using the symbol H or mu for the muon
    # trust me, there is not an easier way!
    muon_symbol = None
    for i_atom, this_atom in enumerate(atoms_mu):
        if this_atom.symbol == 'H':
            muon_symbol = 'H'
        elif this_atom.symbol == 'X':
            muon_symbol = 'X'
            break

    # before we faff with a supercell (which is actually a super-supercell, if we're doing a DFT calculation)
    # apply the DFT correction factor to all the atomic positions
    if dft_correction != 0.0 and dft_correction is not None:
        for i_atom, after_atom in enumerate(atoms_mu):
            if after_atom.symbol != muon_symbol:
                before_atom = before_atoms_mu[i_atom]
                # check the atom type hasn't mysteriously changed...
                assert before_atom.symbol == after_atom.symbol

                # get the positions
                before_atom_position = coord(before_atom.position[0], before_atom.position[1], before_atom.position[2])
                after_atom_position = coord(after_atom.position[0], after_atom.position[1], after_atom.position[2])

                # calculate the new position
                correction = (before_atom_position - after_atom_position)*dft_correction

                after_atom_position = after_atom_position + correction
                after_atom.position = after_atom_position.toarray()

    if supercell_size is None:
        supercell_size = 2 * nnnness + 1
    # supercell must have odd dimensions
    if supercell_size % 2 == 0:
        supercell_size += 1

    # make a supercell
    supercell = build.make_supercell(atoms_mu, np.diag([supercell_size, supercell_size, supercell_size]))

    # find the most central muon -- this will be in the average position of all of them. Also keep track of H atoms, as
    # sometimes DFT uses H instead of mu. H atoms have -ve positions -- so if the array has only negative entries,
    # make them all positive and continue. If there is one +ve entry, scrap the negative entries.
    supercell_muon_indexes = []
    for i_supercell_at in range(0, len(supercell)):
        if supercell[i_supercell_at].symbol == muon_symbol:
            supercell_muon_indexes.append(i_supercell_at)

    muon_pos_x = 0
    muon_pos_y = 0
    muon_pos_z = 0

    for supercell_muon_index in supercell_muon_indexes:
        pos = supercell[supercell_muon_index].position
        muon_pos_x += pos[0] / len(supercell_muon_indexes)
        muon_pos_y += pos[1] / len(supercell_muon_indexes)
        muon_pos_z += pos[2] / len(supercell_muon_indexes)

    # create a new set of atoms, which have no muons
    supercell_onemuon = Atoms()

    # remove all the muons
    for atom in supercell:
        if atom.symbol != muon_symbol:
            supercell_onemuon.append(atom)

    if before_atoms_mu is not None:
        before_supercell = build.make_supercell(before_atoms_mu,
                                                np.diag([supercell_size, supercell_size, supercell_size]))
        before_supercell_onemuon = Atoms()
        for atom in before_supercell:
            before_supercell_onemuon.append(atom)

    del supercell

    # add the muon back. As all the perturbations have been done already, don't need to worry about
    # enforce_nn or anything like that.
    supercell_onemuon = add_muon_to_aseatoms(supercell_onemuon, np.array([muon_pos_x, muon_pos_y, muon_pos_z]))

    # This vector is of the form [id, mu_distance]
    mu_distances = [[i for i in range(0, len(supercell_onemuon))], supercell_onemuon.get_all_distances(mic=False)[-1]]

    mu_distances = list(map(list, zip(*mu_distances)))
    mu_distances = sorted(mu_distances, key=lambda l: l[1])

    # return a new atoms object with only nnnness nearest neighbours
    nearest_neighbour_atoms = Atoms()
    nnnness_atoms = []
    current_nnnness = 1
    current_nnness_dist = 0
    nn_tol = 1e-3
    max_squish_radius = None
    for [atom_cellID, mu_at_dist] in mu_distances:
        # see if we are in a new sphere of nnnness
        if abs(current_nnness_dist - mu_at_dist) > nn_tol or (not nnnness_shells
                                                              and supercell_onemuon[atom_cellID].symbol!='X'):
            # we are in a new nnnness sphere -- so check it's wanted
            current_nnnness += 1
            if current_nnnness > nnnness:
                max_squish_radius = mu_at_dist
                break
            current_nnness_dist = mu_at_dist
        nearest_neighbour_atoms.append(supercell_onemuon[atom_cellID])
        nnnness_atoms.append(current_nnnness)

    # do squishification -- indexes for mu_distances and nearest_neighbour_atoms are the same
    for i_nearest_neighbour in range(0, len(nearest_neighbour_atoms)):
        # if the squish radius is not None, and this atom is not a muon
        if squish_radii is not None and nearest_neighbour_atoms[i_nearest_neighbour].symbol != 'X':
            # squish radius for this nnnness (nnnness_atoms defines this)
            current_nnnness = nnnness_atoms[i_nearest_neighbour]
            this_squish = None
            if len(squish_radii) > current_nnnness - 2:
                this_squish = squish_radii[current_nnnness - 2]
            elif lambda_start_nnnness is not None:
                if current_nnnness >= lambda_start_nnnness:
                    this_squish = mu_distances[i_nearest_neighbour][1] * lambda_squish
                    # if this wants to squish loads, put out a warning
                    if this_squish < squish_radii[-1]:
                        print('The lambda_squish brings in the atoms closer than the most distant squish value. Be careful '
                              'with the results.')
            if this_squish is not None:
                nearest_neighbour_atoms.set_distance(0, i_nearest_neighbour, this_squish, 0)

    return nearest_neighbour_atoms


def aseatoms_to_tdecoatoms(atoms: Atoms, muon_array_id: int = 0, muon_centred_coords: bool = True,
                            included_nuclei: list = None) -> (Matom, list):
    """
    Converts ASE Atoms into an array of TDecoerenceAtom objects
    :param atoms: ASE atoms
    :param muon_array_id: id of the muon location in atoms
    :param muon_centred_coords: centre coordinates on the muon
    :param included_nuclei: list of the symbols of nuclei to be converted. Useful if you want to ignore nuclei of a
                            certain type (e.g if they have small magnetic moments)
    :return: muon, list of TDecoherenceAtoms (with muon in position 0)
    """

    if muon_centred_coords:
        muon_centred_coords = 1
    else:
        muon_centred_coords = 0

    # sort out muon
    muon = None
    muon_location = coord(0, 0, 0)
    if muon_array_id is not None:
        muon_location = coord(atoms[muon_array_id].position[0], atoms[muon_array_id].position[1],
                              atoms[muon_array_id].position[2])
        muon = Matom(position=muon_location - muon_location * muon_centred_coords, name='mu')

    All_spins = [muon]
    for i_atom in range(0, len(atoms)):
        if i_atom != muon_array_id:
            atom_location = coord(atoms[i_atom].position[0], atoms[i_atom].position[1], atoms[i_atom].position[2])
            # if there is no restrictions on the included nuclei, or the nucleus in question is marked for inclusion
            if included_nuclei is None or atoms[i_atom].symbol in included_nuclei:
                All_spins.append(Matom(position=atom_location - muon_location * muon_centred_coords,
                                       name=atoms[i_atom].symbol))

    return muon, All_spins


def add_muon_to_aseatoms(atoms: Atoms, muon_position: np.ndarray, nn_indices: list = None,
                         enforce_nn_dist: float = 0) -> Atoms:
    """
    Add a muon with position muon_position to ASE atoms. Only works for PLANAR F--mu--F.
    :param atoms: ASE atoms to add the muon to
    :param muon_position: numpy array (x, y, z) of the muon position in CARTESIAN coordinates, in Angstroms
    :param nn_indices: indices of the nn atoms to perturb (if requried)
    :param enforce_nn_dist: perturbed distance
    :return: ASE atoms with the muon
    """

    # make a copy of atoms
    atoms = copy.deepcopy(atoms)

    muon_atom_obj = atom.Atom('X', muon_position)
    atoms.append(muon_atom_obj)

    # enforce nnnness
    if enforce_nn_dist > 0 and nn_indices is not None:
        print('WARNING -- setting enforce_nn_dist may lead to inaccurate next-nearest-neighbours. Use with caution.')
        atoms.set_distance(nn_indices[0], -1, enforce_nn_dist, fix=1)
        atoms.set_distance(nn_indices[1], -1, enforce_nn_dist, fix=1)

    return atoms


def add_muon_to_aseatoms_bent(ase_atoms: Atoms, fluorines: list, plane_atom, fmuf_angle: float = 180,
                              swing_angle: float = 0) -> Atoms:
    """
    Adds a muon to ase atoms as a bent F--mu--F bond
    :param ase_atoms: ASE atoms of the structure, without the muon
    :param fluorines: The two fluorine nuclei which the muon primarily interacts with
    :param plane_atom: the atom, with which the 2F+plane_atom make up the plane which the F--mu--F bond is on,
                             or an angle swing_angle away from. (can be int of atom in ase_atoms, or coords)
    :param fmuf_angle: the F--mu--F bond angle
    :param swing_angle: the angle the planar F--mu--F molecule makes with the 2F+plane_atom plane
    :return: ASE atoms, with muon
    """

    # get position of midpoint, M, of the two F atoms
    if isinstance(fluorines[0], int):
        # get the coordinates from ase_atoms
        f0_array = ase_atoms[fluorines[0]].position
        f1_array = ase_atoms[fluorines[1]].position
        f0 = coord(f0_array[0], f0_array[1], f0_array[2])
        f1 = coord(f1_array[0], f1_array[1], f1_array[2])
    else:
        f0 = coord(fluorines[0][0], fluorines[0][1], fluorines[0][2])
        f1 = coord(fluorines[1][0], fluorines[1][1], fluorines[1][2])

    if isinstance(plane_atom, int) or isinstance(plane_atom, np.int64):
        plane_atom_position_array = ase_atoms[plane_atom].position
        plane_atom_position = coord(plane_atom_position_array[0], plane_atom_position_array[1],
                                    plane_atom_position_array[2])
    else:
        plane_atom_position = coord(plane_atom[0], plane_atom[1], plane_atom[2])

    f0_to_midpoint = (f1 - f0) * 0.5
    f_midpoint = f0 + f0_to_midpoint

    # mu_protude is the displacement of the muon along the line plane_atom -> midpoint of F1-F2
    f_mid_dist = f0_to_midpoint.r()
    theta = (fmuf_angle % 360) * math.pi / 180
    # noinspection PyTypeChecker
    mu_protude = f_mid_dist * math.sqrt(2 / (1 - math.cos(theta))) * math.cos(theta / 2) \
                 + f_mid_dist * math.sqrt(abs(1 - 2 / (1 - math.cos(theta)) * math.pow(math.sin(theta / 2), 2)))

    mu_position = f_midpoint + (f_midpoint - plane_atom_position).rhat() * mu_protude

    # rotate the muon position about axis 2F
    if swing_angle != 0:
        print('Sorry! Not implemented swing angle yet...')
        assert False

    mu_atoms = add_muon_to_aseatoms(atoms=ase_atoms, muon_position=mu_position.tonumpyarray())

    # mu_img = Images()
    # mu_img.initialize([mu_atoms])
    # mu_gui = GUI(mu_img)
    # mu_gui.run()

    return mu_atoms

def get_muon_pos_nn_visually(atoms: Atoms) -> (np.ndarray, list):
    """
    Get the muon position by selecting two atoms to be the nn
    :param atoms: ASE atoms of the structure
    :return: (muon position ndarray, list of the index of the atoms the muon is in between
    """
    # get the muon site by asking the user for a list of ASE atoms
    images = Images()
    images.initialize([atoms])
    gui = GUI(images)
    gui.run()

    # images.selected is an array of all the atoms that have been selected when the window is closed
    selected = []

    # convert the selected array into the actual atoms they correspond to
    for i_images_selected_atom in range(0, len(images.selected)):
        if images.selected[i_images_selected_atom]:
            selected.append(i_images_selected_atom)

    assert len(selected) == 2

    # Find the midpoint, by finding the vector that links atom 1 to atom 2, then halving it.
    pos1 = atoms[selected[0]].position
    pos2 = atoms[selected[1]].position

    # the possible muon positions
    muon_pos = (pos1 + pos2) / 2

    print('Muon location in cartesian coordinates is: (' +
          str(muon_pos[0]) + ', ' + str(muon_pos[1]) + ', ' + str(muon_pos[2]) + ')')

    return muon_pos, selected


def add_muon_to_aseatoms_visual(atoms: Atoms, enforce_nn_dist: float = 0) -> Atoms:
    """
    Add a muon to ASE atoms visually (i.e by making the user select the nn atoms)
    :param atoms: ASE atoms of the unit cell the muon is located in
    :param enforce_nn_dist: The enforced distance between the muon and the SELECTED nn atoms (similar to squish)
                            (note: this is NOT the same as squish, because the selected atoms might not be the nn atoms
                            initially. This should only really be used as a way to ensure the selected atoms are the nn,
                            but they need not be...)
    :return: ASE atoms of the entire structure, including the muon
    """

    muon_pos, selected = get_muon_pos_nn_visually(atoms)

    atoms_mu = add_muon_to_aseatoms(atoms.copy(), muon_pos, selected, enforce_nn_dist)

    mu_img = Images()
    mu_img.initialize([atoms_mu])
    mu_gui = GUI(mu_img)
    mu_gui.run()

    return atoms_mu
