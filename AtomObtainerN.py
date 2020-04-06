"""
AtomObtainerN.py -- get atoms using the tools in ASE
Allows getting atoms from .cif files, and for the user to define the muon location using a GUI (easier than faffing
with locations, squish factors, etc!)
Created 23/3/2020 by John Wilkinson (during COVID isolation!)
"""

from ase import Atoms, atom
from ase.gui.gui import GUI
from ase.gui.images import Images
from ase import build
import numpy as np
from MDecoherenceAtom import TDecoherenceAtom as Matom  # import class for decoherence atom
from TCoord3D import TCoord3D as coord
import math


def get_linear_fmuf_atoms(ase_atoms: Atoms, muon_position: np.ndarray, nnnness: int = 2, squish_radii: list = None,
                          enforce_nn_dist: float = 0, enforced_nn_indices: list = None) -> (Matom, list):
    """
    Get the TDecoAtoms muon, All_Spins in a *linear* F--mu--F state for these calculations
    :param ase_atoms: ASE atoms of the structure (without muon)
    :param muon_position: cartesian position of muon (in numpy array [x,y,z])
    :param nnnness: nnnness of the calculation
    :param squish_radii: list of perturbations to the structure for [nn, nnn, nnnn, ...]
    :param enforce_nn_dist: perturbs the selected two atoms which define the muon location
    :param enforced_nn_indices: indices of the two atoms to perturb before nnnness
    :return: TDecoherenceAtom Muon, list[TDecoAtom] All_spins (including muon in pos 0)
    """

    # add the muon, doing the enforced distancing when doing so
    muon_aseatoms = add_muon_to_aseatoms(ase_atoms, muon_position, enforced_nn_indices, enforce_nn_dist)

    # find the nnnness atoms, and squishification
    nnn_aseatoms = ase_nnnfinder(muon_aseatoms, nnnness, squish_radii, 5)

    # return muon, All_Spins
    return aseatoms_to_tdecoatoms(nnn_aseatoms)


def get_bent_fmuf_atoms(ase_atoms: Atoms, fluorines: list, plane_atom, fmuf_angle: float = 180,
                        swing_angle: float = 0, nnnness: int = 2, squish_radii: list = None):
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
    :return: TDecoherenceAtom Muon, list[TDecoAtom] All_spins (including muon in pos 0)
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

    if isinstance(plane_atom, int):
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

    mu_atoms_nnn = ase_nnnfinder(atoms_mu=mu_atoms, nnnness=nnnness, squish_radii=squish_radii, supercell_size=3)

    # return muon, All_Spins
    return aseatoms_to_tdecoatoms(mu_atoms_nnn)


def ase_nnnfinder(atoms_mu: Atoms, nnnness: int, squish_radii: list = None, supercell_size: int = None) -> Atoms:
    """
    Find nearest-neighbours using ASE
    :param atoms_mu: ASE atoms with muon
    :param nnnness: nnnness to get
    :param squish_radii: list of perturbations to the structure for [nn, nnn, nnnn, ...]
    :param supercell_size: size of supercell to look for nearest neighbours in. Should be odd.
    :return: ASE Atoms with just the nnnness asked for
    """

    if supercell_size is None:
        supercell_size = 2 * nnnness + 1
    # supercell must have odd dimensions
    if supercell_size % 2 == 0:
        supercell_size += 1
    # force squish radii to be the right size for nnnness
    if squish_radii is not None:
        squish_radii_len = len(squish_radii)
        if squish_radii_len < nnnness - 1:
            for _ in range(0, nnnness - squish_radii_len - 1):
                squish_radii.append(None)

    # make a supercell
    supercell = build.make_supercell(atoms_mu, np.diag([supercell_size, supercell_size, supercell_size]))

    # find the most central muon -- this will be in the average position of all of them. Also keep track of H atoms, as
    # sometimes DFT uses H instead of mu. H atoms have -ve positions -- so if the array has only negative entries,
    # make them all positive and continue. If there is one +ve entry, scrap the negative entries.
    supercell_muon_indexes = []
    for i_supercell_at in range(0, len(supercell)):
        if supercell[i_supercell_at].symbol == 'mu':
            supercell_muon_indexes.append(i_supercell_at)
        elif supercell[i_supercell_at].symbol == 'H':
            supercell_muon_indexes.append(-1 * i_supercell_at)

    if max(supercell_muon_indexes) < 0:
        # this means there is no mu, only H -- so make these muons
        muon_symbol = 'H'
        supercell_muon_indexes = [-1 * index for index in supercell_muon_indexes]
    else:
        muon_symbol = 'mu'
        # get rid of the H indexes -- we have a muon!
        for index in supercell_muon_indexes:
            if index < 0:
                del index

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
        if abs(current_nnness_dist - mu_at_dist) > nn_tol:
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
        if squish_radii is not None and nearest_neighbour_atoms[i_nearest_neighbour].symbol != 'mu':
            # squish radius for this nnnness (nnnness_atoms defines this)
            current_nnnness = nnnness_atoms[i_nearest_neighbour]
            this_squish = squish_radii[current_nnnness - 2]
            # if the squish radius is defined, and this squish is not bigger than the nnnness beyond that asked for
            if this_squish is not None:  # and this_squish <= max_squish_radius: removed -- why not have things bigger?!
                nearest_neighbour_atoms.set_distance(0, i_nearest_neighbour, squish_radii[current_nnnness - 2], 0)

    return nearest_neighbour_atoms


def aseatoms_to_tdecoatoms(atoms: Atoms, muon_array_id: int = 0, muon_centred_coords: bool = True) -> (Matom, list):
    """
    Converts ASE Atoms into an array of TDecoerenceAtom objects
    :param atoms: ASE atoms
    :param muon_array_id: id of the muon location in atoms
    :param muon_centred_coords: centre coordinates on the muon
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
            All_spins.append(Matom(position=atom_location - muon_location * muon_centred_coords,
                                   name=atoms[i_atom].symbol))

    return muon, All_spins


def add_muon_to_aseatoms(atoms: Atoms, muon_position: np.ndarray, nn_indices: list = None,
                         enforce_nn_dist: float = 0) -> Atoms:
    """
    Add a muon with position muon_position to ASE atoms. Only works for PLANAR F--mu--F.
    :param atoms: ASE atoms to add the muon to
    :param muon_position: numpy array (x, y, z) of the muon position
    :param nn_indices: indices of the nn atoms to perturb (if requried)
    :param enforce_nn_dist: perturbed distance
    :return: ASE atoms with the muon
    """

    muon_atom_obj = atom.Atom('mu', muon_position)
    atoms.append(muon_atom_obj)

    # enforce nnnness
    if enforce_nn_dist > 0 and nn_indices is not None:
        print('WARNING -- setting enforce_nn_dist may lead to inaccurate next-nearest-neighbours. Use with caution.')
        atoms.set_distance(nn_indices[0], -1, enforce_nn_dist, fix=1)
        atoms.set_distance(nn_indices[1], -1, enforce_nn_dist, fix=1)

    return atoms


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
