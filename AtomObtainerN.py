"""
AtomObtainerN.py -- get atoms using the tools in ASE
Allows getting atoms from .cif files, and for the user to define the muon location using a GUI (easier than faffing
with locations, squish factors, etc!)
Created 23/3/2020 by John Wilkinson (during COVID isolation!)
"""

from ase import Atoms, atom
from ase.io import read
from ase.gui.gui import GUI
from ase.gui.images import Images
from ase import build
import numpy as np
from MDecoherenceAtom import TDecoherenceAtom as Matom  # import class for decoherence atom
from TCoord3D import TCoord3D as coord


def main():

    # cif file location
    cif_file = 'YF3.cif'

    atoms = read(cif_file)

    # get the muon site
    muon_position, nn_indices = get_muon_pos_nn_visually(atoms)

    # make the atomic supercell
    muon, All_spins = get_linear_fmuf_atoms(atoms, muon_position, 2, [1.18])

    return 0


def get_linear_fmuf_atoms(ase_atoms: Atoms, muon_position: np.ndarray, nnnness: int = 2, squish_radii: list = None,
                          enforce_nn_dist: float = 0, enforced_nn_indices: list = None) -> (Matom, list):
    """
    Get the TDecoAtoms muon, All_Spins in a *linear* F--mu--F state for these calculations
    :param ase_atoms: ASE atoms of the structure (without muon)
    :param muon_position: cartesian position of muon (in numpy array [x,y,z])
    :param nnnness: nnnness of the calculation
    :param squish_radii: list of perturbations to the structure for [nn, nnn, nnnn, ...]
    :param enforce_nn_dist: perturbs the selected two
    :param enforced_nn_indices: indices of the two atoms to perturb before nnnness
    :return: TDecoherenceAtom Muon, list[TDecoAtom] All_spins (including muon in pos 0)
    """

    # add the muon, doing the enforced distancing when doing so
    muon_aseatoms = add_muon_to_aseatoms(ase_atoms, muon_position, enforced_nn_indices, enforce_nn_dist)

    # find the nnnness atoms, and squishification
    nnn_aseatoms = ase_nnnfinder(muon_aseatoms, nnnness, squish_radii, 5)

    # return muon, All_Spins
    return aseatoms_to_tdecoatoms(nnn_aseatoms)


def ase_nnnfinder(atoms_mu: Atoms, nnnness: int, squish_radii: list = None, supercell_size: int =None) -> Atoms:
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

    # find the most central muon -- this will be in the average position of all of them
    supercell_muon_indexes = []
    for i_supercell_at in range(0, len(supercell)):
        if supercell[i_supercell_at].symbol == 'mu':
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
        if atom.symbol != 'mu':
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
    current_nnnness = 1
    current_nnness_dist = 0
    nn_tol = 1e-3
    for [atom_cellID, mu_at_dist] in mu_distances:
        # see if we are in a new sphere of nnnness
        if abs(current_nnness_dist - mu_at_dist) > nn_tol:
            # we are in a new nnnness sphere -- so check it's wanted
            current_nnnness += 1
            if current_nnnness > nnnness:
                break
            current_nnness_dist = mu_at_dist
        nearest_neighbour_atoms.append(supercell_onemuon[atom_cellID])
        # do squishification
        if squish_radii is not None and mu_at_dist > 0:
            if squish_radii[current_nnnness-2] is not None:
                nearest_neighbour_atoms.set_distance(0, -1, squish_radii[current_nnnness-2], 0)

    return nearest_neighbour_atoms


def aseatoms_to_tdecoatoms(atoms: Atoms, muon_array_id: int = 0, muon_centred_coords: bool = True) -> (Matom, list):
    """
    Converts ASE Atoms into an array of TDecoerenceAtom objects
    :param atoms: ASE atoms
    :param muon_array_id: id of the muon location in the
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
        muon = Matom(position=muon_location - muon_location*muon_centred_coords, name='mu')

    All_spins = [muon]
    for i_atom in range(0, len(atoms)):
        if i_atom != muon_array_id:
            atom_location = coord(atoms[i_atom].position[0], atoms[i_atom].position[1], atoms[i_atom].position[2])
            All_spins.append(Matom(position=atom_location - muon_location*muon_centred_coords,
                             name=atoms[i_atom].symbol))

    return muon, All_spins


def add_muon_to_aseatoms(atoms: Atoms, muon_position: np.ndarray, nn_indices: list = None,
                         enforce_nn_dist: float = 0) -> Atoms:
    """
    Add a muon with position muon_position to ASE atoms
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


if __name__ == '__main__':
    main()


