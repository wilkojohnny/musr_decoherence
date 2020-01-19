# DipolePolarisation.py -- calculates sum_{nnnness} [n atoms in nnnness]/r_{nnnness}^6, until convergence
# John Wilkinson 16/12/19
import AtomObtainer  # for nnnfinder
from MDecoherenceAtom import TDecoherenceAtom as atom  # for atom object
import TCoord3D as coord  # for 3D coordinates


def main():
    #### INPUT ####

    # ## IF WE'RE USING PW_OUTPUT
    # pw_output_file_location = 'CaF2.relax.mu.pwo'
    # no_atoms = 11  # includes muon

    ## IF WE'RE USING AN XTL (crystal fractional coordinates) FILE
    # xtl_input_location = 'CaF2_final_structure_reduced.xtl'
    # (don't forget to define nnnness!)

    squish_radii = [1.172211] # radius of the nn F-mu bond after squishification
    # (1.18 standard, None for no squishification)

    # lattice type: https://www.quantum-espresso.org/Doc/INPUT_PW.html#idm45922794628048
    lattice_type = AtomObtainer.ibrav.CUBIC_FCC  # # can only do fcc and monoclinic (unique axis b)
    # lattice parameters and angles, in angstroms
    lattice_parameter = [5.44542, 0, 0]  # [a, b, c]
    lattice_angles = [90, 0, 0]  # [alpha, beta, gamma] in **degrees**

    # are atomic coordinates provided in terms of alat or in terms of the primitive lattice vectors?
    input_coord_units = AtomObtainer.position_units.ALAT

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

    calc_variace(muon_position=muon_position, squish_radius=squish_radii, start_nnnness=2, end_tol=1e-4,
                 lattice_type=lattice_type, lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
                 input_coord_units=input_coord_units, atomic_basis=atomic_basis,
                 perturbed_distances=perturbed_distances)

    return 1


def calc_variace(muon_position, squish_radius=None, start_nnnness=2, end_tol=1e-6,
                 # arguments for manual input of lattice
                 lattice_type=None, lattice_parameter=None, lattice_angles=None,
                 input_coord_units=AtomObtainer.position_units.ALAT, atomic_basis=None, perturbed_distances=None):

    current_sum = 0
    old_sum = 0

    # for each nnnness until the new/old - 1 > end_tol
    rel_diff = 100
    nnnness = start_nnnness - 1
    while abs(rel_diff)>end_tol:
        # increment nnnness
        nnnness += 1
        old_sum = current_sum

        # get the atoms which are nnn away from the muon
        muon, All_Spins, _ = AtomObtainer.get_spins(muon_position, squish_radius, lattice_type, lattice_parameter,
                                                    lattice_angles, input_coord_units, atomic_basis,
                                                    perturbed_distances,
                                                    nnnness=nnnness, exclusive_nnnness=True)

        # calculate r
        r = (muon.position - All_Spins[1].position).r()

        # calculate this term in the sum
        new_term = (len(All_Spins) - 1)/pow(r, 6)

        # update sum
        current_sum = current_sum + new_term

        # calculate extent of convergence
        if old_sum!=0:
            rel_diff = current_sum/old_sum - 1

        # print out findings
        print(str(len(All_Spins) - 1) + '/' + str(r) + '=' + str(new_term) + '; sum=' + str(current_sum)
              + '; convergence ratio=' + str(rel_diff))


if __name__=='__main__':
    main()
