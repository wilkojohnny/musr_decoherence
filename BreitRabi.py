# BreitRabi.py -- calculates Breit-Rabi diagram for F--\mu--F system

import AtomObtainer as AO  # get atoms
from MDecoherenceAtom import TDecoherenceAtom as atom  # allows atom generation
import TCoord3D as coord  # coordinate tools
from scipy import sparse, linalg  # allows dealing with sparse matrices and linear algebra tools
import numpy as np  # maths tools
from datetime import datetime  # for date and time printing in the output file
import subprocess  # to get the current git version


def main():
    # #### INPUT ####
    #
    # # ## IF WE'RE USING PW_OUTPUT
    # # pw_output_file_location = 'CaF2.relax.mu.pwo'
    # # no_atoms = 11  # includes muon
    #
    # ## IF WE'RE USING AN XTL (crystal fractional coordinates) FILE
    # # xtl_input_location = 'CaF2_final_structure_reduced.xtl'
    # # (don't forget to define nnnness!)
    #
    # squish_radii = [1.172211, None]  # radius of the nn F-mu bond after squishification (1.18 standard, None for no squishification)
    #
    # ## IF WE'RE NOT USING pw output:
    # # nn, nnn, nnnn?
    # # nnnness = 2  # 2 = nn, 3 = nnn etc
    # # exclusive_nnnness - if TRUE, then only calculate nnnness's interactions (and ignore the 2<=i<nnnness interactions)
    # #   exclusive_nnnness = False
    #
    # ## IF NOT PW NOR XTL:
    # # lattice type: https://www.quantum-espresso.org/Doc/INPUT_PW.html#idm45922794628048
    # lattice_type = ibrav.CUBIC_FCC  # # can only do fcc and monoclinic (unique axis b)
    # # lattice parameters and angles, in angstroms
    # lattice_parameter = [5.44542, 0, 0]  # [a, b, c]
    # lattice_angles = [90, 0, 0]  # [alpha, beta, gamma] in **degrees**
    #
    # # are atomic coordinates provided in terms of alat or in terms of the primitive lattice vectors?
    # input_coord_units = position_units.ALAT
    #
    # # atoms and unit cell: dump only the basis vectors in here, the rest is calculated
    # atomic_basis = [
    #     # atom(coord.TCoord3D(0, 0, 0), gyromag_ratio=np.array([18.0038, 0]), II=np.array([7, 0]), name='Ca',
    #     #     abundance=np.array([0.00145, 0.99855])),
    #     atom(coord.TCoord3D(0.25, 0.25, 0.25), gyromag_ratio=251.713, II=1, name='F'),
    #     atom(coord.TCoord3D(0.25, 0.25, 0.75), gyromag_ratio=251.713, II=1, name='F')
    # ]
    #
    # # register the perturbed distances
    # perturbed_distances = []
    #
    # # define muon position
    # muon_position = coord.TCoord3D(.25, 0.25, 0.5)
    # muon_polarisation = coord.TCoord3D(0, 0, 1)
    #
    # # calc_decoherence(muon_position=muon_position, squish_radius=squish_radii, lattice_type=lattice_type,
    # #                  lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
    # #                  input_coord_units=input_coord_units, atomic_basis=atomic_basis,
    # #                  perturbed_distances=perturbed_distances, plot=True, nnnness=3, ask_each_atom=False,
    # #                  fourier=False, fourier_2d=False, tol=1e-3, times=np.arange(0, 10, 0.1))
    #
    # # calc_entropy(muon_position=muon_position, squish_radius=squish_radii, lattice_type=lattice_type,
    # #              lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
    # #              muon_polarisation=muon_polarisation, input_coord_units=input_coord_units, atomic_basis=atomic_basis,
    # #              perturbed_distances=perturbed_distances, nnnness=2, ask_each_atom=False)

    calc_breit_rabi()
        # muon_position=muon_position, squish_radius=squish_radii, lattice_type=lattice_type,
        #             lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
        #             input_coord_units=input_coord_units, atomic_basis=atomic_basis,
        #             perturbed_distances=perturbed_distances, nnnness=3, ask_each_atom=False,
        #             fields=np.arange(9.5, 15, 0.5), field_polarisation=coord.TCoord3D(1,0,0),
        #             outfile_location='/Users/johnny/Documents/University/CaF2/LCR/CaF2_xfield_nnnF_7G1_7G2.dat',
        #             plot=True)


def calc_breit_rabi(
                    # muon_position, squish_radius=None, fields=np.arange(0, 1, 1e3),
                    # field_polarisation=coord.TCoord3D(0, 0, 1),
                    # # arguments for manual input of lattice
                    # lattice_type=None, lattice_parameter=None, lattice_angles=None,
                    # input_coord_units=position_units.ALAT, atomic_basis=None, perturbed_distances=None,
                    # # arguments for XTL
                    # use_xtl_input=False, xtl_input_location=None,
                    # # arguments for XTL or manual input
                    # nnnness=2, exclusive_nnnness=False,
                    # # arguments for pw.x output
                    # use_pw_output=False, pw_output_file_location=None, no_atoms=0,
                    # # other arguments
                    # outfile_location=None, plot=False, ask_each_atom=False
):

    # # if no outfile nor plot is initiated, no point in continuing...
    # assert not (outfile_location is None and plot is False)
    #
    # # normalise the magnetic field polarisation vector
    # field_polarisation = field_polarisation / field_polarisation.r()
    #
    # # get the atoms and the muon
    # muon, All_Spins, got_atoms = get_spins(muon_position, squish_radius, lattice_type, lattice_parameter,
    #                                        lattice_angles,
    #                                        input_coord_units, atomic_basis, perturbed_distances, use_xtl_input,
    #                                        xtl_input_location, nnnness, exclusive_nnnness, use_pw_output,
    #                                        pw_output_file_location, no_atoms, ask_each_atom)
    #
    # # work out how many energies we should have
    # num_energies = 1
    # for spin in All_Spins:
    #     num_energies *= spin.II + 1
    #
    # # open the output file
    # output_file = None
    # if outfile_location is not None:
    #     # set up the output file
    #     output_file = open(outfile_location, 'w+')
    #     breit_rabi_file_preamble(output_file, field_polarisation, fields, muon_position, All_Spins, use_xtl_input,
    #                              xtl_input_location, use_pw_output, perturbed_distances, squish_radius, nnnness,
    #                              exclusive_nnnness, lattice_type, lattice_parameter)
    #     output_file.write('! field (G) ')
    #     for i in range(0, num_energies):
    #         output_file.write('E' + str(i) + ' (MHz) ')
    #     output_file.write('\n')
    #
    # # calculate the dipolar Hamiltonian
    # dipolar_hamiltonian = calc_dipolar_hamiltonian(All_Spins)
    #
    # # if plotting, set up the arrays to plot
    # energies = np.empty((num_energies, len(fields)))
    #
    # gc.enable()
    #
    # hamiltonian = copy.deepcopy(dipolar_hamiltonian) #+ calc_zeeman_hamiltonian(All_Spins, field_v)
    dense_hamiltonian = sparse.rand(2048, 2048, 0.03, dtype=complex).todense()
    dense_hamiltonian = dense_hamiltonian + dense_hamiltonian.H

    for i_field in range(0,100):
                         #len(fields)):
        # calculate the Zeeman terms
        # field_v = field_polarisation*fields[i_field]*1e-4

        # print out the current field as a status update
        #print('Field: ' + str(fields[i_field]) + 'G')

        # diagonalise the Hamiltonian
        _, _ = linalg.eigh(dense_hamiltonian)

        # append the eigenvalues to the list if plotting (no reason to save them otherwise...)
        # if plot:
        #     energies[:, i_field] = E

        # write to file
        # if output_file is not None:
        #     output_file.write(str(fields[i_field]) + ' ')
        #     for energy in energies[:, i_field]:
        #         output_file.write(str(energy) + ' ')
        #     output_file.write('\n')

        #del E
        # gc.collect()
    #
    # if plot:
    #     for i in range(0, num_energies):
    #         pyplot.plot(fields, energies[i, :])
    #     pyplot.xlabel('Field (G)')
    #     pyplot.ylabel('E /MHz')
    #     pyplot.title('F-mu-F, Field in x-direction')
    #     pyplot.show()
    #
    # if output_file is not None:
    #     output_file.close()


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

    AO.atoms_file_preamble(file, muon_position, nn_atoms, use_xtl_input, xtl_input_location, use_pw_output,
                           perturbed_distances, squish_radius, nnnness, exclusive_nnnness, lattice_type,
                           lattice_parameter)





if __name__=='__main__':
    main()
