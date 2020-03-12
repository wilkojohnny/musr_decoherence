# DipolePolarisation.py -- calculates the muon polarisation function (in terms of time or frequency) for lots of
# spins (in the style of F--\mu--F)
# John Wilkinson 15/11/19

import subprocess  # gets git version
from datetime import datetime  # allows one to print out date and time
import DecoherenceCalculator as decoCalc  # allows one to calculate decoherence
from MDecoherenceAtom import TDecoherenceAtom as atom  # for atoms
import AtomObtainer  # allows one to play with atoms
import TCoord3D as coord  # coordinate utilities
import numpy.linalg as linalg  # matrix stuff
import numpy as np  # for numpy arrays
import matplotlib.pyplot as pyplot  # plotting
import os  #


# do decoherence file preamble
def decoherence_file_preamble(file, muon_position, nn_atoms, fourier, starttime=None, endtime=None, timestep=None,
                              fourier_2d=None, tol=None, use_xtl_input=None, xtl_input_location=None,
                              use_pw_output=None, pw_output_location=None, perturbed_distances=None, squish_radius=None,
                              nnnness=None, exclusive_nnnness=None, lattice_type=None, lattice_parameter=None):
    # program name, date and time completed
    file.writelines('! Decoherence Calculator Output - ' + datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + '\n!\n')

    # get the git version
    script_dir = os.path.dirname(os.path.realpath(__file__))
    version_label = subprocess.check_output(["git", "describe", "--always"], cwd=script_dir).strip()
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

    AtomObtainer.atoms_file_preamble(file, muon_position, nn_atoms, use_xtl_input, xtl_input_location, use_pw_output,
                                     pw_output_location, perturbed_distances, squish_radius, nnnness, exclusive_nnnness,
                                     lattice_type, lattice_parameter)

    file.writelines('! start of data: \n')


# batch write data to file
def write_to_file(file, t, P):
    for i in range(0, len(t) - 1):
        file.writelines(str(t[i]) + ' ' + str(P[i]) + '\n')


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


def calc_decoherence(muon_position=None, muon_sample_polarisation=None, squish_radius=None, times=np.arange(0, 10, 0.1),
                     # arguments for manual input of lattice
                     lattice_type=None, lattice_parameter=None, lattice_angles=None,
                     input_coord_units=AtomObtainer.position_units.ALAT, atomic_basis=None, perturbed_distances=None,
                     # arguments for XTL
                     use_xtl_input=False, xtl_input_location=None,
                     # arguments for XTL or manual input
                     nnnness=2, exclusive_nnnness=False,
                     # arguments for pw.x output
                     use_pw_output=False, pw_output_file_location=None, no_atoms=0,
                     # other arguments
                     fourier=False, fourier_2d=False, outfile_location=None, tol=1e-10, plot=False, shutup=False,
                     ask_each_atom=False):
    '''
    Calculate the time- or frequency- dependent muon polarisation function.
    :param muon_position: Muon position
    :param muon_sample_polarisation: Polarisation of the muon wrt the sample: None for polycrystalline sample
    :param squish_radius: [] radius to which the [nn, nnn, nnnn,...] atoms are perturbed to
    :param times: times in \mu s to calculate the polarisation function for
    :param lattice_type: lattice type number, following Espresso pw.x's convention
    :param lattice_parameter: [a, b, c] lattice parameter in Angstroms
    :param lattice_angles: [alpha, beta, gamma] in degrees
    :param input_coord_units: following QE convention
    :param atomic_basis: basis of atoms to be repeated in the cell through translations
    :param perturbed_distances: in format [[old position, new position], [old position, new position]].
                                Better to use squish_radius
    :param use_xtl_input: whether use XTL input file
    :param xtl_input_location: location of XTL input file, if being used
    :param nnnness: maximum next-nerarest neighbour-ness to calculate to (nn=2, nnn=3, etc)
    :param exclusive_nnnness: only calculate for the nnnness in nnnness, ignore nuclei before
    :param use_pw_output: use Espresso pw.x output file
    :param pw_output_file_location: location of pw.x output file
    :param no_atoms: number of atoms to use in pw.x output file (nnnness is not quite so elegant here)
    :param fourier: calculate frequency-dependent plot
    :param fourier_2d: calculate energies and transitions between them
    :param outfile_location: location of the file for the output to be saved to
    :param tol: difference between frequencies to be treated the same
    :param plot: Draw the plot
    :param shutup: True for very little verbosity
    :param ask_each_atom: ask the user for every atom if they want it included in the calculation
    :return: value of the muon polarisation function (in time, frequency or freq1, freq 2 depending on input)
    '''

    # type of calculation - can't do fourier2d if not fourier
    fourier_2d = fourier_2d and fourier

    # get the atoms and the muon
    muon, All_Spins, got_atoms = AtomObtainer.get_spins(muon_position, squish_radius, lattice_type, lattice_parameter,
                                                        lattice_angles, input_coord_units, atomic_basis,
                                                        perturbed_distances, use_xtl_input, xtl_input_location, nnnness,
                                                        exclusive_nnnness, None, use_pw_output, pw_output_file_location,
                                                        no_atoms, ask_each_atom)

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
        muon_spin_x = 2*decoCalc.measure_ith_spin(Spins, 0, Spins[0].pauli_x)
        muon_spin_y = 2*decoCalc.measure_ith_spin(Spins, 0, Spins[0].pauli_y)
        muon_spin_z = 2*decoCalc.measure_ith_spin(Spins, 0, Spins[0].pauli_z)

        # calculate hamiltonian
        hamiltonian = decoCalc.calc_dipolar_hamiltonian(Spins)

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
        if muon_sample_polarisation is None:
            for i in range(0, len(R)):
                    # angular average mode
                    thisconst = thisconst + pow(abs(Rinv[i] * muon_spin_x * R[:, i]), 2) \
                                + pow(abs(Rinv[i] * muon_spin_y * R[:, i]), 2) \
                                + pow(abs(Rinv[i] * muon_spin_z * R[:, i]), 2)
            const = const + probability * thisconst / (6 * (muon_spin_x.shape[0] / 2))
        else:
            for i in range(0, len(R)):
                # single crystal sample
                thisconst = thisconst + \
                            pow(abs(Rinv[i] * muon_sample_polarisation.ortho_x * muon_spin_x * R[:, i]), 2) \
                            + pow(abs(Rinv[i] * muon_sample_polarisation.ortho_y * muon_spin_y * R[:, i]), 2) \
                            + pow(abs(Rinv[i] * muon_sample_polarisation.ortho_z * muon_spin_z * R[:, i]), 2)
            const = const + probability * thisconst / (2 * (muon_spin_x.shape[0] / 2))

        # now calculate oscillating term
        this_amplitude = np.zeros((len(R), len(R)))
        for i in range(0, len(R)):
            if muon_sample_polarisation is None:
                Rx = Rinv[i] * muon_spin_x
                Ry = Rinv[i] * muon_spin_y
                Rz = Rinv[i] * muon_spin_z
            else:
                Rx = Rinv[i] * muon_spin_x * muon_sample_polarisation.ortho_x
                Ry = Rinv[i] * muon_spin_y * muon_sample_polarisation.ortho_y
                Rz = Rinv[i] * muon_spin_z * muon_sample_polarisation.ortho_z

            if not shutup:
                print(str(100 * i / len(R)) + '% complete...')
            if fourier_2d:
                jmin = 0
            else:
                jmin = i + 1
            for j in range(jmin, len(R)):
                if muon_sample_polarisation is None:
                    # do angular averaging
                    this_amplitude[i][j] = (pow(abs(Rx * R[:, j]), 2)
                                            + pow(abs(Ry * R[:, j]), 2)
                                            + pow(abs(Rz * R[:, j]), 2)) * probability \
                                           / (3 * (muon_spin_x.shape[0] / 2))
                else:
                    # single crystal sample
                    this_amplitude[i][j] = (pow(abs(Rx * R[:, j]), 2)
                                            + pow(abs(Ry * R[:, j]), 2)
                                            + pow(abs(Rz * R[:, j]), 2)) * probability \
                                           / (muon_spin_x.shape[0] / 2)

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
                                      lattice_parameter=lattice_parameter, pw_output_location=pw_output_file_location)

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
            P_average.append(decoCalc.calc_p_average_t(time, const, amplitude, E).max())
            # print(P_average[-1])

        if outfile_location is not None:
            # dump results in a file if requested
            outfile = open(outfile_location, "w")
            # do preamble
            decoherence_file_preamble(file=outfile, muon_position=muon_position, nn_atoms=All_Spins, fourier=fourier,
                                      fourier_2d=fourier_2d, tol=tol, use_xtl_input=use_xtl_input,
                                      xtl_input_location=xtl_input_location, use_pw_output=use_pw_output,
                                      pw_output_location=pw_output_file_location,
                                      perturbed_distances=perturbed_distances, squish_radius=squish_radius,
                                      nnnness=nnnness, exclusive_nnnness=exclusive_nnnness, lattice_type=lattice_type,
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


def main():
    #### INPUT ####

    # ## IF WE'RE USING PW_OUTPUT
    pw_output_file_location = None
    no_atoms = 10  # excludes muon

    ## IF WE'RE USING AN XTL (crystal fractional coordinates) FILE
    # xtl_input_location = 'CaF2_final_structure_reduced.xtl'
    # (don't forget to define nnnness!)

    # CaF2:
    squish_radii = [1.172211, None]  # radius of the nn F-mu bond after squishification (1.18 standard, None for no squishification)

    # KPF6:
    # squish_radii = [None, None]  # radius of the nn F-mu bond after squishification (1.18 standard, None for no squishification)

    ## IF WE'RE NOT USING pw output:
    # nn, nnn, nnnn?
    nnnness = 2  # 2 = nn, 3 = nnn etc
    # exclusive_nnnness - if TRUE, then only calculate nnnness's interactions (and ignore the 2<=i<nnnness interactions)
    #   exclusive_nnnness = False

    ## IF NOT PW NOR XTL:
    # lattice type: https://www.quantum-espresso.org/Doc/INPUT_PW.html#idm45922794628048
    # CaF2:
    lattice_type = AtomObtainer.ibrav.CUBIC_FCC  # # can only do fcc and monoclinic (unique axis b)
    # lattice parameters and angles, in angstroms
    lattice_parameter = [5.44542, 0, 0]  # [a, b, c]
    lattice_angles = [90, 0, 0]  # [alpha, beta, gamma] in **degrees**

    # KPF6:
    # lattice_type = AtomObtainer.ibrav.CUBIC_SC  # # can only do sc, fcc and monoclinic (unique axis b)
    # lattice parameters and angles, in angstroms
    # lattice_parameter = [10, 0, 0]  # [a, b, c]
    # lattice_angles = [90, 0, 0]  # [alpha, beta, gamma] in **degrees**

    # are atomic coordinates provided in terms of alat or in terms of the primitive lattice vectors?
    input_coord_units = AtomObtainer.position_units.ALAT

    # atoms and unit cell: dump only the basis vectors in here, the rest is calculated
    atomic_basis = [
        # CaF2:
        # atom(coord.TCoord3D(0, 0, 0), gyromag_ratio=np.array([18.0038, 0]), II=np.array([7, 0]), name='Ca',
        #     abundance=np.array([0.00145, 0.99855])),
        atom(coord.TCoord3D(0.25, 0.25, 0.25), gyromag_ratio=251.713, II=1, name='F'),
        atom(coord.TCoord3D(0.25, 0.25, 0.75), gyromag_ratio=251.713, II=1, name='F')

    ]

    # register the perturbed distances
    perturbed_distances = []

    # define muon position
    muon_position = coord.TCoord3D(.25, 0.25, 0.5)  # CaF2

    # define muon polarisation relative to the sample -- None for polycrystalline
    muon_sample_polarisation = coord.TCoord3D(0, 0, 1)

    calc_decoherence(muon_position=muon_position, muon_sample_polarisation=muon_sample_polarisation,
                     squish_radius=squish_radii, lattice_type=lattice_type,
                     lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
                     input_coord_units=input_coord_units, atomic_basis=atomic_basis,
                     use_pw_output=False, pw_output_file_location=pw_output_file_location, no_atoms=no_atoms,
                     perturbed_distances=perturbed_distances, plot=True, nnnness=3, ask_each_atom=False,
                     fourier=False, fourier_2d=False, tol=1e-3, times=np.arange(0, 25, 0.1), outfile_location=None)
    return 1


if __name__=='__main__':
    main()