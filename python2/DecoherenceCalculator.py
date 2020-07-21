# Hamiltonians.py - Calculate decoherence of muon state in any lattice with any structure

from __future__ import division
from __future__ import absolute_import
from MDecoherenceAtom import TDecoherenceAtom as Tatom  # import class for decoherence atom
import TCoord3D as coord  # 3D coordinates class
import AtomObtainer  # to obtain atoms from pw output file
import numpy as np  # for matrices
import scipy.sparse as sparse  # for sparse matrices
import numpy.linalg as linalg  # for linear algebra
import matplotlib.pyplot as pyplot      # for plotting
from datetime import datetime  # for date and time printing in the output file
from enum import Enum  # for enumerations for the inputs - makes everything a lot easier to read!
import subprocess  # to get the current git version
from io import open


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
def nnn_finder(basis, muon, lattice_translation, nn=2, exclusive_nnnness=False, perturbations=None, squish_radius=None):
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
        for i in xrange(0, len(exact_nml)):
            flr_nml = np.floor(exact_nml[i])
            for nm_or_l in xrange(int(flr_nml) - (nn - 1), int(flr_nml) + (nn + 1)):
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
    closest_F_radius = 0
    for atom in nearestneighbours:
        if atom[2].name == u'F':
            if closest_F_radius == 0:
                closest_F_radius = atom[0]  # if no Fs have been registered yet, use this radius as the reference
            elif (atom[0] - closest_F_radius) < 1e-3:
                # this atom is also a nn
                pass
            else:
                break
            # if we get here, it means this atom needs squishification (if desired)
            if squish_radius is not None:
                atom[0] = squish_radius
                atom[1].set_r(squish_radius, muon.position)
                atom[2].position = atom[1]

    # sort the list by radius (just the beginning atoms might've changed)
    nearestneighbours.sort(key=lambda atom: atom[0])

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
        for i in xrange(0, len(basis)):
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
    for i_spin in xrange(0, i):
        lhs_dim = lhs_dim * Spins[i_spin].pauli_dimension

    # ... and the RHS
    rhs_dim = 1
    for i_spin in xrange(i + 1, len(Spins)):
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
    return A/pow(abs(r.r()), 3)*(i_x*j_x + i_y*j_y + i_z*j_z
                                 - 3*(i_x*r.xhat() + i_y*r.yhat() + i_z*r.zhat())
                                 * (j_x*r.xhat() + j_y*r.yhat() + j_z*r.zhat()))


# calculate entire hamiltonian
def calc_total_hamiltonian(spins):
    current_hamiltonian = 0

    # calculate hamiltonian for each pair and add onto sum
    for i in xrange(0, len(spins)):
        for j in xrange(i+1, len(spins)):
            current_hamiltonian = current_hamiltonian + calc_hamiltonian_term(spins, i, j)
    return current_hamiltonian


# calculate polarisation function for one specific time (used in the integration routine)
def calc_p_average_t(t, const, amplitude, E):
    # calculate the oscillating term
    osc_term = 0
    for isotope_combination in xrange(0, len(amplitude)):
        for i in xrange(0, len(E[isotope_combination])):
            for j in xrange(i+1, len(E[isotope_combination])):
                osc_term = osc_term + amplitude[isotope_combination][i][j]*np.cos((E[isotope_combination][i]
                                                                                   - E[isotope_combination][j])*t)

    # add on the constant, and return, and divide by the size of the space
    return const + osc_term


# do file preamble
def file_preamble(file, muon_position, nn_atoms, fourier, starttime=None, endtime=None, timestep=None, fourier_2d=None,
                  tol=None, use_xtl_input=None, xtl_input_location=None, use_pw_output=None, perturbed_distances=None,
                  squish_radius=None,  nnnness=None, exclusive_nnnness=None, lattice_type=None, lattice_parameter=None):

    # program name, date and time completed
    file.writelines(u'! Decoherence Calculator Output - ' + datetime.now().strftime(u"%d/%m/%Y, %H:%M:%S") + u'\n!\n')

    # get the git version
    version_label = subprocess.check_output([u"git", u"describe", u"--always"]).strip()
    file.writelines(u'! Using version ' + unicode(version_label) + u'\n!\n')

    # type of calculation
    if not fourier:
        file.writelines(u'! time calculation completed between t=' + unicode(starttime) + u' and ' + unicode(endtime) +
                        u' with a timestep of ' + unicode(timestep) + u' microseconds' + u'\n!\n')
    else:
        if fourier_2d:
            file.writelines(u'! 2D fourier calculation, showing the amplitude between each transition pair. \n')
        else:
            file.writelines(u'! 1D fourier calculation, showing the amplitude of each E_i-E_j combination \n')
        file.writelines(u'! absolute tolerance between eigenvalues to treat them as equivalent was ' + unicode(tol)
                        + u'\n!\n')

    # data source, if used
    if use_xtl_input:
        file.writelines(u'! Atom positional data was obtained from XTL (fractional crystal coordinate) file: ' +
                        xtl_input_location + u'\n')
    if use_pw_output:
        file.writelines(u'! Atom positional data was obtained from QE PWSCF file: ' + xtl_input_location + u'\n')

    # Basis atoms, with gyromag ratios and I values
    for atom in nn_atoms:
        lines_to_write = atom.verbose_description(gle_friendly=True)
        for line in lines_to_write:
            file.writelines(line)
    file.writelines(u'!\n')

    # muon position
    file.writelines(u'! muon position: ' + unicode(muon_position) + u' \n! \n')

    # atom perturbations
    if len(perturbed_distances) > 0:
        file.writelines(u'! atom position perturbations: \n')
        for iperturbpair in xrange(0, len(perturbed_distances)):
            file.writelines(u'!\t ' + unicode(perturbed_distances[iperturbpair][0]) + u' to '
                            + unicode(perturbed_distances[iperturbpair][1]) + u'\n')
        file.writelines(u'! \n')
    elif isinstance(squish_radius, float) and not use_pw_output:
        file.writelines(u'! nearest neighbour F-mu radius adjusted to be ' + unicode(squish_radius) + u' angstroms. \n!\n')

    # nnn ness
    file.writelines(u'! Calculated by looking at ')
    for i in xrange(0, nnnness):
        file.writelines(u'n')

    file.writelines(u' interactions \n! \n')

    if exclusive_nnnness == True and not use_pw_output:
        file.writelines(u'! Effects of interactions of atoms spatially closer than ')
        for i in xrange(0, nnnness):
            file.writelines(u'n')
        file.writelines(u' have been ignored. \n! \n')

    # lattice type and parameter
    if not use_xtl_input:
        file.writelines(u'! lattice type: ' + unicode(lattice_type) + u' (based on QE convention) \n')
        file.writelines(u'! lattice parameter: ' + unicode(lattice_parameter) + u' Angstroms \n! \n')

    file.writelines(u'! start of data: \n')


# batch write data to file
def write_to_file(file, t, P):
    for i in xrange(0, len(t)-1):
        file.writelines(unicode(t[i]) + u' ' + unicode(P[i]) + u'\n')


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
                     fourier=False, fourier_2d=False, outfile_location=None, tol=1e-10, plot=False):

    # if told to use both pw and xtl, exit
    if use_pw_output and use_xtl_input:
        print u'Cannot use pw and xtl inputs simultaneously. Aborting...'
        return times * 0

    # check that everything is in order: if not, then leave
    if not use_pw_output and not use_xtl_input:
        # should have manually entered all the details in
        if lattice_type is None or lattice_parameter is None or lattice_angles is None or atomic_basis is None or \
                perturbed_distances is None:
            print u'Not enough information given. Aborting...'
            return times * 0
    elif use_pw_output:
        if pw_output_file_location is None or no_atoms<=0:
            print u'Not enough information given. Aborting...'
            return times * 0
    else:
        if xtl_input_location is None:
            print u'Not enough information given. Aborting...'
            return times * 0


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
            alpha = lattice_angles[0]*np.pi/180.
            beta = lattice_angles[1]*np.pi/180.
            gamma = lattice_angles[2]*np.pi/180.

            # type of calculation - can't do fourier2d if not fourier
            fourier_2d = fourier_2d and fourier

            # define primitive vectors a1, a2 and a3, from pw.x input description
            if lattice_type == ibrav.CUBIC_SC:
                # simple cubic
                a1 = coord.TCoord3D(a, 0, 0)
                a2 = coord.TCoord3D(0, a, 0)
                a3 = coord.TCoord3D(0, 0, a)
            elif lattice_type == ibrav.CUBIC_FCC:
                # fcc cubic
                a1 = coord.TCoord3D(-a*.5, 0, a*.5)
                a2 = coord.TCoord3D(0, a*.5, a*.5)
                a3 = coord.TCoord3D(-a*.5, a*.5, 0)
            elif lattice_type == ibrav.CUBIC_BCC:
                a1 = coord.TCoord3D(.5*a, .5*a, .5*a)
                a2 = coord.TCoord3D(-.5*a, .5*a, .5*a)
                a3 = coord.TCoord3D(-.5*a, -.5*a, .5*a)
            elif lattice_type == ibrav.CUBIC_BCC_EXTRA:
                a1 = coord.TCoord3D(-.5*a, .5*a, .5*a)
                a2 = coord.TCoord3D(.5*a, -.5*a, .5*a)
                a3 = coord.TCoord3D(.5*a, .5*a, -.5*a)
            elif lattice_type == ibrav.MONOCLINIC_UB:
                # monoclinic, unique axis b
                a1 = coord.TCoord3D(a, 0, 0)
                a2 = coord.TCoord3D(0, b, 0)
                a3 = coord.TCoord3D(c*np.cos(beta), 0, c*np.sin(beta))
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
                    basis_atom.position = basis_atom.position*a

                # sort out the perturbed pairs
                for perturbed_pair in perturbed_distances:
                    perturbed_pair[0] = perturbed_pair[0]*a
                    perturbed_pair[1] = perturbed_pair[1]*a

                # finally, do the muon
                muon_position = muon_position*a
            else:
                # we're in cartesian-land with the distances given in angstroms - so do nothing!
                pass

            # create muon
            muon = Tatom(muon_position, gyromag_ratio=851.372, II=1, name=u'mu')
        else:
            # import the fractional coordinates from the XTL
            muon, atomic_basis, [a1, a2, a3] = AtomObtainer.get_atoms_from_xtl(xtl_file_location=xtl_input_location)
            lattice_type = ibrav.OTHER
            muon_position = muon.position

        # now what we want to do is calculate how many of these are nn, nnn, nnnn etc
        nnn_atoms = nnn_finder(atomic_basis, muon, [a1, a2, a3], nnnness, exclusive_nnnness,
                               perturbed_distances, squish_radius)

        print unicode(squish_radius)

        # as before, make a list of spins to calculate (including that of the muon)
        All_Spins = [muon]
        for i_atom in nnn_atoms:
            All_Spins.append(Tatom(i_atom[1], i_atom[2].gyromag_ratio, i_atom[2].II, i_atom[2].name, i_atom[2].abundance))

    # print the atoms in the list
    for i_atom in All_Spins:
        print i_atom

    # turn the spins into a muon-centred basis
    for spin in All_Spins:
        for isotopeid in xrange(0, len(spin)):
            spin[isotopeid].position = spin[isotopeid].position - muon_position

    # count number of spins
    N_spins = len(All_Spins) - 1

    # count the number of combinations of isotopes
    isotope_combinations = 1
    for atoms in All_Spins:
        isotope_combinations = isotope_combinations*len(atoms)
    print unicode(isotope_combinations) + u' isotope combination(s) found'

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
        for atomid in xrange(0, len(All_Spins)):
            Spins.append(All_Spins[atomid][current_isotope_ids[atomid]])
            probability = probability * All_Spins[atomid][current_isotope_ids[atomid]].abundance

        # create measurement operators for the muon's spin
        muon_spin_x = measure_ith_spin(Spins, 0, Spins[0].pauli_x)
        muon_spin_y = measure_ith_spin(Spins, 0, Spins[0].pauli_y)
        muon_spin_z = measure_ith_spin(Spins, 0, Spins[0].pauli_z)

        # calculate hamiltonian
        hamiltonian = calc_total_hamiltonian(Spins)

        # find eigenvalues and eigenvectors of hamiltonian
        print u"Finding eigenvalues..."
        dense_hamiltonian = hamiltonian.todense()
        this_E, R = linalg.eigh(dense_hamiltonian)
        Rinv = R.H
        print u"Found eigenvalues:"
        print this_E

        # Calculate constant (lab book 1 page 105)
        thisconst = 0
        for i in xrange(0, len(R)):
            thisconst = thisconst + pow(abs(Rinv[i]*muon_spin_x*R[:, i]), 2) \
                            + pow(abs(Rinv[i]*muon_spin_y*R[:, i]), 2) \
                            + pow(abs(Rinv[i]*muon_spin_z*R[:, i]), 2)
        const = const + probability*thisconst/(6*(muon_spin_x.shape[0]/2))

        this_amplitude = np.zeros((len(R), len(R)))
        # now calculate oscillating term
        for i in xrange(0, len(R)):
            Rx = Rinv[i] * muon_spin_x
            Ry = Rinv[i] * muon_spin_y
            Rz = Rinv[i] * muon_spin_z
            print unicode(100*i/len(R)) + u'% complete...'
            if fourier_2d:
                jmin = 0
            else:
                jmin = i+1
            for j in xrange(jmin, len(R)):
                this_amplitude[i][j] = (pow(abs(Rx*R[:, j]), 2)
                                        + pow(abs(Ry*R[:, j]), 2)
                                        + pow(abs(Rz*R[:, j]), 2))*probability / (3*(muon_spin_x.shape[0]/2))

        amplitude.append(this_amplitude.tolist())
        E.append(this_E.tolist())

        # increment the isotope ids
        current_isotope_ids = inc_isotope_id(basis=number_isotopes, oldids=current_isotope_ids)


    ## OUTPUT ##

    if fourier:

        # dump all into an array
        fourier_result = []

        # for each isotope
        for isotope_combination in xrange(0, len(amplitude)):
            # noinspection PyTypeChecker
            for i in xrange(0, len(E[isotope_combination])):
                if fourier_2d:
                    # noinspection PyTypeChecker
                    for j in xrange(0, len(E[isotope_combination])):
                        fourier_result.append((amplitude[isotope_combination][i][j], E[isotope_combination][i],
                                               E[isotope_combination][j]))
                else:
                    # noinspection PyTypeChecker
                    for j in xrange(i + 1, len(E[isotope_combination])):
                        fourier_result.append((amplitude[isotope_combination][i][j],
                                               abs(E[isotope_combination][i] - E[isotope_combination][j])))

        # go through the frequencies, if there's degenerate eigenvalues then add together the amplitudes
        if fourier_2d:
            fourier_result = sorted(fourier_result, key=lambda frequency: (frequency[1], frequency[2]))
            i = 0
            while i < len(fourier_result)-1:
                # test for degeneracy (up to a tolerance for machine precision)
                if (abs((fourier_result[i][1]) - (fourier_result[i+1][1])) < tol) \
                        and (abs(fourier_result[i][2] - fourier_result[i+1][2]) < tol):
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
            while i < len(fourier_result)-1:
                # test for degeneracy (up to a tolerance for machine precision)
                if (abs(fourier_result[i][1] - fourier_result[i+1][1]) < tol)\
                        and (abs(fourier_result[i][2] - fourier_result[i+1][2]) < tol):
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
            while i < len(fourier_result)-1:
                # test for degeneracy (up to a tolerance for machine precision)
                if abs((fourier_result[i][1]) - (fourier_result[i+1][1])) < tol:
                    # degenerate eigenvalue: add the amplitudes, keep frequency the same
                    fourier_result[i] = (fourier_result[i][0] + fourier_result[i+1][0], fourier_result[i][1])
                    # remove the i+1th (degenerate) eigenvalue
                    del fourier_result[i+1]
                else:
                    i = i + 1

            i = 0
            # now remove any amplitudes which are less than 1e-15
            while i < len(fourier_result)-1:
                if abs(fourier_result[i][0]) < 1e-7:
                    # remove the entry
                    del fourier_result[i]
                else:
                    i = i + 1

        # dump into file if requested
        if outfile_location is not None:
            outfile = open(outfile_location, u"w")
            # do preamble
            file_preamble(file=outfile, muon_position=muon_position, nn_atoms=All_Spins, fourier=fourier,
                          fourier_2d=fourier_2d, tol=tol, use_xtl_input=use_xtl_input,
                          xtl_input_location=xtl_input_location, use_pw_output=use_pw_output,
                          perturbed_distances=perturbed_distances, squish_radius=squish_radius, nnnness=nnnness,
                          exclusive_nnnness=exclusive_nnnness, lattice_type=lattice_type,
                          lattice_parameter=lattice_parameter)

            if fourier_2d:
                outfile.writelines(u'! frequency1 frequency2 amplitude \n')
                outfile.writelines([unicode(fourier_entry[1]) + u' ' + unicode(fourier_entry[2]) + u' ' + unicode(fourier_entry[0])
                                    + u'\n' for fourier_entry in fourier_result])
            else:
                outfile.writelines(u'! frequency amplitude \n')
                outfile.writelines(u'0 ' + unicode(const[0, 0]) + u'\n')
                outfile.writelines([unicode(fourier_entry[1]) + u' ' + unicode(fourier_entry[0]) + u'\n' for fourier_entry
                                    in fourier_result])
            outfile.close()

        return np.array(fourier_result)

    else: # if not fourier output:

        P_average = []

        # calculate each time separately
        for time in np.nditer(times):
            print u"t=" + unicode(time)
            P_average.append(calc_p_average_t(time, const, amplitude, E).max())
            # print(P_average[-1])

        if outfile_location is not None:
            # dump results in a file if requested
            outfile = open(outfile_location, u"w")
            # do preamble
            file_preamble(file=outfile, muon_position=muon_position, nn_atoms=All_Spins, fourier=fourier,
                          fourier_2d=fourier_2d, tol=tol, use_xtl_input=use_xtl_input,
                          xtl_input_location=xtl_input_location, use_pw_output=use_pw_output,
                          perturbed_distances=perturbed_distances, squish_radius=squish_radius, nnnness=nnnness,
                          exclusive_nnnness=exclusive_nnnness, lattice_type=lattice_type,
                          lattice_parameter=lattice_parameter, starttime=times[0], endtime=times[-1],
                          timestep=times[1]-times[0])
            outfile.writelines(u'! t P_average \n')
            write_to_file(outfile, times, P_average)
            outfile.close()

        # plot the angular averaged muon polarisation
        if plot:
            pyplot.plot(times, P_average)
            pyplot.title(u'Muon Polarisation')
            pyplot.xlabel(u'time (microseconds)')
            pyplot.ylabel(u'Muon Polarisation')
            pyplot.show()

        return np.array(P_average)
