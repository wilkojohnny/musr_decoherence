# DecoherenceEntropy.py - calculates the entropy for muon-dipolar decoherence
# John Wilkinson 15/11/19

import AtomObtainer as AO  # get atoms
from MDecoherenceAtom import TDecoherenceAtom as atom  # allows atom generation
import DecoherenceCalculator  # for Hamiltonian etc
import TCoord3D as coord  # coordinate tools
import scipy.sparse as sparse  # allows dealing with sparse matrices and linear algebra tools
import scipy.linalg as scilinalg  # stops this conflicting with numpy's linalg
import numpy as np  # maths tools
import matplotlib.pyplot as pyplot  # plottting
import subprocess  # git version
from datetime import datetime  # for current date and time


def main():
    #### INPUT ####

    # ## IF WE'RE USING PW_OUTPUT
    # pw_output_file_location = 'CaF2.relax.mu.pwo'
    # no_atoms = 11  # includes muon

    ## IF WE'RE USING AN XTL (crystal fractional coordinates) FILE
    # xtl_input_location = 'CaF2_final_structure_reduced.xtl'
    # (don't forget to define nnnness!)

    squish_radii = [1.172211, None]  # radius of the nn F-mu bond after squishification (1.18 standard, None for no squishification)

    ## IF WE'RE NOT USING pw output:
    # nn, nnn, nnnn?
    # nnnness = 2  # 2 = nn, 3 = nnn etc
    # exclusive_nnnness - if TRUE, then only calculate nnnness's interactions (and ignore the 2<=i<nnnness interactions)
    #   exclusive_nnnness = False

    ## IF NOT PW NOR XTL:
    # lattice type: https://www.quantum-espresso.org/Doc/INPUT_PW.html#idm45922794628048
    lattice_type = AO.ibrav.CUBIC_FCC  # # can only do fcc and monoclinic (unique axis b)
    # lattice parameters and angles, in angstroms
    lattice_parameter = [5.44542, 0, 0]  # [a, b, c]
    lattice_angles = [90, 0, 0]  # [alpha, beta, gamma] in **degrees**

    # are atomic coordinates provided in terms of alat or in terms of the primitive lattice vectors?
    input_coord_units = AO.position_units.ALAT

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
    muon_polarisation = coord.TCoord3D(0, 0, 1)

    # file name
    output_file_name = 'sertest.dat'

    calc_entropy(muon_position=muon_position, squish_radius=squish_radii, lattice_type=lattice_type,
                 lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
                 muon_polarisation=muon_polarisation, input_coord_units=input_coord_units, atomic_basis=atomic_basis,
                 perturbed_distances=perturbed_distances, nnnness=2, ask_each_atom=False, times=np.arange(0, 25, 0.02),
                 output_file_location=output_file_name, plot=False, trace_left_dim=0, trace_right_dim=4)

    return 1


def calc_entropy(muon_position, muon_polarisation: coord, squish_radius=None, times=np.arange(0, 10, 0.1),
                 output_file_location=None, plot=False,
                 # partial trace dimensions - left (right) trace corresponds to tracing over the components which correspond to the
                 # density matrix kronecker product-ed with matrices of these dimensions on the left (right).
                 trace_left_dim=0, trace_right_dim=0,
                 # arguments for manual input of lattice
                 lattice_type=None, lattice_parameter=None, lattice_angles=None,
                 input_coord_units=AO.position_units.ALAT, atomic_basis=None, perturbed_distances=None,
                 # arguments for XTL
                 use_xtl_input=False, xtl_input_location=None,
                 # arguments for XTL or manual input
                 nnnness=2, exclusive_nnnness=False,
                 # arguments for pw.x output
                 use_pw_output=False, pw_output_file_location=None, no_atoms=0, ask_each_atom=False):

    muon, Spins, got_spins = AO.get_spins(muon_position, squish_radius, lattice_type, lattice_parameter, lattice_angles,
                                          input_coord_units, atomic_basis, perturbed_distances, use_xtl_input,
                                          xtl_input_location, nnnness, exclusive_nnnness, use_pw_output,
                                          pw_output_file_location, no_atoms, ask_each_atom)

    # create measurement operators for the muon's spin
    muon_spin_x = 2*Spins[0].pauli_x
    muon_spin_y = 2*Spins[0].pauli_y
    muon_spin_z = 2*Spins[0].pauli_z

    # generate density matrix (=.5(1+muon_rhat) x 1^N)
    muon_spin_polariasation = muon_polarisation.ortho_x * muon_spin_x + muon_polarisation.ortho_y * muon_spin_y + \
                              muon_polarisation.ortho_z * muon_spin_z

    # calcualte the density matrix at time 0
    density_matrix_0 = sparse.kron(0.5*(sparse.identity(2, format="csc") + muon_spin_polariasation),
                                   sparse.identity(pow(2, len(Spins)-1)), format="csc") / pow(2, len(Spins)-1)

    # calculate the Hamiltonian
    hamiltonian = DecoherenceCalculator.calc_dipolar_hamiltonian(Spins, just_muon_interactions=False)

    # diagonalsie the Hamiltonian
    E, R = np.linalg.eigh(hamiltonian.todense())
    Rinv = R.H
    Rsp = sparse.csc_matrix(R)
    Rspinv = sparse.csc_matrix(Rinv)
    E = sparse.diags(E, format='csc')

    Rspinv_density0_Rsp = Rspinv*density_matrix_0*Rsp

    # set up output array
    entropy_out = np.empty(shape=times.shape)

    # file open and preamble (if applicable)
    output_file = None
    if output_file_location is not None:
        # open the output file in write mode
        output_file = open(output_file_location, 'w')

        entropy_file_preamble(output_file, times, trace_left_dim, trace_right_dim, muon_polarisation, muon_position,
                              Spins, use_xtl_input, xtl_input_location, use_pw_output, pw_output_file_location,
                              perturbed_distances, squish_radius, nnnness, exclusive_nnnness, lattice_type,
                              lattice_parameter, 1)

    for i_time in range(0, len(times)):
        # get the time
        time = times[i_time]
        print(time)

        # time evolve
        Ejt = E * 1j * time
        expE = scilinalg.expm(Ejt)
        expEm = scilinalg.expm(-Ejt)
        density_matrix = Rsp*expEm*Rspinv_density0_Rsp*expE*Rspinv

        # do partial traces
        if trace_right_dim != 0:
            if trace_right_dim/density_matrix.shape[0] <= 0.125:  # 256 found experimentally for 2028x2048 matrix
                density_matrix = traces(density_matrix, right_dim=trace_right_dim)
            else:
                density_matrix = trace(density_matrix, right_dim=trace_right_dim)
        if trace_left_dim != 0:
            if trace_left_dim/density_matrix.shape[0] <= 0.125:  # 256 found experimentally for 2028x2048 matrix
                density_matrix = traces(density_matrix, left_dim=trace_left_dim)
            else:
                density_matrix = trace(density_matrix, left_dim=trace_left_dim)

        # do x*log(x)
        entropy = -1*tr_xlogx(density_matrix)

        # if there is a big imaginary part of the trace, something has gone wrong...
        assert entropy.imag < 1e-5

        entropy_out[i_time] = entropy.real/np.log(2)

        if output_file is not None:
            output_file.writelines(str(time) + '\t' + str(entropy.real) + '\n')

    print(entropy_out)
    if output_file is not None:
        output_file.close()
    if plot:
        pyplot.plot(times, entropy_out, 'b')
        pyplot.show()


def entropy_file_preamble(output_file, times, trace_left_dim, trace_right_dim, muon_polarisation, muon_position, Spins,
                          use_xtl_input, xtl_input_location, use_pw_output, pw_output_file_location,
                          perturbed_distances, squish_radius, nnnness, exclusive_nnnness, lattice_type,
                          lattice_parameter, no_processors=1):

    # program name, date and time completed
    output_file.writelines('! Decoherence Calculator Output - ' + datetime.now().strftime("%d/%m/%Y, %H:%M:%S") +
                           '\n!\n')

    # get the git version
    version_label = subprocess.check_output(["git", "describe", "--always"]).strip()
    output_file.writelines('! Using version ' + str(version_label) + ', running on' + str(no_processors) +
                           ' cores' + '\n!\n')

    # type of calculation
    output_file.writelines('! Entropy time calculation completed between t=' + str(times[0]) + ' and ' +
                           str(times[-1]) + ' with a timestep of ' + str(times[1] - times[0]) + ' microseconds' +
                           '\n!\n')

    output_file.writelines('! Muon initial polarisation is in the direction (' + str(muon_polarisation.ortho_x) +
                           ', ' + str(muon_polarisation.ortho_y) + ', ' + str(muon_polarisation.ortho_z) + ') \n ')

    output_file.writelines('! Calculation done by partial traces of left dimension ' + str(trace_left_dim) + ' and '
                           + 'right dimension ' + str(trace_right_dim) + '. \n')

    # do atoms preamble
    AO.atoms_file_preamble(output_file, muon_position, Spins, use_xtl_input, xtl_input_location, use_pw_output,
                           pw_output_file_location, perturbed_distances, squish_radius, nnnness, exclusive_nnnness,
                           lattice_type, lattice_parameter)

    output_file.writelines('! time (microseconds)  von Neumann Entropy (units of log2)\n')


def trace(matrix, left_dim: int = 0, right_dim: int = 0):
    # check matrix is square, and the input arguments are not silly
    assert matrix.shape[0] == matrix.shape[1]
    assert left_dim >= 0
    assert right_dim >= 0

    # declare current trace to shut up interpreter
    current_trace = None

    # if not doing partial trace, then
    if left_dim == 0 and right_dim == 0:  # this usage of 0 is inconsistent -- this would be no trace...
        current_trace = 0
        for i in range(0, matrix.shape[0]):
            current_trace += matrix[i][i]
    if left_dim > 0:
        # trace over left dims
        # calculate what the final dimension should be (will always be an integer)
        res_dim = int(np.round(matrix.shape[0]/left_dim))
        # set up final matrix (and allow it to be complex)
        current_trace = np.zeros((res_dim, res_dim,), dtype=complex)
        # for each element in the final matrix (i,j)
        for i in range(0, res_dim):
            for j in range(0, res_dim):
                # sum over the indices
                for s in range(0, left_dim):
                    current_trace[i][j] = current_trace[i][j] + matrix[i + s*res_dim, j + s*res_dim]
        current_trace = sparse.csc_matrix(current_trace)
    if right_dim > 0:
        # trace over right dims
        # calculate what the final dimension should be (will always be an integer)
        res_dim = int(np.round(matrix.shape[0] / right_dim))
        # set up final matrix (and allow it to be complex)
        current_trace = np.zeros((res_dim, res_dim,), dtype=complex)
        # for each element in the final matrix (i,j)
        for i in range(0, res_dim):
            for j in range(0, res_dim):
                # sum over the indices
                for s in range(0, right_dim):
                    current_trace[i][j] = current_trace[i][j] + matrix[i*right_dim + s, j*right_dim + s]
        current_trace = sparse.csc_matrix(current_trace)

    return current_trace


def traces(matrix, left_dim: int = 1, right_dim: int = 1):
    # sparce traces (complete and partial), useful for big matrices
    # see lab book 2 page 64

    # check matrix is square, and the input arguments are not silly
    assert matrix.shape[0] == matrix.shape[1]
    assert left_dim >= 0
    assert right_dim >= 0
    # check its not being asked to do two traces at the same time
    assert (left_dim > 1) != (right_dim > 1)

    matrix_dim = matrix.shape[0]

    # check its not being asked to trace over bigger matrices than matrix
    assert (left_dim <= matrix_dim) and (right_dim <= matrix_dim)

    # define size of orthonormal basis to do trace with
    ortho_trace_dim = left_dim*(left_dim > 1) + right_dim*(right_dim > 1)
    # left (and right) _identity_dim - size of the identity matrix to use (min these can be is 1...)
    right_identity_dim = int(matrix_dim / left_dim - 1) * (left_dim > 1) + 1
    left_identity_dim = int(matrix_dim / right_dim - 1) * (right_dim > 1) + 1
    final_trace_dim = int(matrix_dim / ortho_trace_dim)

    # declare current trace to shut up interpreter
    current_trace = sparse.dia_matrix((final_trace_dim, final_trace_dim))

    # for each element, find partial trace element, add to current_trace
    for i in range(0, ortho_trace_dim):
        # generate this orthogonal vector for PT
        ortho_trace_elem = sparse.lil_matrix((ortho_trace_dim, 1))
        ortho_trace_elem[i, 0] = 1
        # put this in DIA format because kron likes that
        ortho_trace_elem = sparse.dia_matrix(ortho_trace_elem)
        # generate the right-multiply matrix
        right_trace_mat = sparse.kron(sparse.eye(left_identity_dim),
                                      sparse.kron(ortho_trace_elem, sparse.eye(right_identity_dim)))
        # ... and the left
        left_trace_mat = right_trace_mat.transpose()
        # and find the trace element!
        current_trace = current_trace + left_trace_mat*matrix*right_trace_mat

    return current_trace

# do x*log(x) of a matrix (takes into account lim(x->0)xlogx = 0)
def tr_xlogx(x):
    # diagonalise x
    E, R, Rinv = scilinalg.eig(x.todense(), left=True, right=True)

    # log each element in E, unless it's 0, then make it 0
    for i in range(0, len(E)):
        if E[i] != 0:
            E[i] = E[i]*np.log(E[i])

    # make E a matrix
    E = sparse.diags(E)

    # return trace xlogx
    return np.sum(E)


if __name__=='__main__':
    main()