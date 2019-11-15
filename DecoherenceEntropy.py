# DecoherenceEntropy.py - calculates the entropy for muon-dipolar decoherence
# John Wilkinson 15/11/19

import AtomObtainer as AO  # get atoms
from MDecoherenceAtom import TDecoherenceAtom as atom  # allows atom generation
import TCoord3D as coord  # coordinate tools
from scipy import sparse, linalg  # allows dealing with sparse matrices and linear algebra tools
import numpy as np  # maths tools

def calc_entropy(muon_position, muon_polarisation: coord, squish_radius=None,
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
    density_matrix_0 = sparse.kron(0.5*(sparse.identity(2) + muon_spin_polariasation),
                                   sparse.identity(pow(2, len(Spins)-1))) / pow(2, len(Spins)-1)

    # do a partial trace
    density_matrix_0 = trace(density_matrix_0.tocsr(), left_dim=2)

    # do x*log(x)
    rho_log_rho = xlogx(density_matrix_0)

    # do trace
    print(-1*trace(rho_log_rho))


def trace(matrix, left_dim: int = 0, right_dim: int = 0):
    # check matrix is square, and the input arguments are not silly
    assert matrix.shape[0] == matrix.shape[1]
    assert left_dim >= 0
    assert left_dim >= 0

    # if not doing partial trace, then
    if left_dim == 0 and right_dim == 0:
        current_trace = 0
        for i in range(0, matrix.shape[0]):
            current_trace += matrix[i][i]
    elif left_dim > 0:
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
                    current_trace[i][j] = current_trace[i][j] + matrix[i + s*res_dim,j + s*res_dim]
        current_trace = sparse.coo_matrix(current_trace)
    elif right_dim > 0:
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
        current_trace = sparse.coo_matrix(current_trace)
    else:
        # something has gone horribly wrong!
        assert False

    return current_trace


# do x*log(x) of a matrix (takes into account lim(x->0)xlogx = 0)
def xlogx(x):
    # diagonalise x
    E, R, Rinv = linalg.eig(x.todense(), left=True, right=True)

    # log each element in E, unless it's 0, then make it 0
    for i in range(0, len(E)):
        if E[i] != 0:
            E[i] = E[i]*np.log(E[i])

    # make E a matrix
    E = sparse.diags(E)

    # return xlogx
    return R*E*Rinv


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

    calc_entropy(muon_position=muon_position, squish_radius=squish_radii, lattice_type=lattice_type,
                 lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
                 muon_polarisation=muon_polarisation, input_coord_units=input_coord_units, atomic_basis=atomic_basis,
                 perturbed_distances=perturbed_distances, nnnness=2, ask_each_atom=False)
    return 1


if __name__=='__main__':
    main()