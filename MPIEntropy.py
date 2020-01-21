# MPIEntropy.py -- Calculate the DecoherenceEntropy but using MPI, so can be done in parallel speeding things up a lot!
# John Wilkinson, 20/1/2020

# run this in the conda environment, to replicate the ARC environment as well as possible...
from mpi4py import MPI  # for parallelisation
import AtomObtainer as AO  # atom tools
from MDecoherenceAtom import TDecoherenceAtom as atom  # allows atom generation
import DecoherenceCalculator  # for Hamiltonian etc
import TCoord3D as coord  # coordinate tools
import scipy.sparse as sparse  # allows dealing with sparse matrices and linear algebra tools
import scipy.linalg as scilinalg  # stops this conflicting with numpy's linalg
import numpy as np  # maths tools
import matplotlib.pyplot as pyplot  # for plotting
import DecoherenceEntropy  # serial version still has lots of the meat we can use
import time

def main():
    #### INPUT ####

    squish_radii = [1.172211,
                    None]  # radius of the nn F-mu bond after squishification (1.18 standard, None for no squishification)

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
        atom(coord.TCoord3D(0.25, 0.25, 0.25), gyromag_ratio=251.713, II=1, name='F'),
        atom(coord.TCoord3D(0.25, 0.25, 0.75), gyromag_ratio=251.713, II=1, name='F')
    ]

    # register the perturbed distances
    perturbed_distances = []

    # define muon position
    muon_position = coord.TCoord3D(.25, 0.25, 0.5)
    muon_polarisation = coord.TCoord3D(0, 0, 1)

    # file name
    output_file_name = 'testy.dat'

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
    # set up MPI environment

    MPI_comm = MPI.COMM_WORLD  # MPI communicator
    rank = MPI_comm.Get_rank()  # get what core this is
    no_cores = MPI_comm.Get_size()  # find out how many cores there are

    assert no_cores >= 2  # don't run parallel version if only one core!

    if rank == 0:
        print('We are running on ' + str(no_cores) + ' cores!')

        # core 0 starts things off, by getting the atoms and finding the Hamiltonian etc
        muon, Spins, got_spins = AO.get_spins(muon_position, squish_radius, lattice_type, lattice_parameter,
                                              lattice_angles, input_coord_units, atomic_basis, perturbed_distances,
                                              use_xtl_input, xtl_input_location, nnnness, exclusive_nnnness,
                                              use_pw_output, pw_output_file_location, no_atoms, ask_each_atom)

        # create measurement operators for the muon's spin
        muon_spin_x = 2 * Spins[0].pauli_x
        muon_spin_y = 2 * Spins[0].pauli_y
        muon_spin_z = 2 * Spins[0].pauli_z

        # generate density matrix (=.5(1+muon_rhat) x 1^N)
        muon_spin_polariasation = muon_polarisation.ortho_x * muon_spin_x + muon_polarisation.ortho_y * muon_spin_y + \
                                  muon_polarisation.ortho_z * muon_spin_z

        # calcualte the density matrix at time 0
        density_matrix_0 = sparse.kron(0.5 * (sparse.identity(2, format="csc") + muon_spin_polariasation),
                                       sparse.identity(pow(2, len(Spins) - 1)), format="csc") / pow(2, len(Spins) - 1)

        # calculate the Hamiltonian
        hamiltonian = DecoherenceCalculator.calc_dipolar_hamiltonian(Spins, just_muon_interactions=False)

        # diagonalsie the Hamiltonian
        E, R = np.linalg.eigh(hamiltonian.todense())
        Rinv = R.H
        Rsp = sparse.csc_matrix(R)
        Rspinv = sparse.csc_matrix(Rinv)
        E = sparse.diags(E, format='csc')

        Rspinv_density0_Rsp = Rspinv * density_matrix_0 * Rsp

        # file open and preamble (if applicable)
        if output_file_location is not None:
            # open the output file in write mode
            output_file = open(output_file_location, 'w')

            DecoherenceEntropy.entropy_file_preamble(output_file, times, trace_left_dim, trace_right_dim,
                                                     muon_polarisation,
                                                     muon_position, Spins, use_xtl_input, xtl_input_location,
                                                     use_pw_output,
                                                     pw_output_file_location, perturbed_distances, squish_radius,
                                                     nnnness,
                                                     exclusive_nnnness, lattice_type, lattice_parameter, no_cores)

            output_file.close()
        # steal a copy of times before core 1 messes it up...
        all_times = times

        # print left and right traces
        print('Left trace dim: ' + str(trace_left_dim))
        print('Right trace dim: ' + str(trace_right_dim))
        print('\n BEGIN ENTROPY (format is <core>: time)')
    else:
        # for all other cores, nontype them for now
        E = None
        Rsp = None
        Rspinv = None
        Rspinv_density0_Rsp = None

    if rank == 1:
        # core 1 can distribute the times to everyone to give core 0 a break
        times = np.array_split(times, no_cores - 1)
        times = [np.array([len(arr) for arr in times])] + times

    # now core 0 should broadcast what its found to the other cores
    E = MPI_comm.bcast(E, root=0)
    Rsp = MPI_comm.bcast(Rsp, root=0)
    Rspinv = MPI_comm.bcast(Rspinv, root=0)
    Rspinv_density0_Rsp = MPI_comm.bcast(Rspinv_density0_Rsp, root=0)

    # core 1 should tell the other cores what times they should calculate for
    times = MPI_comm.scatter(times, root=1)

    # if rank is not 0, do the calculation:
    if rank != 0:
        for i_time in range(0, len(times)):
            # get the time
            time = times[i_time]
            print('<' + str(rank) + '>:' + str(time))

            # time evolve
            Ejt = E * 1j * time
            expE = scilinalg.expm(Ejt)
            expEm = scilinalg.expm(-Ejt)
            density_matrix = Rsp * expEm * Rspinv_density0_Rsp * expE * Rspinv

            # do partial traces
            if trace_right_dim != 0:
                if trace_right_dim / density_matrix.shape[0] <= 0.125:  # 256 found experimentally for 2028x2048 matrix
                    density_matrix = DecoherenceEntropy.traces(density_matrix, right_dim=trace_right_dim)
                else:
                    density_matrix = DecoherenceEntropy.trace(density_matrix, right_dim=trace_right_dim)
            if trace_left_dim != 0:
                if trace_left_dim / density_matrix.shape[0] <= 0.125:  # 256 found experimentally for 2028x2048 matrix
                    density_matrix = DecoherenceEntropy.traces(density_matrix, left_dim=trace_left_dim)
                else:
                    density_matrix = DecoherenceEntropy.trace(density_matrix, left_dim=trace_left_dim)

            # do x*log(x)
            entropy = -1 * DecoherenceEntropy.tr_xlogx(density_matrix)

            # if there is a big imaginary part of the trace, something has gone wrong...
            assert entropy.imag < 1e-5

            # divide by log 2
            entropy = entropy.real / np.log(2)

            # send this off to core 0
            MPI_comm.send(entropy, dest=0, tag=i_time)

    elif rank == 0:
        # rank is 0 - so be in charge of data gathering

        # times here is just a list of the size of the arrays the others are on
        times_sizes = times

        # make data gathering lists (remember -- !)
        entropy_outputs = [np.empty(size) for size in times_sizes]

        # find out what the largest size time array one core is working on
        max_time_array_size = max(times_sizes)

        # for time_i from 0 to <biggest time_i on one core>
        for time_i in range(0, max_time_array_size):
            # for each core
            for core in range(1, no_cores):
                # check that this time_i exists
                if time_i < times_sizes[core - 1]:
                    # if it does, then wait for the entropy, and put into array
                    entropy_outputs[core - 1][time_i] = MPI_comm.recv(source=core, tag=time_i)

        # when all done, collate entropy_out, and save into file
        entropy_output = np.empty(0)
        for core in range(1, no_cores):
            entropy_output = np.append(entropy_output, entropy_outputs[core - 1])

        if output_file_location is not None:
            output_file = open(output_file_location, 'a')
            for i in range(0, len(all_times)):
                output_file.writelines(str(all_times[i]) + '\t' + str(entropy_output[i]) + '\n')

        if plot:
            pyplot.plot(all_times, entropy_output, 'b')
            pyplot.show()


if __name__ == '__main__':
    main()
