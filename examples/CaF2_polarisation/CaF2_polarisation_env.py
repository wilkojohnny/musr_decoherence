# Plot muon polarisation of CaF2 using muon_environment
# works with Decoherence_calculator version>ff1a392386b66178c63dbe80dbfe6e7c5b935286

import functools
print = functools.partial(print, flush=True)

import musr_decoherence.muon_environment as mu_env
from musr_decoherence import DipolarPolarisation

from ase.io import read
import numpy as np


def main():

    cif_file = '../CaF2.cif'

    # get the atoms from the CIF file
    CaF2_atoms = read(cif_file)

    # define the muon position as being in between atom 4 and 9 (which are both Fs in the structure).
    nnindices = [4, 9]

    # define how much the muon perturbs the environment. List is [[atom_id, dist], [atom_id, dist], ...]
    perturbations = [[4, 1.18], [9, 1.18]]
    # lambda_squish -- the factor by all the nearest-neighbours above what is
    # defined in squish_radii are perturbed by (in most cases <1)
    lambda_squish = 0.920

    # add the muon to CaF2_atoms
    CaF2_atoms_mu = mu_env.add_muon_to_aseatoms(CaF2_atoms, nn_indices=nnindices)

    # now perturb the atoms
    CaF2_atoms_mu = mu_env.perturb_atoms(CaF2_atoms_mu, perturbations=perturbations)

    # get the dominant nuclei
    nn_nuclei, CaF2_atoms_mu, CaF2_atoms_mu_ids = mu_env.get_dominant_nuclei(CaF2_atoms_mu)

    # model the further nuclei with lambda_squish
    # nn_nuclei = mu_env.model_further_nuclei(nn_nuclei, nn_start=-1, atoms_mu=CaF2_atoms_mu,
    #                                         nn_indices=CaF2_atoms_mu_ids, unperturbed_atoms=CaF2_atoms)

    muon, all_spins = mu_env.aseatoms_to_tdecoatoms(nn_nuclei)

    output_file_name = '../../../CaF2_polarisation_T20.dat'

    # define muon polarisation relative to the sample -- None for polycrystalline
    muon_sample_polarisation = None

    DipolarPolarisation.calc_dipolar_polarisation(all_spins=all_spins, muon=muon, muon_sample_polarisation=muon_sample_polarisation,
                                                  plot=True, fourier=False, fourier_2d=False, tol=1e-3, gpu=True, shutup=False,
                                                  times=np.arange(0, 20, 0.02), outfile_location=output_file_name,
                                                  musr_type=DipolarPolarisation.musr_type.ZF)
    return 0


if __name__=='__main__':
    main()
