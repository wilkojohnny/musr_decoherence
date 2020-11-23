# NaF_nnn_polarisation.py -- calculate muon polarisation function for F-mu-F states

import functools
print = functools.partial(print, flush=True)

from ase.io import read
from musr_decoherence import DipolarPolarisation
from musr_decoherence import muon_environment as mu_env
import numpy as np

def main():
    r_nn = 1.19778
    r_nnn = 2.2926
    r_nnnn = 2.4153789

    cif_file = '../NaF.cif'

    # get the atoms from the CIF file
    naf_atoms = read(cif_file)

    # define the muon position as being in between atom 4 and 9 (which are both Fs in the structure).
    nnindices = [5, 6]

    # define how much the muon perturbs the environment. List is [[atom_id, dist], [atom_id, dist], ...]
    perturbations = [[5, 1.18], [6, 1.18], [0, 2.2926], [3, 2.2926]]
    # lambda_squish -- the factor by all the nearest-neighbours above what is
    # defined in squish_radii are perturbed by (in most cases <1)
    lambda_squish = 0.920

    # add the muon to naf_atoms
    naf_atoms_mu = mu_env.add_muon_to_aseatoms(naf_atoms, nn_indices=nnindices)

    # now perturb the atoms
    naf_atoms_mu = mu_env.perturb_atoms(naf_atoms_mu, perturbations=perturbations)

    # get the dominant nuclei
    nn_nuclei, naf_atoms_mu, nn_indices = mu_env.get_dominant_nuclei(naf_atoms_mu, unperturbed_atoms=naf_atoms)

    # calculate quadrupoles
    efgs, quad_nuc_ids = mu_env.calculate_quadrupoles(naf_atoms_mu, nn_indices, naf_atoms, 50)

    # model the further nuclei with lambda_squish
    nn_nuclei = mu_env.model_further_nuclei(nn_nuclei, draw_in_factor=lambda_squish)

    muon, all_spins = mu_env.aseatoms_to_tdecoatoms(nn_nuclei, efgs=efgs, efg_ids=quad_nuc_ids)
    # file name
    output_file_name = 'NaF_polarisation.dat'

    DipolarPolarisation.calc_dipolar_polarisation(all_spins=all_spins, muon=muon, times=np.arange(0, 20, 0.1), plot=True,
                                                  outfile_location=output_file_name, do_quadrupoles=True, gpu=True)

    return 0


if __name__ == '__main__':
    main()
