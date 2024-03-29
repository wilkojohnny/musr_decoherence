# Plot muon polarisation of CaF2
# works with Decoherence_calculator version>ff1a392386b66178c63dbe80dbfe6e7c5b935286

import functools
print = functools.partial(print, flush=True)

import musr_decoherence.AtomObtainerN as AO
from musr_decoherence import DipolarPolarisation

from ase.io import read
from ase import build
import numpy as np


def main():

    cif_file = '../CaF2.cif'

    # get the atoms from the CIF file
    CaF2_atoms = read(cif_file)

    # make a supecell to make the visual muon location selection easier
    CaF2_atoms = build.make_supercell(CaF2_atoms, np.diag([2,1,1]))

    # get the muon position visually -- to do this, CTRL+click the two F atoms the
    # muon sits between.
    # muon_position, nnindices = AO.get_muon_pos_nn_visually(CaF2_atoms)
    # ^^ but instead of doing the above every time, you can also just type in the muon location manually
    nnindices = [4, 9]
    muon_position = np.array([2.722, 1.361, 4.084])

    # squish_radii -- this is how much the [nn, nnn, nnnn...] atoms should be
    # drawn in/out by (i.e the new bond lengths, in Angstroms)
    squish_radii = [1.172, 1.172]
    # lambda_squish -- the factor by all the nearest-neighbours above what is
    # defined in squish_radii are perturbed by (in most cases <1)
    lambda_squish = 0.920

    # get the muon, All_spins objects to feed the polarisation function
    muon, All_spins = AO.get_linear_fmuf_atoms(ase_atoms=CaF2_atoms, muon_position=muon_position,
                                               nnnness=6, squish_radii=squish_radii, lambda_squish=lambda_squish)

    output_file_name = '../../../CaF2_polarisation_T20.dat'

    # define muon polarisation relative to the sample -- None for polycrystalline
    muon_sample_polarisation = None

    DipolarPolarisation.calc_dipolar_polarisation(all_spins=All_spins, muon=muon, muon_sample_polarisation=muon_sample_polarisation,
                                                  plot=True, fourier=False, fourier_2d=False, tol=1e-3, gpu=True, shutup=False,
                                                  times=np.arange(0, 20, 0.02), outfile_location=output_file_name,
                                                  musr_type=DipolarPolarisation.musr_type.ZF)
    return 0


if __name__=='__main__':
    main()
