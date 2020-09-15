# Fit CaF2 data with DecoFitter.py
# works with Decoherence_calculator version>ff1a392386b66178c63dbe80dbfe6e7c5b935286

# add DecoCalc to PATH
import os
pwd = os.path.dirname(__file__)
deco_path = os.path.join(pwd, '../../') # this should be the path of Decoherence_calculator
import sys
sys.path.append(deco_path)

from DecoFitter import *
import AtomObtainerN as AO
from ase.io import read
from ase import build
import numpy as np
import DipolarPolarisation
from lmfit import Parameters

cif_file = '../CaF2.cif'

# get the atoms from the CIF file
CaF2_atoms = read(cif_file)

# make a supecell to make the visual muon location selection easier
CaF2_atoms = build.make_supercell(CaF2_atoms, np.diag([2,1,1]))

# get the muon position visually -- to do this, CTRL+click the two F atoms the
# muon sits between. (right click+drag to orbit the atoms;
# close the window to start the calculation)
# muon_position, nnindices = AO.get_muon_pos_nn_visually(CaF2_atoms)

muon_position = [2.72, 1.36, 4.08]
nnindices = [4,9]


def main():
    # set up the parameters
    params = Parameters()
    params.add('A', value=10.14)
    params.add('r_nn', value=1.172, min=1, max=2, vary=False)
    params.add('field', value=20.68, vary=True)
    params.add('lambda_squish', value=0.9, min=0.5, max=1, vary=False)
    params.add('A0', value=2.52, min=0)

    # data
    muon_data = {#'asymmetry': './CaF2_data.dat',
                 'N_F': './CaF2_T20_F.dat',
                 'N_B': './CaF2_T20_B.dat',
                 'alpha': 1.52,
                 }


    # try to do a fit
    fit(muon_data=muon_data, end_time=15, fit_function=fit_function, params=params, plot=True, just_plot=True, outfile_location='fit_out.dat')


# fit function
def fit_function(params, x):

    A = params['A']
    r_nn = params['r_nn']
    lambda_squish = params['lambda_squish']
    A0 = params['A0']
    field = params['field']

    squish_radii = [r_nn]

    muon, All_spins = AO.get_linear_fmuf_atoms(ase_atoms=CaF2_atoms, muon_position=muon_position,
                                               nnnness=8, squish_radii=squish_radii, lambda_squish=lambda_squish)

    deco = DipolarPolarisation.calc_dipolar_polarisation(all_spins=All_spins, muon=muon, muon_sample_polarisation=None, times=x,
                                                         plot=False, shutup=True, gpu=True, musr_type=DipolarPolarisation.musr_type.TF,
                                                         field=field)

    return A*deco + A0*np.ones(len(x))


if __name__=='__main__':
    main()
