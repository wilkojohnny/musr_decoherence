# NaF_nnn_polarisation.py -- calculate muon polarisation function for F-mu-F states

import functools
print = functools.partial(print, flush=True)

# add DecoCalc to PATH
import os
pwd = os.path.dirname(__file__)
deco_path = os.path.join(pwd, '../../')  # this should be the path of Decoherence_calculator
import sys
sys.path.append(deco_path)

import DipolarPolarisation
from MDecoherenceAtom import TDecoherenceAtom as atom
from TCoord3D import TCoord3D as coord
import numpy as np
import NaF_EFG


def main():
    r_nn = 1.19778
    r_nnn = 2.2926
    r_nnnn = 2.4153789

    muon = atom(name='mu', position=coord(0, 0, 0))

    # calculate the EFG for the quadrupoles
    efg = NaF_EFG.calc_efg(r_nnn, r_nn) / (2.31 ** 3)

    All_spins = [muon,
                 # nn
                 atom(name='F', position=coord(1, -1, 0).rhat() * r_nn),
                 atom(name='F', position=coord(-1, 1, 0).rhat() * r_nn),
                 # nnn
                 atom(name='Na', position=coord(1, 1, 0).rhat() * r_nnn, efg=efg),
                 atom(name='Na', position=coord(-1, -1, 0).rhat() * r_nnn, efg=efg),
                 # nnnn
                 atom(name='F', position=coord(1, 1, 2).rhat() * r_nnnn),
                 atom(name='F', position=coord(1, 1, -2).rhat() * r_nnnn),
                 atom(name='F', position=coord(-1, -1, 2).rhat() * r_nnnn),
                 atom(name='F', position=coord(-1, -1, -2).rhat() * r_nnnn),
                 ]

    # file name
    output_file_name = 'NaF_polarisation.dat'

    DipolarPolarisation.calc_dipolar_polarisation(all_spins=All_spins, muon=muon, times=np.arange(0, 20, 0.1), plot=True,
                                                  outfile_location=output_file_name, do_quadrupoles=True, gpu=True)

    return 0


if __name__ == '__main__':
    main()
