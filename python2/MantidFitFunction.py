# MantidFitFunction.py - a fit function for DecoherenceCalculator for use in MANTID.
#
# Created 19/6/19, John Wilkinson

from mantid.api import *
import numpy as np
import sys
import TCoord3D as coord
from MDecoherenceAtom import TDecoherenceAtom as atom
import DecoherenceCalculator

# You choose which type you would like by picking the super class
class DecoherenceFunction(IFunction1D): # or IPeakFunction

    def category(self):
        return 'Muon'

    def init(self):
        # register initial amplitude A, and a stretching factor for the time
        self.declareParameter("A", 1.0)
        self.declareParameter("F_mu distance", 1.18)     

    def function1D(self, xvals):
        # set up the calculation
        # lattice parameters and angles, in angstroms
        lattice_parameter = [5.44542, 0, 0]  # [a, b, c]
        lattice_angles = [90, 0, 0]  # [alpha, beta, gamma] in **degrees**

        # are atomic coordinates provided in terms of alat or in terms of the primitive lattice vectors?
        input_coord_units = DecoherenceCalculator.position_units.ALAT

        lattice_type = DecoherenceCalculator.ibrav.CUBIC_FCC

        # atoms and unit cell: dump only the basis vectors in here, the rest is calculated
        atomic_basis = [#atom(coord.TCoord3D(0, 0, 0), gyromag_ratio=np.array([18.0038, 0]), II=np.array([7, 0]), name='Ca',
                        #     abundance=np.array([0.00145, 0.99855])),
                        atom(coord.TCoord3D(0.25, 0.25, 0.25), gyromag_ratio=251.713, II=1, name=u'F'),
                        atom(coord.TCoord3D(0.25, 0.25, 0.75), gyromag_ratio=251.713, II=1, name=u'F')
                        ]
                        
        # define muon position
        muon_position = coord.TCoord3D(0.25, 0.25, 0.50)
        
        # do the calculation:
        # linearly interpolate between points to get the function value
        A = self.getParameterValue("A")
        squish_radius = self.getParameterValue("F_mu distance")
        vals = DecoherenceCalculator.calc_decoherence(muon_position=muon_position, squish_radius=squish_radius, lattice_type=lattice_type,
                 lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
                 input_coord_units=input_coord_units, atomic_basis=atomic_basis,
                 perturbed_distances=[], nnnness=3, times=xvals)
        return A*vals

   
# Register with Mantid
FunctionFactory.subscribe(DecoherenceFunction)