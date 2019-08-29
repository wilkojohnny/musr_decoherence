# MantidFitFunction.py - a fit function for DecoherenceCalculator for use in MANTID.
#
# Created 19/6/19, John Wilkinson

from mantid.api import *
import numpy as np
import sys
import TCoord3D as coord
import DecoherenceCalculator
from MDecoherenceAtom import TDecoherenceAtom as atom

# You choose which type you would like by picking the super class
class DecoherenceFunction(IFunction1D): # or IPeakFunction

    def category(self):
        return 'Muon'

    def init(self):
        # register initial amplitude A, and a stretching factor for the time
        self.declareParameter("A", 1.0)
        self.declareParameter("F_mu distance", 1.18)

        # lattice parameters and angles, in angstroms
        self.lattice_parameter = [5.44542, 0, 0]  # [a, b, c]
        self.lattice_angles = [90, 0, 0]  # [alpha, beta, gamma] in **degrees**

        # are atomic coordinates provided in terms of alat or in terms of the primitive lattice vectors?
        self.input_coord_units = DecoherenceCalculator.position_units.ALAT

        # atoms and unit cell: dump only the basis vectors in here, the rest is calculated
        self.atomic_basis = [#atom(coord.TCoord3D(0, 0, 0), gyromag_ratio=np.array([18.0038, 0]), II=np.array([7, 0]), name='Ca',
                        #     abundance=np.array([0.00145, 0.99855])),
                        atom(coord.TCoord3D(0.25, 0.25, 0.25), gyromag_ratio=251.713, II=1, name=u'F'),
                        atom(coord.TCoord3D(0.25, 0.25, 0.75), gyromag_ratio=251.713, II=1, name=u'F')
                        ]
                        
        # define muon position
        self.muon_position = coord.TCoord3D(.25, 0.25, 0.5)
        

    def function1D(self, xvals):
        # linearly interpolate between points to get the function value
        A = self.getParameterValue("A")
        squish_radius = self.getParameterValue("F_mu distance")
        vals = calc_decoherence(muon_position=self.muon_position, squish_radius=squish_radius, lattice_type=lattice_type,
                 lattice_parameter=lattice_parameter, lattice_angles=lattice_angles,
                 input_coord_units=input_coord_units, atomic_basis=atomic_basis,
                 perturbed_distances=perturbed_distances, plot=True, nnnness=2, times=xvals)
        return A*vals

    def importGLEdata(self, file_location):
        # open the file
        GLE_data_file = open(file_location, "r")

        time_data = []
        amplitude_data = []

        # for each line in the file...
        for file_line in GLE_data_file:
            # check the line doesnt start with ! (comment in GLE)
            if file_line[0] != "!":
                # read in the data (if we cant convert, pass)
                try:
                    output_data = [float(entry) for entry in file_line.split()]
                    time_data.append(output_data[0])
                    amplitude_data.append(output_data[1])
                except ValueError:
                    pass

        # close the input file
        GLE_data_file.close()
        
        # convert into numpy arrays
        time_data = np.array(time_data)
        amplitude_data = np.array(amplitude_data)
        return time_data, amplitude_data

# Register with Mantid
FunctionFactory.subscribe(DecoherenceFunction)