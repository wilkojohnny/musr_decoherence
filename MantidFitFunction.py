# MantidFitFunction.py - a fit function for DecoherenceCalculator for use in MANTID.
#
# Created 19/6/19, John Wilkinson

from mantid.api import *
import numpy as np


# You choose which type you would like by picking the super class
class DecoherenceFunction(IFunction1D): # or IPeakFunction

    def category(self):
        return 'Muon'

    def init(self):
        # register initial amplitude A, and a stretching factor for the time
        self.declareParameter("A", 1.0)
        self.declareParameter("t_stretch", 1.0)

        # open the file, and load the data into numpy arrays
        self.time_data, self.amplitude_data = self.importGLEdata("/Users/johnny/Documents/University/CaF2/CaF2_simulated_data.dat")

        # normalise by making amplitude 1 (for now - multiply by A in the actual function)
        self.amplitude_data = self.amplitude_data/self.amplitude_data[0]


    def function1D(self, xvals):
        # linearly interpolate between points to get the function value
        A = self.getParameterValue("A")
        t_stretch = self.getParameterValue("t_stretch")
        return A*np.interp(xvals, self.time_data*t_stretch, self.amplitude_data, 0, 0)

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