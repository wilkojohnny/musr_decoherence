# DecoFitter.py - fits experimental musr data to that calculated by DecoherenceCalculator

# load modules
import numpy as np  # for numpy arrays
from scipy import optimize  # for nls curve fitting
import matplotlib.pyplot as pyplot  # for plotting
import matplotlib.colors as color


def fit(data_file_location: str, fit_function, initial_params: list, plot: bool, end_time=None):
    """
    :param data_file_location: location of the muon data file
    :param fit_function: function to be fitted
    :param initial_params: initial parameters in array
    :param plot: True == do a plot of the result
    :return: params, chi squared per dof
    """
    # load in the data (expect it of the form x y yerr)
    x, y, y_error = load_muon_data(data_file_location, end_time=end_time)

    params, covariances = optimize.curve_fit(fit_function, xdata=x, ydata=y, p0=initial_params, sigma=y_error)
    params_errors = np.sqrt(np.diag(covariances))

    for i in range(0, len(params)):
        print(str(params[i]) + "\t" + str(params_errors[i]) + "\t", end='')
    print("\n")

    # calculate the fit function one last time
    fit_func = fit_function(x, params[0], params[1], params[2], params[3], params[4])
    # find the chi squared
    chi_squared = (((fit_func - y)/y_error)**2).sum()
    chi_squared_dof = chi_squared/(len(y) - len(params))

    # print out the chi squared
    print('Chi-squared: ' + str(chi_squared) + ' (' + str(chi_squared_dof) + ' per dof)')

    # plot the data
    if plot:
        pyplot.errorbar(x, y, y_error, ecolor=color.cnames['red'], marker='.', linestyle='none')
        pyplot.plot(x, fit_func, color=color.cnames['black'])
        pyplot.ylim((0, 30))
        pyplot.show()


def load_muon_data(filename: str, end_time=None, encoding='iso-8859-1'):
    # loads muon data, spits it out as three numpy arrays: x, y, yerr

    # open the file
    data_file = open(filename, 'r', encoding=encoding)

    # set up the arrays
    x = []
    y = []
    y_error = []

    for line in data_file.readlines():
        # if line doesn't start with !...
        if not line.startswith('!'):
            # split the line based on spaces
            split_line = line.split()
            # put these into the arrays
            this_x = float(split_line[0])
            this_y = float(split_line[1])
            this_yerror = float(split_line[2])
            if end_time is not None:
                if this_x > end_time:
                    break
            x.append(this_x)
            y.append(this_y)
            y_error.append(this_yerror)

    return np.array(x), np.array(y), np.array(y_error)
