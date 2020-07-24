# DecoFitter.py - fits experimental musr data to that calculated by DecoherenceCalculator

# load modules

# flush printing cache (useful for ARC)
import functools
print = functools.partial(print, flush=True)

import numpy as np  # for numpy arrays
try:
	import matplotlib.pyplot as pyplot  # for plotting
	import matplotlib.colors as color  # for colourful plots
except ModuleNotFoundError:
	no_plot = True

from lmfit import *  # for nls curve fitting


def fit(data_file_location: str, fit_function, params: Parameters, plot: bool, end_time=None, just_plot=False):
    """
    :param data_file_location: location of the muon data file
    :param fit_function: function to be fitted
    :param params: fit parameters
    :param plot: True == do a plot of the result
    :param end_time: cutoff time for fitting
    :param just_plot: if TRUE, just plots the parameters instead of doing an actual fit
    :return: lmfit.parameter object of the fit result
    """

    if just_plot and not plot:
        return params

    # load in the data (expect it of the form x y yerr)
    x, y, y_error = load_muon_data(data_file_location, end_time=end_time)

    fitted_params = params

    if not just_plot:
        fit_result = minimize(residual, params, args=(fit_function, x, y, y_error), iter_cb=print_iteration)

        print(fit_result.message)
        print(fit_report(fit_result))

        fitted_params = fit_result.params

    # calculate the fit function one last time
    fit_func = fit_function(fitted_params, x)

    # plot the data
    if plot:
        pyplot.errorbar(x, y, y_error, ecolor=color.cnames['red'], marker='.', linestyle='none')
        pyplot.plot(x, fit_func, color=color.cnames['black'])
        pyplot.ylim((-10, 30))
        pyplot.show()

    return fitted_params


def gle_friendly_out(fit_parameters):
    # do labels
    print('!\t', end='')
    for name, _ in fit_parameters.items():
        print(name + '\terr(' + name + ')\t', end='')
    print('')
    # print output
    for _, parameter in fit_parameters.items():
        print('\t' + str(parameter.value) + '\t' + str(parameter.stderr), end='')


def print_iteration(params, iter, residuals, *args, **kwargs):
    # this function is run at every iteration of the fit
    print('Iteration ' + str(iter))
    print(params.pretty_print())
    return False


def residual(params, fit_function, x, y, yerr):
    """
    Calculates the residuals of the fit function
    :param params: parameter object of the fitting parameters
    :param fit_function: function to be fitted
    :param x: xdata
    :param y: ydata
    :param yerr: error(ydata)
    :return: residual of fit_function's description of y
    """
    y_func = fit_function(params, x)
    return (y - y_func) / yerr


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
