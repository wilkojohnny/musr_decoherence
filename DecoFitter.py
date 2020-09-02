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


def fit(muon_data: dict, fit_function, params: Parameters, plot: bool, start_time=None,
        end_time=None, just_plot=False, outfile_location=None, algorithm='leastsq', epsfcn=1e-4):
    """
    :param muon_data: dict with keys N_F and N_B, the location of the files containing the forward and backward counts,
                      and alpha, OR key 'asymmetry' with the location of the file containing the full asymmetry data
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
    x, y, y_error = load_muon_data(muon_data, start_time=start_time, end_time=end_time)

    fitted_params = params

    if not just_plot:
        fit_result = minimize(residual, params, args=(fit_function, x, y, y_error), iter_cb=print_iteration,
                              method=algorithm, epsfcn=epsfcn)

        print(fit_result.message)
        print(fit_report(fit_result))

        fitted_params = fit_result.params

    # calculate the fit function one last time
    fit_func = fit_function(fitted_params, x)

    # save the fit function to file
    if outfile_location is not None:
        save_fit(x, fit_func, filename=outfile_location, params=fitted_params)

    # plot the data
    if plot:
        pyplot.errorbar(x, y, y_error, ecolor=color.cnames['red'], marker='.', linestyle='none')
        pyplot.plot(x, fit_func, color=color.cnames['black'])
        pyplot.ylim((-10, 30))
        pyplot.show()

    return fitted_params


def save_fit(x, fit_function, filename, params):
    with open(filename, 'w') as file:
        file.write('! Fitting output for MuSR data \n')
        file.write('!\n! Fitting parameters: \n')
        for name, parameter in params.items():
            file.write('! ' + name + ': ' + str(parameter.value) + ' +/- ' + str(parameter.stderr) + '\n')
        file.write('! t\tfitting function output\n')
        for i_x in range(0, len(x)):
            file.write(str(x[i_x]) + '\t' + str(fit_function[i_x]) + '\n')


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


def load_muon_data(muon_data: dict, start_time=None, end_time=None, encoding='iso-8859-1'):
    # loads muon data, spits it out as three numpy arrays: x, y, yerr

    if start_time is None:
        start_time = 0

    # set up the arrays
    x = []
    y = []
    y_error = []

    # see what data we've been given
    if 'asymmetry' in muon_data:
        asym_file_location = muon_data['asymmetry']
        # open the file
        data_file = open(asym_file_location, 'r', encoding=encoding)
        for line in data_file.readlines():
            # if line doesn't start with !...
            if not line.startswith('!'):
                # split the line based on spaces
                split_line = line.split()
                # put these into the arrays
                this_x = float(split_line[0])
                if this_x < start_time:
                    continue
                this_y = float(split_line[1])
                this_yerror = float(split_line[2])
                if end_time is not None:
                    if this_x > end_time:
                        break
                x.append(this_x)
                y.append(this_y)
                y_error.append(this_yerror)
        data_file.close()
        return np.array(x), np.array(y), np.array(y_error)
    else:
        # need to deal with N_F and N_B individually
        n_f_file_location = muon_data['N_F']
        n_b_file_location = muon_data['N_B']
        alpha = muon_data['alpha']

        # open both files:
        n_f_file = open(n_f_file_location, 'r', encoding=encoding)
        n_b_file = open(n_b_file_location, 'r', encoding=encoding)

        # zip them both together, and iterate the lines
        for data in list(zip(n_f_file.readlines(), n_b_file.readlines())):
            if not data[0].startswith('!'):
                # split both the n_f and n_b data
                n_f = data[0].split()
                n_b = data[1].split()
                # check the times are the same,
                assert abs(float(n_f[0]) - float(n_b[0])) < 1e-3
                # save the time
                this_x = float(n_f[0])
                if this_x < start_time:
                    continue
                if end_time is not None:
                    if this_x > end_time:
                        break
                x.append(this_x)
                # save the asymmetry
                n_f_counts = float(n_f[1])
                n_b_counts = float(n_b[1])
                try:
                    asymmetry = 100*(n_b_counts - alpha*n_f_counts)/(n_b_counts + alpha*n_f_counts)
                except ZeroDivisionError:
                    x.pop(-1)
                    continue
                y.append(asymmetry)
                err_n_f = float(n_f[2])
                err_n_b = float(n_b[2])
                asymmetry_error = 2*100*alpha/(n_b_counts + alpha*n_f_counts) ** 2 *\
                                  np.sqrt((n_f_counts*err_n_b) ** 2 + (n_b_counts*err_n_f) ** 2)
                y_error.append(asymmetry_error)
        return np.array(x), np.array(y), np.array(y_error)



