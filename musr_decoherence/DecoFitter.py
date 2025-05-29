# DecoFitter.py - fits experimental musr data to that calculated by DecoherenceCalculator

# load modules

# flush printing cache (useful for ARC)
import functools
import sys
from pathlib import Path

print = functools.partial(print, flush=True)
import numpy as np  # for numpy arrays
try:
    import matplotlib.pyplot as pyplot  # for plotting
    import matplotlib.colors as color  # for colourful plots
except ModuleNotFoundError:
    no_plot = True
from lmfit import *  # for nls curve fitting
import re  # for regular expressions


def fit(muon_data: dict, fit_function, params: Parameters, plot: bool, start_time=None,
        end_time=None, just_plot=False, outfile_location=None, algorithm='leastsq', epsfcn=1e-4,
        plot_xlim=(0,15), plot_ylim=(None, None)):
    """
    :param muon_data: dict with keys N_F and N_B, the location of the files containing the forward and backward counts,
                      and alpha, OR key 'asymmetry' with the location of the file containing the full asymmetry data.
                      Alteratively, keys 'x', 'y', and 'y_error' with numpy arrays for each. If using x y yerrror,
                      you can also use 'i' for the dataset instance.
    :param fit_function: function to be fitted
    :param params: fit parameters
    :param plot: True == do a plot of the result
    :param end_time: cutoff time for fitting
    :param just_plot: if TRUE, just plots the parameters instead of doing an actual fit
    :return: lmfit.parameter object of the fit result
    """

    if just_plot and not plot:
        return params

    global_fitting = False

    if 'asymmetry' or 'N_F' in muon_data.keys():
        # load in the data (expect it of the form x y yerr)
        x, y, y_error = load_muon_data(muon_data, start_time=start_time, end_time=end_time)
    else:
        # the data isn't in files -- so take it of the form x y y_error
        x, y, y_error = muon_data['x'], muon_data['y'], muon_data['y_error']
        # i identifies the dataset
        if 'i' in muon_data.keys():
            i = muon_data['i']
            global_fitting = True

    # check the array lengths are the same
    assert len(x) == len(y_error) == len(y)

    fitted_params = params

    if not just_plot:
        if global_fitting:
            fit_args = (fit_function, x, y, y_error, i)
        else:
            fit_args = (fit_function, x, y, y_error)

        fit_result = minimize(residual, params, args=fit_args, iter_cb=print_iteration,
                              method=algorithm, epsfcn=epsfcn)

        print(fit_result.message)
        print(fit_report(fit_result))

        fitted_params = fit_result.params

    # calculate the fit function one last time
    if global_fitting:
        fit_func = fit_function(fitted_params, x, i)
    else:
        fit_func = fit_function(fitted_params, x)

    # if global fitting, split fit_func up by i
    unique_i = np.unique(i)
    x = {val: x[i==val] for i in unique_i}
    fit_func = {val: fit_func[i==val] for i in unique_i}

    # save the fit function to file
    if outfile_location is not None:
        if global_fitting:
            filepath = Path(outfile_location)
            for this_i in unique_i:
                outfile_location_i = filepath.with_name(f"{filepath_stem}_{this_i}{filepath.suffix}")
                save_fit(x, fit_func[this_i], filename=outfile_location_i, params=fitted_params)
        else:
            save_fit(x, fit_func, filename=outfile_location, params=fitted_params)

    # plot the data
    if plot:
        if global_fitting:
            y = {val: y[i==val] for i in unique_i}
            y_error = {val: y_error[i==val] for i in unique_i}
            for this_i in unique_i:
                pyplot.errorbar(x[this_i], y[this_i], y_error[this_i], ecolor=color.cnames['red'], marker='.', linestyle='none')
                pyplot.plot(x[this_i], fit_func[this_i], color=color.cnames['black'])
                pyplot.title(str(this_i))
                pyplot.xlim(plot_xlim)
                pyplot.ylim(plot_ylim)
                pyplot.show()
        else:
            pyplot.errorbar(x, y, y_error, ecolor=color.cnames['red'], marker='.', linestyle='none')
            pyplot.plot(x, fit_func, color=color.cnames['black'])
            pyplot.title(str(this_i))
            pyplot.xlim(plot_xlim)
            pyplot.ylim(plot_ylim)
            pyplot.show()


    return fitted_params


def fit_dt(muon_data: dict, asymmetry_function, params: Parameters, plot: bool, outfile_location=None,
           algorithm='leastsq', epsfcn=1e-4, plot_xlim=(0, 30), plot_ylim=(None, None)):
    """
    Fit the deadtime correction to F B data, assuming that asymmetry_function is constant
    :param muon_data: dict with MuSR data, with keys N_F, N_B, alpha
    :param asymmetry_function: Expected MuSR asymmetry function. None of these parameters will be varied
    :param params: Parameters for the fit, with keys DT0_{F/B}, DTC2_{F/B} which mean the same as they do in WiMDA
    :param plot: if True, do a plot of the result
    :param outfile_location: file to write the corrected asymmetry data to
    :param algorithm: algorithm for the fit
    :param epsfcn: fitting algorithm step size
    :return: lmfit.parameter object of fit result
    """

    # extract the data into arrays x_f, x_b, N_f, N_b, N_f_error, N_b_error, and dict run_info
    x_f, N_f, N_f_error, f_run_info = load_count_data(muon_data['N_F'])
    x_b, N_b, N_b_error, b_run_info = load_count_data(muon_data['N_B'])

    # check both f and b belong to the same data set (crudely)
    assert f_run_info == b_run_info
    assert x_f.all() == x_b.all()

    # evaluate the asymmetry function -- (doing it now assumes the parameters are fixed for speed)
    asym = asymmetry_function(params, x_f)

    # zip together the arrays
    xx = np.concatenate([x_f, x_b])
    yy = np.concatenate([N_f, N_b])
    yy_errerr = np.concatenate([N_f_error, N_b_error])

    # get alpha
    try:
        alpha = muon_data['alpha']
    except KeyError:
        alpha = 1

    # fit the parameters
    fit_result = minimize(fb_residual, params, args=(asym, xx, yy, yy_errerr, f_run_info, alpha),
                          iter_cb=print_iteration, method=algorithm, epsfcn=epsfcn)

    # do a decay + dead time correction with the final parameters
    fb_higher_orders = get_dt_corrections(fit_result.params)
    N_dc_f = apply_dead_time_dc(x_f, N_f, f_run_info, fit_result.params['DT0_f'], fb_higher_orders[0]) / np.sqrt(alpha)
    N_dc_b = apply_dead_time_dc(x_b, N_b, b_run_info, fit_result.params['DT0_b'], fb_higher_orders[1]) * np.sqrt(alpha)

    print(fit_result.message)
    print(fit_report(fit_result))

    # calculate the new errors on N_f and N_b -- they've changed, as the DT correction has an error associated with it
    N_dc_f_error = calc_dt_dc_error(x_f, N_f, N_f_error, fit_result.params, f_run_info, 'f') / np.sqrt(alpha)
    N_dc_b_error = calc_dt_dc_error(x_b, N_b, N_b_error, fit_result.params, b_run_info, 'b') * np.sqrt(alpha)

    # plot (if wanted)
    if plot:
        pyplot.errorbar(x_f, N_dc_f, N_dc_f_error, mec='darkred', mfc='darkred', ecolor=color.cnames['red'], marker='.', linestyle='none', label='F')
        pyplot.errorbar(x_b, N_dc_b, N_dc_b_error, mec='darkblue', mfc='darkblue', ecolor=color.cnames['blue'], marker='.', linestyle='none', label='B')
        pyplot.legend()
        pyplot.plot(x_f, fit_result.params['N0']*(1 + 0.01*asym), color='m')
        pyplot.plot(x_b, fit_result.params['N0']*(1 - 0.01*asym), color='c')
        pyplot.xlim(plot_xlim)
        pyplot.ylim(plot_ylim)
        pyplot.show()

    # if outfiile_location specified, then calculate the asymmetry and save this
    if outfile_location is not None:
        # get the data ready
        A = 100 * (N_dc_f - N_dc_b) / (N_dc_f + N_dc_b)
        A_err = 200 / (N_dc_b + N_dc_f) ** 2 * np.sqrt((N_dc_f * N_dc_b_error) ** 2 + (N_dc_b * N_dc_f_error) ** 2)

        with open(outfile_location, 'w+') as f:
            f.write('! Asymmetry data calculated by DecoFitter.py\n')
            f.write('! Calculated from files ' + muon_data['N_F'] + ' and ' + muon_data['N_B'] + '\n')
            f.write('!\n! Fitted deadtime parameters: \n')
            for name, parameter in fit_result.params.items():
                f.write('! ' + name + ': ' + str(parameter.value) + ' +/- ' + str(parameter.stderr) + '\n')
            f.write('!\n! t A(t) err\n')
            for i_t, t in enumerate(x_f):
                f.write('{:.3f}\t{:.5f}\t{:.5f}\n'.format(t, A[i_t], A_err[i_t]))

    return fit_result.params



def save_fit(x, fit_function, filename, params):
    with open(filename, 'w') as file:
        file.write('! Fitting output for MuSR data \n')
        file.write('!\n! Fitting parameters: \n')
        for name, parameter in params.items():
            file.write('! ' + name + ': ' + str(parameter.value) + ' +/- ' + str(parameter.stderr) + '\n')
        file.write('! t\tfitting function output\n')
        for i_x in range(0, len(x)):
            file.write(str(x[i_x]) + '\t' + str(fit_function[i_x]) + '\n')


def gle_friendly_out(fit_parameters, preamble='', print_headings=True, fileout=sys.stdout):
    if print_headings:
        # do labels
        print('!\t', end='', file=fileout)
        for name, _ in fit_parameters.items():
            print(' {:12.10} {:12.10}'.format(name, 'err'), end='', file=fileout)
    # print output
    print('\n', end='', file=fileout)
    print(preamble + ' \t', end='', file=fileout)
    for _, parameter in fit_parameters.items():
        print(' {:<12.5g} {:<12.5g}'.format(parameter.value, parameter.stderr), end='',
              file=fileout)


def print_iteration(params, iter, residuals, *args, **kwargs):
    # this function is run at every iteration of the fit
    print('Iteration ' + str(iter))
    print(params.pretty_print())
    return False


def residual(params, fit_function, x, y, yerr, i=None):
    """
    Calculates the residuals of the fit function
    :param params: parameter object of the fitting parameters
    :param fit_function: function to be fitted
    :param x: xdata
    :param y: ydata
    :param yerr: error(ydata)
    :param i: identifier of the dataset, None if not global fitting
    :return: residual of fit_function's description of y
    """
    if i is None:
        y_func = fit_function(params, x)
    else:
        y_func = fit_function(params, x, i)
    return (y - y_func) / yerr


def fb_residual(params, asym_func, xx, yy, yy_errerr, run_info, alpha=1):
    """
    Calculate the residuals for F B muSR data, for DT fitting
    :param params: parameters for the fit
    :param asym_func: asymmetry function
    :param xx: times the data corresponds to (first half x_F, second x_B)
    :param yy: array with the first half containing N_F, second N_B
    :param yy_errerr: errors of yy
    :param run_info: info of the runs, given by the asym function
    :param alpha: detector efficiency parameter
    :return: residuals (asym_func-y)/y_err
    """

    fb_array_crossover = round(len(xx) / 2)

    # apply the dead time and decay correction
    dt_higher_orders = get_dt_corrections(params)
    n_f_dt = apply_dead_time_dc(xx[:fb_array_crossover], yy[:fb_array_crossover], run_info, params['DT0_f'],
                                dt_higher_orders[0]) / np.sqrt(alpha)
    n_f_err = yy_errerr[:fb_array_crossover]
    n_b_dt = apply_dead_time_dc(xx[fb_array_crossover:], yy[fb_array_crossover:], run_info, params['DT0_b'],
                                dt_higher_orders[1]) * np.sqrt(alpha)
    n_b_err = yy_errerr[fb_array_crossover:]

    # calcualte the asymmetry function for F and B
    n_f_func = params['N0'] * (1 + 0.01*asym_func)
    n_b_func = params['N0'] * (1 - 0.01*asym_func)

    residuals = []
    for i_n_f_func, this_n_f_func in enumerate(n_f_func):
        residuals.append((this_n_f_func - n_f_dt[i_n_f_func])/n_f_err[i_n_f_func])
    for i_n_b_func, this_n_b_func in enumerate(n_b_func):
        residuals.append((this_n_b_func - n_b_dt[i_n_b_func])/n_b_err[i_n_b_func])
    return np.array(residuals)*np.exp(-xx/2.19698)


def get_dt_corrections(params: Parameters, do_error=False):
    """
    turns the params into a list of deadtime corrections for 2nd order and beyond
    :param params: Parameters object for the fitting parameters, with the >=2nd order corrections being called
                   DTC{order}_{f/b}. Do not skip any orders, or they will be ignored!
    :param do_error: set to True to also output the errors in a separate array
    :return: 2d array, [0][:] containing the forward DT, [1][:] with backward, if do_error also outputs the errors
    """

    # do f
    f_corrections = []
    f_error = []
    while True:
        i = len(f_corrections)+2
        try:
            this_correction = params['DTC' + str(i) + '_f']
            f_corrections.append(this_correction)
            if do_error:
                f_error.append(this_correction.stderr)
        except KeyError:
            break

    # do b
    b_corrections = []
    b_error = []
    while True:
        i = len(b_corrections)+2
        try:
            this_correction = params['DTC' + str(i) + '_b']
            b_corrections.append(this_correction)
            if do_error:
                b_error.append(this_correction.stderr)
        except KeyError:
            break

    if do_error:
        return [f_corrections, b_corrections], [f_error, b_error]
    else:
        return [f_corrections, b_corrections]


def apply_dead_time_dc(t, N, run_info, DT0, DTn=None):
    """
    applies dead time and decay correction to the count data N, in the same way as WIMDA. Does not deal with errors.
    :param t: time data
    :param N: count data
    :param run_info: dict with the MuSR run information, which load_caunt_data() gives out
    :param DT0: DT0 dead time correction in ns, defined in the same way as WiMDA
    :param DTn: array of higher-order corrections, defined in the same way as WiMDA
    :return: the count data corrected for the dead time
    """

    # this parameter, q, characterises the dead time correction (see lab book 3 page 139)
    q = 1/(run_info['n_frames']*run_info['n_detectors']*run_info['tres']*1e3)

    dt_denominator = 1 - DT0 * q * N

    if DTn is not None:
        for i_DT, DTx in enumerate(DTn):
            dt_denominator -= DTx * (1e3 ** (i_DT + 1)) * (q * N) ** (i_DT + 2)

    N_dt = N * np.exp(t/2.19698) / dt_denominator

    return N_dt


def calc_dt_dc_error(t, M, M_err, params, run_info, fb):
    """
    calculates the erorrs on measured count rate data M after dead time and decay correction is applied
    see lab book 3 page 140
    :param t: time data
    :param M:  measured count rate data, *without decay or dead time corrections*
    :param M_err: error in measured count rate
    :param params: parameter array containing all the data
    :param run_info: run info dictionary
    :param fb: 'f' to calculate for f detectors, 'b' for backward
    :return: error in the dead time and decay corrected counts
    """

    # work out what dt corrections we have, and place them and the errors into an array
    dt_corrections, dt_errors = get_dt_corrections(params, do_error=True)
    if fb=='f':
        dt_corrections = dt_corrections[0]
        dt_errors = dt_errors[0]
    else:
        dt_corrections = dt_corrections[1]
        dt_errors = dt_errors[1]

    N = apply_dead_time_dc(t, M, run_info, params['DT0_'+fb], dt_corrections)

    N_denominator = M * np.exp(t / 2.19698) / N

    # this parameter, q, characterises the dead time correction (see lab book 3 page 139)
    q = 1/(run_info['n_frames']*run_info['n_detectors']*run_info['tres']*1e3)

    dN_dM = 1
    for i_dt_correction, dt_correction in enumerate(dt_corrections):
        dN_dM += (i_dt_correction + 1) * dt_correction * 1e3 ** (i_dt_correction + 1) * (q * M) ** (i_dt_correction + 2)
    dN_dM *= np.exp(t / 2.19698) / (N_denominator ** 2)

    dN_dDTO = q * M * N / N_denominator

    dN_dDTCi = [N * (1e3 ** (i+1) * (q * M) ** (i+2)) / N_denominator for i, _ in enumerate(dt_corrections)]

    err_sq = (dN_dM * M_err) ** 2 + (dN_dDTO * params['DT0_'+fb].stderr) ** 2
    for i_dN_dTC, this_dN_dTC in enumerate(dN_dDTCi):
        err_sq += (this_dN_dTC * dt_errors[i_dN_dTC]) ** 2

    err = np.sqrt(err_sq)

    return err


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
                # if there are three entries in the line, do the error
                if len(split_line) == 3:
                    this_yerror = float(split_line[2])
                if end_time is not None:
                    if this_x > end_time:
                        break
                x.append(this_x)
                y.append(this_y)
                if len(split_line) == 3:
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
                    asymmetry = 100*(n_f_counts - alpha*n_b_counts)/(n_f_counts + alpha*n_b_counts)
                except ZeroDivisionError:
                    x.pop(-1)
                    continue
                y.append(asymmetry)
                err_n_f = float(n_f[2])
                err_n_b = float(n_b[2])
                asymmetry_error = 2*100*alpha/(n_f_counts + alpha*n_b_counts) ** 2 *\
                                  np.sqrt((n_b_counts*err_n_f) ** 2 + (n_f_counts*err_n_b) ** 2)
                y_error.append(asymmetry_error)
        return np.array(x), np.array(y), np.array(y_error)


def load_count_data(file_name, encoding='iso-8859-1'):
    """
    Load raw counts data
    :param file_name: filename of the file which contains the raw counts data
    :return: arrays x, n_counts, err_n_counts, run_info (with keys 'n_frames', 'n_detectors', 'tres')
    """

    # extract the run information and stick it in run_info
    run_info = {}

    # do the regex
    number_re = re.compile(r'[\d.]+')

    x = []
    y = []
    y_err = []
    with open(file_name, 'r', encoding=encoding) as file:
        # for each line in the file
        for file_line in file.readlines():
            if file_line.startswith('!'):
                # if the file line talks about histograms, get the number of detectors out (and /2 as F and B), and tres
                if 'Histograms' in file_line:
                    hist_data = number_re.findall(file_line)
                    run_info.update({'n_detectors': round(int(hist_data[1])/2)})
                    run_info.update({'tres': float(hist_data[-2])*1e-6})
                # get the number of frames from the line about 'Events'
                if 'Events' in file_line:
                    event_data = number_re.findall(file_line)
                    run_info.update({'n_frames': int(event_data[-2])})
            else:
                # now load the data
                data = file_line.split()

                x.append(float(data[0]))
                y.append(float(data[1]))

                this_error = float(data[2])
                if this_error == 0:
                    this_error = max(1e-3, np.sqrt(y[-1]))
                y_err.append(this_error)

    return np.array(x), np.array(y), np.array(y_err), run_info

