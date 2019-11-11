# DecoFitter.py - fits experimental musr data to that calculated by DecoherenceCalculator

# load modules
import DecoherenceCalculator  # for the decoherence routines
from TCoord3D import TCoord3D as coord  # for coordinates
import numpy as np  # for numpy arrays
from scipy import optimize  # for nls curve fitting
from MDecoherenceAtom import TDecoherenceAtom as atom  # import class for decoherence atom
import matplotlib.pyplot as pyplot  # for plotting
import matplotlib.colors as color


# squish_radius initial values (made global so that we can fix)
squish_radii_initial = [1.172211, 2.79623945]
# which nnnness are we squishing?
squish_radius_nnnness_vary = 3

def main():

    # load in the data (expect it of the form x y yerr)
    x, y, y_error = load_muon_data('76785.dat')

    # work out which squish radius to vary
    squish_radius = squish_radii_initial[squish_radius_nnnness_vary - 2]

    # try to do a fit
    #initial_guess = [35.0795, squish_radius, 0.01122]
    initial_guess = [8.70375687, squish_radius, 0.04102239]
    #params, covariances = optimize.curve_fit(fit_function, xdata=x, ydata=y, p0=initial_guess, sigma=y_error)
    # params_errors = np.sqrt(np.diag(covariances))
    #
    # print(params)
    # print(params_errors)

    params = initial_guess

    # calculate the fit function one last time
    fit_func = fit_function(x, params[0], params[1], params[2])
    # find the chi squared
    chi_squared = (((fit_func - y)/y_error)**2).sum()
    chi_squared_dof = chi_squared/(len(y) - len(params))

    # print out the chi squared
    print('Chi-squared: ' + str(chi_squared) + ' (' + str(chi_squared_dof) + ' per dof)')

    # plot the data
    pyplot.errorbar(x, y, y_error, ecolor=color.cnames['red'], marker='.', linestyle='none')
    pyplot.plot(x, fit_func, color=color.cnames['black'])
    pyplot.ylim([-10, 20])
    pyplot.xlim([0, 25])
    pyplot.show()


def fit_function(x, A, squish, A0):

    lattice_type = DecoherenceCalculator.ibrav.CUBIC_FCC  # # can only do fcc and monoclinic (unique axis b)
    lattice_parameter = [5.44542, 0, 0]  # [a, b, c]
    lattice_angles = [90, 0, 0]  # [alpha, beta, gamma] in **degrees**

    input_coord_units = DecoherenceCalculator.position_units.ALAT

    # atoms and unit cell: dump only the basis vectors in here, the rest is calculated
    atomic_basis = [
        atom(coord(0.25, 0.25, 0.25), gyromag_ratio=251.713, II=1, name='F'),
        atom(coord(0.25, 0.25, 0.75), gyromag_ratio=251.713, II=1, name='F')
    ]

    # register the perturbed distances
    perturbed_distances = []

    # define muon position
    muon_position = coord(.25, 0.25, 0.5)

    # make the squish radii
    squish_radii = squish_radii_initial
    squish_radii[squish_radius_nnnness_vary - 2] = squish

    deco = DecoherenceCalculator.calc_decoherence(muon_position=muon_position, times=x, squish_radius=squish_radii,
                                                  lattice_type=lattice_type, lattice_parameter=lattice_parameter,
                                                  lattice_angles=lattice_angles, input_coord_units=input_coord_units,
                                                  atomic_basis=atomic_basis, perturbed_distances=perturbed_distances,
                                                  plot=False, nnnness=3, shutup=False,
                                                  outfile_location='CaF2_nnnsquish_output.dat')

    return A*deco + A0*np.ones(len(x))


def load_muon_data(filename: str, encoding='iso-8859-1'):
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
            split_line = line.split(' ')
            # put these into the arrays
            x.append(float(split_line[0]))
            y.append(float(split_line[1]))
            y_error.append(float(split_line[2]))

    return np.array(x), np.array(y), np.array(y_error)


if __name__=='__main__':
    main()