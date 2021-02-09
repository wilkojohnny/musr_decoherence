"""
PercolationPolarisation.py -- to calculate the muon polarisation using ideas percolation theory
Created by John Wilkinson 9/2/21
"""

from . import DipolarPolarisation
from .MDecoherenceAtom import TDecoherenceAtom as atom
import numpy as np
import math
from matplotlib import pyplot


def calc_equiv_percolated_polarisation(all_spins:list, muon: atom, p: float, times:np.ndarray, gpu=False, plot=False) \
        -> np.ndarray:
    """
    Calculate the polarisation, assuming that the probability of each site being occupied is p, with each site being
     equivalent.
    :param all_spins: list of all the spins (each a TDecoherenceAtom object), with the muon in position 0
    :param muon:  TDecoherenceAtom object of the muon
    :param p: probability of each site being VACANT. Should be between 0 and 1
    :param times: np.ndarray of the times to calculate for
    :param gpu: if True, uses GPU parallelisation
    :param plot: if True, does a plot at the end
    """

    # count the number of sites (i.e 1 less than all_spins, as all_spins contains the muon)
    n_sites = len(all_spins) - 1

    # do 0 vacancies first
    result = (1-p) ** n_sites * DipolarPolarisation.calc_dipolar_polarisation(all_spins=all_spins, muon=muon,
                                                                              times=times, shutup=True, gpu=gpu)

    # for each site
    for n_vacancy in range(1, n_sites):
        these_spins = all_spins[:,-n_vacancy]
        probability = (1 - p) ** (n_sites - n_vacancy) * p ** n_vacancy
        print(probability)
        probability *= math.factorial(n_sites) / (math.factorial(n_sites - n_vacancy)* math.factorial(n_vacancy))
        print(probability)
        result += probability * DipolarPolarisation.calc_dipolar_polarisation(all_spins=all_spins, muon=muon,
                                                                              times=times, shutup=True, gpu=gpu)

    # add on all vacancies
    result += p ** n_sites * np.ones(shape=times.shape)

    pyplot.plot(times, result)
    pyplot.plot.show()

    return result
