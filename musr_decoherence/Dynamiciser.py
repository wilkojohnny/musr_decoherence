"""
Dynamiciser.py -- adds dynamics to any MuSR polarisation function
By John Wilkinson, 6/10/2020
13/5/2024 -- added in radical->mu+
"""

import numpy as np
from matplotlib import pyplot as plt
from typing import Callable
from scipy import integrate

def main():

    delta = 0.3
    KT = lambda x: 1/3 + 2/3 * (1-(delta*x) ** 2) * np.exp(-(delta * x) ** 2 / 2)

    nu = 0.01

    t = np.arange(0.0, 35.0, 0.01)

    dynamic_function = radical_to_muplus(muplus_function=KT, radical_lambda=0.005, nu=nu, t=t)

    plt.plot(t, dynamic_function)
    plt.show()


def dynamicise(static_function: np.ndarray, t: np.ndarray, dt: float, nu:float) -> np.ndarray:
    """
    dynamicise: add dynamics to a static MuSR funciton. Based on a Markovian strong collision model approach.
    :param static_function: numpy array of the static function to dynamicise
    :param t: numpy array of the corresponding times for the static_functin
    :param dt: time step of t
    :param nu: probability per unit time of a muon hop
    :return: numpy array of the dynamicised function
    """

    dynamic_function = np.zeros(static_function.shape)

    # start the dynamic function at 1
    remove_extra = False
    if t[0] != 0:
        t = np.insert(t, 0, 0)
        dynamic_function = np.insert(dynamic_function,0, 0)
        static_function = np.insert(static_function, 0, 1)
        remove_extra = True

    dynamic_function[0] = 1

    for i_t in range(1, len(t)):
        this_t = t[i_t]
        sum = 0
        for i_t1 in range(1, i_t):
            sum += np.exp(-1*nu*(this_t - t[i_t1])) * static_function[i_t - i_t1] * dynamic_function[i_t1]
        sum *= nu*dt
        dynamic_function[i_t] = 1/(1 - nu*dt/2) * (static_function[i_t]*(1 + nu*dt/2)*np.exp(-nu*this_t) + sum)

    if remove_extra:
        dynamic_function = np.delete(dynamic_function, 0)

    return dynamic_function


def radical_to_muplus(muplus_function: Callable[[np.ndarray], np.ndarray], radical_lambda: float, nu: float,
                      t: np.ndarray) -> np.ndarray:
    """
    Calculates the polarization function for a muon starting off in a muonium radical state (with polarisation function
    exp(-lambda*t), and then hops with rate nu into a mu+ polarization function (which could be e.g. Kubo-Toyabe).
    :param muplus_function: function of the form F(t), where t is a numpy array, and returns a numpy array.
    :param radical_lambda: decay rate of the radical state
    :param nu: hop rate from radical -> muplus_function
    :param t: time
    :return: numpy array of the polarization function at the times t
    """
    result = np.zeros(t.shape)
    for i_t, this_t in enumerate(t):
        integral = integrate.quad(lambda int_time: muplus_function(this_t - int_time) *
                                                   np.exp(-(nu + radical_lambda)*int_time),
                                  0, this_t)
        result[i_t] = integral[0]

    return np.exp(-(radical_lambda + nu) * t) + nu*result


if __name__=='__main__':
    main()
