"""
Dynamiciser.py -- adds dynamics to any MuSR polarisation function
By John Wilkinson, 6/10/2020
"""

import numpy as np
from matplotlib import pyplot as plt


def main():

    dt = 0.1
    t = np.arange(0, 10, dt)
    static_function = 1/3*np.ones(t.shape) + 2/3*(1-t**2)*np.exp(-.5*(t**2))

    nu = 0.1

    dynamic_function = dynamicise(static_function, t, dt, nu)

    plt.plot(t, static_function)
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
    dynamic_function[0] = 1

    for i_t in range(1, len(t)):
        this_t = t[i_t]
        sum = 0
        for i_t1 in range(1, i_t):
            sum += np.exp(-1*nu*(this_t - t[i_t1])) * static_function[i_t - i_t1] * dynamic_function[i_t1]
        sum *= nu*dt
        dynamic_function[i_t] = 1/(1 - nu*dt/2) * (static_function[i_t]*(1 + nu*dt/2)*np.exp(-nu*this_t) + sum)

    return dynamic_function


if __name__=='__main__':
    main()
