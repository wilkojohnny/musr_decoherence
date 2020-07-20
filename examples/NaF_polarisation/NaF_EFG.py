import numpy as np


def main():
    # print(calc_efg(r_f_mu=1.637, r_na_mu=1.637))

    print(calc_mugrid_efg())

    return 0


def calc_mugrid_efg():
    na_position = np.array([-.5, -.5, 0])

    r = 60

    nml = []

    for n in range(-r, r):
        for m in range(-r, r):
            for l in range(-r, r):
                if (n - na_position[0]) ** 2 + (m - na_position[1]) ** 2 + (l - na_position[2]) ** 2 <= r ** 2:
                    # within radius -- so add to the list
                    nml.append([4*n, 4*m, 4*l])

    lattice_efg = np.zeros((3, 3))

    for i_nml, this_nml in enumerate(nml):
        xyz = this_nml - na_position
        lattice_efg += efg_matrix(xyz)

    return lattice_efg



def calc_efg(r_na_mu: float, r_f_mu: float):
    """
    Calculates the EFG the Na ions experience in NaF
    :param r_na_mu: mu--Na distance (in Angstroms)
    :param r_f_mu: mu--F distance (in Angstroms)
    :return: EFG tensor as numpy array, in units of (a/2)^-3.
    """
    d_na = (1.637 - r_na_mu)/(2*1.637)
    r = 50

    # position of sodium
    na_position = np.array([d_na, d_na, 0])

    lattice_efg = np.zeros((3, 3))

    nml = []

    for n in range(-r, r):
        for m in range(-r, r):
            for l in range(-r, r):
                if (n - na_position[0]) ** 2 + (m - na_position[1]) ** 2 + (l - na_position[2]) ** 2 <= r ** 2:
                    # within radius -- so add to the list
                    nml.append([n, m, l])

    for i_nml, this_nml in enumerate(nml):
        if this_nml == [0, 0, 0]:
            # skip the origin (that's where the Na is...)
            continue
        # see if its a sodium or fluorine (i.e if n+m+l even, Na, else F)
        charge = 1 - 2 * ((this_nml[0] + this_nml[1] + this_nml[2]) % 2)
        xyz = this_nml - na_position
        lattice_efg += charge * efg_matrix(xyz)

    muon_position = np.array((0.5, 0.5, 0))
    muon_efg = efg_matrix(muon_position - na_position)

    # make the perturbation matrix
    # nn fluorines
    fluorine_hole = efg_matrix(np.array((1, 0, 0)) - na_position)
    fluorine_hole += efg_matrix(np.array((0, 1, 0)) - na_position)

    d_f_ortho = (1.637 - r_f_mu) / (2 * 1.637)
    new_fluorines = -1 * efg_matrix(np.array((1 - d_f_ortho, d_f_ortho, 0)) - na_position)
    new_fluorines += -1 * efg_matrix(np.array((d_f_ortho, 1 - d_f_ortho, 0)) - na_position)

    fluorine_contribution = fluorine_hole + new_fluorines

    sodium_hole = -1 * efg_matrix(np.array((1, 1, 0)) - na_position)
    new_sodium = efg_matrix(np.array((1, 1, 0)) - 2 * na_position)
    sodium_contribution = sodium_hole + new_sodium

    total_efg = lattice_efg + muon_efg + sodium_contribution + fluorine_contribution

    return total_efg


def efg_matrix(xyz: np.ndarray):
    (x, y, z) = xyz
    r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return np.array([[3*x**2 - r**2, 3*x*y, 3*x*z], [3*x*y, 3*y**2 - r**2, 3*y*z],
                     [3*x*z, 3*y*z, 3*z**2 - r**2]]) / (r ** 5)


if __name__ == '__main__':
    main()
