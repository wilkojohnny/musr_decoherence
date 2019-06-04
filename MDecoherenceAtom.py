# MDecoherenceAtom.py - module with a class (TDecoherenceAtom) for adding an atom/muon (anything with spin) which
# provides some sort of decoherence to the muon polarisation.
# If multiple isotopes of one atom, this code creates new versions of itself as child objects (in self.isotopes)
# and then all the results work as if it was one atom when calling self[i] (i.e it acts as a list for isotopes)

import scipy.sparse as spmat  # for sparse matrices
import numpy as np  # for sqrt
import TCoord3D # for position


class TDecoherenceAtom:

    def __init__(self, position: TCoord3D, gyromag_ratio, II, name='atom', abundance=1.):
        # we require the nuclear spin to create the Pauli matrices, and also the magnetic moment
        # (II = 2*spin, to prevent floating point problems)
        self.II = II
        self.I = II/2
        self.gyromag_ratio = gyromag_ratio
        self.position = position
        self.name = name
        self.abundance = abundance
        self.isotopes = []

        if type(abundance) is np.ndarray:
            # if there's more than one isotope, register them as a list
            for i in range(0, len(abundance)):
                self.isotopes.append(TDecoherenceAtom(self.position, self.gyromag_ratio[i], self.II[i], self.name, self.abundance[i]))
            # don't define any of the pauli matrices
            self.pauli_x = None
            self.pauli_y = None
            self.pauli_z = None
        else:
            # single isotope
            # Calculate the spin measurement matrices depending on I
            # pauli z
            self.pauli_z = .5*np.array(spmat.diags([m for m in range(-II, II+2, 2)])) # the self.II+2 is due to Python using exclusive ranges
            # pauli +
            self.pauli_plus = np.array(spmat.diags([np.sqrt(self.I*(self.I+1) - .5*m*(.5*m+1)) for m in range(-II, II, 2)], 1))
            # pauli -
            self.pauli_minus = np.array(spmat.diags([np.sqrt(self.I * (self.I + 1) - .5 * m * (.5 * m + 1)) for m in range(-II, II, 2)], -1))

            # calculate pauli x and y
            self.pauli_x = .5*(self.pauli_plus + self.pauli_minus)
            self.pauli_y = .5/1j * (self.pauli_plus - self.pauli_minus)

        # also save dimension of these (need this for identity matrices)
        self.pauli_dimension = self.II + 1

    def duplicate(self, new_position=None):
        if new_position is None:
            new_position = self.position
        return TDecoherenceAtom(new_position, self.gyromag_ratio, self.II, self.name, self.abundance)

    def __str__(self):
        return self.name + ' at ' + str(self.position) + ' (with ' + str(len(self)) + ' isotope(s))'

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        else:
            return NotImplemented

    def __getitem__(self, item):
        if len(self.isotopes) > 1 and len(self.isotopes) > item:
            return self.isotopes[item]
        else:
            return self

    def __len__(self):
        if len(self.isotopes) > 1:
            return len(self.isotopes)
        else:
            return 1

    def change_position(self, new_position: TCoord3D):
        self.position = new_position

    # write out a full description of what this atom is
    def verbose_description(self, gle_friendly=True):
        if gle_friendly:
            linestart = '! '
        else:
            linestart = ''

        outstring = []

        # write an introductory line
        outstring.append(linestart + self.name + ' at position ' + str(self.position) + ' with ' + str(len(self))
                         + ' isotope(s):' + '\n')
        # for each isotope...
        if len(self) > 1:
            for i in range(0, len(self)):
                # ... output the atom type,
                outstring.append(linestart + '\t Isotope ' + str(i) + ':\n')
                outstring.append(linestart + '\t \t Abundance: ' + str(self.abundance[i]*100) + '% \n')
                outstring.append(linestart + '\t \t I: ' + str(self.II[i]) + '/2 \n')
                outstring.append(linestart + '\t \t Gyromagnetic ratio (*2pi): ' + str(self.gyromag_ratio[i]) + ' \n')
        else:
            outstring.append(linestart + '\t I: ' + str(self.II) + '/2 \n')
            outstring.append(linestart + '\t Gyromagnetic ratio (*2pi): ' + str(self.gyromag_ratio) + ' \n')

        return outstring