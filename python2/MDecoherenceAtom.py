# MDecoherenceAtom.py - module with a class (TDecoherenceAtom) for adding an atom/muon (anything with spin) which
# provides some sort of decoherence to the muon polarisation.
# If multiple isotopes of one atom, this code creates new versions of itself as child objects (in self.isotopes)
# and then all the results work as if it was one atom when calling self[i] (i.e it acts as a list for isotopes)

from __future__ import division
from __future__ import absolute_import
import scipy.sparse as spmat  # for sparse matrices
import numpy as np  # for sqrt
import TCoord3D # for position


class TDecoherenceAtom(object):

    def __init__(self, position, gyromag_ratio, II, name=u'atom', abundance=1.):
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
            for i in xrange(0, len(abundance)):
                self.isotopes.append(TDecoherenceAtom(self.position, self.gyromag_ratio[i], self.II[i], self.name, self.abundance[i]))
            # don't define any of the pauli matrices
            self.pauli_x = None
            self.pauli_y = None
            self.pauli_z = None
        else:
            # single isotope
            # Calculate the spin measurement matrices depending on I
            # pauli z
            self.pauli_z = .5*np.array(spmat.diags([m for m in xrange(-II, II+2, 2)], 0)) # the self.II+2 is due to Python using exclusive ranges
            # pauli +
            self.pauli_plus = np.array(spmat.diags([np.sqrt(self.I*(self.I+1) - .5*m*(.5*m+1)) for m in xrange(-II, II, 2)], 1))
            # pauli -
            self.pauli_minus = np.array(spmat.diags([np.sqrt(self.I * (self.I + 1) - .5 * m * (.5 * m + 1)) for m in xrange(-II, II, 2)], -1))

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
        return self.name + u' at ' + unicode(self.position) + u' (with ' + unicode(len(self)) + u' isotope(s))'

    def __eq__(self, other):
        if isinstance(other, unicode):
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

    def change_position(self, new_position):
        self.position = new_position

    # write out a full description of what this atom is
    def verbose_description(self, gle_friendly=True):
        if gle_friendly:
            linestart = u'! '
        else:
            linestart = u''

        outstring = []

        # write an introductory line
        outstring.append(linestart + self.name + u' at position ' + unicode(self.position) + u' with ' + unicode(len(self))
                         + u' isotope(s):' + u'\n')
        # for each isotope...
        if len(self) > 1:
            for i in xrange(0, len(self)):
                # ... output the atom type,
                outstring.append(linestart + u'\t Isotope ' + unicode(i) + u':\n')
                outstring.append(linestart + u'\t \t Abundance: ' + unicode(self.abundance[i]*100) + u'% \n')
                outstring.append(linestart + u'\t \t I: ' + unicode(self.II[i]) + u'/2 \n')
                outstring.append(linestart + u'\t \t Gyromagnetic ratio (*2pi): ' + unicode(self.gyromag_ratio[i]) + u' \n')
        else:
            outstring.append(linestart + u'\t I: ' + unicode(self.II) + u'/2 \n')
            outstring.append(linestart + u'\t Gyromagnetic ratio (*2pi): ' + unicode(self.gyromag_ratio) + u' \n')

        return outstring
