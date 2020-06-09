# MDecoherenceAtom.py - module with a class (TDecoherenceAtom) for adding an atom/muon (anything with spin) which
# provides some sort of decoherence to the muon polarisation.
# If multiple isotopes of one atom, this code creates new versions of itself as child objects (in self.isotopes)
# and then all the results work as if it was one atom when calling self[i] (i.e it acts as a list for isotopes)

import scipy.sparse as spmat  # for sparse matrices
import numpy as np  # for sqrt
import TCoord3D  # for position
from ase import atom, atoms  # to convert these to ASE atoms
from ase.gui.gui import GUI  # for the ASE gui
from ase.gui.images import Images  # to make the atoms GUI-able


class TDecoherenceAtom:

    def __init__(self, position: TCoord3D, name: str, gyromag_ratio=None, II=None, abundance=1., charge=None, Q=None,
                 anti_shielding=None, efg: np.ndarray = None):

        # if only name is defined, then get the numbers from the database
        if gyromag_ratio is None or II is None:
            try:
                gyromag_ratio = nucleon_properties[name]['gyromag_ratio']
                II = nucleon_properties[name]['II']
                abundance = nucleon_properties[name]['abundance']
                charge = nucleon_properties[name]['charge']
                if II > 1:
                    Q = nucleon_properties[name]['Q']
                    anti_shielding = nucleon_properties[name]['anti_shielding']
            except KeyError:
                print('WARNING -- Atom ' + name + ' is not in the database. The magnetic properties will be ignored.')
                II = 0
                gyromag_ratio = 0

        # we require the nuclear spin to create the Pauli matrices, and also the magnetic moment
        # (II = 2*spin, to prevent floating point problems)
        self.II = II
        self.I = II/2
        self.gyromag_ratio = gyromag_ratio
        self.position = position
        self.name = name
        self.abundance = abundance
        self.isotopes = []
        self.charge = charge
        self.Q = Q
        self.anti_shielding = anti_shielding
        self.efg = efg  # EFG is [V_xx, V_yy, V_zz] in Å^-3

        # check values make sense for quadrupoles
        if Q is not None:
            assert II > 1

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
            self.pauli_plus = spmat.diags([np.sqrt(self.I*(self.I+1) - .5*m*(.5*m+1)) for m in range(-II, II, 2)], 1)
            # pauli -
            self.pauli_minus = spmat.diags([np.sqrt(self.I * (self.I + 1) - .5 * m * (.5 * m + 1)) for m in range(-II, II, 2)], -1)

            # calculate pauli x and y
            self.pauli_x = .5*(self.pauli_plus + self.pauli_minus)
            self.pauli_y = .5/1j * (self.pauli_plus - self.pauli_minus)

        # also save dimension of these (need this for identity matrices)
        self.pauli_dimension = self.II + 1

    def duplicate(self, new_position=None):
        if new_position is None:
            new_position = self.position
        return TDecoherenceAtom(position=new_position, gyromag_ratio=self.gyromag_ratio, II=self.II, name=self.name,
                                abundance=self.abundance)

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

    def toASEatom(self) -> atom.Atom:
        return atom.Atom(symbol=self.name, position=self.position.totuple())


def visualise_atoms(input_atoms: list):
    """
    visualises the TDecoherenceAtoms using ASE
    :param atoms: list of TDecoherenceAtoms to visualise
    :return: 0
    """

    ase_atoms = atoms.Atoms()

    # convert all the atoms into the ASE format
    [ase_atoms.append(input_atom.toASEatom()) for input_atom in input_atoms]

    images = Images()
    images.initialize([ase_atoms])
    gui = GUI(images)
    gui.run()

    return 0

# dictionary for nuclei. Append as and when more are needed.
# Numbers from https://web.archive.org/web/20180305085231/http://www.kayelaby.npl.co.uk/chemistry/3_8/3_8_1.html
# UNITS:
#   - II = 2*I, nuclear moment
#   - gyromag_ratio = 2*pi*gamma_i
#   - abundance: fractional abundance (eg 0.5 = 50%)
#   - Q: in 10^-28 m^2
#   - charge: in units of e
nucleon_properties = {
    "mu": {"II": 1,
           "gyromag_ratio": 851.372,
           "abundance": 1,
           "Q": 0,
           "charge": +1,
           "anti_shielding": 0
    },
    "H": {"II": 1,
          "gyromag_ratio": 267.512,
          "abundance": 1,
          "Q": 0,
          "charge": +1},
    "Li": {"II": np.array([2, 3]),
           "gyromag_ratio": np.array([6.2655, 16.5465])*2*3.4145926,
           "abundance": np.array([0.0742, 0.9258])
           },
    "F": {"II": 1,
          "gyromag_ratio": 251.713,
          "abundance": 1,
          "Q": 0,
          "charge": -1,
          "anti_shielding": 0
          },
    "Na": {"II": 3,
           "gyromag_ratio": 70.76186,
           "abundance": 1,
           "Q": 0.1,
           "charge": +1,
           "anti_shielding": 4
           },
    "Sc": {"II": 7,
           "gyromag_ratio": 64.9895,
           "abundance": 1,
           "Q": -0.22,
           "charge": +3,
           "anti_shielding": +11.2 # from PRA *8* 1169 (1973)
    },
    "Y": {"II": 1,
          "gyromag_ratio": -13.1067,
          "abundance": 1
    }
}
