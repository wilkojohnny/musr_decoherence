# TCoord3d.py - class for 3D coordinates
#
# set basis to None for cartesian
#
# crystal_{x,y,z}: position in the crystal relative to the crystal lattice basis vectors [basis]
# ortho_{x,y,z}: position in the crystal relative to an orthonormal set of basis vectors, i.e ihat jhat khat
#
# for most cases, we want the actual x y and z rather than the one in the crystal x y and z, so call ortho_{x,y,z}
# unless there is a good reason to want the crystal_{x,y,z} (the only reason i've bothered with crystal coordinates is
# to be consistent with the literature and quantum espresso...)

import math
import numpy as np


class TCoord3D:

    def __init__(self, x: float = 0, y: float = 0, z: float = 0, basis=None):
        self.crystal_x = float(x)
        self.crystal_y = float(y)
        self.crystal_z = float(z)

        # define the basis, and the ortho coordinates
        self.basis = basis
        self.ortho_x = 0
        self.ortho_y = 0
        self.ortho_z = 0

        # set up the basis
        self.set_basis(basis)

    def set_basis(self, basis=None):
        if basis:
            self.basis = basis
            # set the orthogonal x y z to be the coordinates in terms of ihat jkat khat
            ortho_xyz = basis[0] * self.crystal_x + basis[1] * self.crystal_y + basis[2] * self.crystal_z
            self.ortho_x = ortho_xyz.ortho_x
            self.ortho_y = ortho_xyz.ortho_y
            self.ortho_z = ortho_xyz.ortho_z
        else:
            self.basis = None
            self.ortho_x = self.crystal_x
            self.ortho_y = self.crystal_y
            self.ortho_z = self.crystal_z

    # update this TCoord3D by copying another
    def update(self, other):
        self.crystal_x = other.crystal_x
        self.crystal_y = other.crystal_y
        self.crystal_z = other.crystal_z
        self.set_basis(other.basis)

    def __str__(self):
        return '(' + str(self.ortho_x) + ', ' + str(self.ortho_y) + ', ' + str(self.ortho_z) + ')'

    def __repr__(self):
        return '(' + str(self.ortho_x) + ', ' + str(self.ortho_y) + ', ' + str(self.ortho_z) + ')'

    # get normalised variables: defined as self*TCoord3D(0,1,0)/abs(this) ((* <-> dot product))
    def xhat(self):
        if self.r() != 0:
            return self * TCoord3D(1, 0, 0) / self.r()
        else:
            return 0

    def yhat(self):
        if self.r() != 0:
            return self * TCoord3D(0, 1, 0) / self.r()
        else:
            return 0

    def zhat(self):
        if self.r() != 0:
            return self * TCoord3D(0, 0, 1) / self.r()
        else:
            return 0

    def get_perpendicular_vector(self, normalise=True):
        while True:
            cross_vector = np.random.rand(3)
            perp_vector = np.cross(self.tonumpyarray(), cross_vector)
            if not np.all((perp_vector == 0)):
                perp_vector = TCoord3D(perp_vector[0], perp_vector[1], perp_vector[2])
                if normalise:
                    return perp_vector.rhat()
                else:
                    return perp_vector

    def totuple(self):
        return self.ortho_x, self.ortho_y, self.ortho_z,

    def toarray(self):
        return [self.ortho_x, self.ortho_y, self.ortho_z]

    def tonumpyarray(self):
        return np.array(self.toarray())

    # set r to be a predefined value, squashing this vector to make this happen
    def set_r(self, r, other=None):
        r_vector = self
        # if other is set, define a TCoord3D with the difference between these two vectors
        if other is not None:
            r_vector = self - other

        # now we want to multiply the vector by r/r_vector.r() to find the scale factor
        scale_factor = r/r_vector.r()
        r_vector *= scale_factor

        # if we defined other, add it back on, and return
        if other is not None:
            r_vector += other

        # update this vector
        self.update(other=r_vector)

    # get r
    def r(self):
        return math.sqrt(self * self)

    def rhat(self):
        return self / self.r()

    # define operators
    def __sub__(self, other):
        if self.basis == other.basis:
            return TCoord3D(self.crystal_x - other.crystal_x, self.crystal_y - other.crystal_y,
                            self.crystal_z - other.crystal_z, self.basis)
        if self.basis != other.basis:
            # do subtraction, but return something without a basis
            return TCoord3D(self.ortho_x - other.ortho_x, self.ortho_y - other.ortho_y, self.ortho_z - other.ortho_z)
        else:
            return NotImplemented

    def __add__(self, other):
        if self.basis == other.basis:
            return TCoord3D(self.crystal_x + other.crystal_x, self.crystal_y + other.crystal_y,
                            self.crystal_z + other.crystal_z, self.basis)
        if self.basis != other.basis:
            # if the bases are different, just add together the ortho coordinates and have a cartesian basis
            return TCoord3D(self.ortho_x + other.ortho_x, self.ortho_y + other.ortho_y, self.ortho_z + other.ortho_z)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, int):
            return TCoord3D(self.crystal_x * float(other), self.crystal_y * float(other), self.crystal_z * float(other)
                            , self.basis)
        elif isinstance(other, float):
            return TCoord3D(self.crystal_x * other, self.crystal_y * other, self.crystal_z * other, self.basis)
        elif isinstance(other, TCoord3D):
            # do the dot product
            return other.ortho_x * self.ortho_x + other.ortho_y * self.ortho_y + other.ortho_z * self.ortho_z
        else:
            return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, float):
            return TCoord3D(self.crystal_x / other, self.crystal_y / other, self.crystal_z / other, self.basis)
        else:
            return NotImplemented

    def __round__(self, n=None):
        return TCoord3D(round(self.ortho_x, n), round(self.ortho_y, n), round(self.ortho_z, n))

    def __eq__(self, other):
        if self.ortho_x == other.ortho_x and self.ortho_y == other.ortho_y and self.ortho_z == other.ortho_z:
            return True
        else:
            return False

    def __index__(self, index):
        if isinstance(index, int):
            if index == 0 or index == -1:
                return self.ortho_x
            elif index == 1:
                return self.ortho_y
            elif index == 2:
                return self.ortho_z
            else:
                return NotImplemented
        elif isinstance(index, str):
            if index == 'x':
                return self.ortho_x
            elif index == 'y':
                return self.ortho_y
            elif index == 'z':
                return self.ortho_z
            else:
                return NotImplemented
        else:
            return NotImplemented


cartesian_basis = [TCoord3D(1, 0, 0), TCoord3D(0, 1, 0), TCoord3D(0, 0, 1)]
