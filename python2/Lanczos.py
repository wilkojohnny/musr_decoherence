# Lanczos.py - investigations into diagonalizing with the Lanczos method
#
# Created 3/7/19 by John Wilkinson

from __future__ import absolute_import
import numpy as np
from scipy.linalg import eig, eigh
import scipy.sparse.linalg
import time


def main():
    # dimensions of the square matrix
    for n_atoms in xrange(6, 23):
        n = pow(2, n_atoms)

        a = np.random.rand(n)  # leading diagonal
        b = np.random.rand(n - 1)   # off diagonal
        c = np.random.rand(n - 2)   # off-off diagonal

        # create the matrix
        matrix = scipy.sparse.diags(a) + scipy.sparse.diags(b, 1) + scipy.sparse.diags(b, -1)
        matrix = matrix + scipy.sparse.diags(c, 2) + scipy.sparse.diags(c, -2)

        tic = time.time()
        values, vectors = scipy.linalg.eig(matrix.todense())
        toc = time.time()
        # print the number of atoms and the time taken (in seconds) to find 50 eigenvalues
        print u'! ' + unicode(n_atoms) + u' ' + unicode(n) + u' ' + unicode(toc-tic)

    return 1


if __name__==u'__main__':
    main()