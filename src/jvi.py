
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import math
import numpy as np
import scipy.special as sp
from scipy import misc
import functools
import itertools

# Compute the Sharot coefficients a_{j,k,n,r}.
def sharot(k,n,r):
    p = float(r)/float(n)
    res = np.zeros(k+1)
    for j in xrange(0,k+1):
        c = sp.binom(k,j)*((1.0-j*p)**k)
        if (j % 2) == 1:
            c *= -1
        res[j] = c / ((p**k)*math.factorial(k))
    return res

def jvi_size(n,k):
    return sum([int(sp.binom(n,j)) for j in xrange(0,k+1)])

# Compute the JVI weighting vector and weighting matrix.
#
# Return A,B, where:
#   A: (M,) vector of weights, one for each of M sets.
#   B: (M,n) matrix; each row corresponds to a subset of n samples,
#      withing a row, the weights correspond to the weighted sum over samples.
def jvi_matrix(n,k):
    if n <= k:
        raise ValueError("JVI order must be smaller than number of samples.")

    sc = jvi_size(n,k)
    B = np.zeros((sc,n))
    A = np.zeros(sc)
    SH = sharot(k,n,1)

    j = 0
    for setsize in xrange(0,k+1):
        for index_set in itertools.combinations(range(n), n-setsize):
            B[j,index_set] = 1.0/float(n-setsize)
            A[j] = SH[setsize] / sp.binom(n, n-setsize)
            j += 1

    return A, B
