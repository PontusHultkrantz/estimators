''' Monte Carlo with Welford online mean and variance  '''

import numpy as np
import putils
import math
from itertools import chain
import honey_shrink as covest


class Welford:
    '''
        Welford's online algorithm for online mean and variance.
    '''
    def __init__(self, shape):
        self._count_mean_M2 = np.zeros(shape + (3,))

    def update(self, new_value, loc):
        state = self._count_mean_M2[loc].copy() # Copy (Exception safe method)
        state[0] += 1 # ++count
        delta = new_value - state[1]
        state[1] += delta / state[0]
        delta2 = new_value - state[1]
        state[2] += delta * delta2
        self._count_mean_M2[loc] = state
        return

    def count(self, loc=None):
        if loc:
            return self._count_mean_M2[loc, 0]
        else:
            return self._count_mean_M2[:, 0]

    def mean(self, loc=None):
        if loc:
            return self._count_mean_M2[loc, 1]
        else:
            return self._count_mean_M2[:, 1]

    def samplevariance(self, loc=None):
        if loc:
            return self._count_mean_M2[loc, 2] / (self._count_mean_M2[loc, 0] - 1)
        else:
            return self._count_mean_M2[:, 2] / (self._count_mean_M2[:, 0] - 1)

    def MSE(self, pop_mean, loc=None):
        count = self.count(loc)
        mu = self.mean(loc)
        var = self.samplevariance(loc)
        return (mu - pop_mean)**2 + (1 - 1 / count) * var


class MonteCarlo:

    def __init__(self, rndgen, nsim, evalfunc, welfords):
        self.rndgen = rndgen
        self.evalfunc = evalfunc
        self.nsim = nsim
        self.welfords = welfords

    def run(self):
        for k in range(self.nsim):
            X = self.rndgen()
            self.evalfunc(self.welfords, X)

        return self.welfords