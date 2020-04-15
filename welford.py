import numpy as np

class Welford:
    '''
        Welford's online algorithm for online mean and variance.
    '''
    def __init__(self, nsteps):
        self._count_mean_M2 = np.zeros((nsteps, 3))

    def update(self, new_value, istep):
        state = self._count_mean_M2[istep].copy() # Copy (Exception safe method)
        state[0] += 1 # ++count
        delta = new_value - state[1]
        state[1] += delta / state[0]
        delta2 = new_value - state[1]
        state[2] += delta * delta2
        self._count_mean_M2[istep] = state
        return

    def count(self, istep=None):
        if istep:
            return self._count_mean_M2[istep, 0]
        else:
            return self._count_mean_M2[:, 0]

    def mean(self, istep=None):
        if istep:
            return self._count_mean_M2[istep, 1]
        else:
            return self._count_mean_M2[:, 1]

    def samplevariance(self, istep=None):
        if istep:
            return self._count_mean_M2[istep, 2] / (self._count_mean_M2[istep, 0] - 1)
        else:
            return self._count_mean_M2[:, 2] / (self._count_mean_M2[:, 0] - 1)

    def MSE(self, pop_mean, istep=None):
        count = self.count(istep)
        mu = self.mean(istep)
        var = self.samplevariance(istep)
        return (mu - pop_mean)**2 + (1 - 1 / count) * var