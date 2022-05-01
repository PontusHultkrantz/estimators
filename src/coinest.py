import warnings
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import scipy.stats as stats

def cmle_loglike(p, nvec, kvec):
    loglike = 0
    for i in range(p.shape[0]):
        loglike += -np.log(stats.binom.pmf(kvec[i], nvec[i], p[i]))
    return loglike

def cmle_est(ntrials, nsucc):
    ''' Conditional MLE argmax f(p|n,x) s.t. p monotonically incr parameters. '''
    ncoins = ntrials.shape[0]
    constraints = {'type':'ineq', 'fun':lambda x : np.diff(x)} # np.diff(x) > 0 constraint
    bounds = [(0.001, 0.999)] * ncoins
    x0 = [i/(ncoins+1) for i in range(ncoins)]
    res = minimize(cmle_loglike, x0, args=(ntrials, nsucc), bounds=bounds, constraints=constraints)
    if not res.success:
        warnings.warn(res.message, RuntimeWarning)
    return res.x

def mle_est(ntrials, ksucc):
    ''' MLE argmax f(p|n,x) '''
    return ksucc / ntrials

def sample_true_p(ncoins, dist):
    return np.sort(dist.rvs(ncoins))