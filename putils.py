''' Portfolio statistics '''

import numpy as np
import putils
import math
from itertools import chain
import honey_shrink as covest


class Portfolio:
    def __init__(self, position):
        self.pos = position
    def __add__(self, other):
        return Portfolio(self.pos + other.pos)

def portfolio_var(p, mcov):
    '''
    Variance of linear combination of random variables (portfolio variance).
    '''
    return p.pos @ mcov @ p.pos

def mvp_hedge(p, mcov):
    '''
    Hedge ratio for minimum variance portfolio (MVP). Assuming hedging instruments are those with no current position in.
    '''
    idx_target = np.ravel( np.argwhere(p.pos != 0.0) )
    idx_hedge = np.ravel( np.argwhere(p.pos == 0.0) )
    mcov_hh = mcov[idx_hedge,:][:,idx_hedge]
    mcov_hk = mcov[idx_hedge,:][:,idx_target]
    hedge_weights = np.zeros(p.pos.shape)
    hedge_weights[idx_hedge] = -np.linalg.inv(mcov_hh) @ mcov_hk @ p.pos[idx_target]
    return Portfolio(hedge_weights)

def hedged_var(p, cov, cov_act):
    ''' 
    Variance of minimum variance portfolio (MVP).
    '''
    hdg = mvp_hedge(p, cov)
    mpv_var_est = portfolio_var(p + hdg, cov)
    mpv_var_act = portfolio_var(p + hdg, cov_act)
    return mpv_var_est, mpv_var_act


def portfolio_measures(p, mcov_est, mcov_exact):
    # Est: Hedged Portfolio Variance.
    mpv_var_est, mpv_var_act = hedged_var(p, mcov_est, mcov_exact)
    return (mpv_var_est, mpv_var_act)

def portfolio_est(p, mcov_est, mcov_act):
    # Est: Hedged Portfolio Variance.
    hdg = mvp_hedge(p, mcov_est)
    mpv_var_est = portfolio_var(p + hdg, mcov_est)
    mpv_var_act = portfolio_var(p + hdg, mcov_act)
    cov_relnorm = np.linalg.norm(mcov_est - mcov_act, ord='fro') / np.linalg.norm(mcov_act, ord='fro')
    return (mpv_var_est, mpv_var_act, cov_relnorm)
