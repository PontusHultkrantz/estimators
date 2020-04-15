import numpy as np

def portfolio_var(p_weights, mcov):
    '''
    Variance of linear combination of random variables (portfolio variance).
    '''
    return p_weights @ mcov @ p_weights

def hedge_ratio(p_weights, mcov):
    '''
    Hedge ratio for minimum variance portfolio (MVP). Assuming hedging instruments are those with no current position in.
    '''
    idx_target = np.ravel( np.argwhere(p_weights != 0.0) )
    idx_hedge = np.ravel( np.argwhere(p_weights == 0.0) )
    mcov_hh = mcov[idx_hedge,:][:,idx_hedge]
    mcov_hk = mcov[idx_hedge,:][:,idx_target]
    hedge_weights = np.zeros(p_weights.shape)
    hedge_weights[idx_hedge] = -np.linalg.inv(mcov_hh) @ mcov_hk @ p_weights[idx_target]
    return hedge_weights

def hedged_var(hdg_target, cov):
    ''' 
    Variance of minimum variance portfolio (MVP).
    '''
    hdg_weights = hedge_ratio(hdg_target, cov)
    var_hedged_est = portfolio_var(hdg_target + hdg_weights, cov)
    return var_hedged_est


def portfolio_statistics(p_weights, hedge_target, mcov_est):
    # Est: Portfolio var.
    p_var = portfolio_var(p_weights, mcov_est)
    # Est: Hedged Portfolio Variance.
    p_hdg_var = hedged_var(hedge_target, mcov_est)
    return (p_var, p_hdg_var)    