''' Helper module for monte carlo estimation of portfolio statistics '''
import numpy as np
import putils
import math
from itertools import chain
import honey_shrink as covest
import montecarlo as mc

def init_sparse_grid(lbound=1, max_sample_size=1):
    # Grid of number of samples. Don't allow nsamples == p_dim, too avoid possibly singular covariance matrices.
    ubound = int(math.log(max_sample_size)/math.log(2) - 1) # largest k such that 2^(k+1) <= max_samples_size.
    T_grid = np.array( list(chain.from_iterable([range(lbound * 2**k, lbound * 2**(k+1), 4**k) for k in range(ubound)]))[1::] )
    return T_grid


estimations = lambda time_dim: {
    'mle.mvp.var.est':mc.Welford(shape=(time_dim,)),
    'mle.mvp.var.act': mc.Welford(shape=(time_dim,)),
    'mle.cov.relnorm.est': mc.Welford(shape=(time_dim,)),
    'shrink.mvp.var.est':mc.Welford(shape=(time_dim,)),
    'shrink.mvp.var.act': mc.Welford(shape=(time_dim,)),
    'shrink.cov.relnorm.est': mc.Welford(shape=(time_dim,)),
    'rie.mvp.var.est':mc.Welford(shape=(time_dim,)),
    'rie.mvp.var.act': mc.Welford(shape=(time_dim,)),
    'rie.cov.relnorm.est': mc.Welford(shape=(time_dim,))    
    }


from src.estimation.cov_cleaning import RIE, mp_denoise

# mle_hdgvar, mle_hdgvar_act, mle_norm, shrink_hdgvar, shrink_hdgvar_act, shrink_norm = (Welford(len(T_grid)) for i in range(6))
def mc_path_eval(T_grid, p, mcov, est, path):
    for t_idx, t in enumerate(T_grid):
        # ==== MLE (Sample cov) =====
        mcov_mle = covest.sample_cov(path[0:t,:])
        hdg_var, hdg_var_act, relnorm = putils.portfolio_est(p, mcov_mle, mcov)
        est['mle.mvp.var.est'].update(hdg_var, t_idx)
        est['mle.mvp.var.act'].update(hdg_var_act, t_idx)
        est['mle.cov.relnorm.est'].update(relnorm, t_idx)
        
        # ==== Shrinkage =====
        mcov_honey = covest.honey_shrink(path[0:t,:])
        hdg_var, hdg_var_act, relnorm = putils.portfolio_est(p, mcov_honey, mcov)
        est['shrink.mvp.var.est'].update(hdg_var, t_idx)
        est['shrink.mvp.var.act'].update(hdg_var_act, t_idx)
        est['shrink.cov.relnorm.est'].update(relnorm, t_idx)
        
        # ==== RIE =====
        #mcov_rie = RIE(mcov_mle, mcov_mle.shape[0], t, corrmode=False)
        #mcov_rie = mp_denoise(mcov_mle, mcov_mle.shape[0], t)
        #hdg_var, hdg_var_act, relnorm = putils.portfolio_est(p, mcov_rie, mcov)
        #est['rie.mvp.var.est'].update(hdg_var, t_idx)
        #est['rie.mvp.var.act'].update(hdg_var_act, t_idx)
        #est['rie.cov.relnorm.est'].update(relnorm, t_idx)        