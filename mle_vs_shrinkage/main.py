import numpy as np
import scipy.stats as stats
import putils
import montecarlo as mc
import pmontecarlo as pmc
import present

# =========================
# ===== Daily Returns =====
# =========================
# Some 4 OMX stocks....
mcorr = np.array(
    [
        [1.00, 0.26, 0.41, 0.57],
        [0.26, 1.00, 0.19, 0.33],
        [0.41, 0.19, 1.00, 0.52],
        [0.57, 0.33, 0.52, 1.00]
    ])

vol = np.array([0.01531293, 0.01802784, 0.02480944, 0.01251876])
mcov = np.diag(vol) @ mcorr @ np.diag(vol)
obs_gen = stats.multivariate_normal(cov=mcov)
p_dim = mcorr.shape[0]

# ====================================
# ==== Covariance Frobenius Norm =====
# ====================================
norm_exact = np.linalg.norm(mcov, 'fro')

# ====================================
# ===== Portfolio Hedging Inits ======
# ====================================
# Used for estimating hedged portfolio variance, involing inverse of covariance matrix.
p = putils.Portfolio(np.array([1, 0, 0, 0]))
p_var = putils.portfolio_var(p, mcov)
hdg = putils.mvp_hedge(p, mcov)
mvp = p + hdg
mvp_var = putils.portfolio_var(mvp, mcov)
truevals = {'mvp.var':mvp_var, 'cov.relnorm':norm_exact, 'p.var':p_var}

# ====================================
# =========== Monte Carlo ============
# ====================================

T_grid = pmc.init_sparse_grid(lbound=p_dim, max_sample_size = 64)
mc = mc.MonteCarlo(
    rndgen=lambda: obs_gen.rvs(T_grid[-1]),
    nsim=1000,
    evalfunc=lambda welfords,path : pmc.mc_path_eval(T_grid, p, mcov, welfords, path),
    welfords=pmc.estimations(len(T_grid))
    )
est = mc.run()

# =====================
# ===== Plotting ======
# =====================
present.result(T_grid, est, truevals, alpha_conf = 0.01)