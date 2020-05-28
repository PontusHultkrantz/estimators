'''
Implementation of algorithm from 
    Ledoit, O., and M. Wolf, 2003, “Honey, I Shrunk the Sample Covariance Matrix,” Working
    Paper, Department of Economics and Business, Universitat Pompeu Fabra.
Access: https://ssrn.com/abstract=433840
'''

import numpy as np
from numba import jit

def sample_cov(samples):
    return np.cov(samples, rowvar=0, ddof=1)

@jit(nopython=True)
def xcov_err_foo(y_norm, s, k, l, i, j):
    res = 0
    T = y_norm.shape[1]
    for t in range(T):
        res += ( y_norm[k,t]*y_norm[l,t] - s[k,l] )*( y_norm[i,t]*y_norm[j,t] - s[i,j] )
    return res/T

@jit(nopython=True)
def v_kkij(y_norm, s, k, i, j):
    return xcov_err_foo(y_norm, s, k, k, i, j)

@jit(nopython=True)
def pi_ij(y_norm, s, i, j):
    return xcov_err_foo(y_norm, s, i, j, i, j)

# RHO
@jit(nopython=True)
def calc_rho(y_norm, s, r_bar):
    N = y_norm.shape[0]
    rho_hat = 0
    for i in range(N):
        rho_hat += pi_ij(y_norm,s,i,i)
        for j in range(N):
            if j != i:
                rho_hat += r_bar*0.5*(np.sqrt(s[j,j]/s[i,i])*v_kkij(y_norm,s,i,i,j) + np.sqrt(s[i,i]/s[j,j])*v_kkij(y_norm,s,j,i,j))
    return rho_hat

@jit(nopython=True)
def calc_pi(y_norm, s):
    N = y_norm.shape[1]
    res = 0
    for i in range(N):
        for j in range(N):
            res += pi_ij(y_norm, s,i,j)
    return res

def honey_shrink(samples):
    # y_it: asset return series with asset 1<=i<=N and sample 1<=t<=T.
    y = samples.T
    T, N = samples.shape # (#Samples, #Variables).
    y_bar = np.mean(y, axis=1)
    y_norm = y - y_bar[:,np.newaxis]
    s = np.cov(y, rowvar=1, ddof=1)
    s_diag = np.diag(s)
    r = np.diag(1/np.sqrt(s_diag)) @ s @ np.diag(1/np.sqrt(s_diag))
    r_bar = (np.sum(r) - N)/(N*(N-1)) # Avg of xcorr.

    f = r_bar * np.sqrt( np.outer(s_diag, s_diag) )
    f[np.diag_indices_from(f)] = s_diag

    pi_hat = sum(pi_ij(y_norm, s,i,j) for i in range(N) for j in range(N))
    # pi_hat = calc_pi(y_norm, s)

    rho_hat = calc_rho(y_norm, s, r_bar)

    gamma_hat = np.sum((f-s)**2)
    kappa_hat = (pi_hat - rho_hat) / gamma_hat
    delta_hat = max(0, min(kappa_hat/T, 1))
    
    
    cov_shrink = delta_hat * f + (1-delta_hat)*s
    return cov_shrink