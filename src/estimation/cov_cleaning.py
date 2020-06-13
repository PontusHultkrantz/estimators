import numpy as np

def cov2corr(cov):
    vol = np.diag(cov)**0.5
    corr = np.diag(1/vol) @ cov @ np.diag(1/vol)
    return corr, vol

def corr2cov(corr, vols):
    volm = np.diag(vols)
    return volm @ corr @ volm

def mp_denoise(cov, p, n):
    #vol = np.diag(cov)**0.5
    #corr = np.diag(1/vol) @ cov @ np.diag(1/vol)
    corr, vol = cov2corr(cov)
    
    mp_ubound = (1 + (p/n)**0.5)**2
    u, s, vt = np.linalg.svd(corr)
    noise_idx = s < mp_ubound
    s[noise_idx] = s[noise_idx].mean()
    corr_denoised = u @ np.diag(s) @ vt
    #cov_denoised = np.diag(vol) @ corr_denoised @ np.diag(vol)
    cov_denoised = corr2cov(corr_denoised, vol)
    return cov_denoised

def RIE(cov, N, T, corrmode=True):
    if corrmode:
        corr, vol = cov2corr(cov)
    else:
        corr = cov
    q = N / T
    u, eigs, vt = np.linalg.svd(corr)
    z = eigs - 1/N**0.5 * 1j
    s = 1/N * sum(1/(z[k]-eigs[j]) for k in range(N) for j in range(N) if k!=j)
    xi_rie = eigs / np.abs(1 - q + q*z*s)**2
    lambda_N = eigs[-1]
    sigma2 = lambda_N / (1-q**0.5)**2
    lambda_plus = lambda_N * ((1+np.sqrt(q))/(1-np.sqrt(q)))**2
    gmp = (z + sigma2*(q-1) - np.sqrt(z-lambda_N)*np.sqrt(z-lambda_plus)) / (2*q*z*sigma2)
    Gamma = sigma2 * np.abs(1 - q + q*z*gmp)**2 / eigs
    Gamma[Gamma <= 1] = 1.0
    xi = xi_rie * Gamma
    corr_denoised = u @ np.diag(xi) @ vt
    
    if corrmode:
        cov_denoised = corr2cov(corr_denoised, vol)
    else:
        cov_denoised = corr_denoised
    return cov_denoised