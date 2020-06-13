
''' Plot portfolio statistics estimations '''

import scipy.stats as stats
from matplotlib import pyplot as plt


ALPHA_CONF = 0.01
color = {'mle':'#1f77b4', 'shrink':'#ff7f0e', 'rie':'#2ca02c'}
conf_label = dict((name, '{}$_{{\\alpha={:.3g}\%}}$'.format(name, 100*ALPHA_CONF)) for name in ['mle', 'shrink', 'rie'])

def mean_confint(welford_obj, conf_alpha):
    mean = welford_obj.mean()
    mean_std = welford_obj.samplevariance() / welford_obj.count()
    shift = stats.norm.ppf(1.0 - conf_alpha/2.0) * mean_std # CLT -> gaussian
    return {'mean.LB':mean - shift, 'mean':mean, 'mean.UB':mean + shift}

def relerr_mean_var(welford_obj, rel_benchmark):
    ''' V[est/exact-1] '''
    return (welford_obj.samplevariance() / welford_obj.count()) / rel_benchmark**2

def relerr_mean_confint(welford_obj, rel_benchmark, conf_alpha):
    rerr = welford_obj.mean() / rel_benchmark -1
    rerr_std = relerr_mean_var(welford_obj, rel_benchmark) ** 0.5
    shift = stats.norm.ppf(1.0 - conf_alpha/2.0) * rerr_std # CLT -> gaussian
    return {'relerr.LB':rerr - shift, 'relerr.mean':rerr, 'relerr.UB':rerr + shift}

def rel_mean_confint(welford_obj, rel_benchmark, conf_alpha):
    relmean = welford_obj.mean() / rel_benchmark
    relmean_std = relerr_mean_var(welford_obj, rel_benchmark) ** 0.5
    shift = stats.norm.ppf(1.0 - conf_alpha/2.0) * relmean_std # CLT -> gaussian
    return {'relmean.LB':relmean - shift, 'relmean':relmean, 'relmean.UB':relmean + shift}


def result(x, y, truevals, alpha_conf):

    fig, axs = plt.subplots(4, 1, sharex=False, figsize=(15,20))
    plt.subplots_adjust(hspace=0.4)
   
    
    # === Present normalized Frobenius norm, which is what most shrinkage estimators are minimizing. ===
    axs[0].set_title('Normalized Frob.norm of cov. est. error')
    axs[0].set(xlabel='sample size', ylabel='Rel. Err.')
    mle = mean_confint(y['mle.cov.relnorm.est'], conf_alpha=ALPHA_CONF)
    axs[0].plot(x, mle['mean'], color=color['mle'], label=conf_label['mle'])
    axs[0].fill_between(x, mle['mean.LB'], mle['mean.UB'], color=color['mle'], alpha=0.1)

    shrink = mean_confint(y['shrink.cov.relnorm.est'], conf_alpha=ALPHA_CONF)
    axs[0].plot(x, shrink['mean'], color=color['shrink'], label=conf_label['shrink'])
    axs[0].fill_between(x, shrink['mean.LB'], shrink['mean.UB'], color=color['shrink'], alpha=0.1)

    #fig, axs = plt.subplots(2, 1, sharex=True, figsize=(15,10))
    # === [0,1] Min.Var.Portf. Rel.Err. with conf. ===
    if 0:
        axs[1].set_title('MVP variance est. errors')
        axs[1].set(xlabel='sample size', ylabel='Rel. Err.')
        mle = relerr_mean_confint(y['mle.mvp.var.est'], truevals['mvp.var'], conf_alpha=ALPHA_CONF)
        axs[1].plot(x, mle['relerr.mean'], color=color['mle'], label=conf_label['mle'])
        axs[1].fill_between(x, mle['relerr.LB'], mle['relerr.UB'], color=color['mle'], alpha=0.1)

        shrink = relerr_mean_confint(y['shrink.mvp.var.est'], truevals['mvp.var'], conf_alpha=ALPHA_CONF)
        axs[1].plot(x, shrink['relerr.mean'], color=color['shrink'], label=conf_label['shrink'])
        axs[1].fill_between(x, shrink['relerr.LB'], shrink['relerr.UB'], color=color['shrink'], alpha=0.1)

    # == Relative Rooth Mean Squared Error (RRMSE == RMSRE) ==
    if 0:
        axs[2].set_title('')
        axs[2].set(xlabel='sample size', ylabel='Rel. RMSE')
        rerr_mle = y['mle.mvp.var.est'].MSE(truevals['mvp.var'])**0.5 / truevals['mvp.var']
        axs[2].plot(x, rerr_mle, color=color['mle'], label='mle')
        rerr_shrink = y['shrink.mvp.var.est'].MSE(truevals['mvp.var'])**0.5 / truevals['mvp.var']
        axs[2].plot(x, rerr_shrink, color=color['shrink'], label='shrink')

    # === Variance Relative Unhedged Portfolio. ===
    axs[3].set_title('Actual MVP performance $\\frac{V[MVP]}{V[P]}$', fontsize=15)
    axs[3].set(xlabel='sample size', ylabel='Relative Variance')
    mvp_theoretic_optim = truevals['mvp.var'] / truevals['p.var']
    axs[3].axhline(mvp_theoretic_optim, label='$MVP_{LB}$', color='g')
    axs[3].axhline(1.0, label='P', color='r')
    axs[3].set_ylim(int(mvp_theoretic_optim *10)/10, 1.05)

    mle = rel_mean_confint(y['mle.mvp.var.act'], truevals['p.var'], conf_alpha=ALPHA_CONF)
    axs[3].plot(x, mle['relmean'], color=color['mle'], label='mle')
    axs[3].fill_between(x, mle['relmean.LB'], mle['relmean.UB'], color=color['mle'], alpha=0.1)

    shrink = rel_mean_confint(y['shrink.mvp.var.act'], truevals['p.var'], conf_alpha=ALPHA_CONF)
    axs[3].plot(x, shrink['relmean'], color=color['shrink'], label='shrink')
    axs[3].fill_between(x, shrink['relmean.LB'], shrink['relmean.UB'], color=color['mle'], alpha=0.1)
    
    #rie = rel_mean_confint(y['rie.mvp.var.act'], truevals['p.var'], conf_alpha=ALPHA_CONF)
    #axs[3].plot(x, rie['relmean'], color=color['rie'], label='rie')
    #axs[3].fill_between(x, rie['relmean.LB'], rie['relmean.UB'], color=color['rie'], alpha=0.1)
    
    for ax in axs.flatten():
        ax.legend(fontsize=15)
        ax.grid(True)
        ax.tick_params(labelsize=15)
        ax.xaxis.get_label().set_fontsize(15)
        ax.yaxis.get_label().set_fontsize(15)
    
    plt.show()