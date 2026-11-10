"""Paper S v0.7, worklist item 3: a class of priors on the common-cause dispersion, combined with the trade records.
Likelihood of alpha_c per origin group = profile likelihood of the main model (margins and beta profiled out), on a grid
of 200 points in log alpha_c (0.05 to 1e4). Priors on log alpha_c:
  P1 independence-leaning N(log 20, 1), P2 neutral N(log 2, 1.5^2), P3 strong-common-cause N(log 0.5, 1),
  P4 hierarchical: log alpha_c,g ~ N(nu, tau^2), nu ~ N(log 2, 1.5^2), tau ~ half-N(0, 1) (grid integration).
Posterior of beta_J = 1/(1+alpha_c) per group and prior; posterior predictive P(two or more inputs disrupted, ordinary
month) by the first-order (pairwise) formula, which is additive across groups; lower and upper posterior expectations
over the class. Output: work/app/app_v07_priors.json
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, time
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np
from scipy import stats
import app_model2 as am
from app_model2 import Model2, GROUPS, multi, origins, H1, W, S1_STARTS
import ccffit2 as cf
import ccfmodel as cm

t_start = time.time()
MAIN = json.load(open(f'{W}/app_v07_main.json'))
SCOPE = 'China'
x1 = np.load(f'{W}/app_v07_x.npy')
M1 = Model2('S1', H=H1, scope=SCOPE)
nll1 = float(M1.f(x1))
assert abs(nll1 - MAIN['fits']['nll']['S1']) < 1e-6, (nll1, MAIN['fits']['nll']['S1'])
NG = 200
la_grid = np.linspace(np.log(0.05), np.log(1e4), NG)        # log alpha_c
dla = la_grid[1] - la_grid[0]
ac_grid = np.exp(la_grid); bj_grid = 1 / (1 + ac_grid)

# ---------------------------------------------------------------- 1. profile log-likelihood per group on the grid
ell = {}
for o in multi:
    j = M1.gp[o]
    vals = []
    x_prev = x1.copy()
    for la in la_grid[::-1]:                                   # from independence towards strong dependence, warm starts
        r = M1.fit(x0=x_prev, fixed={j: -la})
        vals.append(-float(r.fun)); x_prev = r.x
    ell[o] = np.array(vals[::-1])                               # aligned with la_grid
    ell[o] -= ell[o].max()
    print(f'{o}: profile done; max at alpha_c = {ac_grid[np.argmax(ell[o])]:.2f}; log-lik drop at 0.05: {ell[o][0]:.2f}, at 1e4: {ell[o][-1]:.2f}  ({time.time() - t_start:.0f}s)')

# ---------------------------------------------------------------- 2. group importance I_g(alpha) on the grid (pairwise formula, ordinary month)
ac_hat = {o: MAIN['alpha_c'][o]['ac_hat'] for o in multi}
live = lambda a: np.isfinite(a) and a < 5e3
shares = MAIN['shares']
mu_hat = {o: np.array([MAIN['mu_S1'][o][s] for s in GROUPS[o][0]]) for o in origins.index}
AP = am.AP_FIXED


def marg_hazard(o, mu):
    a = ac_hat.get(o, np.inf)
    return a * np.log1p(mu / a) if live(a) else mu


def mu_for(o, mu, a):
    m = marg_hazard(o, mu)
    return a * np.expm1(m / a) if live(a) else m


sids_all = [s for o in origins.index for s in GROUPS[o][0]]
inputs2 = {}
for sid in sids_all:
    inputs2.setdefault(sid.split('_')[0], []).append(sid)
h2 = cm.ReliabilityPolynomial(sids_all, inputs=inputs2, shares=shares, k_fail=2, thr=0.5)
p = {s: float(np.exp(-marg_hazard(o, mu1))) for o in origins.index for s, mu1 in zip(GROUPS[o][0], mu_hat[o])}
P_IND = 1 - h2(p)
I = {}
for o in multi:
    vals = []
    for a in ac_grid:
        if a > 5e3:
            vals.append(0.0); continue
        C = cm.margins_and_pair_cov(mu_for(o, mu_hat[o], a), AP, a)[1]
        vals.append(-cm.importance_pairwise(h2, p, [list(GROUPS[o][0])], [C])['total'])
    I[o] = np.array(vals)                                      # increase in P(two or more) due to group o at dispersion a
    print(f'{o}: I_g at alpha_c=0.05 {100 * I[o][0]:.3f} pp, at 0.5 {100 * np.interp(np.log(0.5), la_grid, I[o]):.3f}, at 2 {100 * np.interp(np.log(2), la_grid, I[o]):.3f}, at 20 {100 * np.interp(np.log(20), la_grid, I[o]):.3f}')
check = P_IND + sum(np.interp(np.log(MAIN['alpha_c'][o]['ac_lo96.5']), la_grid, I[o]) for o in multi)
print(f'check additivity: P at the 96.5% corner by sum of I_g {100 * check:.4f}% vs joint pairwise {100 * MAIN["system"]["ordinary"]["most96.5"]["pairwise"]:.4f}%; P_IND {100 * P_IND:.4f}% vs {100 * MAIN["system"]["ordinary"]["independence"]["pairwise"]:.4f}%')

# ---------------------------------------------------------------- 3. priors and posteriors
PRIORS = {'P1': (np.log(20.0), 1.0), 'P2': (np.log(2.0), 1.5), 'P3': (np.log(0.5), 1.0)}
rng = np.random.default_rng(5)
NDRAW = 40000


def summarise_post(w, o):
    """weights on the grid -> beta_J median, 90% interval, mean of alpha; also posterior mean importance"""
    cdf = np.cumsum(w)
    def quant(qv):   # quantile of beta_J (decreasing in alpha): use the alpha quantile mapping
        k = int(np.searchsorted(cdf, qv)); k = min(max(k, 0), len(w) - 1)
        return float(bj_grid[k])
    return dict(bj_median=quant(0.5), bj_lo90=quant(0.95), bj_hi90=quant(0.05), bj_mean=float(np.sum(w * bj_grid)),
                ac_median=float(np.exp(np.sum(w * la_grid))), p_bj_gt_02=float(np.sum(w[bj_grid > 0.2])), p_bj_gt_05=float(np.sum(w[bj_grid > 0.5])),
                I_mean=float(np.sum(w * I[o])))


res = {'grid': dict(n=NG, lo=0.05, hi=1e4), 'P_IND': P_IND, 'posteriors': {}, 'predictive': {}, 'prior_only': {}}
# prior-only summaries (what the prior alone says about beta_J)
for name, (m, s) in PRIORS.items():
    w = stats.norm.pdf(la_grid, m, s); w /= w.sum()
    res['prior_only'][name] = dict(bj_median=float(1 / (1 + np.exp(m))), bj_lo90=float(1 / (1 + np.exp(m + 1.645 * s))), bj_hi90=float(1 / (1 + np.exp(m - 1.645 * s))))
for name, (m, s) in PRIORS.items():
    draws = np.zeros(NDRAW)
    res['posteriors'][name] = {}
    for o in multi:
        lw = ell[o] + stats.norm.logpdf(la_grid, m, s)
        w = np.exp(lw - lw.max()); w /= w.sum()
        res['posteriors'][name][o] = summarise_post(w, o)
        k = rng.choice(NG, size=NDRAW, p=w)
        draws += I[o][k]
    Pd = P_IND + draws
    res['predictive'][name] = dict(mean=float(Pd.mean()), q05=float(np.quantile(Pd, 0.05)), q50=float(np.quantile(Pd, 0.5)), q95=float(np.quantile(Pd, 0.95)))
    print(name, {o: (round(res['posteriors'][name][o]['bj_median'], 3), round(res['posteriors'][name][o]['bj_lo90'], 3), round(res['posteriors'][name][o]['bj_hi90'], 3)) for o in multi},
          'P:', {k: round(100 * v, 3) for k, v in res['predictive'][name].items()})

# P4 hierarchical by grid integration over (nu, tau)
nu_grid = np.linspace(np.log(2) - 6.0, np.log(2) + 6.0, 97); tau_grid = np.linspace(0.02, 4.0, 80)
log_prior_nu = stats.norm.logpdf(nu_grid, np.log(2), 1.5); log_prior_tau = stats.halfnorm.logpdf(tau_grid, 0, 1)
# m_g(nu, tau) = sum_k exp(ell_gk) N(la_k | nu, tau) dla  ->  log
NU, TAU = np.meshgrid(nu_grid, tau_grid, indexing='ij')            # (n_nu, n_tau)
logN = stats.norm.logpdf(la_grid[None, None, :], NU[:, :, None], TAU[:, :, None])   # (n_nu, n_tau, NG)
log_m = {}
for o in multi:
    lw = ell[o][None, None, :] + logN + np.log(dla)
    log_m[o] = np.logaddexp.reduce(lw, axis=2)
log_joint = log_prior_nu[:, None] + log_prior_tau[None, :] + sum(log_m[o] for o in multi)
wj = np.exp(log_joint - log_joint.max()); wj /= wj.sum()
# marginal posteriors of alpha_g and posterior predictive
res['posteriors']['P4'] = {}
res['P4_hyper'] = dict(nu_mean=float(np.sum(wj * NU)), tau_median=float(tau_grid[np.searchsorted(np.cumsum(wj.sum(0)), 0.5)]),
                       tau_q05=float(tau_grid[np.searchsorted(np.cumsum(wj.sum(0)), 0.05)]), tau_q95=float(tau_grid[min(np.searchsorted(np.cumsum(wj.sum(0)), 0.95), len(tau_grid) - 1)]))
flat = wj.ravel()
idx = rng.choice(flat.size, size=NDRAW, p=flat)
inu, itau = np.unravel_index(idx, wj.shape)
draws = np.zeros(NDRAW)
for o in multi:
    cond = np.exp(ell[o][None, None, :] + logN - log_m[o][:, :, None] + np.log(dla))   # conditional posterior of alpha_g given (nu, tau): (n_nu, n_tau, NG)
    cond /= cond.sum(2, keepdims=True)
    wmarg = np.tensordot(wj, cond, axes=([0, 1], [0, 1]))
    wmarg /= wmarg.sum()
    res['posteriors']['P4'][o] = summarise_post(wmarg, o)
    cdf = np.cumsum(cond, axis=2)
    u = rng.random(NDRAW)
    k = np.array([int(np.searchsorted(cdf[a, b], uu)) for a, b, uu in zip(inu, itau, u)])
    k = np.minimum(k, NG - 1)
    draws += I[o][k]
Pd = P_IND + draws
res['predictive']['P4'] = dict(mean=float(Pd.mean()), q05=float(np.quantile(Pd, 0.05)), q50=float(np.quantile(Pd, 0.5)), q95=float(np.quantile(Pd, 0.95)))
print('P4', {o: (round(res['posteriors']['P4'][o]['bj_median'], 3), round(res['posteriors']['P4'][o]['bj_lo90'], 3), round(res['posteriors']['P4'][o]['bj_hi90'], 3)) for o in multi},
      'P:', {k: round(100 * v, 3) for k, v in res['predictive']['P4'].items()}, 'hyper', res['P4_hyper'])
means = {k: res['predictive'][k]['mean'] for k in res['predictive']}
res['lower_upper_expectation'] = dict(lower=min(means.values()), upper=max(means.values()), argmin=min(means, key=means.get), argmax=max(means, key=means.get))
# how much the prior moves the posterior median of beta_J, by group (data-informed vs boundary groups)
res['prior_sensitivity'] = {o: dict(bj_median_range=[min(res['posteriors'][k][o]['bj_median'] for k in res['posteriors']), max(res['posteriors'][k][o]['bj_median'] for k in res['posteriors'])],
                                    prior_median_range=[min(res['prior_only'][k]['bj_median'] for k in PRIORS), max(res['prior_only'][k]['bj_median'] for k in PRIORS)]) for o in multi}
res['profile'] = {o: dict(log_alpha=la_grid.tolist(), ell=ell[o].tolist(), I=I[o].tolist()) for o in multi}
res['elapsed_s'] = time.time() - t_start
json.dump(res, open(f'{W}/app_v07_priors.json', 'w'), indent=1, default=float)
print('lower/upper posterior expectation of P:', {k: round(100 * v, 3) if isinstance(v, float) else v for k, v in res['lower_upper_expectation'].items()})
print('prior sensitivity (beta_J median range across priors):', {o: [round(v, 3) for v in res['prior_sensitivity'][o]['bj_median_range']] for o in multi}, 'prior medians', [round(v, 3) for v in res['prior_sensitivity']['China']['prior_median_range']])
print('done', res['elapsed_s'])
