"""Paper S v0.7, worklist item 2: sampling uncertainty of the stream margins (outer layer of the two-layer report).
Parametric bootstrap of B panels under the main model (S-0 frailty by origin + Lunar New Year covariate on the China
group), activity mask as observed, dispersions at their estimates (Japan independent). In each replicate the margins
and beta are re-estimated with the dispersions fixed; the allowed set Theta (96.5% box of the original fit) is kept.
For each replicate: P(two or more inputs disrupted) in an ordinary month at the least dependent corner (independence,
exact polynomial), at the fitted dispersions and at the most dependent corner (pairwise formula, corrected by the
Monte Carlo / pairwise ratio of the original fit at that corner); 50 replicates checked by Monte Carlo; coarse grid check;
ranking stability of the re-sourcing priorities.
Output: work/app/app_v07_bootstrap.json
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, itertools, time
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
import app_model2 as am
from app_model2 import GROUPS, multi, origins, periods, T, H1, W
import ccffit2 as cf
import ccfmodel as cm
jax.config.update('jax_enable_x64', True)

t_start = time.time()
B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
NMC = int(sys.argv[2]) if len(sys.argv) > 2 else 50
MAIN = json.load(open(f'{W}/app_v07_main.json'))
SCOPE = 'China'
beta_hat = MAIN['fits']['beta']['hat']
ac_hat = {o: MAIN['alpha_c'][o]['ac_hat'] for o in multi}
live = lambda a: np.isfinite(a) and a < 5e3
shares = MAIN['shares']; inputs = MAIN['inputs']
ORDER = list(origins.index)
mu_hat = {o: np.array([MAIN['mu_S1'][o][s] for s in GROUPS[o][0]]) for o in ORDER}
corner_most = {o: MAIN['alpha_c'][o]['ac_lo96.5'] for o in multi}
ratio_corner = MAIN['system']['ordinary']['most96.5']['mc']['p_two'] / MAIN['system']['ordinary']['most96.5']['pairwise']
ratio_fitted = MAIN['system']['ordinary']['fitted']['mc']['p_two'] / MAIN['system']['ordinary']['fitted']['pairwise']
print('MC/pairwise ratio at the most dependent corner', round(ratio_corner, 4), 'at the fitted values', round(ratio_fitted, 4))

# ---------------------------------------------------------------- fixed-shape likelihood (S-0: P(no onset | eta) = exp(-mu eta))
Hj = jnp.asarray(H1, dtype=jnp.float64)
Xs = {o: jnp.asarray(GROUPS[o][1]) for o in ORDER}        # T x m, -1 inactive
m_of = {o: GROUPS[o][1].shape[1] for o in ORDER}
slot = {}; k = 0
for o in ORDER:
    slot[o] = (k, k + m_of[o]); k += m_of[o]
I_BETA = k; NPAR = k + 1
log_thc_fixed = {o: float(-np.log(ac_hat[o])) for o in multi if live(ac_hat[o])}
eta_nodes = {o: cf.eta_logw(np.exp(-log_thc_fixed[o])) for o in log_thc_fixed}     # (s = log eta nodes, log weights)


def ll_origin(X, log_mu, hmul, nodes):
    """X: T x m (-1 masked); log_mu: m; hmul: T multiplier (exp(beta H)); nodes: (s, logw) or None (independent)."""
    mu_t = jnp.exp(log_mu)[None, :] * hmul[:, None]                      # T x m
    if nodes is None:
        lq = -mu_t                                                        # T x m
        l1 = am.log1mexp(lq)
        x1 = (X == 1); x0 = (X == 0)
        return jnp.sum(jnp.where(x1, l1, 0.0) + jnp.where(x0, lq, 0.0))
    s, logw = nodes
    lq = -mu_t[:, :, None] * jnp.exp(s)[None, None, :]                    # T x m x Z
    l1 = am.log1mexp(lq)
    x1 = (X == 1)[:, :, None]; x0 = (X == 0)[:, :, None]
    lp = jnp.sum(jnp.where(x1, l1, 0.0) + jnp.where(x0, lq, 0.0), axis=1)   # T x Z
    return jnp.sum(logsumexp(lp + logw[None, :], axis=1))


def make_negll(Xdict):
    def negll(th):
        tot = 0.0
        hm_china = jnp.exp(th[I_BETA] * Hj); ones = jnp.ones(T)
        for o in ORDER:
            a, b = slot[o]
            hm = hm_china if o == SCOPE else ones
            tot += ll_origin(Xdict[o], th[a:b], hm, eta_nodes.get(o))
        return -tot
    return negll


negll = make_negll(Xs)
f = jax.jit(negll); g = jax.jit(jax.grad(negll))
# generic version taking the data as arguments (one compile for all replicates)
def negll_data(th, *Xlist):
    Xd = dict(zip(ORDER, Xlist))
    return make_negll(Xd)(th)
fd = jax.jit(negll_data); gd = jax.jit(jax.grad(negll_data, argnums=0))


def fit_margins(Xlist, x0):
    bnds = [(-12, 3)] * I_BETA + [(-6, 6)]
    r = minimize(lambda z: float(fd(jnp.asarray(z), *Xlist)), x0, jac=lambda z: np.asarray(gd(jnp.asarray(z), *Xlist)),
                 method='L-BFGS-B', bounds=bnds, options=dict(maxiter=3000, ftol=1e-12, gtol=1e-8))
    return r


# check: the fixed-shape likelihood reproduces the main fit
x_hat = np.zeros(NPAR)
for o in ORDER:
    a, b = slot[o]; x_hat[a:b] = np.log(mu_hat[o])
x_hat[I_BETA] = beta_hat
r_check = fit_margins([Xs[o] for o in ORDER], x_hat)
print(f"check: nll at main-fit margins {float(f(jnp.asarray(x_hat))):.4f}; refit {r_check.fun:.4f} (main {MAIN['fits']['nll']['S1']:.4f}); max |d log mu| {np.abs(r_check.x - x_hat).max():.4f}  ({time.time() - t_start:.0f}s)")


# ---------------------------------------------------------------- system machinery (ordinary month)
def marg_hazard(o, mu):
    a = ac_hat.get(o, np.inf)
    return a * np.log1p(mu / a) if live(a) else mu


def mu_for(o, mu, a):
    m = marg_hazard(o, mu)
    return a * np.expm1(m / a) if live(a) else m


AP = am.AP_FIXED
sids_all = [s for o in ORDER for s in GROUPS[o][0]]
inputs2 = {}
for sid in sids_all:
    inputs2.setdefault(sid.split('_')[0], []).append(sid)
h2 = cm.ReliabilityPolynomial(sids_all, inputs=inputs2, shares=shares, k_fail=2, thr=0.5)


def outputs(mu_rep, ac_map):
    """P(two or more) by the pairwise formula (independence term exact) and the priorities, margins from mu_rep (dict origin -> array)."""
    p = {}
    for o in ORDER:
        for sid, mu1 in zip(GROUPS[o][0], mu_rep[o]):
            p[sid] = float(np.exp(-marg_hazard(o, mu1)))
    base = h2(p)
    groups, covs = [], []
    for o in multi:
        a = ac_map.get(o, np.inf)
        if not live(a):
            continue
        groups.append(list(GROUPS[o][0])); covs.append(cm.margins_and_pair_cov(mu_for(o, mu_rep[o], a), AP, a)[1])
    imp = cm.importance_pairwise(h2, p, groups, covs) if groups else dict(total=0.0, per_group=[], priority={})
    return dict(p_ind=1 - base, p=1 - (base + imp['total']), priority=imp['priority'],
                per_origin={g[0].split('_')[1]: -v for g, v in zip(groups, imp['per_group'])})


def simulate(mu_rep, ac_map, n=400_000, seed=7):
    rng = np.random.default_rng(seed)
    lam = []; grp = []
    for o in ORDER:
        a = ac_map.get(o, np.inf)
        for mu1 in mu_rep[o]:
            if live(a):
                lam.append(mu_for(o, mu1, a)); grp.append(o)
            else:
                lam.append(marg_hazard(o, mu1)); grp.append(None)
    lam = np.array(lam); K = len(lam)
    etas = {o: (rng.gamma(ac_map[o], 1 / ac_map[o], size=n) if live(ac_map.get(o, np.inf)) else np.ones(n)) for o in multi}
    lamt = np.tile(lam, (n, 1))
    for k_ in range(K):
        if grp[k_] is not None:
            lamt[:, k_] *= etas[grp[k_]]
    interrupted = rng.random((n, K)) < -np.expm1(-lamt)
    idx = {s: i for i, s in enumerate(sids_all)}
    nd = np.zeros(n, int)
    for c in inputs:
        cols = [idx[s] for s in sids_all if s.startswith(f'{c}_')]
        w = np.array([shares[sids_all[i]] for i in cols])
        nd += ((interrupted[:, cols] * w).sum(1) > 0.5)
    return float((nd >= 2).mean())


orig = dict(ind=outputs(mu_hat, {}), fit=outputs(mu_hat, ac_hat), most=outputs(mu_hat, corner_most))
print('original: ind', round(orig['ind']['p_ind'], 5), 'fitted pw', round(orig['fit']['p'], 5), 'most pw', round(orig['most']['p'], 5), '(main json:', round(MAIN['system']['ordinary']['most96.5']['pairwise'], 5), ')')


def top_within(pri):
    out = {}
    for o in multi:
        ks = sorted([k for k in pri if k.endswith('_' + o)], key=lambda k: -pri[k])
        out[o] = ks[:2]
    return out


top_orig = {key: top_within(orig[key]['priority']) for key in ('fit', 'most')}
lead_orig = {key: (max(orig[key]['priority'], key=orig[key]['priority'].get) if orig[key]['priority'] else None) for key in ('fit', 'most')}
print('original leading streams within groups (most dependent corner):', top_orig['most'], '| overall', lead_orig['most'])

# ---------------------------------------------------------------- bootstrap
rng = np.random.default_rng(2027)
act = {o: GROUPS[o][1] >= 0 for o in ORDER}
hm_china = np.exp(beta_hat * H1)
rows = []; mc_rows = []; nonconv = 0
for b in range(B):
    Xb = {}
    for o in ORDER:
        a = ac_hat.get(o, np.inf)
        eta = rng.gamma(a, 1 / a, size=T) if live(a) else np.ones(T)
        hm = hm_china if o == SCOPE else np.ones(T)
        pon = 1 - np.exp(-np.outer(eta * hm, mu_hat[o]))
        x = (rng.random(pon.shape) < pon).astype(int)
        x[~act[o]] = -1
        Xb[o] = x
    r = fit_margins([jnp.asarray(Xb[o]) for o in ORDER], x_hat)
    if not r.success:
        nonconv += 1
    mu_b = {o: np.exp(r.x[slot[o][0]:slot[o][1]]) for o in ORDER}
    o_ind = outputs(mu_b, {}); o_fit = outputs(mu_b, ac_hat); o_most = outputs(mu_b, corner_most)
    rec = dict(b=b, beta=float(r.x[I_BETA]), n_onsets=int(sum((Xb[o] == 1).sum() for o in ORDER)),
               L=o_ind['p_ind'], F_pw=o_fit['p'], U_pw=o_most['p'], F=o_fit['p'] * ratio_fitted, U=o_most['p'] * ratio_corner,
               per_origin_most={o: o_most['per_origin'].get(o, 0.0) for o in multi})
    tw = top_within(o_most['priority']); twf = top_within(o_fit['priority'])
    rec['lead_same_most'] = {o: tw[o][:1] == top_orig['most'][o][:1] for o in multi}
    rec['top2_same_most'] = {o: tw[o] == top_orig['most'][o] for o in multi}
    rec['top2set_same_most'] = {o: set(tw[o]) == set(top_orig['most'][o]) for o in multi}
    rec['lead_same_fit'] = {o: twf[o][:1] == top_orig['fit'][o][:1] for o in multi if top_orig['fit'][o]}
    rec['overall_lead_same_most'] = (max(o_most['priority'], key=o_most['priority'].get) == lead_orig['most'])
    if b < NMC:
        mc_most = simulate(mu_b, corner_most, seed=100 + b); mc_fit = simulate(mu_b, ac_hat, seed=200 + b)
        # coarse grid (3 points per origin) of the 96.5% box with the pairwise formula: is the corner the maximum?
        gvals = []
        for combo in itertools.product(*[[corner_most[o], np.sqrt(corner_most[o] * 50.0), np.inf] for o in multi]):
            gvals.append(outputs(mu_b, dict(zip(multi, combo)))['p'])
        rec['mc_most'] = mc_most; rec['mc_fit'] = mc_fit; rec['grid_max_at_corner'] = bool(np.argmax(gvals) == 0); rec['grid_min_at_independence'] = bool(np.argmin(gvals) == len(gvals) - 1)
        mc_rows.append(rec)
    rows.append(rec)
    if b % 100 == 0 or b == B - 1:
        print(f'b={b}: onsets {rec["n_onsets"]} beta {rec["beta"]:.2f} L {100 * rec["L"]:.3f} F {100 * rec["F"]:.3f} U {100 * rec["U"]:.3f}  ({time.time() - t_start:.0f}s)')

R = pd.DataFrame(rows)
q = lambda s, a: float(np.quantile(s, a))
summ = {}
for key in ['L', 'F', 'U', 'F_pw', 'U_pw', 'beta', 'n_onsets']:
    s = R[key].values
    summ[key] = dict(mean=float(s.mean()), sd=float(s.std(ddof=1)), q05=q(s, 0.05), q25=q(s, 0.25), q50=q(s, 0.5), q75=q(s, 0.75), q95=q(s, 0.95))
summ['width'] = dict(mean=float((R.U - R.L).mean()), q05=q(R.U - R.L, 0.05), q95=q(R.U - R.L, 0.95))
summ['per_origin_most'] = {o: dict(q05=q([r[o] for r in R.per_origin_most], 0.05), q50=q([r[o] for r in R.per_origin_most], 0.5), q95=q([r[o] for r in R.per_origin_most], 0.95)) for o in multi}
rank = dict(lead_same_most={o: float(np.mean([r[o] for r in R.lead_same_most])) for o in multi},
            top2_same_most={o: float(np.mean([r[o] for r in R.top2_same_most])) for o in multi},
            top2set_same_most={o: float(np.mean([r[o] for r in R.top2set_same_most])) for o in multi},
            lead_same_fit={o: float(np.mean([r[o] for r in R.lead_same_fit if o in r])) for o in multi if top_orig['fit'][o]},
            overall_lead_same_most=float(R.overall_lead_same_most.mean()), top_orig=top_orig, lead_orig=lead_orig)
mc = {}
if mc_rows:
    M = pd.DataFrame(mc_rows)
    mc = dict(n=len(M), most_pw_corrected_vs_mc=dict(mean_diff=float((M.U - M.mc_most).mean()), max_abs_diff=float((M.U - M.mc_most).abs().max()),
                                                     corr=float(np.corrcoef(M.U, M.mc_most)[0, 1])),
              fit_pw_corrected_vs_mc=dict(mean_diff=float((M.F - M.mc_fit).mean()), max_abs_diff=float((M.F - M.mc_fit).abs().max())),
              mc_most=dict(q05=q(M.mc_most, 0.05), q95=q(M.mc_most, 0.95)), U_same=dict(q05=q(M.U, 0.05), q95=q(M.U, 0.95)),
              grid_max_at_corner=float(M.grid_max_at_corner.mean()), grid_min_at_independence=float(M.grid_min_at_independence.mean()))
OUT = dict(B=B, nonconverged=nonconv, ratio_corner=ratio_corner, ratio_fitted=ratio_fitted, corner_most=corner_most, original=dict(L=orig['ind']['p_ind'], F_pw=orig['fit']['p'], U_pw=orig['most']['p'],
           F=orig['fit']['p'] * ratio_fitted, U=orig['most']['p'] * ratio_corner), summary=summ, ranking=rank, mc_check=mc, elapsed_s=time.time() - t_start,
           replicates=R[['b', 'beta', 'n_onsets', 'L', 'F', 'U']].to_dict('list'))
json.dump(OUT, open(f'{W}/app_v07_bootstrap.json', 'w'), indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ['nonconverged', 'original', 'summary', 'ranking', 'mc_check']}, indent=1, default=float))
print('done', OUT['elapsed_s'])
