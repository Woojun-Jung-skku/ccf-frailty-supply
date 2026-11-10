"""Paper S v0.7: the application under the main model
   S-0 frailty per origin group (item dispersion at its Poisson limit) + Lunar New Year covariate on the China group,
   mu_it = mu_i exp(beta H_t), binary onset likelihood.
Computes everything Sections 5.2-5.5 report:
 1. fits (S1, IND, BFR with the same covariate), AIC, per-origin profile curves and bounds (90/95/96.5%), beta profile
 2. period diagnostic c_gt at the fitted values and at the 90% lower bounds
 3. pairwise co-onset pattern (parametric bootstrap) by origin
 4. system outputs (ordinary month and Lunar New Year month): Monte Carlo and pairwise formula at independence, the
    fitted values and the corners of the allowed boxes; per-origin importance; re-sourcing priorities (first-order and
    Monte Carlo with common random numbers); grid scan of the 96.5% box; ranking over the grid
 5. quarterly refit
Output: work/app/app_v07_main.json, work/app/app_v07_diagnostic.csv, work/app/app_v07_x.npy
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, itertools, time
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np, pandas as pd
from scipy import stats
import app_model2 as am
from app_model2 import Model2, GROUPS, multi, origins, periods, T, H1, W, S1_STARTS, BFR_STARTS, profile_interval
import ccffit2 as cf
import ccfmodel as cm

t_start = time.time()
OUT = {}
LEVELS = {'90': 0.90, '95': 0.95, '96.5': 0.965}
CUT = {k: float(stats.chi2.ppf(v, 1)) for k, v in LEVELS.items()}
OUT['cutoffs'] = CUT
AP = am.AP_FIXED
SCOPE = 'China'

# ---------------------------------------------------------------- 1. fits
M1 = Model2('S1', H=H1, scope=SCOPE); r1 = M1.fit(starts=S1_STARTS)
M0 = Model2('IND', H=H1, scope=SCOPE); r0 = M0.fit()
MB = Model2('BFR', H=H1, scope=SCOPE); rb = MB.fit(starts=BFR_STARTS)
x1 = r1.x; nll1 = float(r1.fun)
np.save(f'{W}/app_v07_x.npy', x1)
beta = float(x1[M1.i_beta])
fits = dict(nll={'S1': nll1, 'IND': float(r0.fun), 'BFR': float(rb.fun)}, npar={'S1': M1.npar(), 'IND': M0.npar(), 'BFR': MB.npar()})
fits['aic'] = {k: 2 * fits['nll'][k] + 2 * fits['npar'][k] for k in fits['nll']}
fits['lr_all'] = 2 * (float(r0.fun) - nll1)
fits['beta'] = dict(hat=beta, hat_IND=float(r0.x[M0.i_beta]))
L, U, bcurve = profile_interval(M1, M1.i_beta, x1, nll1, cut=CUT['90'], lo=-6, hi=6, grid=np.linspace(-0.5, 2.5, 31))
fits['beta'].update(lo90=L, hi90=U, lr_zero=float(2 * (M1.fit(x0=x1, fixed={M1.i_beta: 0.0}).fun - nll1)), curve=bcurve)
L95, U95, _ = profile_interval(M1, M1.i_beta, x1, nll1, cut=CUT['95'], lo=-6, hi=6)
fits['beta'].update(lo95=L95, hi95=U95)
fits['bfr'] = {o: dict(w=float(1 / (1 + np.exp(-rb.x[MB.gp[o][0]]))), p=float(1 / (1 + np.exp(-rb.x[MB.gp[o][1]])))) for o in multi}
print(f"fits: nll S1 {nll1:.3f} IND {r0.fun:.3f} BFR {rb.fun:.3f}; AIC {fits['aic']}; beta {beta:.3f} [{L:.3f}, {U:.3f}] ({time.time() - t_start:.0f}s)")

# profile curves and bounds per origin
grid = np.linspace(cf.LOG_TH_MIN, np.log(50.0), 48)
prof = {}
for o in multi:
    j = M1.gp[o]
    curve = [float(2 * (M1.fit(x0=x1, fixed={j: v}).fun - nll1)) for v in grid]
    d = dict(ac_hat=M1.alpha_c(x1, o), log_thc=grid.tolist(), lr=curve, lr_boundary=curve[0],
             n_streams=int(origins[o]), n_onsets=int((GROUPS[o][1] == 1).sum()),
             months_ge2=int(((GROUPS[o][1] == 1).sum(1) >= 2).sum()))
    for lev, c in CUT.items():
        Lo, Up, _ = profile_interval(M1, j, x1, nll1, cut=c)
        d[f'ac_lo{lev}'] = float(np.exp(-Up))
        d[f'ac_hi{lev}'] = float(np.exp(-Lo)) if Lo > cf.LOG_TH_MIN + 1e-6 else float('inf')
    prof[o] = d
    print(f"{o}: ac_hat {d['ac_hat']:.3f}; 90 [{d['ac_lo90']:.3f}, {d['ac_hi90']:.2f}] 95 [{d['ac_lo95']:.3f}, {d['ac_hi95']:.2f}] 96.5 [{d['ac_lo96.5']:.3f}, {d['ac_hi96.5']:.2f}]; LR {d['lr_boundary']:.2f}; onsets {d['n_onsets']} months>=2 {d['months_ge2']}")
OUT['fits'] = fits; OUT['alpha_c'] = prof
ac_hat = {o: prof[o]['ac_hat'] for o in multi}
ac_lo90 = {o: prof[o]['ac_lo90'] for o in multi}
mu_hat = {o: M1.mu(x1, o) for o in origins.index}
OUT['mu_S1'] = {o: dict(zip(GROUPS[o][0], mu_hat[o].round(6).tolist())) for o in origins.index}
OUT['mu_IND'] = {o: dict(zip(GROUPS[o][0], M0.mu(r0.x, o).round(6).tolist())) for o in origins.index}


def live(a):
    return np.isfinite(a) and a < 5e3


def hol(o, t_mask):
    """multiplier on the intensity in the months t_mask (covariate scope)"""
    return np.where(t_mask & (o == SCOPE), np.exp(beta), 1.0)


def marg_hazard(o, mu):
    """-log P(no onset) under the fitted dispersion of origin o"""
    a = ac_hat.get(o, np.inf)
    return a * np.log1p(mu / a) if live(a) else mu


def mu_for(o, mu, a):
    """stream means that keep the fitted marginal onset probabilities when origin o has dispersion a"""
    m = marg_hazard(o, mu)
    return a * np.expm1(m / a) if live(a) else m


# ---------------------------------------------------------------- 2. period diagnostic c_gt = P(eta_gt > 3 | x_gt)
def post_exceed(mu, x, ac, thr=3.0, n=120):
    if not live(ac):
        return 0.0
    eta, w = cm.eta_nodes(ac, n)
    q = cm.q_cond(mu, eta, AP)
    lik = np.prod(np.where(x[:, None] == 1, 1 - q, np.where(x[:, None] == 0, q, 1.0)), axis=0)
    post = w * lik
    return float(post[eta > thr].sum() / post.sum())


diag = []
for o in multi:
    sids, X = GROUPS[o]
    mu = mu_hat[o]
    mu_lo = mu_for(o, mu, ac_lo90[o])
    for t, per in enumerate(periods):
        h = np.exp(beta) if (H1[t] == 1 and o == SCOPE) else 1.0
        xrow = X[t]
        diag.append(dict(origin=o, period=str(pd.Timestamp(per).date()), k=int((xrow == 1).sum()), active=int((xrow >= 0).sum()),
                         lny=int(H1[t]), c_hat=post_exceed(mu * h, xrow, ac_hat[o]), c_lo=post_exceed(mu_lo * h, xrow, ac_lo90[o]),
                         streams=';'.join(s for s, v in zip(sids, xrow) if v == 1)))
D = pd.DataFrame(diag)
D.to_csv(f'{W}/app_v07_diagnostic.csv', index=False)
OUT['diag'] = dict(top_hat=D.sort_values('c_hat', ascending=False).head(8)[['origin', 'period', 'k', 'lny', 'c_hat', 'c_lo', 'streams']].to_dict('records'),
                   flagged_lo=D[D.c_lo >= 0.5].groupby('origin').size().to_dict(),
                   flagged_lo_rows=D[D.c_lo >= 0.5][['origin', 'period', 'k', 'lny', 'c_hat', 'c_lo', 'streams']].to_dict('records'),
                   flagged_hat=D[D.c_hat >= 0.5].groupby('origin').size().to_dict(),
                   july2019=D[(D.origin == 'Japan') & (D.period == '2019-07-01')][['k', 'c_hat', 'c_lo', 'streams']].to_dict('records'))
print('diagnostic: flagged at fitted', OUT['diag']['flagged_hat'], 'at 90% lower bound', OUT['diag']['flagged_lo'])
print('  top at fitted:', [(r['origin'], r['period'][:7], r['k'], round(r['c_hat'], 2)) for r in OUT['diag']['top_hat'][:5]])
print('  flagged at bound:', [(r['origin'], r['period'][:7], r['k'], r['lny']) for r in OUT['diag']['flagged_lo_rows']])
print('  July 2019 Japan:', OUT['diag']['july2019'])

# ---------------------------------------------------------------- 3. pairwise co-onset pattern by origin (parametric bootstrap)
rng = np.random.default_rng(11)
B = 4000
conc = {}
for o in multi:
    sids, X = GROUPS[o]
    act = X >= 0; xo = (X == 1) & act
    m = len(sids)
    pairs = list(itertools.combinations(range(m), 2)); triples = list(itertools.combinations(range(m), 3))

    def stats_of(xb):
        N = np.array([(xb[:, a] & xb[:, b]).sum() for a, b in pairs])
        tot = N.sum()
        t3 = max((xb[:, a] & xb[:, b] & xb[:, c]).sum() for a, b, c in triples)
        return int(tot), N, int(N.max()), int(t3)
    tot_o, N_o, t2_o, t3_o = stats_of(xo)
    out = dict(streams=list(sids), n_pairs=len(pairs), observed_total=tot_o, observed_max_pair=t2_o, observed_max_triple=t3_o,
               n_pairs_with_coonset=int((N_o > 0).sum()),
               observed_pairs={f'{sids[a]}|{sids[b]}': int(n) for (a, b), n in zip(pairs, N_o) if n > 0},
               expected_total_independence=None)
    # expectation under independence with the fitted margins (hazard preserved)
    mu = mu_hat[o]
    hmul = hol(o, H1 == 1)
    a_hat = ac_hat[o]
    mu_t = np.outer(hmul, mu)                                        # month-specific intensities mu_i exp(beta H_t)
    p_on = (1 - (1 + mu_t / a_hat) ** (-a_hat)) if live(a_hat) else (1 - np.exp(-mu_t))   # marginal onset probability under the fitted model
    out['expected_total_independence'] = float(sum((p_on[:, a] * p_on[:, b] * (act[:, a] & act[:, b])).sum() for a, b in pairs))
    out['observed_in_holiday_months'] = int(sum(int((xo[H1 == 1][:, a] & xo[H1 == 1][:, b]).sum()) for a, b in pairs))
    out['expected_independence_in_holiday_months'] = float(sum((p_on[H1 == 1][:, a] * p_on[H1 == 1][:, b] * (act[H1 == 1][:, a] & act[H1 == 1][:, b])).sum() for a, b in pairs))
    scen = [('fitted', ac_hat[o]), ('lower90', ac_lo90[o]), ('ac1', 1.0), ('ac2', 2.0)]
    for name, a in scen:
        mu_a = mu_for(o, mu, a) if live(a) else marg_hazard(o, mu)
        tots, t2s, t3s = [], [], []
        for b in range(B):
            eta = rng.gamma(a, 1 / a, size=T) if live(a) else np.ones(T)
            pon = 1 - np.exp(-np.outer(eta * hmul, mu_a))
            xb = (rng.random(X.shape) < pon) & act
            t, _, t2, t3 = stats_of(xb)
            tots.append(t); t2s.append(t2); t3s.append(t3)
        tots = np.array(tots); t2s = np.array(t2s); t3s = np.array(t3s)
        out[name] = dict(alpha_c=float(a), expected_total=float(tots.mean()), p_total_le_obs=float(np.mean(tots <= tot_o)),
                         p_total_ge_obs=float(np.mean(tots >= tot_o)), p_max_pair_ge_obs=float(np.mean(t2s >= t2_o)),
                         p_max_triple_ge_obs=float(np.mean(t3s >= t3_o)), p_triple_repeat=float(np.mean(t3s >= 2)),
                         max_triple_dist={str(k): float(np.mean(t3s == k)) for k in range(0, 5)})
    conc[o] = out
    print(f"{o}: pair co-onsets {tot_o} over {out['n_pairs_with_coonset']} pairs; max pair {t2_o}; max triple {t3_o}; indep. expectation {out['expected_total_independence']:.1f}")
    for name, _ in scen:
        d = out[name]
        print(f"   {name}: ac={d['alpha_c']:.2f} E[total] {d['expected_total']:.1f} P(total<=obs) {d['p_total_le_obs']:.3f} P(total>=obs) {d['p_total_ge_obs']:.3f} P(triple repeat) {d['p_triple_repeat']:.3f} P(max triple>=obs) {d['p_max_triple_ge_obs']:.3f}")
OUT['concentration'] = conc

# ---------------------------------------------------------------- 4. system
df = pd.read_csv(f'{W}/korea_imports_long.csv', parse_dates=['period'])
df = df[(df.partnerCode != 0) & (df.period >= '2015-01-01')]
NAMES_INV = {'Japan': 392, 'China': 156, 'USA': 842, 'Taiwan': 490, 'Germany': 276, 'Singapore': 702, 'Qatar': 634,
             'Canada': 124, 'United Kingdom': 826, 'Belgium': 56, 'Russia': 643, 'Ukraine': 804, 'France': 251}
tot_val = df.groupby('cmdCode').primaryValue.sum()
shares = {}
for o in origins.index:
    for sid in GROUPS[o][0]:
        code = int(sid.split('_')[0])
        v = df[(df.cmdCode == code) & (df.partnerCode == NAMES_INV[o])].primaryValue.sum()
        shares[sid] = float(v / tot_val[code])
inputs = sorted({int(s.split('_')[0]) for o in origins.index for s in GROUPS[o][0]})
OUT['shares'] = shares; OUT['inputs'] = inputs
all_streams = []                                 # (sid, origin, mu_S1, mu_IND)
for o in origins.index:
    for j, sid in enumerate(GROUPS[o][0]):
        all_streams.append((sid, o, float(mu_hat[o][j]), float(M0.mu(r0.x, o)[j])))
sid_idx = {s[0]: k for k, s in enumerate(all_streams)}
K = len(all_streams)
in_cols = {c: [sid_idx[s] for s in shares if s.startswith(f'{c}_')] for c in inputs}
in_w = {c: np.array([shares[all_streams[k][0]] for k in in_cols[c]]) for c in inputs}


def hazards(ac_map, month='ordinary', moved=()):
    """per-stream Poisson rates lambda_i and the origin whose frailty multiplies them (None for independent streams).
    Margins are held at their fitted values for the month type; streams in `moved` leave their group (own frailty: none)."""
    lam = np.zeros(K); grp = [None] * K
    hm = np.exp(beta) if month == 'lny' else 1.0
    for k, (sid, o, mu1, mui) in enumerate(all_streams):
        h = hm if o == SCOPE else 1.0
        a = ac_map.get(o, np.inf)
        if sid in moved or not live(a):
            lam[k] = marg_hazard(o, mu1 * h); grp[k] = None
        else:
            lam[k] = mu_for(o, mu1 * h, a); grp[k] = o
    return lam, grp


def simulate(ac_map, n=400_000, seed=7, month='ordinary', moved=(), return_nd=False):
    rng = np.random.default_rng(seed)
    etas = {o: (rng.gamma(ac_map[o], 1 / ac_map[o], size=n) if live(ac_map.get(o, np.inf)) else np.ones(n)) for o in multi}
    u = rng.random((n, K))
    lam, grp = hazards(ac_map, month, moved)
    lamt = np.tile(lam, (n, 1))
    for k in range(K):
        if grp[k] is not None:
            lamt[:, k] *= etas[grp[k]]
    interrupted = u < -np.expm1(-lamt)
    dis = np.zeros((n, len(inputs)), bool)
    for j, c in enumerate(inputs):
        dis[:, j] = (interrupted[:, in_cols[c]] * in_w[c]).sum(1) > 0.5
    nd = dis.sum(1)
    res = dict(p_any=float((nd >= 1).mean()), p_two=float((nd >= 2).mean()), p_two_3m=float(1 - (1 - (nd >= 2).mean()) ** 3),
               se_two=float(np.sqrt((nd >= 2).mean() * (1 - (nd >= 2).mean()) / n)))
    return (res, nd) if return_nd else res


# pairwise formula
def pairwise(ac_map, month='ordinary'):
    hm = np.exp(beta) if month == 'lny' else 1.0
    p = {}
    for sid, o, mu1, mui in all_streams:
        h = hm if o == SCOPE else 1.0
        p[sid] = float(np.exp(-marg_hazard(o, mu1 * h)))
    inputs2 = {}
    for sid in p:
        inputs2.setdefault(sid.split('_')[0], []).append(sid)
    h2 = cm.ReliabilityPolynomial(list(p), inputs=inputs2, shares=shares, k_fail=2, thr=0.5)
    h1 = cm.ReliabilityPolynomial(list(p), inputs=inputs2, shares=shares, k_fail=1, thr=0.5)
    groups, covs = [], []
    for o in multi:
        a = ac_map.get(o, np.inf)
        if not live(a):
            continue
        sids = GROUPS[o][0]
        h = hm if o == SCOPE else 1.0
        mu_a = mu_for(o, mu_hat[o] * h, a)
        pg, C = cm.margins_and_pair_cov(mu_a, AP, a)
        assert np.allclose(pg, [p[s] for s in sids], atol=3e-4), (o, pg[:3], [p[s] for s in sids][:3])
        groups.append(list(sids)); covs.append(C)
    out = {}
    for name, hh in [('two', h2), ('any', h1)]:
        base = hh(p)
        imp = cm.importance_pairwise(hh, p, groups, covs) if groups else dict(total=0.0, per_group=[], priority={})
        out[name] = dict(independence=1 - base, p=1 - (base + imp['total']), priority=imp['priority'],
                         per_origin={g[0].split('_')[1]: -v for g, v in zip(groups, imp['per_group'])})
    return out


system = {}
for month in ['ordinary', 'lny']:
    S = {}
    ind_map = {}
    S['independence'] = dict(mc=simulate(ind_map, month=month), pairwise=pairwise(ind_map, month)['two']['independence'],
                             pairwise_any=pairwise(ind_map, month)['any']['independence'])
    pw_hat = pairwise(ac_hat, month)
    S['fitted'] = dict(ac=ac_hat, mc=simulate(ac_hat, month=month), pairwise=pw_hat['two']['p'], pairwise_any=pw_hat['any']['p'],
                       priority=pw_hat['two']['priority'], per_origin=pw_hat['two']['per_origin'])
    for lev in LEVELS:
        lo = {o: prof[o][f'ac_lo{lev}'] for o in multi}
        hi = {o: prof[o][f'ac_hi{lev}'] for o in multi}
        pw_lo = pairwise(lo, month); pw_hi = pairwise(hi, month)
        S[f'most{lev}'] = dict(ac=lo, mc=simulate(lo, month=month), pairwise=pw_lo['two']['p'], pairwise_any=pw_lo['any']['p'],
                               priority=pw_lo['two']['priority'], per_origin=pw_lo['two']['per_origin'])
        S[f'least{lev}'] = dict(ac={o: (v if np.isfinite(v) else None) for o, v in hi.items()}, mc=simulate(hi, month=month),
                                pairwise=pw_hi['two']['p'], pairwise_any=pw_hi['any']['p'])
        print(f"[{month}] {lev}%: most dependent MC {S[f'most{lev}']['mc']['p_two']:.5f} pw {pw_lo['two']['p']:.5f} | least MC {S[f'least{lev}']['mc']['p_two']:.5f} pw {pw_hi['two']['p']:.5f} | any MC least {S[f'least{lev}']['mc']['p_any']:.4f} most {S[f'most{lev}']['mc']['p_any']:.4f}")
    print(f"[{month}] independence MC {S['independence']['mc']['p_two']:.5f} pw {S['independence']['pairwise']:.5f}; fitted MC {S['fitted']['mc']['p_two']:.5f} pw {S['fitted']['pairwise']:.5f}")
    # per-origin importance by Monte Carlo at the 96.5% most dependent corner (common random numbers) and at the fitted values
    for key in ['most96.5', 'fitted']:
        acm = S[key]['ac']
        base, nd0 = simulate(acm, month=month, return_nd=True)
        imp_mc = {}
        for o in multi:
            if not live(acm.get(o, np.inf)):
                imp_mc[o] = 0.0; continue
            acm_o = dict(acm); acm_o[o] = np.inf
            _, nd1 = simulate(acm_o, month=month, return_nd=True)
            imp_mc[o] = float(((nd0 >= 2).mean() - (nd1 >= 2).mean()))
        S[key]['per_origin_mc'] = imp_mc
        # Monte Carlo re-sourcing priorities with common random numbers: move one stream out of its group
        pri_mc = {}
        for sid, o, mu1, mui in all_streams:
            if o in multi and live(acm.get(o, np.inf)):
                _, nd1 = simulate(acm, month=month, moved=(sid,), return_nd=True)
                pri_mc[sid] = float((nd0 >= 2).mean() - (nd1 >= 2).mean())
        S[key]['priority_mc'] = pri_mc
        top = sorted(S[key]['priority'].items(), key=lambda kv: -kv[1])[:7]
        print(f"[{month}] {key}: per-origin importance MC {imp_mc} pw {S[key]['per_origin']}")
        print(f"   top priorities pw {[(k, round(100 * v, 3)) for k, v in top]}")
        print(f"   same streams MC {[(k, round(100 * pri_mc.get(k, 0), 3)) for k, _ in top]}")
        print(f"   MC order {[(k, round(100 * v, 3)) for k, v in sorted(pri_mc.items(), key=lambda kv: -kv[1])[:7]]}  ({time.time() - t_start:.0f}s)")
    system[month] = S
OUT['system'] = system

# grid scan of the 96.5% box (pairwise formula), ordinary month: extremes and ranking
grid_ac = {}
for o in multi:
    lo, hi = prof[o]['ac_lo96.5'], prof[o]['ac_hi96.5']
    hi_eff = hi if np.isfinite(hi) else 1e4
    grid_ac[o] = list(np.exp(np.linspace(np.log(lo), np.log(hi_eff), 7)))
pts = []; within = {o: set() for o in multi}; lead2 = {}; lead1 = {}
for combo in itertools.product(*[grid_ac[o] for o in multi]):
    acm = {o: (a if a < 5e3 else np.inf) for o, a in zip(multi, combo)}
    pw = pairwise(acm, 'ordinary')
    pr = pw['two']['priority']
    pts.append(dict(ac=[float(a) for a in combo], p_two=pw['two']['p'], p_any=pw['any']['p']))
    for o in multi:
        ks = [k for k in pr if k.endswith('_' + o)]
        if ks:
            within[o].add(tuple(sorted(ks, key=lambda k: -pr[k])))
    srt = [k for k, _ in sorted(pr.items(), key=lambda kv: -kv[1])]
    if srt:
        lead1[srt[0]] = lead1.get(srt[0], 0) + 1
        lead2[' & '.join(srt[:2])] = lead2.get(' & '.join(srt[:2]), 0) + 1
p2 = np.array([q['p_two'] for q in pts]); p1 = np.array([q['p_any'] for q in pts])
OUT['grid'] = dict(n=len(pts), p_two_min=float(p2.min()), p_two_max=float(p2.max()), at_min=pts[int(p2.argmin())]['ac'], at_max=pts[int(p2.argmax())]['ac'],
                   p_any_min=float(p1.min()), p_any_max=float(p1.max()),
                   within_origin_orders={o: len(within[o]) for o in multi}, leading_stream_counts=lead1, leading_pair_counts=lead2)
print('grid 96.5%:', {k: v for k, v in OUT['grid'].items() if k not in ('leading_pair_counts',)})
print('   leading pairs', lead2)

# ---------------------------------------------------------------- 5. quarterly refit (S-0, Q1 covariate for the China group)
Eq = am.E.copy()
Eq['q'] = Eq.period.dt.to_period('Q').dt.start_time
Eq = (Eq.assign(ev1=(Eq.event == 1).astype(int), act=Eq.active.astype(int))
      .groupby(['q', 'sid', 'origin', 'code', 'input']).agg(ev=('ev1', 'max'), act=('act', 'max')).reset_index().rename(columns={'q': 'period'}))
Eq['event'] = np.where(Eq.act == 1, Eq.ev, -1); Eq['active'] = Eq.act == 1
qperiods = sorted(Eq.period.unique())


def qmatrix(o):
    sub = Eq[Eq.origin == o]; sids = sorted(sub.sid.unique())
    X = sub.pivot(index='period', columns='sid', values='event').reindex(qperiods)[sids].fillna(-1).astype(int).values
    return sids, X


GQ = {o: qmatrix(o) for o in origins.index}
HQ = np.array([1 if pd.Timestamp(q).month == 1 else 0 for q in qperiods])
MQ1 = Model2('S1', groups=GQ, H=HQ, scope=SCOPE); rq1 = MQ1.fit(starts=S1_STARTS)
MQ0 = Model2('IND', groups=GQ, H=HQ, scope=SCOPE); rq0 = MQ0.fit()
q = dict(T=len(qperiods), nll_S1=float(rq1.fun), nll_IND=float(rq0.fun), aic_S1=2 * float(rq1.fun) + 2 * MQ1.npar(), aic_IND=2 * float(rq0.fun) + 2 * MQ0.npar(),
         beta=float(rq1.x[MQ1.i_beta]), origins={})
for o in multi:
    j = MQ1.gp[o]
    Lo, Up, _ = profile_interval(MQ1, j, rq1.x, rq1.fun, cut=CUT['90'])
    q['origins'][o] = dict(ac_hat=MQ1.alpha_c(rq1.x, o), ac_lo90=float(np.exp(-Up)), ac_hi90=float(np.exp(-Lo)) if Lo > cf.LOG_TH_MIN + 1e-6 else float('inf'),
                           lr_boundary=float(2 * (MQ1.fit(x0=rq1.x, fixed={j: cf.LOG_TH_MIN}).fun - rq1.fun)))
OUT['quarterly'] = q
print('quarterly:', {o: {k: round(v, 3) for k, v in q['origins'][o].items()} for o in multi}, 'dAIC', q['aic_S1'] - q['aic_IND'])

OUT['elapsed_s'] = time.time() - t_start
json.dump(OUT, open(f'{W}/app_v07_main.json', 'w'), indent=1, default=float)
print('done', OUT['elapsed_s'])
