"""Paper S v0.7, worklist item 1: is the common cause a calendar or demand effect?
 A. Lunar New Year covariate mu_it = mu_i exp(beta H_t): H = holiday month / holiday month + next; scope China / all.
 B. Supply- vs demand-side onsets: an onset is demand-side when the other-origin import total of the same HS code
    (all partners except the stream's own origin) is below 75% (or 50%) of its own trailing-12-month median.
    Refit with supply-side onsets only (demand-side onsets set to 0).
 C. Combined: holiday covariate + supply-side onsets.
Output: work/app/app_v07_calendar.json, work/app/onset_classification.csv
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, itertools, time
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np, pandas as pd
import app_model2 as am
from app_model2 import Model2, GROUPS, multi, origins, periods, T, H1, H2, W, CUT90, S1_STARTS, profile_interval, summarise_S1
import ccffit2 as cf

OUT = {'lny_months': [str(pd.Timestamp(periods[t]).date())[:7] for t in np.where(H1 == 1)[0]], 'n_H1': int(H1.sum()), 'n_H2': int(H2.sum())}
t_start = time.time()


def fit_pair(groups, H=None, scope=None, label=''):
    """S1 and IND fits with the same covariate; returns dict with estimates, intervals, LRs, beta interval."""
    m1 = Model2('S1', groups=groups, H=H, scope=scope); r1 = m1.fit(starts=S1_STARTS)
    m0 = Model2('IND', groups=groups, H=H, scope=scope); r0 = m0.fit()
    d = dict(nll_S1=float(r1.fun), nll_IND=float(r0.fun), npar_S1=m1.npar(), npar_IND=m0.npar(),
             aic_S1=float(2 * r1.fun + 2 * m1.npar()), aic_IND=float(2 * r0.fun + 2 * m0.npar()))
    d['origins'] = summarise_S1(m1, r1.x, r1.fun, r0.fun)
    if m1.i_beta is not None:
        j = m1.i_beta
        L, U, _ = profile_interval(m1, j, r1.x, r1.fun, cut=CUT90, lo=-6.0, hi=6.0)
        d['beta'] = dict(hat=float(r1.x[j]), lo90=L, hi90=U, lr_zero=float(2 * (m1.fit(x0=r1.x, fixed={j: 0.0}).fun - r1.fun)),
                         hat_IND=float(r0.x[m0.i_beta]))
        # 95% interval too
        L95, U95, _ = profile_interval(m1, j, r1.x, r1.fun, cut=am.CUT95, lo=-6.0, hi=6.0)
        d['beta']['lo95'] = L95; d['beta']['hi95'] = U95
    d['mu_S1'] = {o: dict(zip(groups[o][0], m1.mu(r1.x, o).round(6).tolist())) for o in m1.order}
    d['x_S1'] = r1.x.tolist()
    print(f'[{label}] nll S1 {r1.fun:.3f} IND {r0.fun:.3f} | ' + ' '.join(
        f"{o}: a={d['origins'][o]['ac_hat']:.2f} [{d['origins'][o]['ac_lo']:.3f},{d['origins'][o]['ac_hi']:.1f}] LR {d['origins'][o]['lr_boundary']:.2f}" for o in multi)
          + (f" | beta {d['beta']['hat']:.3f} [{d['beta']['lo90']:.3f},{d['beta']['hi90']:.3f}] LR0 {d['beta']['lr_zero']:.2f}" if 'beta' in d else '')
          + f'  ({time.time() - t_start:.0f}s)')
    return d


# ---------------------------------------------------------------- A. baseline and four holiday variants
OUT['baseline'] = fit_pair(GROUPS, label='baseline')
OUT['holiday'] = {}
for hname, H in [('H1', H1), ('H2', H2)]:
    for scope in ['China', 'all']:
        OUT['holiday'][f'{hname}_{scope}'] = fit_pair(GROUPS, H=H, scope=scope, label=f'{hname} {scope}')

# descriptive: onsets in holiday months by origin
desc = {}
for o in origins.index:
    sids, X = GROUPS[o]
    on = (X == 1); act = (X >= 0)
    for hname, H in [('H1', H1), ('H2', H2)]:
        hm = H == 1
        desc.setdefault(o, {})[hname] = dict(onsets_in=int(on[hm].sum()), active_in=int(act[hm].sum()),
                                             onsets_out=int(on[~hm].sum()), active_out=int(act[~hm].sum()))
    desc[o]['months_ge2_H1'] = int(((on.sum(1) >= 2) & (H1 == 1)).sum()); desc[o]['months_ge2'] = int((on.sum(1) >= 2).sum())
OUT['holiday_descriptives'] = desc
for o in multi:
    d = desc[o]
    r_in = d['H1']['onsets_in'] / d['H1']['active_in']; r_out = d['H1']['onsets_out'] / d['H1']['active_out']
    print(f"{o}: onset rate in LNY months {r_in:.3f} ({d['H1']['onsets_in']}/{d['H1']['active_in']}) vs other months {r_out:.3f} ({d['H1']['onsets_out']}/{d['H1']['active_out']}); months with >=2 onsets in LNY {d['months_ge2_H1']}/{d['months_ge2']}")

# ---------------------------------------------------------------- B. supply- vs demand-side onsets
df = pd.read_csv(f'{W}/korea_imports_long.csv', parse_dates=['period'])
df = df[df.partnerCode != 0]
months = pd.date_range('2014-01-01', '2025-12-01', freq='MS')
usd = df.pivot(index='period', columns=['cmdCode', 'partnerCode'], values='primaryValue').reindex(months)
kg = df.pivot(index='period', columns=['cmdCode', 'partnerCode'], values='netWgt').reindex(months)
has_rec = usd.notna()
uv = (usd / kg.where(kg > 0)).median()
kg_imp = kg.where(kg.notna(), usd / uv)
val = kg_imp.where(has_rec, 0.0).fillna(0.0)
tot_kg = val.T.groupby(level=0).sum().T                      # all-origin net weight by code and month
NAMES_INV = {'Japan': 392, 'China': 156, 'USA': 842, 'Taiwan': 490, 'Germany': 276, 'Singapore': 702, 'Qatar': 634,
             'Canada': 124, 'United Kingdom': 826, 'Belgium': 56, 'Russia': 643, 'Ukraine': 804, 'France': 251}
Eall = am.E.copy()
rows = []
for (sid, o), sub in Eall[Eall.event == 1].groupby(['sid', 'origin']):
    code = int(sid.split('_')[0]); p = NAMES_INV[o]
    other = tot_kg[code] - val[(code, p)]
    med = other.shift(1).rolling(12, min_periods=12).median()
    own_share = val[(code, p)] / tot_kg[code].replace(0, np.nan)
    for t in sub.period:
        rows.append(dict(sid=sid, origin=o, code=code, period=t, other_kg=float(other[t]), other_med12=float(med[t]),
                         ratio=float(other[t] / med[t]) if med[t] > 0 else np.nan, own_share_t=float(own_share[t]) if not np.isnan(own_share[t]) else np.nan,
                         own_share_12=float(val[(code, p)].shift(1).rolling(12).sum()[t] / tot_kg[code].shift(1).rolling(12).sum()[t])))
CL = pd.DataFrame(rows)
CL['demand75'] = CL.ratio < 0.75
CL['demand50'] = CL.ratio < 0.50
CL.to_csv(f'{W}/onset_classification.csv', index=False)
OUT['classification'] = dict(n_onsets=int(len(CL)), demand75=int(CL.demand75.sum()), demand50=int(CL.demand50.sum()),
                             by_origin={o: dict(n=int(len(g)), demand75=int(g.demand75.sum()), demand50=int(g.demand50.sum()))
                                        for o, g in CL.groupby('origin')},
                             ratio_quantiles={q: float(CL.ratio.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})
# how many of the co-onset months (>= 2 onsets in a group) lose onsets
co = {}
for o in multi:
    g = CL[CL.origin == o]
    per_m = g.groupby('period').agg(n=('sid', 'size'), d75=('demand75', 'sum'), d50=('demand50', 'sum'))
    co[o] = dict(months_ge2=int((per_m.n >= 2).sum()), months_ge2_after75=int(((per_m.n - per_m.d75) >= 2).sum()),
                 months_ge2_after50=int(((per_m.n - per_m.d50) >= 2).sum()),
                 lny_onsets_demand75=int(g[g.period.isin([periods[t] for t in np.where(H1 == 1)[0]])].demand75.sum()),
                 lny_onsets=int(g.period.isin([periods[t] for t in np.where(H1 == 1)[0]]).sum()))
OUT['classification']['co_onset_months'] = co
print('classification:', json.dumps({k: v for k, v in OUT['classification'].items() if k != 'co_onset_months'}, indent=0))
print('co-onset months:', co)


def groups_supply(thr_col):
    """event matrices with demand-side onsets (per thr_col) set to 0"""
    E2 = Eall.copy()
    dem = CL[CL[thr_col]][['sid', 'period']]
    key = set(zip(dem.sid, dem.period))
    mask = [(s, t) in key for s, t in zip(E2.sid, E2.period)]
    E2.loc[mask, 'event'] = 0
    return {o: am.group_matrix(o, E2) for o in origins.index}, E2


OUT['supply'] = {}
for thr in ['demand75', 'demand50']:
    G, E2 = groups_supply(thr)
    OUT['supply'][thr] = fit_pair(G, label=f'supply-side ({thr})')
    OUT['supply'][thr]['n_onsets'] = int((E2.event == 1).sum())
    OUT['supply'][thr]['onsets_by_origin'] = {o: int(((E2.origin == o) & (E2.event == 1)).sum()) for o in multi}
    # C. combined with the holiday covariate (China scope and all scope, H1)
    for scope in ['China', 'all']:
        OUT['supply'][thr][f'H1_{scope}'] = fit_pair(G, H=H1, scope=scope, label=f'supply-side ({thr}) + H1 {scope}')

OUT['elapsed_s'] = time.time() - t_start
json.dump(OUT, open(f'{W}/app_v07_calendar.json', 'w'), indent=1, default=float)
print('done', OUT['elapsed_s'])
