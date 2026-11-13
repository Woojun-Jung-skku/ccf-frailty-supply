"""Sensitivity of the case study to the onset-detection rule, and the onset rate by calendar year
(Section 5.1 and Supplementary S8, Table S7).

The rule of Section 5.1 records an onset when the net weight of an active stream falls below THRESH times its
trailing WINDOW-month median after a month in which it did not. The main analysis uses THRESH = 0.5 and WINDOW = 12.
This script rebuilds the panel for other thresholds and windows and repeats, for each variant,
 1. the fits of the main model (S-0 frailty per origin group, Lunar New Year covariate on the China group) and of
    independence with the same covariate: Akaike criteria, calendar coefficient, per-origin dispersions with their 90%
    profile intervals, 96.5% lower bounds and likelihood ratios against independence;
 2. the ordinary-month probability that two or more inputs are disrupted, by Monte Carlo (400,000 months) under
    independence, at the fitted dispersions and at the most dependent corner of the 96.5% box, margins held at their
    fitted values (the definitions of app_v07_main.py).
With THRESH = 0.5 and WINDOW = 12 the script must reproduce the archived panel and the archived results exactly; it
stops if it does not.

The variant panels are built from the import table work/app/korea_imports_long.csv (not redistributed, see README).
Their onset indicators are archived in work/app/app_v08_variants.csv, and the script reads that file when the import
table is absent.

Output: work/app/app_v08_detection.json, work/app/app_v08_variants.csv
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, time
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np, pandas as pd
from scipy import stats
import app_model2 as am
from app_model2 import Model2, multi, origins, periods, H1, W, S1_STARTS, profile_interval
import ccffit2 as cf

t_start = time.time()
VARIANTS = [(0.3, 12), (0.4, 12), (0.5, 12), (0.6, 12), (0.7, 12), (0.5, 6), (0.5, 18), (0.5, 24)]   # (THRESH, WINDOW)
MAIN = (0.5, 12)
CUT90 = float(stats.chi2.ppf(0.90, 1))
CUT965 = float(stats.chi2.ppf(0.965, 1))
SCOPE = 'China'
NAMES = {392: 'Japan', 156: 'China', 842: 'USA', 490: 'Taiwan', 276: 'Germany', 702: 'Singapore', 634: 'Qatar',
         124: 'Canada', 826: 'United Kingdom', 56: 'Belgium', 643: 'Russia', 804: 'Ukraine', 251: 'France',
         704: 'Viet Nam', 699: 'India', 344: 'Hong Kong', 458: 'Malaysia'}
OUT = {}


def key(v):
    return f'q{int(round(100 * v[0]))}_w{v[1]}'


# ---------------------------------------------------------------- 0. onset rate by calendar year (main panel)
A = am.E[am.E.active]
by_year = A.assign(year=A.period.dt.year).groupby('year').agg(active=('event', 'size'), onsets=('event', 'sum'))
pan = A.period.dt.year.between(2020, 2022)
OUT['by_year'] = {int(y): dict(active=int(r.active), onsets=int(r.onsets), rate=float(r.onsets / r.active)) for y, r in by_year.iterrows()}
OUT['pandemic'] = dict(years='2020-2022', active=int(pan.sum()), onsets=int(A[pan].event.sum()), rate=float(A[pan].event.mean()),
                       other_active=int((~pan).sum()), other_onsets=int(A[~pan].event.sum()), other_rate=float(A[~pan].event.mean()))
e = OUT['pandemic']
print(f"onset rate 2020-2022: {e['onsets']}/{e['active']} = {e['rate']:.4f}; other years: {e['other_onsets']}/{e['other_active']} = {e['other_rate']:.4f}")
print('  by year:', {y: round(d['rate'], 3) for y, d in OUT['by_year'].items()})


# ---------------------------------------------------------------- 1. variant panels
def build_panels():
    """onset indicators (-1 inactive, 0, 1) of every variant, by the code of app_events.py with THRESH and WINDOW free.
    A window longer than twelve months uses the months available (at least twelve), so that every panel starts in
    January 2015."""
    df = pd.read_csv(f'{W}/korea_imports_long.csv', parse_dates=['period'])
    df = df[df.partnerCode != 0]
    months = pd.date_range('2014-01-01', '2025-12-01', freq='MS')
    usd = df.pivot(index='period', columns=['cmdCode', 'partnerCode'], values='primaryValue').reindex(months)
    kg = df.pivot(index='period', columns=['cmdCode', 'partnerCode'], values='netWgt').reindex(months)
    has_rec = usd.notna()
    uv = (usd / kg.where(kg > 0)).median()
    val = kg.where(kg.notna(), usd / uv).where(has_rec, 0.0).fillna(0.0)
    share_base = usd.fillna(0.0)
    tot = share_base.T.groupby(level=0).sum().T
    trail_sum = share_base.shift(1).rolling(12, min_periods=12).sum()
    trail_tot = tot.shift(1).rolling(12, min_periods=12).sum()
    streams = [(c, p) for (c, p) in val.columns if share_base[(c, p)].sum() / tot[c].sum() >= 0.05]
    mon = months[months >= '2015-01-01']
    cols = {}
    for v in VARIANTS:
        thresh, window = v
        trail_med = val.shift(1).rolling(window, min_periods=min(window, 12)).median()
        for (c, p) in streams:
            active = (trail_sum[(c, p)] / trail_tot[c] >= 0.05) & (trail_med[(c, p)] > 0)
            low = (val[(c, p)] < thresh * trail_med[(c, p)]) & active
            ev = low & ~low.shift(1, fill_value=False)
            cols.setdefault((c, p), {})[key(v)] = np.where(active[mon], ev[mon].astype(int), -1)
    recs = []
    for (c, p), d in cols.items():
        f = pd.DataFrame(d); f.insert(0, 'period', mon); f.insert(1, 'code', c); f.insert(2, 'origin', NAMES.get(p, str(p)))
        recs.append(f)
    return pd.concat(recs, ignore_index=True)


if _os.path.exists(f'{W}/korea_imports_long.csv'):
    V = build_panels()
    V.to_csv(f'{W}/app_v08_variants.csv', index=False)
    OUT['panels'] = 'rebuilt from the import table'
else:
    V = pd.read_csv(f'{W}/app_v08_variants.csv', parse_dates=['period'])
    OUT['panels'] = 'read from work/app/app_v08_variants.csv'
V['sid'] = V.code.astype(str) + '_' + V.origin
print('variant panels:', OUT['panels'])

# the main variant must be the archived panel
chk = am.E.merge(V[['period', 'sid', key(MAIN)]], on=['period', 'sid'], how='outer')
assert len(chk) == len(am.E) == len(V) and (chk.event == chk[key(MAIN)]).all(), 'the main variant differs from stream_events.csv'

R0 = json.load(open(f'{W}/app_v07_main.json'))            # archived results: value shares of the streams, main fit
shares = R0['shares']; inputs = R0['inputs']


def live(a):
    return np.isfinite(a) and a < 5e3


def system_p_two(mu_hat, ac_hat, ac_map, n=400_000, seed=7):
    """ordinary-month probability that two or more inputs are disrupted, by Monte Carlo as in app_v07_main.simulate:
    margins held at their fitted values, origin o given dispersion ac_map[o] (independent when absent or infinite)."""
    all_streams = [(sid, o, float(mu_hat[o][j])) for o in origins.index for j, sid in enumerate(G[o][0])]
    sid_idx = {s[0]: k for k, s in enumerate(all_streams)}
    K = len(all_streams)
    in_cols = {c: [sid_idx[s] for s in shares if s.startswith(f'{c}_')] for c in inputs}
    in_w = {c: np.array([shares[all_streams[k][0]] for k in in_cols[c]]) for c in inputs}
    rng = np.random.default_rng(seed)
    etas = {o: (rng.gamma(ac_map[o], 1 / ac_map[o], size=n) if live(ac_map.get(o, np.inf)) else np.ones(n)) for o in multi}
    u = rng.random((n, K))
    lamt = np.empty((n, K))
    for k, (sid, o, mu1) in enumerate(all_streams):
        a0 = ac_hat.get(o, np.inf)
        m = a0 * np.log1p(mu1 / a0) if live(a0) else mu1                 # fitted marginal hazard
        a = ac_map.get(o, np.inf)
        lamt[:, k] = a * np.expm1(m / a) * etas[o] if live(a) else m     # rate that keeps the margin under dispersion a
    interrupted = u < -np.expm1(-lamt)
    nd = np.zeros(n, int)
    for c in inputs:
        nd += (interrupted[:, in_cols[c]] * in_w[c]).sum(1) > 0.5
    return float((nd >= 2).mean()), float((nd >= 1).mean())


# ---------------------------------------------------------------- 2. fits and system outputs by variant
OUT['variants'] = {}
for v in VARIANTS:
    t0 = time.time()
    Ev = V[['period', 'sid', 'origin']].copy(); Ev['event'] = V[key(v)]
    G = {o: am.group_matrix(o, Ev) for o in origins.index}
    act = Ev.event >= 0
    d = dict(thresh=v[0], window=v[1], streams=int(Ev[act].sid.nunique()), active=int(act.sum()), onsets=int((Ev.event == 1).sum()))
    M1 = Model2('S1', groups=G, H=H1, scope=SCOPE); r1 = M1.fit(starts=S1_STARTS)
    M0 = Model2('IND', groups=G, H=H1, scope=SCOPE); r0 = M0.fit()
    x1 = r1.x; nll1 = float(r1.fun)
    d['aic'] = dict(S1=2 * nll1 + 2 * M1.npar(), IND=2 * float(r0.fun) + 2 * M0.npar())
    d['lr_all'] = 2 * (float(r0.fun) - nll1)
    L, U, _ = profile_interval(M1, M1.i_beta, x1, nll1, cut=CUT90, lo=-6, hi=6)
    d['beta'] = dict(hat=float(x1[M1.i_beta]), lo90=L, hi90=U, lr_zero=float(2 * (M1.fit(x0=x1, fixed={M1.i_beta: 0.0}).fun - nll1)))
    d['origin'] = {}
    for o in multi:
        j = M1.gp[o]; X = G[o][1]
        Lo, Up, _ = profile_interval(M1, j, x1, nll1, cut=CUT90)
        _, Up965, _ = profile_interval(M1, j, x1, nll1, cut=CUT965)
        d['origin'][o] = dict(n_streams=int(origins[o]), n_onsets=int((X == 1).sum()), months_ge2=int(((X == 1).sum(1) >= 2).sum()),
                              ac_hat=M1.alpha_c(x1, o), ac_lo90=float(np.exp(-Up)),
                              ac_hi90=(float(np.exp(-Lo)) if Lo > cf.LOG_TH_MIN + 1e-6 else float('inf')),
                              ac_lo965=float(np.exp(-Up965)),
                              lr_boundary=float(2 * (M1.fit(x0=x1, fixed={j: cf.LOG_TH_MIN}).fun - nll1)))
    mu_hat = {o: M1.mu(x1, o) for o in origins.index}
    ac_hat = {o: d['origin'][o]['ac_hat'] for o in multi}
    corner = {o: d['origin'][o]['ac_lo965'] for o in multi}
    p_ind, any_ind = system_p_two(mu_hat, ac_hat, {})
    p_fit, any_fit = system_p_two(mu_hat, ac_hat, ac_hat)
    p_cor, any_cor = system_p_two(mu_hat, ac_hat, corner)
    d['system'] = dict(p_two_independence=p_ind, p_two_fitted=p_fit, p_two_corner965=p_cor, ratio_corner=p_cor / p_ind,
                       p_any_independence=any_ind, p_any_corner965=any_cor)
    d['seconds'] = round(time.time() - t0, 1)
    OUT['variants'][key(v)] = d
    print(f"{key(v)}: onsets {d['onsets']} | beta {d['beta']['hat']:.2f} (LR {d['beta']['lr_zero']:.1f}) | "
          + ' '.join(f"{o} ac {d['origin'][o]['ac_hat']:.3g} lo90 {d['origin'][o]['ac_lo90']:.2f} LR {d['origin'][o]['lr_boundary']:.2f}" for o in multi)
          + f" | AIC S1-IND {d['aic']['S1'] - d['aic']['IND']:+.1f} | P2 {100 * p_ind:.2f}-{100 * p_cor:.2f}% (x{p_cor / p_ind:.2f}) ({d['seconds']}s)")

# the main variant must reproduce the archived fit and system outputs
m = OUT['variants'][key(MAIN)]; S0 = R0['system']['ordinary']
assert abs(m['aic']['S1'] - R0['fits']['aic']['S1']) < 1e-6 and abs(m['aic']['IND'] - R0['fits']['aic']['IND']) < 1e-6
assert abs(m['system']['p_two_independence'] - S0['independence']['mc']['p_two']) < 1e-12
assert abs(m['system']['p_two_corner965'] - S0['most96.5']['mc']['p_two']) < 1e-12
print('main variant reproduces app_v07_main.json')

# ---------------------------------------------------------------- 3. summary over the variants other than the main one
oth = [OUT['variants'][key(v)] for v in VARIANTS if v != MAIN]
OUT['summary'] = dict(
    n_variants=len(oth),
    beta_hat_range=[min(d['beta']['hat'] for d in oth), max(d['beta']['hat'] for d in oth)],
    beta_lr_range=[min(d['beta']['lr_zero'] for d in oth), max(d['beta']['lr_zero'] for d in oth)],
    aic_prefers_independence=int(sum(d['aic']['IND'] < d['aic']['S1'] for d in oth)),
    lr_above_cutoff={o: [f"q{int(round(100 * d['thresh']))}_w{d['window']}" for d in oth if d['origin'][o]['lr_boundary'] > CUT90] for o in multi},
    ac_lo90_range={o: [min(d['origin'][o]['ac_lo90'] for d in oth), max(d['origin'][o]['ac_lo90'] for d in oth)] for o in multi},
    ratio_corner_range=[min(d['system']['ratio_corner'] for d in oth), max(d['system']['ratio_corner'] for d in oth)])
print('summary:', json.dumps(OUT['summary']))
OUT['elapsed_s'] = time.time() - t_start
json.dump(OUT, open(f'{W}/app_v08_detection.json', 'w'), indent=1, default=float)
print('done', round(OUT['elapsed_s']))
