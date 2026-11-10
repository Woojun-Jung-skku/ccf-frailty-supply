"""Tokens and markdown tables for Section 6 from the simulation outputs."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, glob
import numpy as np, pandas as pd

WK = f'{SP_ROOT}/work'
LOG_TH_MIN = np.log(1e-4)


def load(pat):
    rows = []
    for f in glob.glob(pat):
        rows += [json.loads(l) for l in open(f)]
    d = pd.DataFrame(rows)
    if 'error' in d:
        d = d[d['error'].isna()]
    key = [k for k in ['I', 'T', 'mub', 'ac', 'rep', 'mech', 'm'] if k in d.columns]
    return d.drop_duplicates(subset=key).reset_index(drop=True)


pct = lambda x: f'{100 * x:.0f}%'
TOK = {}
# ------------------------------------------------ Study C
C = load(f'{WK}/studyC/C_w*.jsonl')
C['finite'] = np.isfinite(C.ac)
C['lt_true'] = np.where(C.finite, -np.log(C.ac.where(C.finite, 1.0)), LOG_TH_MIN)
C['cover'] = (C.lo <= C.lt_true + 1e-9) & (C.lt_true - 1e-9 <= C.hi)
C['rej'] = C.lr > 2.705543
C['err_ac'] = np.where(C.finite, np.log(np.clip(C.ac_hat, 1e-3, 1e4)) - np.log(C.ac.where(C.finite, 1.0)), np.nan)
C['rS_D2'] = C.sD2 / C.tD2; C['rI_D2'] = C.iD2 / C.tD2
# joint loss at the true stream rates with each model's estimated dispersions (separates dependence error from rate noise)
import sys
sys.path.insert(0, f'{SP_ROOT}/scripts')
import ccfmodel as cm
def true_mu12(I, T, mub, ac, rep):
    seed = abs(hash((int(I), int(T), float(mub), float(ac), int(rep)))) % (2**32)
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 0.3, int(I)); z -= z.mean()
    return mub * np.exp(z)[:2]
kS, kI = [], []
for r in C.itertuples():
    m12 = true_mu12(r.I, r.T, r.mub, r.ac, r.rep)
    t = cm.allfail_group(m12, 2.0, r.ac)
    kS.append(cm.allfail_group(m12, r.ap_hat, r.ac_hat if r.ac_hat < 5e3 else np.inf) / t)
    pI = 1 - (1 + m12 / r.ap_ind) ** (-r.ap_ind)
    kI.append(float(pI[0] * pI[1]) / t)
C['kS_D2'] = kS; C['kI_D2'] = kI
C[['I', 'T', 'mub', 'ac', 'rep', 'kS_D2', 'kI_D2']].to_csv(f'{WK}/studyC_joint_true_rates.csv', index=False)
chk = max(abs(r.tD2 - cm.allfail_group(true_mu12(r.I, r.T, r.mub, r.ac, r.rep), 2.0, r.ac)) for r in C.head(40).itertuples())
print('seed check (should be ~0):', chk)
C['rS_R3'] = C.sR12 / C.tR12; C['rI_R3'] = C.iR12 / C.tR12
fin = C[C.finite]
TOK['C_N'] = f'{len(C):,}'
TOK['C_COVER'] = f'{fin.cover.mean():.2f}'
byI = fin.groupby('I').cover.mean()
TOK['C_COVER_RANGE_I'] = f'{byI.min():.2f}–{byI.max():.2f}'
TOK['C_COVER_INF'] = pct(C[~C.finite].cover.mean())
TOK['C_SIZE'] = f'{100 * C[~C.finite].rej.mean():.1f}%'
pw = C[C.finite].groupby(['ac', 'I']).rej.mean()
andjoin = lambda xs: ', '.join(xs[:-1]) + ' and ' + xs[-1]
TOK['C_POWER_A2'] = andjoin([pct(pw.loc[(2.0, i)]) for i in [2, 5, 10, 30]])
TOK['C_POWER_A05'] = andjoin([pct(pw.loc[(0.5, i)]) for i in [2, 5, 10, 30]])
nc = C[~C.conv.astype(bool)]
atb = (nc.ac_hat >= 5e3) | (nc.ac_hat <= 1e-3) | (nc.ap_hat >= 5e3)
TOK['C_NONCONV'] = pct(1 - C.conv.astype(bool).mean())
TOK['C_NONCONV_BOUND'] = pct(atb.mean())
TOK['C_COVER_NONCONV'] = f"{fin[~fin.conv.astype(bool)].cover.mean():.2f}"
TOK['C_COVER_CONV'] = f"{fin[fin.conv.astype(bool)].cover.mean():.2f}"
rm = fin.groupby(['ac', 'I']).err_ac.apply(lambda x: float(np.sqrt(np.mean(x ** 2))))
TOK['C_RMSE_A2_2'] = f'{rm.loc[(2.0, 2)]:.1f}'; TOK['C_RMSE_A2_30'] = f'{rm.loc[(2.0, 30)]:.2f}'
ga = C.groupby('ac')
TOK['C_APIND_05'] = f'{ga.ap_ind.median().loc[0.5]:.2f}'; TOK['C_APIND_2'] = f'{ga.ap_ind.median().loc[2.0]:.2f}'
for a, s in [(0.5, '05'), (2.0, '2')]:
    TOK[f'C_D2_IND_{s}'] = f'{ga.kI_D2.median().loc[a]:.2f}'
    TOK[f'C_D2_S1_{s}'] = f'{ga.kS_D2.median().loc[a]:.2f}'
    TOK[f'C_D2P_IND_{s}'] = f'{ga.rI_D2.median().loc[a]:.2f}'
    TOK[f'C_D2P_S1_{s}'] = f'{ga.rS_D2.median().loc[a]:.2f}'
    TOK[f'C_R3_IND_{s}'] = f'{ga.rI_R3.median().loc[a]:.2f}'
    TOK[f'C_R3_S1_{s}'] = f'{ga.rS_R3.median().loc[a]:.2f}'
# Table 3
rows = ['| $\\alpha_c$ | Streams per group | Coverage, $\\theta_c$ | Rejects $\\theta_c = 0$ | RMSE, $\\log\\hat\\alpha_c$ | Joint loss, S-1 | Joint loss, IND |',
        '|---|---|---|---|---|---|---|']
for a in [0.5, 2.0, np.inf]:
    for I in [2, 5, 10, 30]:
        g = C[(C.ac == a) & (C.I == I)]
        r = '—' if not np.isfinite(a) else f'{np.sqrt(np.nanmean(g.err_ac ** 2)):.2f}'
        al = '∞' if not np.isfinite(a) else f'{a:g}'
        rows.append(f'| {al} | {I} | {g.cover.mean():.2f} | {g.rej.mean():.2f} | {r} | {g.kS_D2.median():.2f} | {g.kI_D2.median():.2f} |')
TOK['TABLE3'] = '\n'.join(rows)

# ------------------------------------------------ Study D
D = load(f'{WK}/studyD/D_w*.jsonl')
meth = [('item_each_union', '(1) Union, each stream at 24 (item model)'), ('union_ind', '(2) Union, calibrated under independence'),
        ('union_dep', '(3) Union, calibrated under fitted S-1'), ('sys_ind', '(4) System, calibrated under independence'),
        ('sys_dep', '(5) System, calibrated under fitted S-1')]
arl = {}
for (I, a), g in D.groupby(['I', 'ac']):
    for m, _ in meth:
        arl[(m, a, I)] = 1 / max(g[f'{m}_frac'].mean(), 1e-9)
def rng_str(m, a):
    v = [arl[(m, a, I)] for I in [5, 10, 30]]
    lo, hi = min(v), max(v)
    f = (lambda x: f'{x:.1f}') if hi < 10 else (lambda x: f'{x:.0f}')
    return f'{f(lo)}–{f(hi)}'
rows = ['| Rule | $\\text{ARL}_0$, $\\alpha_c = 0.5$ | $\\text{ARL}_0$, $\\alpha_c = 2$ | $\\text{ARL}_0$, independence | Common shift ×2: detected (median delay) | One stream ×3: detected (median delay) |',
        '|-------------|--------|--------|--------|----------|----------|']
for m, lab in meth:
    dc = D[f'{m}_common_delay']; ds = D[f'{m}_single_delay']
    rows.append(f'| {lab} | {rng_str(m, 0.5)} | {rng_str(m, 2.0)} | {rng_str(m, np.inf)} | {pct(dc.notna().mean())} ({dc.median():.0f}) | {pct(ds.notna().mean())} ({ds.median():.0f}) |')
TOK['TABLE4A'] = '(a) Study D\n\n' + '\n'.join(rows)
TOK['D_ITEM_RANGE'] = f"{min(arl[('item_each_union', a, I)] for a in [0.5, 2.0, np.inf] for I in [5, 10, 30]):.1f} to {max(arl[('item_each_union', a, I)] for a in [0.5, 2.0, np.inf] for I in [5, 10, 30]):.1f}"
TOK['D_SYSIND_05'] = rng_str('sys_ind', 0.5)
TOK['D_SYSDEP_05'] = rng_str('sys_dep', 0.5)
TOK['D_SYSDEP_INF'] = rng_str('sys_dep', np.inf)
TOK['D_SYSIND_INF'] = rng_str('sys_ind', np.inf)
TOK['D_SYS_COMMON'] = pct(D.sys_dep_common_delay.notna().mean())
TOK['D_SYS_COMMON_DELAY'] = f'{D.sys_dep_common_delay.median():.0f}'
TOK['D_SYS_SINGLE'] = pct(D.sys_dep_single_delay.notna().mean())
TOK['D_UDEP_SINGLE'] = pct(D.union_dep_single_delay.notna().mean())

# ------------------------------------------------ Study E
E = load(f'{WK}/studyE/E_w*.jsonl')
E['resp_true'] = E.jh_true / E.j1_true
for k in ['S1', 'BFR', 'IND']:
    E[f'resp_{k}'] = E[f'jh_{k}'] / E[f'j1_{k}']
E['resp_sel'] = np.where(E.sel_aic == 'S1', E.resp_S1, np.where(E.sel_aic == 'BFR', E.resp_BFR, E.resp_IND))
E['err_IND'] = E.j1_IND / E.j1_true
rows = ['| Mechanism | Streams per group | Selects S-1 | Selects BFR | Selects IND | Halving response, true | Halving response, selected model |',
        '|----------|-----|-----|-----|-----|-------|-------|']
names = {'frailty': 'frailty ($\\alpha_c = 1$)', 'bfr': 'BFR common shock', 'indep': 'independence'}
for mech in ['frailty', 'bfr', 'indep']:
    for m in [3, 8]:
        g = E[(E.mech == mech) & (E.m == m)]
        rows.append(f"| {names[mech]} | {m} | {pct((g.sel_aic == 'S1').mean())} | {pct((g.sel_aic == 'BFR').mean())} | {pct((g.sel_aic == 'IND').mean())} | {g.resp_true.median():.2f} | {g.resp_sel.median():.2f} |")
TOK['TABLE4B'] = '(b) Study E, pooled over $T \\in \\{60, 120\\}$ (120 replications per row)\n\n' + '\n'.join(rows)
sel = lambda mech, m, k: (E[(E.mech == mech) & (E.m == m)].sel_aic == k).mean()
TOK['E_SEL_F8'] = pct(sel('frailty', 8, 'S1')); TOK['E_SEL_B8'] = pct(sel('bfr', 8, 'BFR'))
TOK['E_SEL_F3'] = pct(sel('frailty', 3, 'S1')); TOK['E_SEL_B3'] = pct(sel('bfr', 3, 'BFR'))
TOK['E_SEL_I'] = pct((E[E.mech == 'indep'].sel_aic == 'IND').mean())
TOK['E_RESP_F'] = f"{1 / E[E.mech == 'frailty'].resp_true.median():.1f}"
TOK['E_RESP_B'] = f"{1 / E[E.mech == 'bfr'].resp_true.median():.1f}"
TOK['E_IND_ERR'] = f"{E[E.mech != 'indep'].err_IND.median():.2f}"
json.dump(TOK, open(f'{WK}/tokens.json', 'w'), indent=1)
for k, v in TOK.items():
    if not k.startswith('TABLE'):
        print(k, '=', v)
print(TOK['TABLE3']); print(TOK['TABLE4A']); print(TOK['TABLE4B'])
