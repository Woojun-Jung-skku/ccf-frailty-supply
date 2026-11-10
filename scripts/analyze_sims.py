"""Summaries, tables and figures for Studies C, D and E."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, glob
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np, pandas as pd
import sfig as S
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

WK = f'{SP_ROOT}/work'
LOG_TH_MIN = np.log(1e-4)
OUT = {}


def load(pat):
    rows = []
    for f in glob.glob(pat):
        rows += [json.loads(l) for l in open(f)]
    d = pd.DataFrame(rows)
    key = [k for k in ['I', 'T', 'mub', 'ac', 'rep', 'mech', 'm'] if k in d.columns]
    return d.drop_duplicates(subset=key).reset_index(drop=True)


# ----------------------------------------------------------------- Study C
C = load(f'{WK}/studyC/C_w*.jsonl')
OUT['C_n'] = int(len(C)); OUT['C_errors'] = int(C['error'].notna().sum()) if 'error' in C else 0
if 'error' in C:
    C = C[C['error'].isna()]
C['finite'] = np.isfinite(C.ac)
C['lt_true'] = np.where(C.finite, -np.log(C.ac.where(C.finite, 1.0)), LOG_TH_MIN)
C['cover'] = (C.lo <= C.lt_true + 1e-9) & (C.lt_true - 1e-9 <= C.hi)
C['bound_lo'] = C.lo <= LOG_TH_MIN + 1e-6          # interval reaches independence
C['rej'] = C.lr > 2.705543                         # 5% boundary test (chi2 mixture)
C['err_ac'] = np.where(C.finite, np.log(np.clip(C.ac_hat, 1e-3, 1e4)) - np.log(C.ac.where(C.finite, 1.0)), np.nan)
mu_pair = C.mub ** 2  # proxy; exposures centred
C['pairinfo'] = C['T'] * C.I * (C.I - 1) * C.mub ** 2 / 2   # Lemma 1 at the boundary, per group, x3 groups below
C['pairinfo'] *= 3
for q, nm in [('R1', 'R1'), ('R12', 'R3'), ('D2', 'D2')]:
    C[f'eS_{nm}'] = np.log(C[f's{q}'] / C[f't{q}'])
    C[f'eI_{nm}'] = np.log(C[f'i{q}'] / C[f't{q}'])
gC = C.groupby(['ac', 'I', 'T', 'mub'])
tabC = gC.agg(n=('rep', 'size'), cover=('cover', 'mean'), rej=('rej', 'mean'), bound=('bound_lo', 'mean'),
              med_err_ac=('err_ac', 'median'),
              eS_D2=('eS_D2', 'median'), eI_D2=('eI_D2', 'median'),
              eS_R3=('eS_R3', 'median'), eI_R3=('eI_R3', 'median'),
              eS_R1=('eS_R1', 'median'), eI_R1=('eI_R1', 'median'),
              ap_hat=('ap_hat', 'median'), ap_ind=('ap_ind', 'median')).reset_index()
tabC.to_csv(f'{WK}/tabC_full.csv', index=False)
# condensed summaries
fin = C[C.finite]
OUT['C_cover_finite_all'] = float(fin.cover.mean())
OUT['C_cover_by_I'] = fin.groupby('I').cover.mean().round(3).to_dict()
OUT['C_cover_by_ac'] = fin.groupby('ac').cover.mean().round(3).to_dict()
OUT['C_cover_inf_all'] = float(C[~C.finite].cover.mean())
OUT['C_size_inf'] = float(C[~C.finite].rej.mean())
OUT['C_power_by_ac_I'] = C[C.finite].groupby(['ac', 'I']).rej.mean().round(2).unstack().to_dict()
OUT['C_bound_by_ac_I'] = C.groupby(['ac', 'I']).bound_lo.mean().round(2).unstack().to_dict()
OUT['C_rmse_logac_by_ac_I'] = fin.groupby(['ac', 'I']).err_ac.apply(lambda x: float(np.sqrt(np.mean(x ** 2)))).round(2).unstack().to_dict()
OUT['C_D2_err'] = C.groupby(['ac']).agg(S1=('eS_D2', 'median'), IND=('eI_D2', 'median')).round(3).to_dict()
OUT['C_R3_err'] = C.groupby(['ac']).agg(S1=('eS_R3', 'median'), IND=('eI_R3', 'median')).round(3).to_dict()
OUT['C_D2_err_byI'] = C.groupby(['ac', 'I']).agg(S1=('eS_D2', 'median'), IND=('eI_D2', 'median')).round(3).reset_index().values.tolist()
OUT['C_apind'] = C.groupby(['ac']).ap_ind.median().round(2).to_dict()
OUT['C_aphat'] = C.groupby(['ac']).ap_hat.median().round(2).to_dict()
OUT['C_conv'] = float(C.conv.mean())

# ----------------------------------------------------------------- Study D
D = load(f'{WK}/studyD/D_w*.jsonl')
if 'error' in D:
    OUT['D_errors'] = int(D['error'].notna().sum()); D = D[D['error'].isna()]
meth = ['item_each_union', 'union_ind', 'union_dep', 'sys_ind', 'sys_dep']
rows = []
for (I, ac), g in D.groupby(['I', 'ac']):
    for m in meth:
        frac = g[f'{m}_frac'].values
        arl = 1 / max(frac.mean(), 1e-9)
        bs = []
        rng = np.random.default_rng(1)
        for b in range(400):
            s = frac[rng.integers(0, len(frac), len(frac))]
            bs.append(1 / max(s.mean(), 1e-9))
        lo, hi = np.quantile(bs, [0.05, 0.95])
        dc = g[f'{m}_common_delay']; ds = g[f'{m}_single_delay']
        rows.append(dict(I=I, ac=ac, method=m, n=len(g), ARL0=arl, ARL0_lo=lo, ARL0_hi=hi,
                         det_common=float(dc.notna().mean()), delay_common=float(dc.median()),
                         det_single=float(ds.notna().mean()), delay_single=float(ds.median())))
tabD = pd.DataFrame(rows)
tabD.to_csv(f'{WK}/tabD_full.csv', index=False)
OUT['D_n'] = int(len(D))
OUT['D_ARL0'] = tabD.pivot_table(index=['ac', 'I'], columns='method', values='ARL0').round(1).reset_index().values.tolist()
OUT['D_ac_hat_median'] = D.groupby(['I', 'ac']).ac_hat.median().round(2).reset_index().values.tolist()

# ----------------------------------------------------------------- Study E
E = load(f'{WK}/studyE/E_w*.jsonl')
if 'error' in E:
    E = E[E['error'].isna()]
E['resp_true'] = E.jh_true / E.j1_true
for k in ['S1', 'BFR', 'IND']:
    E[f'resp_{k}'] = E[f'jh_{k}'] / E[f'j1_{k}']
    E[f'err1_{k}'] = np.log(E[f'j1_{k}'] / E.j1_true)
E['resp_sel'] = np.where(E.sel_aic == 'S1', E.resp_S1, np.where(E.sel_aic == 'BFR', E.resp_BFR, E.resp_IND))
E['err1_sel'] = np.where(E.sel_aic == 'S1', E.err1_S1, np.where(E.sel_aic == 'BFR', E.err1_BFR, E.err1_IND))
tabE = E.groupby(['mech', 'm', 'T']).agg(
    n=('rep', 'size'),
    sel_S1=('sel_aic', lambda s: np.mean(s == 'S1')), sel_BFR=('sel_aic', lambda s: np.mean(s == 'BFR')),
    sel_IND=('sel_aic', lambda s: np.mean(s == 'IND')),
    resp_true=('resp_true', 'median'), resp_S1=('resp_S1', 'median'), resp_BFR=('resp_BFR', 'median'),
    resp_sel=('resp_sel', 'median'), err1_IND=('err1_IND', 'median'), err1_sel=('err1_sel', 'median')).reset_index()
tabE.to_csv(f'{WK}/tabE_full.csv', index=False)
OUT['E_table'] = tabE.round(3).values.tolist()
OUT['E_cols'] = list(tabE.columns)

json.dump(OUT, open(f'{WK}/sims_summary.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in OUT.items() if not k.startswith('E_table')}, indent=1, default=float)[:6000])

# ----------------------------------------------------------------- Figure 5 (Study C)
KR = pd.read_csv(f'{WK}/studyC_joint_true_rates.csv')
C = C.merge(KR, on=['I', 'T', 'mub', 'ac', 'rep'], how='left')
C['lkS'] = np.log(C.kS_D2); C['lkI'] = np.log(C.kI_D2)
fig, axs = plt.subplots(1, 2, figsize=(S.W2, 2.35))
ax = axs[0]
for (col, mk, ls), ac in zip(S.SERIES, [0.5, 2.0]):
    for T, lsT in [(60, '--'), (120, '-')]:
        g = fin[(fin.ac == ac) & (fin['T'] == T)].groupby('I').cover.mean()
        ax.plot(g.index, g.values, color=col, ls=lsT, marker=mk, mfc='white' if T == 60 else col, mew=0.9)
gi = C[~C.finite].groupby('I').cover.mean()
ax.plot(gi.index, gi.values, color=S.SERIES[2][0], ls='-', marker=S.SERIES[2][1], mfc=S.SERIES[2][0], mew=0.9)
ax.axhline(0.9, color=S.GREY, lw=0.6, ls=':')
ax.set_xscale('log'); ax.set_xticks([2, 5, 10, 30]); ax.set_xticklabels(['2', '5', '10', '30'])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(0.6, 1.02)
ax.set_xlabel('Streams per group, m')
ax.set_ylabel(r'Coverage of 90% interval for $\theta_c$')
ax.text(33, gi.values[-1], 'independence', color=S.SERIES[2][0], fontsize=6.8, va='center')
ends = {ac: fin[(fin.ac == ac) & (fin['T'] == 120)].groupby('I').cover.mean().values[-1] for ac in [0.5, 2.0]}
ylab = dict(ends)
if abs(ends[0.5] - ends[2.0]) < 0.03:                      # spread overlapping direct labels
    mid = 0.5 * (ends[0.5] + ends[2.0]); lo_ac = min(ends, key=ends.get); hi_ac = max(ends, key=ends.get)
    ylab[lo_ac] = mid - 0.016; ylab[hi_ac] = mid + 0.016
for (col, mk, ls), ac in zip(S.SERIES, [0.5, 2.0]):
    ax.text(33, ylab[ac], rf'$\alpha_c={ac:g}$', color=col, fontsize=6.8, va='center')
ax.set_xlim(1.8, 60)
ax.text(0.03, 0.04, 'solid: 120 months   dashed: 60 months', transform=ax.transAxes, fontsize=6.6, color=S.INK, va='bottom')
S.panel_label(ax, '(a)')

ax = axs[1]
pos = 0
xt, xl = [], []
for j, ac in enumerate([0.5, 2.0, np.inf]):
    sub = C[C.ac == ac]
    for k, (col_name, colr, lab) in enumerate([('lkI', S.GREY, 'item model'), ('lkS', S.C['red'], 'frailty')]):
        v = sub[col_name].dropna().values
        q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
        x = pos + k * 0.9
        ax.plot([x, x], [q[0], q[4]], color=colr, lw=0.8)
        ax.add_patch(plt.Rectangle((x - 0.3, q[1]), 0.6, q[3] - q[1], facecolor='white' if k == 0 else colr, edgecolor=colr, lw=0.9, alpha=1 if k == 0 else 0.85))
        ax.plot([x - 0.3, x + 0.3], [q[2], q[2]], color=S.INK if k else colr, lw=1.1)
    xt.append(pos + 0.45); xl.append('independence' if not np.isfinite(ac) else rf'$\alpha_c={ac:g}$')
    pos += 2.6
ax.axhline(0, color=S.GREY, lw=0.6, ls=':')
ax.set_xticks(xt); ax.set_xticklabels(xl)
ax.set_ylabel('log(estimated / true)\nP(both sources lost)')
ax.text(0.70, 0.30, 'item model', color=S.GREY, transform=ax.transAxes, fontsize=6.8, va='top')
ax.text(0.70, 0.22, 'frailty (S-1)', color=S.C['red'], transform=ax.transAxes, fontsize=6.8, va='top')
S.panel_label(ax, '(b)')
fig.tight_layout(w_pad=2.5)
S.save(fig, 'fig5_studyC')

# ----------------------------------------------------------------- Figure 6 (Study D)
fig, axs = plt.subplots(1, 2, figsize=(S.W2, 2.4))
ax = axs[0]
labs = {'item_each_union': 'union, items at 24', 'union_ind': 'union, independence', 'union_dep': 'union, fitted frailty',
        'sys_ind': 'system, independence', 'sys_dep': 'system, fitted frailty'}
styl = {'item_each_union': (S.GREY, 'o', '-'), 'union_ind': (S.C['gold'], 's', '-'), 'union_dep': (S.C['teal'], '^', '-'),
        'sys_ind': (S.C['purple'], 'v', '--'), 'sys_dep': (S.C['red'], 'D', '-')}
xs = {0.5: 0, 2.0: 1, np.inf: 2}
for m in ['item_each_union', 'union_ind', 'union_dep', 'sys_ind', 'sys_dep']:
    col, mk, ls = styl[m]
    for I, off in [(5, -0.22), (10, 0.0), (30, 0.22)]:
        sub = tabD[(tabD.method == m) & (tabD.I == I)].set_index('ac')
        xx = np.array([xs[a] for a in sub.index]) + off
        ax.errorbar(xx, sub.ARL0, yerr=[sub.ARL0 - sub.ARL0_lo, sub.ARL0_hi - sub.ARL0], fmt=mk, color=col,
                    mfc='white', ms=3.2, mew=0.8, elinewidth=0.6, capsize=0)
ax.axhline(24, color=S.GREY, lw=0.6, ls=':')
ax.set_yscale('log'); ax.set_yticks([1, 3, 10, 24, 50]); ax.set_yticklabels(['1', '3', '10', '24', '50'])
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r'$\alpha_c=0.5$', r'$\alpha_c=2$', 'independence'])
ax.set_ylabel(r'Realised ARL$_0$ (months)')
ax.set_xlim(-0.5, 2.5)
S.panel_label(ax, '(a)')
ax = axs[1]
for m in ['item_each_union', 'union_dep', 'sys_dep']:
    col, mk, ls = styl[m]
    sub = tabD[(tabD.method == m)].groupby('I')[['det_common', 'det_single']].mean()
    ax.plot(sub.index, sub.det_common, color=col, marker=mk, ls='-', mfc=col, mew=0.8)
    ax.plot(sub.index, sub.det_single, color=col, marker=mk, ls='--', mfc='white', mew=0.8)
ax.set_xscale('log'); ax.set_xticks([5, 10, 30]); ax.set_xticklabels(['5', '10', '30'])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(0, 1.08)
ax.text(5.2, 1.035, 'solid: all streams ×2', fontsize=6.6, color=S.INK, va='bottom')
ax.text(5.2, 0.30, 'dashed: one stream ×3', fontsize=6.6, color=S.INK, va='bottom')
ax.set_xlabel('Streams monitored, m')
ax.set_ylabel('Shifts detected within 96 months')
S.panel_label(ax, '(b)')
h = [plt.Line2D([], [], color=styl[m][0], marker=styl[m][1], ls='none', mfc='white', mew=0.9) for m in labs]
fig.legend(h, list(labs.values()), loc='lower center', ncol=5, fontsize=6.6, handletextpad=0.3, columnspacing=1.0,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(w_pad=2.5, rect=(0, 0.08, 1, 1))
S.save(fig, 'fig6_studyD')
