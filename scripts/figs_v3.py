"""Paper S v0.3: figures that recombine v0.2 panels.
Fig. 5  (fig5_identification): (a) group-periods needed (Lemma 1), (b) coverage vs m (simulation), (c) joint-loss error boxes.
Fig. S1 (figS1_alarms): (a) union run length (Prop. S1), (b) realised ARL0 of five rules, (c) detection shares.
Other figures (fig1_framework, fig3_ccf, fig2_system, fig4_workflow, fig6_application, fig7_system) are drawn by
fig1_framework.py, fig4_workflow.py and figs_v2.py."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import figstyle as fs
import ccfmodel as cm
import figs_v2 as f2
from figs_v2 import figure, logx, logy, load, AC_COL, AC_MK, AC_LAB, MODEL, SMALL, AP, WK, OUT


def study_c():
    C = load(f'{WK}/studyC/C_w*.jsonl')
    C['finite'] = np.isfinite(C.ac)
    C['lt_true'] = np.where(C.finite, -np.log(C.ac.where(C.finite, 1.0)), np.log(1e-4))
    C['cover'] = (C.lo <= C.lt_true + 1e-9) & (C.lt_true - 1e-9 <= C.hi)
    KR = pd.read_csv(f'{WK}/studyC_joint_true_rates.csv')
    return C.merge(KR, on=['I', 'T', 'mub', 'ac', 'rep'], how='left')


def fig5():
    C = study_c()
    fig, axs = figure(180, 58, 3)
    # (a) group-periods needed for SE(theta) = 0.5  (Lemma 1)
    ax = axs[0]
    mus = np.logspace(np.log10(0.02), np.log10(0.5), 40)
    for k, m in enumerate([2, 5, 10, 30]):
        Tn = 8 / (m * (m - 1) * mus ** 2)
        fs.series(ax, mus, Tn, k, 4, marks=4, mrange=(0.025, 0.4))
        ax.text(0.53, Tn[-1], f'{m} streams', color=fs.INK, fontsize=SMALL, va='center')
    ax.plot(mus, 8 / mus ** 2, color=fs.INK, lw=1.0, ls=':')
    ax.text(0.055, 2.3e4, 'item dispersion,\none stream', color=fs.INK, fontsize=SMALL, va='center')
    ax.axhline(120, **fs.REF)
    ax.text(0.95, 140, '120 months', color=fs.GREY, fontsize=SMALL, va='bottom', ha='right')
    logx(ax, [0.02, 0.05, 0.1, 0.2, 0.5], ['0.02', '0.05', '0.1', '0.2', '0.5']); ax.set_xlim(0.018, 1.1)
    logy(ax, [1e-1, 1, 10, 100, 1e3, 1e4], [r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$']); ax.set_ylim(0.012, 6e4)
    ax.set_xlabel(r'Stream interruption rate $\mu$')
    ax.set_ylabel(r'Group-periods for SE($\theta_c$) = 0.5')
    fs.panel(ax, 'a')
    # (b) coverage
    ax = axs[1]
    for ac in [0.5, 2.0]:
        for T, ls in [(120, '-'), (60, '--')]:
            g = C[(C.ac == ac) & (C['T'] == T)].groupby('I').cover.mean()
            ax.plot(g.index, g.values, color=AC_COL[ac], ls=ls, lw=1.1, marker=AC_MK[ac], mfc=AC_COL[ac], mec=AC_COL[ac], ms=3.8, mew=0.8)
    gi = C[~C.finite].groupby('I').cover.mean()
    ax.plot(gi.index, gi.values, color=AC_COL[np.inf], lw=1.1, marker=AC_MK[np.inf], mfc=AC_COL[np.inf], mec=AC_COL[np.inf], ms=3.8)
    ax.axhline(0.9, **fs.REF)
    logx(ax, [2, 5, 10, 30]); ax.set_xlim(1.8, 36); ax.set_ylim(0.76, 1.075); ax.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    ax.set_xlabel('Streams per group, m'); ax.set_ylabel('Coverage of the 90%\ninterval for ' + r'$\theta_c$')
    h = [Line2D([], [], color=AC_COL[a], lw=1.1, marker=AC_MK[a], mfc=AC_COL[a], ms=3.8, label=AC_LAB[a]) for a in [0.5, 2.0, np.inf]]
    h2 = [Line2D([], [], color=fs.INK, lw=1.1, ls='-', label='120 months'), Line2D([], [], color=fs.INK, lw=1.1, ls='--', label='60 months')]
    leg1 = ax.legend(handles=h, loc='lower left', handlelength=1.6, fontsize=SMALL, borderaxespad=0.3); ax.add_artist(leg1)
    ax.legend(handles=h2, loc='upper right', handlelength=1.6, fontsize=SMALL, borderaxespad=0.3)
    fs.panel(ax, 'b')
    # (c) joint-loss error boxes
    ax = axs[2]
    pos = 0; xt, xl = [], []
    for ac in [0.5, 2.0, np.inf]:
        sub = C[C.ac == ac]
        for k, (col_name, mdl) in enumerate([('kI_D2', 'IND'), ('kS_D2', 'S1')]):
            v = np.log(sub[col_name].dropna().values)
            q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
            colr, _, filled = MODEL[mdl]
            x = pos + k * 0.9
            ax.plot([x, x], [q[0], q[1]], color=colr, lw=0.8); ax.plot([x, x], [q[3], q[4]], color=colr, lw=0.8)
            ax.add_patch(Rectangle((x - 0.3, q[1]), 0.6, q[3] - q[1], facecolor=fs.tint(colr, 0.35) if mdl == 'S1' else 'white', edgecolor=colr, lw=0.9))  # open: item model; filled: frailty model
            ax.plot([x - 0.3, x + 0.3], [q[2], q[2]], color=colr, lw=1.2)
        xt.append(pos + 0.45); xl.append(AC_LAB[ac] if np.isfinite(ac) else 'indep.')
        pos += 2.6
    ax.axhline(0, **fs.REF)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel('log(estimated / true),\nP(both sources lost)')
    h = [Patch(facecolor='white', edgecolor=fs.BLUE, lw=0.9, label='item model'), Patch(facecolor=fs.tint(fs.RED, 0.35), edgecolor=fs.RED, lw=0.9, label='frailty model')]
    ax.legend(handles=h, loc='lower right', handlelength=1.4, fontsize=SMALL)
    fs.panel(ax, 'c')
    fs.save(fig, f'{OUT}/fig5_identification')


def figS1():
    tabD = pd.read_csv(f'{WK}/tabD_full.csv')
    fig = plt.figure(figsize=(180 * fs.MM, 108 * fs.MM), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, wspace=0.06, hspace=0.08)
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - 3.5 / 108))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[0, 1])]
    # (a) union run length (Prop. S1)
    ax = axs[0]
    Is = np.arange(1, 31); a_target = 1 / 24; mu = 0.1
    for ac in [np.inf, 2.0, 0.5]:
        pN = 1 - cm.sri_marginal(mu, AP, ac); r = a_target / pN
        eta, wq = cm.eta_nodes(ac, 160)
        aeta = r * (1 - cm.q_cond(np.array([mu]), eta, AP)[0])
        arl = np.array([1 / (1 - np.sum(wq * (1 - aeta) ** I)) for I in Is])
        ax.plot(Is, arl, color=AC_COL[ac], lw=1.1)
        sel = [0, 4, 9, 19, 29]
        ax.plot(Is[sel], arl[sel], ls='none', marker=AC_MK[ac], mfc='white', mec=AC_COL[ac], mew=0.9, ms=3.8)
    ax.axhline(24, **fs.REF)
    logx(ax, [1, 2, 5, 10, 30]); ax.set_xlim(0.95, 33)
    logy(ax, [1, 2, 5, 10, 24]); ax.set_ylim(0.9, 30)
    ax.set_xlabel('Streams monitored, m'); ax.set_ylabel('Union run length\n(periods)')
    h = [Line2D([], [], color=AC_COL[a], lw=1.1, marker=AC_MK[a], mfc='white', mec=AC_COL[a], ms=3.8, label=AC_LAB[a]) for a in [0.5, 2.0, np.inf]]
    ax.legend(handles=h, loc='lower left', handlelength=1.6, fontsize=SMALL)
    fs.panel(ax, 'a')
    # (b) ARL0 by rule
    ax = axs[1]
    rules = {'item_each_union': (fs.LIGHT, 'o', False, 'union, each stream at 24'),
             'union_ind': (fs.INK, 's', False, 'union, independence calibration'),
             'union_dep': (fs.INK, 's', True, 'union, frailty calibration'),
             'sys_ind': (fs.GOLD, 'D', False, 'system, independence calibration'),
             'sys_dep': (fs.GOLD, 'D', True, 'system, frailty calibration')}
    xs = {0.5: 0, 2.0: 1, np.inf: 2}
    offs = {'item_each_union': -0.34, 'union_ind': -0.17, 'union_dep': 0.0, 'sys_ind': 0.17, 'sys_dep': 0.34}
    for mth, (col, mk, filled, lab) in rules.items():
        for I, dI in [(5, -0.05), (10, 0.0), (30, 0.05)]:
            sub = tabD[(tabD.method == mth) & (tabD.I == I)].set_index('ac')
            xx = np.array([xs[a] for a in sub.index]) + offs[mth] + dI
            ax.errorbar(xx, sub.ARL0, yerr=[sub.ARL0 - sub.ARL0_lo, sub.ARL0_hi - sub.ARL0], fmt=mk, color=col,
                        mfc=col if filled else 'white', mec=col, ms=2.5 + 0.4 * (I == 30), mew=0.8, elinewidth=0.6, capsize=0)
    ax.axhline(24, **fs.REF)
    logy(ax, [1, 3, 10, 24, 50], ['1', '3', '10', '24', '50']); ax.set_ylim(0.9, 80)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([AC_LAB[0.5], AC_LAB[2.0], 'independence']); ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel(r'Realised ARL$_0$ (months)')
    h = [Line2D([], [], ls='none', marker=mk, color=col, mfc=col if filled else 'white', mec=col, ms=3.6, mew=0.8, label=lab) for col, mk, filled, lab in rules.values()]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3, handlelength=1.0, fontsize=SMALL, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)
    fs.panel(ax, 'c')
    # (c) detection
    ax = axs[2]
    for mth in ['item_each_union', 'union_dep', 'sys_dep']:
        col, mk, filled, lab = rules[mth]
        sub = tabD[tabD.method == mth].groupby('I')[['det_common', 'det_single']].mean()
        ax.plot(sub.index, sub.det_common, color=col, lw=1.1, ls='-', marker=mk, mfc=col, mec=col, ms=3.8, mew=0.8)
        ax.plot(sub.index, sub.det_single, color=col, lw=1.1, ls='--', marker=mk, mfc=col, mec=col, ms=3.8, mew=0.8)
    logx(ax, [5, 10, 30]); ax.set_xlim(4.5, 33); ax.set_ylim(0, 1.05)
    ax.set_xlabel('Streams monitored, m'); ax.set_ylabel('Shifts detected\nwithin 96 months')
    h = [Line2D([], [], color=rules[m][0], lw=1.1, marker=rules[m][1], mfc=rules[m][0], ms=3.8, label=l) for m, l in
         [('item_each_union', 'union, each at 24'), ('union_dep', 'union, frailty calib.'), ('sys_dep', 'system, frailty calib.')]]
    h += [Line2D([], [], color=fs.GREY, lw=1.1, ls='-', label='all streams ×2'), Line2D([], [], color=fs.GREY, lw=1.1, ls='--', label='one stream ×3')]
    ax.legend(handles=h, loc='lower left', ncol=1, handlelength=1.6, fontsize=SMALL, borderaxespad=0.3)
    fs.panel(ax, 'b')
    fs.save(fig, f'{OUT}/figS1_alarms')


def fig7():
    """Fig. 7 (v0.5): (a) P(two or more inputs disrupted) as intervals over the allowed set of dispersions;
    (b) re-sourcing priorities at the most dependent corner (bars) and at the fitted values (open markers)."""
    import json
    IV = json.load(open(f'{WK}/app/app_interval.json'))
    IMP = json.load(open(f'{WK}/app/app_importance.json'))
    fig, axs = figure(180, 58, 2, gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axs[0]
    ind_mc = 100 * IV['independence']['mc']['p_two']
    ax.axhline(ind_mc, **fs.REF)
    ax.text(1.5, ind_mc + 0.09, 'independence', color=fs.GREY, fontsize=SMALL, ha='center', va='bottom')
    # fitted value
    DX = 0.13
    ax.plot([0], [100 * IV['mle']['mc']['p_two']], ls='none', marker='o', mfc=fs.INK, mec='white', mew=0.6, ms=5.0, zorder=5)
    ax.plot([DX], [100 * IV['mle']['pairwise']['two']], ls='none', marker='D', mfc='white', mec=fs.INK, mew=0.9, ms=4.2, zorder=6)
    # intervals
    for k, lev in enumerate(['90', '96.5']):
        c = IV['corners'][lev]
        lo, hi = 100 * c['mc']['least']['p_two'], 100 * c['mc']['most']['p_two']
        x = k + 1
        ax.plot([x, x], [lo, hi], color=fs.INK, lw=1.2, solid_capstyle='butt', zorder=3)
        ax.plot([x, x], [lo, hi], ls='none', marker='o', mfc=fs.INK, mec='white', mew=0.6, ms=5.0, zorder=5)
        ax.plot([x + DX, x + DX], [100 * c['pairwise']['least']['two'], 100 * c['pairwise']['most']['two']], ls='none', marker='D', mfc='white', mec=fs.INK, mew=0.9, ms=4.2, zorder=6)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['fitted\nvalue', '90%\nintervals', '96.5%\nintervals']); ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 3.8)
    ax.set_ylabel('P(two or more inputs\ndisrupted in a month), %')
    h = [Line2D([], [], ls='none', marker='o', mfc=fs.INK, mec='white', ms=5.0, label='Monte Carlo'),
         Line2D([], [], ls='none', marker='D', mfc='white', mec=fs.INK, ms=4.2, label='pairwise formula')]
    ax.legend(handles=h, loc='lower left', handlelength=1.4)
    fs.panel(ax, 'a')
    ax = axs[1]
    im = IMP['importance']
    pr_lo = im['lower90']['two_or_more']['priority']; pr_hat = im['mle']['two_or_more']['priority']
    top = list(pr_lo)[:7]
    y = np.arange(len(top))[::-1]
    vals_lo = [100 * pr_lo[s] for s in top]; vals_hat = [100 * pr_hat.get(s, 0.0) for s in top]
    colr = [f2.ORG_COL[s.split('_')[1]] for s in top]
    ax.barh(y, vals_lo, height=0.6, color=colr)
    ax.plot(vals_hat, y, ls='none', marker='D', mfc='white', mec=fs.INK, mew=0.9, ms=4.0, clip_on=False, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels([f"{f2.SHORT[s.split('_')[0]]}, {f2.ORG_LAB[s.split('_')[1]]}" for s in top])
    ax.set_xlabel('Reduction in P(two or more inputs disrupted)\nif moved to its own origin, percentage points')
    ax.set_xlim(0, max(vals_lo) * 1.15)
    h = [Patch(facecolor=fs.tint(fs.INK, 0.6), label='at the most dependent corner'),
         Line2D([], [], ls='none', marker='D', mfc='white', mec=fs.INK, ms=4.0, label='at the fitted values')]
    ax.legend(handles=h, loc='lower right', handlelength=1.4)
    fs.panel(ax, 'b', dx=-62)
    fs.save(fig, f'{OUT}/fig7_system')


if __name__ == '__main__':
    which = sys.argv[1:] or ['5', 'S1', '7']
    if '5' in which:
        fig5(); print('fig 5 done', fs.margins(f'{OUT}/fig5_identification.png'))
    if 'S1' in which:
        figS1(); print('fig S1 done', fs.margins(f'{OUT}/figS1_alarms.png'))
    if '7' in which:
        fig7(); print('fig 7 done', fs.margins(f'{OUT}/fig7_system.png'))
