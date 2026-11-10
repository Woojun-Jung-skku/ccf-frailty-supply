"""Paper S v0.7: Figs 6 and 7 of the application under the main model (S-0 + Lunar New Year covariate on the China group).
Reads work/app/app_v07_main.json, app_v07_diagnostic.csv, app_v07_bootstrap.json; writes fig6_application and fig7_system
into work/figs_v2 (same file names as before; the v0.6 versions are kept in work/figs_v2/v06_backup)."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import figstyle as fs
import figs_v2 as f2
from figs_v2 import figure, SMALL, WK, OUT, ORG_COL, ORG_LAB, SHORT, MATRIX

MM = fs.MM
MAIN = json.load(open(f'{WK}/app/app_v07_main.json'))


def fig6():
    D = pd.read_csv(f'{WK}/app/app_v07_diagnostic.csv', parse_dates=['period'])
    fig = plt.figure(figsize=(180 * MM, 118 * MM), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, wspace=0.06, hspace=0.1)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    # (a) profile likelihood ratio vs beta_J
    ax = fig.add_subplot(gs[0, 0])
    for o in ['China', 'Japan', 'USA']:
        pc = MAIN['alpha_c'][o]
        th = np.exp(np.array(pc['log_thc'])); bj = th / (1 + th); lr = np.array(pc['lr'])
        ok = lr <= 9.5
        ax.plot(bj[ok], lr[ok], color=ORG_COL[o], lw=1.1, label=ORG_LAB[o])
        ah = pc['ac_hat']; bh = 0 if ah > 5e3 else 1 / (1 + ah)
        ax.plot([bh], [0], marker='o', ms=4.2, mfc=ORG_COL[o], mec='white', mew=0.6, ls='none', clip_on=False, zorder=5)
    ax.axhline(MAIN['cutoffs']['90'], **fs.REF)
    ax.set_xlim(0, 1); ax.set_ylim(0, 8)
    ax.set_xlabel(r'Common-cause fraction $\beta_J = 1/(1+\alpha_c)$'); ax.set_ylabel('Profile likelihood ratio')
    hh, ll = ax.get_legend_handles_labels()   # compact legend in the empty upper left; the long name is set on two lines to clear the curves
    ax.legend(hh, [l.replace('United States', 'United\nStates') for l in ll], loc='upper left', handlelength=1.0, fontsize=SMALL, borderaxespad=0.2, handletextpad=0.5)
    fs.panel(ax, 'a')
    # (b) months with two or more onsets: diagnostic at the 90% lower bound (open) and at the fitted value (filled)
    ax = fig.add_subplot(gs[0, 1:])
    rows = {'China': 2.2, 'Japan': 1.1, 'USA': 0}   # rows 1.1 apart; each runs from 0 to 1 (0.82 units) with 0.5 dashed
    for o in ['China', 'Japan', 'USA']:
        d = D[(D.origin == o) & (D.k >= 2)]
        y0 = rows[o]
        for _, r in d.iterrows():
            x = r.period
            ax.plot([x, x], [y0, y0 + 0.82 * r.c_lo], color=fs.tint(ORG_COL[o], 0.55), lw=0.9, zorder=1)
            ax.plot(x, y0 + 0.82 * r.c_lo, marker='o', ms=2.6 + 0.9 * r.k, mfc='white', mec=ORG_COL[o], mew=0.8, ls='none', zorder=3)
            if r.c_hat >= 0.05:
                ax.plot(x, y0 + 0.82 * r.c_hat, marker='o', ms=2.6 + 0.9 * r.k, mfc=ORG_COL[o], mec='white', mew=0.5, ls='none', zorder=4)
        ax.axhline(y0, color=fs.HAIR, lw=0.5, zorder=0)
        ax.axhline(y0 + 0.82 * 0.5, color=fs.LIGHT, lw=0.6, ls=(0, (3, 1.6)), zorder=0)
    # Lunar New Year months as faint vertical hairlines
    lny = sorted(D[D.lny == 1].period.unique())
    for x in lny:
        ax.axvline(x, color=fs.HAIR, lw=0.5, zorder=0)
    ax.set_yticks([v + 0.41 for v in rows.values()]); ax.set_yticklabels([ORG_LAB[o] for o in rows])
    ax.set_ylim(-0.1, 3.2)
    ax.set_xlim(pd.Timestamp('2014-09-01'), pd.Timestamp('2026-03-01'))
    ax.set_xlabel('Months with two or more onsets (hairlines: Lunar New Year months)')
    ax.set_ylabel(r'Diagnostic $c_{gt}$')
    # scale of c_gt on the right: each origin's row runs from 0 to 1, with 0.5 dashed
    ax2 = ax.twinx(); ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks([v + d for v in rows.values() for d in (0, 0.41, 0.82)]); ax2.set_yticklabels(['0', '0.5', '1'] * 3, fontsize=SMALL, color=fs.GREY)
    ax2.spines['right'].set_visible(True); ax2.spines['right'].set_color(fs.LIGHT); ax2.spines['top'].set_visible(False)
    ax2.tick_params(axis='y', length=2.0, width=0.5, pad=1.5, color=fs.LIGHT)
    h = [Line2D([], [], ls='none', marker='o', mfc='white', mec=fs.INK, ms=4.4, label=r'at the 90% lower bound of $\alpha_c$'),
         Line2D([], [], ls='none', marker='o', mfc=fs.INK, mec='white', ms=4.4, label='at the fitted value'),
         Line2D([], [], ls='none', marker='o', mfc='white', mec=fs.INK, ms=2.6 + 0.9 * 2, label='2 onsets'),
         Line2D([], [], ls='none', marker='o', mfc='white', mec=fs.INK, ms=2.6 + 0.9 * 4, label='4 onsets')]
    ax.legend(handles=h, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, handlelength=1.2, fontsize=SMALL, borderaxespad=0.2, columnspacing=1.0)
    fs.panel(ax, 'b')
    # (c)-(e) pairwise co-onset matrices by origin
    letters = ['c', 'd', 'e']
    for j, o in enumerate(['China', 'Japan', 'USA']):
        ax = fig.add_subplot(gs[1, j])
        info = MAIN['concentration'][o]
        sids = info['streams']
        codes = [s.split('_')[0] for s in sids]
        order = np.argsort([list(SHORT).index(c) for c in codes])
        codes = [codes[i] for i in order]; sids_o = [sids[i] for i in order]
        m = len(sids_o)
        M = np.zeros((m, m))
        for key, n in info['observed_pairs'].items():
            a, b = key.split('|')
            ia, ib = sids_o.index(a), sids_o.index(b)
            M[ia, ib] = M[ib, ia] = n
        for r in range(m):
            for c in range(m):
                if c >= r:
                    continue
                n = M[r, c]
                fc = 'white' if n == 0 else fs.tint(ORG_COL[o], max(0.0, 0.78 - 0.13 * min(n, 6)))
                off = 8 - (m - 1)
                ax.add_patch(Rectangle((c, m - 1 - r + off), 1, 1, facecolor=fc, edgecolor=fs.HAIR, lw=0.4))
                if n > 0:
                    ax.text(c + 0.5, m - 1 - r + off + 0.5, f'{int(n)}', ha='center', va='center', fontsize=MATRIX,
                            color='white' if n >= 4 else fs.INK)
        ax.set_xlim(0, 8); ax.set_ylim(0, 8)
        ax.set_xticks(np.arange(m - 1) + 0.5); ax.set_xticklabels([SHORT[c] for c in codes[:-1]], rotation=90, fontsize=MATRIX)
        ax.set_yticks(np.arange(m - 1) + 0.5 + (8 - (m - 1))); ax.set_yticklabels([SHORT[c] for c in codes[1:]][::-1], fontsize=MATRIX)
        ax.tick_params(length=0)
        for s in ['left', 'bottom']:
            ax.spines[s].set_visible(False)
        ax.set_aspect('equal')
        ax.set_xlabel(ORG_LAB[o]); ax.xaxis.set_label_coords(0.5, -0.5)
        fs.panel(ax, letters[j], dx=-38)
    fs.save(fig, f'{OUT}/fig6_application')


def fig7():
    """(a) P(two or more inputs disrupted) in an ordinary month: fitted value, ranges over the 90% and 96.5% boxes
    (Monte Carlo filled, pairwise open), independence dashed, 90% bootstrap ranges of the ends as grey bands;
    (b) re-sourcing priorities at the most dependent corner of the 96.5% box (bars) and at the fitted values (open)."""
    S = MAIN['system']['ordinary']
    BOOT = json.load(open(f'{WK}/app/app_v07_bootstrap.json'))
    bs = BOOT['summary']
    fig, axs = figure(180, 58, 2, gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axs[0]
    ind_mc = 100 * S['independence']['mc']['p_two']
    ax.axhline(ind_mc, **fs.REF)
    ax.text(0.58, ind_mc - 0.07, 'independence', color=fs.GREY, fontsize=SMALL, ha='center', va='top')
    DX = 0.13; BW = 0.42
    band = dict(facecolor=fs.tint(fs.INK, 0.86), edgecolor='none', alpha=0.75, zorder=0.8)
    # fitted value, with its bootstrap band
    ax.add_patch(Rectangle((0 - BW / 2, 100 * bs['F']['q05']), BW, 100 * (bs['F']['q95'] - bs['F']['q05']), **band))
    ax.plot([0], [100 * S['fitted']['mc']['p_two']], ls='none', marker='o', mfc=fs.INK, mec='white', mew=0.6, ms=5.0, zorder=5)
    ax.plot([DX], [100 * S['fitted']['pairwise']], ls='none', marker='D', mfc='white', mec=fs.INK, mew=0.9, ms=4.2, zorder=6)
    # intervals over the boxes
    for k, lev in enumerate(['90', '96.5']):
        lo, hi = 100 * S[f'least{lev}']['mc']['p_two'], 100 * S[f'most{lev}']['mc']['p_two']
        x = k + 1
        if lev == '96.5':
            # bootstrap bands of the two ends, side by side: lower end to the left, upper end to the right
            ax.add_patch(Rectangle((x - 0.17 - 0.12, 100 * bs['L']['q05']), 0.24, 100 * (bs['L']['q95'] - bs['L']['q05']), **band))
            ax.add_patch(Rectangle((x + 0.17 - 0.12, 100 * bs['U']['q05']), 0.24, 100 * (bs['U']['q95'] - bs['U']['q05']), **band))
        ax.plot([x, x], [lo, hi], color=fs.INK, lw=1.2, solid_capstyle='butt', zorder=3)
        ax.plot([x, x], [lo, hi], ls='none', marker='o', mfc=fs.INK, mec='white', mew=0.6, ms=5.0, zorder=5)
        ax.plot([x + DX, x + DX], [100 * S[f'least{lev}']['pairwise'], 100 * S[f'most{lev}']['pairwise']], ls='none', marker='D', mfc='white', mec=fs.INK, mew=0.9, ms=4.2, zorder=6)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['fitted\nvalue', '90%\nintervals', '96.5%\nintervals']); ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 3.5); ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel('P(≥ 2 inputs disrupted\nin an ordinary month), %')
    h = [Line2D([], [], ls='none', marker='o', mfc=fs.INK, mec='white', ms=5.0, label='Monte Carlo'),
         Line2D([], [], ls='none', marker='D', mfc='white', mec=fs.INK, ms=4.2, label='pairwise formula'),
         Patch(facecolor=fs.tint(fs.INK, 0.82), label='90% bootstrap range (margins)')]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.47, 1.02), handlelength=1.4, fontsize=SMALL, borderaxespad=0.1)
    fs.panel(ax, 'a')
    ax = axs[1]
    pr_lo = S['most96.5']['priority']; pr_hat = S['fitted']['priority']
    top = [k for k, _ in sorted(pr_lo.items(), key=lambda kv: -kv[1])[:7]]
    y = np.arange(len(top))[::-1]
    vals_lo = [100 * pr_lo[s] for s in top]; vals_hat = [100 * pr_hat.get(s, 0.0) for s in top]
    colr = [ORG_COL[s.split('_')[1]] for s in top]
    ax.barh(y, vals_lo, height=0.6, color=colr)
    ax.plot(vals_hat, y, ls='none', marker='D', mfc='white', mec=fs.INK, mew=0.9, ms=4.0, clip_on=False, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels([f"{SHORT[s.split('_')[0]]}, {ORG_LAB[s.split('_')[1]]}" for s in top])
    ax.set_xlabel('Reduction in P(two or more inputs disrupted), %-points,\nif the stream is given a second origin outside its group')
    ax.set_xlim(0, max(vals_lo) * 1.15)
    h = [Patch(facecolor=fs.tint(fs.INK, 0.6), label='at the most dependent corner'),
         Line2D([], [], ls='none', marker='D', mfc='white', mec=fs.INK, ms=4.0, label='at the fitted values')]
    ax.legend(handles=h, loc='lower right', handlelength=1.4)
    fs.panel(ax, 'b', dx=-62)
    fs.save(fig, f'{OUT}/fig7_system')


if __name__ == '__main__':
    which = sys.argv[1:] or ['6', '7']
    if '6' in which:
        fig6(); print('fig 6 done', fs.margins(f'{OUT}/fig6_application.png'))
    if '7' in which:
        fig7(); print('fig 7 done', fs.margins(f'{OUT}/fig7_system.png'))
