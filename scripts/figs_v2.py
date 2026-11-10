"""Paper S v0.2 figures 2-7 in the shared style (figstyle, RESS: labels 8 pt, double column 180 mm).
Numbers come from the same result files as the text; theory curves are recomputed from ccfmodel."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, glob, itertools
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np, pandas as pd
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, LogLocator, FixedLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import figstyle as fs
import ccfmodel as cm

SP = f'{SP_ROOT}'
WK = f'{SP}/work'; OUT = f'{WK}/figs_v2'
fs.apply(base=9)
SMALL = 7.4   # legends with many entries (6.6 pt when displayed at 160 mm)
MATRIX = 7.4
AP = 2.0
MM = fs.MM
# fixed meanings: common-cause strength (ordered): independence BLUE, alpha_c = 2 TEAL, alpha_c = 0.5 RED
AC_COL = {np.inf: fs.BLUE, 2.0: fs.TEAL, 0.5: fs.RED}
AC_MK = {np.inf: '^', 2.0: 's', 0.5: 'o'}
AC_LAB = {np.inf: 'independence', 2.0: r'$\alpha_c = 2$', 0.5: r'$\alpha_c = 0.5$'}
# models: frailty S-1 red filled circle, item model (independence) blue filled triangle, BFR gold open square
MODEL = {'S1': (fs.RED, 'o', True), 'IND': (fs.BLUE, '^', True), 'BFR': (fs.GOLD, 's', False)}
# origins (application)
ORG_COL = {'China': fs.RED, 'Japan': fs.BLUE, 'USA': fs.TEAL}
ORG_LAB = {'China': 'China', 'Japan': 'Japan', 'USA': 'United States'}
NUM = {}


def pow_labels(ticks):
    out = []
    for t in ticks:
        e = int(round(np.log10(t)))
        out.append('1' if e == 0 else (f'{t:g}' if (t >= 1 and t < 1000 and abs(t - round(t)) < 1e-9) else rf'$10^{{{e}}}$'))
    return out


def logx(ax, ticks, labels=None):
    ax.set_xscale('log')
    ax.set_xticks(ticks); ax.set_xticklabels(labels or [f'{t:g}' for t in ticks])
    ax.xaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10))); ax.xaxis.set_minor_formatter(NullFormatter())


def logy(ax, ticks, labels=None):
    ax.set_yscale('log')
    ax.set_yticks(ticks); ax.set_yticklabels(labels or [f'{t:g}' for t in ticks])
    ax.yaxis.set_minor_locator(LogLocator(subs=np.arange(2, 10))); ax.yaxis.set_minor_formatter(NullFormatter())


def mstyle(col, mk, filled, ms=4.0):
    return dict(color=col, marker=mk, ms=ms, mfc=col if filled else 'white', mec=col, mew=0.8)


def figure(w=180, h=60, ncols=2, nrows=1, **kw):
    fig, axs = plt.subplots(nrows, ncols, figsize=(w * MM, h * MM), constrained_layout=True, **kw)
    fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, wspace=0.06, hspace=0.08)
    return fig, np.atleast_1d(axs).ravel()


# ================================================================== Figure 2: system reliability and importance
def fig2():
    acs = np.logspace(np.log10(0.1), 2, 60)
    fig, axs = figure(180, 58, 3)
    ax = axs[0]; mu = 0.05
    for k, m in enumerate([2, 5, 10, 20]):
        vals = np.array([(1 - cm.sri_marginal(mu, AP, a) ** m) / (1 - cm.series_group(np.full(m, mu), AP, a)) for a in acs])
        fs.series(ax, acs, vals, k, 4, label=f'{m} inputs', marks=4, mrange=(0.12, 4))
        NUM[f'fig2a_m{m}_ac0.5'] = float(np.interp(np.log(0.5), np.log(acs), vals))
        NUM[f'fig2a_m{m}_ac2'] = float(np.interp(np.log(2), np.log(acs), vals))
    ax.axhline(1, **fs.REF)
    logx(ax, [0.1, 1, 10, 100]); ax.set_xlim(0.09, 110)
    ax.set_ylim(0.97, 2.4)
    ax.set_xlabel(r'Common-cause dispersion $\alpha_c$')
    ax.set_ylabel('Independence / true,\nP(at least one input lost)')
    ax.legend(loc='upper right', handlelength=1.6)
    fs.panel(ax, 'a')

    ax = axs[1]
    for k, mu_ in enumerate([0.01, 0.05, 0.2]):
        vals = np.array([cm.beta_joint(mu_, mu_, AP, a)[1] for a in acs])
        fs.series(ax, acs, vals, k, 3, label=rf'$\mu = {mu_:g}$', marks=4, mrange=(0.12, 4))
        NUM[f'fig2b_mu{mu_}_ac0.5'] = float(np.interp(np.log(0.5), np.log(acs), vals))
        NUM[f'fig2b_mu{mu_}_ac2'] = float(np.interp(np.log(2), np.log(acs), vals))
    ax.plot(acs, 1 + 1 / acs, **fs.REF)
    logx(ax, [0.1, 1, 10, 100]); ax.set_xlim(0.09, 110)
    logy(ax, [1, 2, 5, 10]); ax.set_ylim(0.9, 12)
    ax.set_xlabel(r'Common-cause dispersion $\alpha_c$')
    ax.set_ylabel('True / independence,\nP(both sources lost)')
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=fs.LIGHT, lw=0.7, ls=(0, (3, 1.6)))); l.append(r'$1 + 1/\alpha_c$')
    ax.legend(h, l, loc='upper right', handlelength=1.6)
    fs.panel(ax, 'b')

    ax = axs[2]
    V = json.load(open(f'{WK}/verify_importance.json'))['fig2c']
    cols = {'series': fs.BLUE, 'fewer_than_two': fs.RED, 'parallel': fs.TEAL}
    labs = {'series': 'all six needed', 'fewer_than_two': 'fewer than two lost', 'parallel': 'any one suffices'}
    ax.plot([0.1, 100], [0, 0], **fs.REF)
    ends = {}
    for name, d in V.items():
        ac = np.array(d['ac']); ex = 100 * np.array(d['exact']); pw = 100 * np.array(d['pairwise'])
        ax.plot(ac, ex, color=cols[name], lw=1.1)
        ax.plot(ac, pw, ls='none', marker='o', mfc='white', mec=cols[name], mew=0.9, ms=3.8)
        ends[name] = ex[-1]
    logx(ax, [0.1, 1, 10, 100]); ax.set_xlim(0.09, 110)
    ax.set_ylim(-11, 13); ax.set_yticks([-10, -5, 0, 5, 10])
    ax.set_xlabel(r'Common-cause dispersion $\alpha_c$')
    ax.set_ylabel('Common-cause importance of\nthe group (percentage points)')
    h = [Line2D([], [], color=cols[n], lw=1.1, label=labs[n]) for n in ['series', 'fewer_than_two', 'parallel']]
    h.append(Line2D([], [], ls='none', marker='o', mfc='white', mec=fs.INK, mew=0.9, ms=3.8, label='pairwise formula'))
    ax.legend(handles=h, loc='upper right', handlelength=1.4, fontsize=SMALL, borderaxespad=0.2, handletextpad=0.5)
    fs.panel(ax, 'c')
    fs.save(fig, f'{OUT}/fig2_system')


# ================================================================== Figure 3: CCF parameters
def fig3():
    fig, axs = figure(180, 58, 2)
    ax = axs[0]
    m, mu = 4, 0.05; ks = np.arange(1, m + 1); width = 0.26
    for j, a in enumerate([0.5, 2.0, np.inf]):
        pmf = cm.multiplicity_pmf(m, mu, AP, a)
        af = cm.alpha_factors(pmf)
        col = AC_COL[a]
        ax.bar(ks + (j - 1) * width, af, width=width * 0.9, color=col if np.isfinite(a) else 'white', edgecolor=col, lw=0.8, label=AC_LAB[a])
        NUM[f'fig3a_alpha_factors_ac{a}'] = af.tolist()
    logy(ax, [1e-6, 1e-4, 1e-2, 1], [r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$', r'$10^{0}$'])
    ax.set_ylim(1e-6, 3)
    ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel('Streams interrupted in the same period, $k$')
    ax.set_ylabel(r'$\alpha$-factor $\alpha_k$')
    ax.legend(loc='upper right', handlelength=1.4)
    fs.panel(ax, 'a')

    ax = axs[1]
    mus = np.logspace(np.log10(0.003), np.log10(0.3), 40); p_hit = 0.5
    for a in [0.5, 2.0]:
        a2 = np.array([cm.alpha_factors(cm.multiplicity_pmf(m, x, AP, a))[1] for x in mus])
        ax.plot(mus, a2, color=AC_COL[a], lw=1.1)
    mu_ref, thc = 0.05, 0.5
    w = thc * mu_ref ** 2 / p_hit ** 2
    def bfr_alpha2(x):
        r = 1 - (1 + max(x - w * p_hit, 1e-6) / AP) ** (-AP)
        rs = 1 - (1 - r) * (1 - p_hit)
        pk = lambda k, pp: special.comb(m, k) * pp ** k * (1 - pp) ** (m - k)
        P = np.array([(1 - w) * pk(k, r) + w * pk(k, rs) for k in range(m + 1)])
        return P[2] / P[1:].sum()
    b2 = np.array([bfr_alpha2(x) for x in mus])
    ax.plot(mus, b2, color=fs.GOLD, lw=1.1, ls='--')
    logx(ax, [0.003, 0.01, 0.03, 0.1, 0.3], ['0.003', '0.01', '0.03', '0.1', '0.3']); ax.set_xlim(0.0028, 0.33)
    logy(ax, [1e-3, 1e-2, 1e-1, 1], [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$'])
    ax.set_xlabel(r'Stream interruption rate $\mu$ (per period)')
    ax.set_ylabel(r'Share of double events, $\alpha_2$')
    h = [Line2D([], [], color=AC_COL[0.5], lw=1.1, label=r'frailty, $\alpha_c = 0.5$'),
         Line2D([], [], color=AC_COL[2.0], lw=1.1, label=r'frailty, $\alpha_c = 2$'),
         Line2D([], [], color=fs.GOLD, lw=1.1, ls='--', label='common shock (BFR)')]
    ax.legend(handles=h, loc='lower right', handlelength=2.0)
    fs.panel(ax, 'b')
    NUM['fig3b_bfr_w'] = w
    fs.save(fig, f'{OUT}/fig3_ccf')


# ================================================================== Figure 4: identification and union collapse
def fig4():
    fig, axs = figure(180, 58, 2)
    ax = axs[0]
    mus = np.logspace(np.log10(0.02), np.log10(0.5), 40)
    for k, m in enumerate([2, 5, 10, 30]):
        Tn = 8 / (m * (m - 1) * mus ** 2)
        c = fs.series(ax, mus, Tn, k, 4, marks=4, mrange=(0.025, 0.4))
        ax.text(0.53, Tn[-1], f'{m} streams', color=fs.INK, fontsize=SMALL, va='center')
    ax.plot(mus, 8 / mus ** 2, color=fs.INK, lw=1.0, ls=':')
    ax.text(0.043, 2.3e4, 'item dispersion, one stream', color=fs.INK, fontsize=SMALL, va='center')
    ax.axhline(120, **fs.REF)
    ax.text(0.5, 150, '120 months', color=fs.GREY, fontsize=SMALL, va='bottom', ha='right')
    logx(ax, [0.02, 0.05, 0.1, 0.2, 0.5], ['0.02', '0.05', '0.1', '0.2', '0.5']); ax.set_xlim(0.018, 0.95)
    logy(ax, [1e-1, 1, 10, 100, 1e3, 1e4], [r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$']); ax.set_ylim(0.03, 6e4)
    ax.set_xlabel(r'Stream interruption rate $\mu$ (per period)')
    ax.set_ylabel(r'Group-periods for SE($\theta$) = 0.5')
    fs.panel(ax, 'a')
    NUM['fig4a_T_m10_mu0.05'] = 8 / (10 * 9 * 0.05 ** 2); NUM['fig4a_T_item_mu0.05'] = 8 / 0.05 ** 2

    ax = axs[1]
    Is = np.arange(1, 31); a_target = 1 / 24; mu = 0.1
    for ac in [np.inf, 2.0, 0.5]:
        pN = 1 - cm.sri_marginal(mu, AP, ac); r = a_target / pN
        eta, wq = cm.eta_nodes(ac, 160)
        aeta = r * (1 - cm.q_cond(np.array([mu]), eta, AP)[0])
        arl = np.array([1 / (1 - np.sum(wq * (1 - aeta) ** I)) for I in Is])
        ax.plot(Is, arl, color=AC_COL[ac], lw=1.1, label=AC_LAB[ac])
        sel = [0, 4, 9, 19, 29]
        ax.plot(Is[sel], arl[sel], ls='none', marker=AC_MK[ac], mfc='white', mec=AC_COL[ac], mew=0.9, ms=3.8)
        NUM[f'fig4b_unionARL_ac{ac}'] = {int(i): float(v) for i, v in zip(Is, arl) if i in (5, 10, 30)}
    ax.axhline(24, **fs.REF)
    logx(ax, [1, 2, 5, 10, 30]); ax.set_xlim(0.95, 33)
    logy(ax, [1, 2, 5, 10, 24]); ax.set_ylim(0.9, 30)
    ax.set_xlabel('Streams monitored, m')
    ax.set_ylabel('Run length of the union (periods)')
    h = [Line2D([], [], color=AC_COL[a], lw=1.1, marker=AC_MK[a], mfc='white', mec=AC_COL[a], ms=3.8, label=AC_LAB[a]) for a in [0.5, 2.0, np.inf]]
    ax.legend(handles=h, loc='lower left', handlelength=1.8)
    fs.panel(ax, 'b')
    fs.save(fig, f'{OUT}/fig4_ident_union')


# ================================================================== Figure 5: simulation studies C and D
def load(pat):
    rows = []
    for f in glob.glob(pat):
        rows += [json.loads(l) for l in open(f)]
    d = pd.DataFrame(rows)
    if 'error' in d:
        d = d[d['error'].isna()]
    key = [k for k in ['I', 'T', 'mub', 'ac', 'rep'] if k in d.columns]
    return d.drop_duplicates(subset=key).reset_index(drop=True)


def fig5():
    C = load(f'{WK}/studyC/C_w*.jsonl')
    C['finite'] = np.isfinite(C.ac)
    C['lt_true'] = np.where(C.finite, -np.log(C.ac.where(C.finite, 1.0)), np.log(1e-4))
    C['cover'] = (C.lo <= C.lt_true + 1e-9) & (C.lt_true - 1e-9 <= C.hi)
    KR = pd.read_csv(f'{WK}/studyC_joint_true_rates.csv')
    C = C.merge(KR, on=['I', 'T', 'mub', 'ac', 'rep'], how='left')
    tabD = pd.read_csv(f'{WK}/tabD_full.csv')
    fig, axs = figure(180, 124, 2, 2)
    # (a) coverage
    ax = axs[0]
    for ac in [0.5, 2.0]:
        for T, ls in [(120, '-'), (60, '--')]:
            g = C[(C.ac == ac) & (C['T'] == T)].groupby('I').cover.mean()
            ax.plot(g.index, g.values, color=AC_COL[ac], ls=ls, lw=1.1, marker=AC_MK[ac], mfc=AC_COL[ac], mec=AC_COL[ac], ms=3.8, mew=0.8)
    gi = C[~C.finite].groupby('I').cover.mean()
    ax.plot(gi.index, gi.values, color=AC_COL[np.inf], lw=1.1, marker=AC_MK[np.inf], mfc=AC_COL[np.inf], mec=AC_COL[np.inf], ms=3.8)
    ax.axhline(0.9, **fs.REF)
    logx(ax, [2, 5, 10, 30]); ax.set_xlim(1.8, 36); ax.set_ylim(0.78, 1.01)
    ax.set_xlabel('Streams per group, m'); ax.set_ylabel('Coverage of the 90%\ninterval for ' + r'$\theta_c$')
    h = [Line2D([], [], color=AC_COL[a], lw=1.1, marker=AC_MK[a], mfc=AC_COL[a], ms=3.8, label=AC_LAB[a]) for a in [0.5, 2.0, np.inf]]
    h2 = [Line2D([], [], color=fs.INK, lw=1.1, ls='-', label='120 months'), Line2D([], [], color=fs.INK, lw=1.1, ls='--', label='60 months')]
    leg1 = ax.legend(handles=h, loc='lower left', handlelength=1.8); ax.add_artist(leg1)
    ax.legend(handles=h2, loc='lower right', handlelength=1.8)
    fs.panel(ax, 'a')
    # (b) joint-loss error boxes
    ax = axs[1]
    pos = 0; xt, xl = [], []
    for ac in [0.5, 2.0, np.inf]:
        sub = C[C.ac == ac]
        for k, (col_name, mdl) in enumerate([('kI_D2', 'IND'), ('kS_D2', 'S1')]):
            v = np.log(sub[col_name].dropna().values)
            q = np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9])
            colr, _, filled = MODEL[mdl]
            x = pos + k * 0.9
            ax.plot([x, x], [q[0], q[1]], color=colr, lw=0.8); ax.plot([x, x], [q[3], q[4]], color=colr, lw=0.8)
            ax.add_patch(Rectangle((x - 0.3, q[1]), 0.6, q[3] - q[1], facecolor=fs.tint(colr, 0.35) if filled else 'white', edgecolor=colr, lw=0.9))
            ax.plot([x - 0.3, x + 0.3], [q[2], q[2]], color=colr, lw=1.2)
        xt.append(pos + 0.45); xl.append(AC_LAB[ac])
        pos += 2.6
    ax.axhline(0, **fs.REF)
    ax.set_xticks(xt); ax.set_xticklabels(xl)
    ax.set_ylabel('log(estimated / true),\nP(both sources lost)')
    h = [Patch(facecolor='white', edgecolor=fs.BLUE, lw=0.9, label='item model'), Patch(facecolor=fs.tint(fs.RED, 0.35), edgecolor=fs.RED, lw=0.9, label='frailty model')]
    ax.legend(handles=h, loc='lower right', handlelength=1.4)
    fs.panel(ax, 'b')
    # (c) ARL0 by rule
    ax = axs[2]
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
                        mfc=col if filled else 'white', mec=col, ms=2.7 + 0.4 * (I == 30), mew=0.8, elinewidth=0.6, capsize=0)
    ax.axhline(24, **fs.REF)
    logy(ax, [1, 3, 10, 24, 50], ['1', '3', '10', '24', '50']); ax.set_ylim(0.9, 80)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([AC_LAB[0.5], AC_LAB[2.0], 'independence']); ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel(r'Realised ARL$_0$ (months)')
    h = [Line2D([], [], ls='none', marker=mk, color=col, mfc=col if filled else 'white', mec=col, ms=3.6, mew=0.8, label=lab) for col, mk, filled, lab in rules.values()]
    ax.legend(handles=h, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, handlelength=1.0, fontsize=SMALL, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.0)
    fs.panel(ax, 'c')
    # (d) detection
    ax = axs[3]
    for mth in ['item_each_union', 'union_dep', 'sys_dep']:
        col, mk, filled, lab = rules[mth]
        sub = tabD[tabD.method == mth].groupby('I')[['det_common', 'det_single']].mean()
        ax.plot(sub.index, sub.det_common, color=col, lw=1.1, ls='-', marker=mk, mfc=col, mec=col, ms=3.8, mew=0.8)
        ax.plot(sub.index, sub.det_single, color=col, lw=1.1, ls='--', marker=mk, mfc=col, mec=col, ms=3.8, mew=0.8)
    logx(ax, [5, 10, 30]); ax.set_xlim(4.5, 33); ax.set_ylim(0, 1.05)
    ax.set_xlabel('Streams monitored, m'); ax.set_ylabel('Shifts detected\nwithin 96 months')
    h = [Line2D([], [], color=rules[m][0], lw=1.1, marker=rules[m][1], mfc=rules[m][0], ms=3.8, label=l) for m, l in
         [('item_each_union', 'union, each stream at 24'), ('union_dep', 'union, frailty calibration'), ('sys_dep', 'system, frailty calibration')]]
    h += [Line2D([], [], color=fs.GREY, lw=1.1, ls='-', label='all streams ×2'), Line2D([], [], color=fs.GREY, lw=1.1, ls='--', label='one stream ×3')]
    ax.legend(handles=h, loc='lower left', ncol=1, handlelength=1.8, fontsize=SMALL)
    fs.panel(ax, 'd')
    fs.save(fig, f'{OUT}/fig5_simulations')


# ================================================================== Figure 6: application, estimates and mechanism
SHORT = {'281111': 'HF', '370790': 'Resist', '392099': 'PI film', '811292': 'Ga/Ge metal', '282560': 'Ge oxide',
         '381800': 'Si wafer', '280461': 'Si', '281290': 'Halides', '282619': 'Fluorides', '280429': 'Rare gases'}


def fig6():
    fit = json.load(open(f'{WK}/app/app_fit_monthly.json'))
    D = pd.read_csv(f'{WK}/app/app_diagnostic.csv', parse_dates=['period'])
    IMP = json.load(open(f'{WK}/app/app_importance.json'))
    fig = plt.figure(figsize=(180 * MM, 118 * MM), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=4 / 72, h_pad=4 / 72, wspace=0.06, hspace=0.1)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    # (a) profile likelihood ratio vs beta_J
    ax = fig.add_subplot(gs[0, 0])
    for o in ['China', 'Japan', 'USA']:
        pc = fit['profile_curves'][o]
        th = np.exp(np.array(pc['log_thc'])); bj = th / (1 + th)
        ax.plot(bj, pc['lr'], color=ORG_COL[o], lw=1.1, label=ORG_LAB[o])
        ah = fit['alpha_c'][o]['ac_hat']; bh = 0 if ah > 5e3 else 1 / (1 + ah)
        ax.plot([bh], [0], marker='o', ms=4.2, mfc=ORG_COL[o], mec='white', mew=0.6, ls='none', clip_on=False, zorder=5)
    ax.axhline(2.7055, **fs.REF)
    ax.set_xlim(0, 1); ax.set_ylim(0, 8)
    ax.set_xlabel(r'Common-cause fraction $\beta_J = 1/(1+\alpha_c)$'); ax.set_ylabel('Profile likelihood ratio')
    ax.legend(loc='upper left', handlelength=1.6)
    fs.panel(ax, 'a')
    # (b) months with two or more onsets, diagnostic at the lower bound (open) and at the MLE (filled)
    ax = fig.add_subplot(gs[0, 1:])
    rows = {'China': 2, 'Japan': 1, 'USA': 0}
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
    ax.set_yticks([v + 0.41 for v in rows.values()]); ax.set_yticklabels([ORG_LAB[o] for o in rows])
    ax.set_ylim(-0.1, 3.0)
    ax.set_xlim(pd.Timestamp('2014-09-01'), pd.Timestamp('2026-03-01'))
    ax.set_xlabel('Months with two or more onsets')
    ax.set_ylabel(r'Diagnostic $c_{gt}$')
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
        info = IMP['concentration'][o]
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
        mx = 7
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


# ================================================================== Figure 7: system consequences
def fig7():
    IMP = json.load(open(f'{WK}/app/app_importance.json'))
    fig, axs = figure(180, 58, 2, gridspec_kw=dict(width_ratios=[1, 1.35]))
    ax = axs[0]
    mc = IMP['mc']; im = IMP['importance']
    scen = ['independence', 'fitted frailty', '90% lower bound']
    mcv = [100 * mc['independence_two'], 100 * mc['mle_two'], 100 * mc['lower_two']]
    pwv = [100 * im['mle']['two_or_more']['independence'], 100 * im['mle']['two_or_more']['pairwise'], 100 * im['lower90']['two_or_more']['pairwise']]
    cols = [fs.BLUE, fs.TEAL, fs.RED]
    ax.bar(range(3), mcv, width=0.6, color=cols)
    ax.plot(range(3), pwv, ls='none', marker='D', mfc='white', mec=fs.INK, mew=0.9, ms=4.2, zorder=5)
    ax.set_xticks(range(3)); ax.set_xticklabels(['independence', 'fitted\nfrailty', '90% lower\nbound'])
    ax.set_ylabel('P(two or more inputs\ndisrupted in a month), %')
    ax.set_ylim(0, 3.6)
    h = [Patch(facecolor=fs.tint(fs.INK, 0.6), label='Monte Carlo'),
         Line2D([], [], ls='none', marker='D', mfc='white', mec=fs.INK, ms=4.2, label='pairwise formula')]
    ax.legend(handles=h, loc='upper left', handlelength=1.4)
    fs.panel(ax, 'a')
    ax = axs[1]
    pr_lo = im['lower90']['two_or_more']['priority']; pr_hat = im['mle']['two_or_more']['priority']
    top = list(pr_lo)[:7]
    y = np.arange(len(top))[::-1]
    vals_lo = [100 * pr_lo[s] for s in top]; vals_hat = [100 * pr_hat.get(s, 0.0) for s in top]
    colr = [ORG_COL[s.split('_')[1]] for s in top]
    ax.barh(y, vals_lo, height=0.6, color=colr)
    ax.plot(vals_hat, y, ls='none', marker='D', mfc='white', mec=fs.INK, mew=0.9, ms=4.0, clip_on=False, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels([f"{SHORT[s.split('_')[0]]}, {ORG_LAB[s.split('_')[1]]}" for s in top])
    ax.set_xlabel('Reduction in P(two or more inputs disrupted)\nif moved to its own origin, percentage points')
    ax.set_xlim(0, max(vals_lo) * 1.15)
    h = [Patch(facecolor=fs.tint(fs.INK, 0.6), label=r'at the 90% lower bound of $\alpha_c$'),
         Line2D([], [], ls='none', marker='D', mfc='white', mec=fs.INK, ms=4.0, label='at the fitted value')]
    ax.legend(handles=h, loc='lower right', handlelength=1.4)
    fs.panel(ax, 'b', dx=-62)
    fs.save(fig, f'{OUT}/fig7_system')


if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1:] or ['2', '3', '4', '5', '6', '7']
    for w in which:
        globals()[f'fig{w}']()
        print('fig', w, 'done', fs.margins(f'{OUT}/' + {'2': 'fig2_system', '3': 'fig3_ccf', '4': 'fig4_ident_union', '5': 'fig5_simulations', '6': 'fig6_application', '7': 'fig7_system'}[w] + '.png'))
    json.dump(NUM, open(f'{WK}/fig_numbers_v2.json', 'w'), indent=1, default=float)
