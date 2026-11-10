"""Tokens for paper S v0.7: Tables S5 (calendar and demand variants), S6 (bootstrap), S7 (class of priors) and the
bootstrap numbers quoted in the main text. Reads work/app/app_v07_*.json, writes work/tokens_v07.json."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, numpy as np
W = f'{SP_ROOT}/work'
cal = json.load(open(f'{W}/app/app_v07_calendar.json'))
main = json.load(open(f'{W}/app/app_v07_main.json'))
pri = json.load(open(f'{W}/app/app_v07_priors.json'))
try:
    boot = json.load(open(f'{W}/app/app_v07_bootstrap.json'))
except FileNotFoundError:
    boot = None
tok = {}
ORIG = ['China', 'Japan', 'USA']
LAB = {'China': 'China', 'Japan': 'Japan', 'USA': 'United States'}


def fa(x, nd=2):
    return '∞' if (x is None or not np.isfinite(x) or x > 5e3) else (f'{x:.{nd}f}' if x < 100 else f'{x:.0f}')


def interval(d):
    return f"{fa(d['ac_hat'])} ({fa(d['ac_lo'])} – {fa(d['ac_hi'], 1)})"


def daic(d):
    return d['aic_S1'] - d['aic_IND']


rows = []
def row(name, d, n_onsets=None):
    b = d.get('beta')
    bs = f"{b['hat']:.2f} ({b['lo90']:.2f} – {b['hi90']:.2f}); {b['lr_zero']:.1f}" if b else '–'
    cells = [name, str(n_onsets) if n_onsets else '272', bs]
    for o in ORIG:
        cells.append(f"{interval(d['origins'][o])}; {max(d['origins'][o]['lr_boundary'], 0):.2f}")
    cells.append(f"{daic(d):+.1f}")
    rows.append('| ' + ' | '.join(cells) + ' |')


row('All onsets, no holiday term', cal['baseline'])
row('Holiday month, China group (main model)', cal['holiday']['H1_China'])
row('Holiday month, all streams', cal['holiday']['H1_all'])
row('Holiday month and next, China group', cal['holiday']['H2_China'])
row('Holiday month and next, all streams', cal['holiday']['H2_all'])
for thr, lab in [('demand75', '75%'), ('demand50', '50%')]:
    s = cal['supply'][thr]
    row(f'Supply-side onsets ({lab} rule), no holiday term', s, s['n_onsets'])
    row(f'Supply-side onsets ({lab} rule), holiday month, China group', s['H1_China'], s['n_onsets'])
head = ('| Fit | Onsets | $\\hat\\beta$ (90% interval); LR against $\\beta = 0$ | China: $\\hat\\alpha_c$ (90% interval); LR | Japan: $\\hat\\alpha_c$ (90% interval); LR | United States: $\\hat\\alpha_c$ (90% interval); LR | AIC, frailty − independence |\n'
        '|------------------------|------|---------------|--------------|-------------|--------------|-------|\n')
tok['TABLE_S5'] = head + '\n'.join(rows)

# descriptives for S6
desc = cal['holiday_descriptives']
tok['S6_RATES'] = '; '.join(f"{LAB[o]} {100 * desc[o]['H1']['onsets_in'] / desc[o]['H1']['active_in']:.1f}% ({desc[o]['H1']['onsets_in']}/{desc[o]['H1']['active_in']}) against {100 * desc[o]['H1']['onsets_out'] / desc[o]['H1']['active_out']:.1f}% ({desc[o]['H1']['onsets_out']}/{desc[o]['H1']['active_out']})" for o in ORIG)
cl = cal['classification']
by_o = cl['by_origin']
parts = [f"{LAB[o]} {by_o[o]['demand75']} of {by_o[o]['n']}" for o in ORIG]
tok['S6_CLASS'] = f"{cl['demand75']} of {cl['n_onsets']} onsets at the 75% threshold ({', '.join(parts)}) and {cl['demand50']} at 50%"
co = cl['co_onset_months']
tok['S6_CO'] = '; '.join(f"{LAB[o]} {co[o]['months_ge2']} to {co[o]['months_ge2_after75']} (75%) and {co[o]['months_ge2_after50']} (50%)" for o in ORIG)

# Table S7: class of priors
P = pri['posteriors']; PR = pri['prior_only']; PP = pri['predictive']
names = {'P1': 'P1, towards independence: $N(\\log 20, 1^2)$', 'P2': 'P2, neutral: $N(\\log 2, 1.5^2)$', 'P3': 'P3, towards a strong common cause: $N(\\log 0.5, 1^2)$',
         'P4': 'P4, hierarchical: $N(\\nu_c, \\tau_c^2)$, $\\nu_c \\sim N(\\log 2, 1.5^2)$, $\\tau_c \\sim \\text{half-}N(0, 1)$'}
rows = []
for k in ['P1', 'P2', 'P3', 'P4']:
    pm = f"{PR[k]['bj_median']:.2f} ({PR[k]['bj_lo90']:.2f} – {PR[k]['bj_hi90']:.2f})" if k in PR else 'group-specific'
    cells = [names[k], pm]
    for o in ORIG:
        d = P[k][o]
        cells.append(f"{d['bj_median']:.2f} ({d['bj_lo90']:.2f} – {d['bj_hi90']:.2f})")
    cells.append(f"{100 * PP[k]['mean']:.2f} ({100 * PP[k]['q05']:.2f} – {100 * PP[k]['q95']:.2f})")
    rows.append('| ' + ' | '.join(cells) + ' |')
head = ('| Prior on $\\log\\alpha_c$ | Prior median of $\\beta_J$ (90% interval) | China: posterior median (90%) | Japan | United States | $P$(two or more inputs disrupted), %: posterior mean (90%) |\n'
        '|----------------------|------------|------------|----------|----------|--------------|\n')
tok['TABLE_S7'] = head + '\n'.join(rows)
tok['S8_HYPER'] = f"{pri['P4_hyper']['tau_median']:.2f} (90% interval {pri['P4_hyper']['tau_q05']:.2f} – {pri['P4_hyper']['tau_q95']:.2f})"
lu = pri['lower_upper_expectation']
tok['S8_LU'] = f"{100 * lu['lower']:.2f}% ({lu['argmin']}) and {100 * lu['upper']:.2f}% ({lu['argmax']})"

# bootstrap
if boot is not None:
    s = boot['summary']; r = boot['ranking']; m = boot['mc_check']; og = boot['original']
    pc = lambda v: f'{100 * v:.2f}'
    tok['BL'] = f"{pc(s['L']['q05'])}–{pc(s['L']['q95'])}"
    tok['BU'] = f"{pc(s['U']['q05'])}–{pc(s['U']['q95'])}"
    tok['BF'] = f"{pc(s['F']['q05'])}–{pc(s['F']['q95'])}"
    # half-width of the 90% bootstrap range, averaged over the two ends, in percentage points
    hw_l = (s['L']['q95'] - s['L']['q05']) / 2; hw_u = (s['U']['q95'] - s['U']['q05']) / 2; hw = (hw_l + hw_u) / 2
    tok['BABS'] = f"{100 * hw_l:.1f} to ±{100 * hw_u:.1f}"
    tok['BWID'] = f"{100 * hw_l:.1f} to {100 * hw_u:.1f} percentage points"
    width_theta = og['U'] - og['L']
    tok['BCMP'] = ('larger than' if 2 * hw > 1.3 * width_theta else ('comparable to' if 2 * hw > 0.7 * width_theta else 'smaller than'))
    tok['BR'] = ', '.join(f"{100 * r['lead_same_most'][o]:.0f}% ({LAB[o]})" for o in ORIG)
    tok['BR2'] = f"{100 * r['top2set_same_most']['China']:.0f}%"
    tok['BRALL'] = f"{100 * r['overall_lead_same_most']:.0f}%"
    tok['BSD_L'] = pc(s['L']['sd']); tok['BSD_U'] = pc(s['U']['sd'])
    tok['BMC'] = f"{100 * m['most_pw_corrected_vs_mc']['mean_diff']:+.3f}".replace('-', '−')
    tok['BMC_MAX'] = f"{100 * m['most_pw_corrected_vs_mc']['max_abs_diff']:.2f}"
    tok['BMC_CORR'] = f"{m['most_pw_corrected_vs_mc']['corr']:.3f}"
    tok['BGRID'] = f"{100 * m['grid_max_at_corner']:.0f}%"
    tok["BBETA"] = f"{s['beta']['q05']:.2f} and {s['beta']['q95']:.2f}"
    tok['BONS'] = f"{s['n_onsets']['q05']:.0f} to {s['n_onsets']['q95']:.0f}"
    tok['BNONCONV'] = str(boot['nonconverged'])
    tok['B_N'] = f"{boot['B']:,}"; tok['BONS_INTRO'] = f"{boot['B']:,}"
    rows = []
    for key, lab in [('L', 'Least dependent corner (independence)'), ('F', 'Fitted dispersions'), ('U', 'Most dependent corner of the 96.5% box')]:
        d = s[key]
        rows.append(f"| {lab} | {pc(og[key])} | {pc(d['q50'])} | {pc(d['sd'])} | {pc(d['q05'])} – {pc(d['q95'])} | {pc(d['q25'])} – {pc(d['q75'])} |")
    head = ('| Point of $\\Theta$ | Original fit | Bootstrap median | Bootstrap s.d. | 90% bootstrap range | 50% range |\n|------------------------|------------|------------|------------|--------------|------------|\n')
    tok['TABLE_S6'] = head + '\n'.join(rows)
json.dump(tok, open(f'{W}/tokens_v07.json', 'w'), indent=1, ensure_ascii=False)
for k, v in tok.items():
    print(k, ':', v if len(v) < 400 else v[:400] + ' …')
