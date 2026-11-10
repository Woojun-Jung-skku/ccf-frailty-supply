"""Fig. 1 (concept), one panel, glyph schematic: data -> model -> what follows -> decisions.
Type scale x1.125 relative to figstyle's schematic defaults so that, displayed at 160 mm, the smallest text is >= 6.2 pt
(header 9, bold title 8.3, grey line 7.6, glyph label 7.0; drawn at 180 mm)."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import figstyle as fs
import ccfmodel as cm

OUT = f'{SP_ROOT}/work/figs_v2/fig1_framework'
fs.apply(base=9)
W, H = 180, 66
FH, FT, FSUB, FG, FNODE = 9.0, 8.3, 7.6, 7.0, 6.3
fig, ax = fs.canvas(W, H)
AP = 2.0
TEXTS = []
# stage columns (mm): [x0, x1]; gaps of 6 mm carry the arrows
S1, S2, S3, S4 = (3, 38), (44, 82), (88, 141), (147, 178)
Y0, HB = 41.5, 14.0          # row-1 glyph band: boxes from Y0 to Y0 + HB
LAB1, LAB2 = 34.0, 8.5       # label rows (bold title at +2.3, grey line at -2.4)
ARROW_Y = Y0 + HB / 2        # arrows between stages at glyph mid-height


def T(x, y, s, **kw):
    t = fs.txt(ax, x, y, s, **kw); TEXTS.append(t); return t


def label(x, y, title, sub, ha='left'):
    TEXTS.append(fs.txt(ax, x, y + 2.3, title, fs=FT, c=fs.INK, w='bold', ha=ha))
    TEXTS.append(fs.txt(ax, x, y - 2.4, sub, fs=FSUB, c=fs.GREY, ha=ha))


def stage(span, num, title, y=62.5):
    x0, x1 = span
    T(x0, y, f'{num}  {title}', fs=FH, c=fs.INK, w='bold', ha='left', va='center')
    fs.hline(ax, x0, x1, y - 2.6, c=fs.DARK, lw=0.6)


def between(x1, x2, y=ARROW_Y):
    fs.arrow(ax, x1, y, x2, y, c=fs.LIGHT, lw=0.7, ms=6)


def glyph_axes(b, xlabel=None, ylabel=None):
    """figstyle.Box.axes with the larger glyph-label size."""
    from matplotlib.patches import FancyArrowPatch
    for end in ((b.x0 + b.w + 1.2, b.y0), (b.x0, b.y0 + b.h + 1.2)):
        ax.add_patch(FancyArrowPatch((b.x0, b.y0), end, arrowstyle='-|>', mutation_scale=3.6, lw=0.5, color=fs.LIGHT,
                                     shrinkA=0, shrinkB=0, zorder=2))
    if xlabel:
        T(b.x0 + b.w + 1.2, b.y0 - 2.4, xlabel, fs=FG, ha='right')
    if ylabel:
        T(b.x0 - 1.6, b.y0 + b.h + 0.6, ylabel, fs=FG, ha='right')


# ---------------------------------------------------------------- stage 1: streams and groups
stage(S1, '1', 'Streams and groups')
pitch = 2.1; x0, y0 = 7, Y0
rows, cols = 6, 13
ev = {(0, 2): 1, (0, 8): 1, (1, 8): 1, (1, 11): 1, (2, 5): 1, (2, 8): 1, (3, 4): 1, (3, 12): 1, (4, 1): 1, (4, 7): 1, (5, 9): 1}
for i in range(rows):
    for j in range(cols):
        n = ev.get((i, j), 0); c = fs.BLUE if i < 3 else fs.RED
        x, y = x0 + j * pitch, y0 + (rows - 1 - i) * pitch
        ax.add_patch(Rectangle((x, y), pitch, pitch, fc=fs.tint(c, 0.72) if n else 'white', ec=fs.HAIR, lw=0.3, zorder=2))
        if n:
            fs.dot(ax, x + pitch / 2, y + pitch / 2, 0.4, c)
ax.add_patch(Rectangle((x0, y0), cols * pitch, rows * pitch, fc='none', ec=fs.LIGHT, lw=0.5, zorder=3))
fs.hline(ax, x0, x0 + cols * pitch, y0 + 3 * pitch, c=fs.DARK, lw=0.6)
T(x0 - 1.2, y0 + 4.5 * pitch, 'J', fs=FG, c=fs.GREY, ha='right')
T(x0 - 1.2, y0 + 1.5 * pitch, 'C', fs=FG, c=fs.GREY, ha='right')
xc = x0 + 8.5 * pitch
fs.arrow(ax, xc, y0 + rows * pitch + 3.6, xc, y0 + rows * pitch + 0.8, c=fs.DARK, lw=0.7, ms=5)
T(x0 + cols * pitch, y0 - 0.8, 'months', fs=FG, ha='right', va='top')
label(S1[0], LAB1, 'Interruption onsets', 'streams × months, by origin')
between(S1[1] + 1, S2[0] - 1)

# ---------------------------------------------------------------- stage 2: the frailty
stage(S2, '2', 'Common-cause frailty')
b = fs.Box(53.5, Y0, 25, HB, xl=(0, 3.4), yl=(0, 1.85))
glyph_axes(b, xlabel=r'$\eta_{gt}$', ylabel='density')
e = np.linspace(0.005, 3.4, 1500)
curves = [(20.0, fs.BLUE), (2.0, fs.TEAL), (0.5, fs.RED)]
for a, c in curves:
    b.line(ax, e, stats.gamma.pdf(e, a, scale=1 / a), color=c, lw=1.0)
ax.plot([b.X(1), b.X(1)], [b.Y(0), b.Y(1.85)], **fs.REF)
for k, (a, c) in enumerate(curves):      # direct labels, stacked where all three densities are near zero
    yy = b.Y(1.72 - 0.3 * k); xx = b.X(1.75)
    ax.plot([xx, xx + 3.0], [yy, yy], color=c, lw=1.0, solid_capstyle='round', zorder=3)
    T(xx + 4.0, yy, rf'$\alpha_c = {a:g}$', fs=FG, c=fs.INK)
label(S2[0], LAB1, 'One frailty per group', r'$\lambda_{it} = R_{it}\,\xi_{it}\,\eta_{g(i)t}$,  Var $\eta = 1/\alpha_c$')
between(S2[1] + 1, S3[0] - 1)

# ---------------------------------------------------------------- stage 3: what follows (two columns, two rows)
stage(S3, '3', 'What follows')
COL2 = 115
# (i) alpha-factors (row 1, left)
bx = fs.Box(92, Y0, 13, HB, xl=(0.4, 4.6), yl=(1e-6, 2), logy=True)
glyph_axes(bx, xlabel=r'$k$', ylabel=r'log $\alpha_k$')
ks = np.arange(1, 5)
af_f = cm.alpha_factors(cm.multiplicity_pmf(4, 0.05, AP, 0.5)); af_i = cm.alpha_factors(cm.multiplicity_pmf(4, 0.05, AP, np.inf))
for k, vf, vi in zip(ks, af_f, af_i):
    for dx, v, c, fc in [(-0.36, vf, fs.RED, fs.RED), (0.02, vi, fs.BLUE, 'white')]:
        ax.add_patch(Rectangle((bx.X(k + dx), bx.Y(1e-6)), bx.X(k + dx + 0.34) - bx.X(k + dx), bx.Y(v) - bx.Y(1e-6), fc=fc, ec=c, lw=0.6, zorder=3))
label(S3[0], LAB1, 'CCF parameters', r'one $\alpha_c$ sets every $\alpha_k$')
# (ii) system structure (row 1, right)
sx, sy = COL2 + 9.5, Y0 + HB / 2
nodes = {'AJ': (sx - 7, sy + 5.5, fs.BLUE), 'AK': (sx - 7, sy + 0.5, fs.RED), 'BJ': (sx - 7, sy - 6, fs.BLUE),
         'A': (sx, sy + 3, fs.INK), 'B': (sx, sy - 6, fs.INK), 'P': (sx + 7, sy - 1.5, fs.INK)}
for a_, b_ in [('AJ', 'A'), ('AK', 'A'), ('BJ', 'B'), ('A', 'P'), ('B', 'P')]:
    (xa, ya, _), (xb, yb, _) = nodes[a_], nodes[b_]
    ax.plot([xa, xb], [ya, yb], color=fs.LIGHT, lw=0.6, zorder=1)
for k, (x, y, c) in nodes.items():
    ax.add_patch(Circle((x, y), 1.85, fc='white', ec=c, lw=0.9 if len(k) == 2 else 0.7, zorder=3))
    T(x, y, {'AJ': r'A$_J$', 'AK': r'A$_C$', 'BJ': r'B$_J$'}.get(k, k), fs=FNODE, c=fs.INK, ha='center', va='center', zorder=4)
label(COL2, LAB1, 'System reliability', r'$h(p) + \sum$ JRI · cov')
# (iii) identification: all pairs of a group (row 2, left column, centred under its label's column glyph)
cx, cy, r = bx.x0 + bx.w / 2, 22.0, 6.0
pts = [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 6)[:-1]]
for i in range(5):
    for j in range(i + 1, 5):
        ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color=fs.tint(fs.BLUE, 0.45), lw=0.6, zorder=1)
for x, y in pts:
    fs.dot(ax, x, y, 0.8, fs.BLUE, z=3)
label(S3[0], LAB2, 'Identification', 'the common cause is seen in pairs')
between(S3[1] + 1, S4[0] - 1)

# ---------------------------------------------------------------- stage 4: decisions (two rows)
stage(S4, '4', 'Decisions')
bu = fs.Box(156, Y0, 17, HB, xl=(0.1, 100), yl=(0, 1.08), logx=True)
glyph_axes(bu, xlabel=r'$\alpha_c$', ylabel='share kept')
acs = np.logspace(-1, 2, 400)
ax.plot([bu.X(0.1), bu.X(100)], [bu.Y(1), bu.Y(1)], **fs.REF)      # independence: full protection
bu.line(ax, acs, acs / (1 + acs), color=fs.INK, lw=1.0)
label(S4[0], LAB1, 'Diversification value', r'protection kept: $\alpha_c/(1+\alpha_c)$')
bx0, by0 = 151, 17.5
for i, v in enumerate([1.0, 0.83, 0.21, 0.19]):
    yy = by0 + (3 - i) * 2.5
    ax.add_patch(Rectangle((bx0, yy), 17 * v, 1.8, fc=fs.tint(fs.RED, 0.35), ec=fs.RED, lw=0.5, zorder=3))
fs.arrow(ax, bx0, by0 - 0.8, bx0 + 18.5, by0 - 0.8, c=fs.LIGHT, lw=0.5, ms=3.6)
T(bx0 + 18.5, by0 - 2.9, 'gain', fs=FG, ha='right')
label(S4[0], LAB2, 'Re-sourcing priority', 'where to add a source first')
fs.save(fig, OUT)
print(fs.margins(OUT + '.png'))
# text extents check (mm): nothing beyond the right edge; nothing spilling out of its stage column
fig2 = plt.figure(figsize=(W * fs.MM, H * fs.MM)); ax2 = fig2.add_axes([0, 0, 1, 1]); ax2.set_xlim(0, W); ax2.set_ylim(0, H); ax2.axis('off')
r = fig2.canvas.get_renderer()
for t in TEXTS:
    t2 = ax2.text(*t.get_position(), t.get_text(), fontsize=t.get_fontsize(), weight=t.get_weight(), ha=t.get_ha(), va=t.get_va())
    bb = t2.get_window_extent(renderer=r)
    x0m, x1m = bb.x0 / fig2.dpi * 25.4, bb.x1 / fig2.dpi * 25.4
    if x1m > W - 0.5:
        print('OVERFLOW right:', t.get_text(), round(x1m, 1))
    for (lo, hi), name in zip([S1, S2, S3, S4], ['S1', 'S2', 'S3', 'S4']):
        if lo - 0.5 <= x0m <= hi and x1m > hi + 0.6:
            print(f'exceeds {name}:', t.get_text(), round(x0m, 1), round(x1m, 1))
print('smallest text (pt, drawn):', min(t.get_fontsize() for t in TEXTS))
