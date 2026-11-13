"""Fig. 4 (research framework, Section 3): a box workflow in the paper-figure-style rules — white boxes, black square
borders, one bold title + one grey line per box, stage titles with an underline, -|> arrows, one panel, no panel letters.
Emphasis by border weight only (the system-analysis box)."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys
sys.path.insert(0, f'{SP_ROOT}/scripts')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import figstyle as fs

OUT = f'{SP_ROOT}/work/figs_v2/fig4_workflow'
fs.apply(base=9)
W, H = 180, 55
FH, FT, FSUB = 9.0, 8.3, 7.3
BW, BH, GAP, X0 = 40.5, 14.0, 3.0, 5.0          # box width/height, gap, left margin  -> 5 + 4 * 40.5 + 3 * 3 = 176
ROW_Y = {1: 32.0, 2: 2.0}                       # bottom y of the boxes in each row
HEAD_DY = 5.2                                   # stage title above the boxes
fig, ax = fs.canvas(W, H)
TEXTS = []


def T(x, y, s, **kw):
    t = fs.txt(ax, x, y, s, **kw); TEXTS.append(t); return t


def box(col, row, title, sub, strong=False):
    x = X0 + col * (BW + GAP); y = ROW_Y[row]
    ax.add_patch(Rectangle((x, y), BW, BH, fc='white', ec=fs.INK, lw=1.6 if strong else 0.8, zorder=2))
    T(x + BW / 2, y + BH / 2 + 2.4, title, fs=FT, c=fs.INK, w='bold', ha='center')
    T(x + BW / 2, y + BH / 2 - 2.5, sub, fs=FSUB, c=fs.GREY, ha='center')
    return x, y


def stage(cols, row, label):
    x0 = X0 + cols[0] * (BW + GAP); x1 = X0 + cols[1] * (BW + GAP) + BW
    y = ROW_Y[row] + BH + HEAD_DY
    T(x0, y, label, fs=FH, c=fs.INK, w='bold', ha='left', va='center')
    fs.hline(ax, x0, x1, y - 2.6, c=fs.DARK, lw=0.6)


def arrow(p, q, lw=0.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=7, lw=lw, color=fs.INK, shrinkA=0, shrinkB=0, zorder=3))


# row 1: data and structure
stage((0, 1), 1, 'Stage 1  Data')
stage((2, 3), 1, 'Stage 2  Structure')
b1 = box(0, 1, 'Trade records', 'UN Comtrade; lunar calendar')
b2 = box(1, 1, 'Streams and onsets', 'flow falls below half its median')
b3 = box(2, 1, 'Common-cause groups', r'origins with $\geq$ 5 streams')
b4 = box(3, 1, 'Supply structure', r'inputs, sources, shares $\rightarrow$ $\varphi$')
# row 2: inference and decisions
stage((0, 1), 2, 'Stage 3  Inference')
stage((2, 3), 2, 'Stage 4  System decisions')
b5 = box(0, 2, 'Estimation', r'profile likelihood (12); covariates')
b6 = box(1, 2, 'Identification and pattern', 'pair rule (13); co-onset pattern')
b7 = box(2, 2, 'Reliability and importance', r'exact (7) or pairwise (10); $I_g$, $\Delta_i$', strong=True)
b8 = box(3, 2, 'Decision outputs', r'ranges over $\alpha_c$; bands; ranking')
# arrows between consecutive boxes
for a, b in [(b1, b2), (b2, b3), (b3, b4), (b5, b6), (b6, b7), (b7, b8)]:
    arrow((a[0] + BW, a[1] + BH / 2), (b[0], b[1] + BH / 2))
# return path from the end of row 1 to the start of row 2
xm = b4[0] + BW / 2; ym = ROW_Y[1] - 4.0; xl = 2.3; yl = ROW_Y[2] + BH / 2
ax.plot([xm, xm, xl, xl], [ROW_Y[1], ym, ym, yl], color=fs.INK, lw=0.8, solid_capstyle='butt', solid_joinstyle='miter', zorder=3)
arrow((xl, yl), (b5[0], yl), lw=0.8)
fs.save(fig, OUT)
print(fs.margins(OUT + '.png'))
# text width check against the box width
fig2 = plt.figure(figsize=(W * fs.MM, H * fs.MM)); ax2 = fig2.add_axes([0, 0, 1, 1]); ax2.set_xlim(0, W); ax2.set_ylim(0, H); ax2.axis('off')
r = fig2.canvas.get_renderer()
for t in TEXTS:
    t2 = ax2.text(*t.get_position(), t.get_text(), fontsize=t.get_fontsize(), weight=t.get_weight(), ha=t.get_ha(), va=t.get_va())
    bb = t2.get_window_extent(renderer=r); wmm = bb.width / fig2.dpi * 25.4
    if t.get_ha() == 'center' and wmm > BW - 2.0:
        print(f'TOO WIDE for box ({wmm:.1f} mm):', t.get_text())
print('smallest text (pt):', min(t.get_fontsize() for t in TEXTS))
