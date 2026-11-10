"""Unified figure style for the paper (one module, used by every figure)."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

FIGDIR = f'{SP_ROOT}/work/figs'
import os
os.makedirs(FIGDIR, exist_ok=True)

MM = 1 / 25.4
W2 = 190 * MM     # double column
W1 = 90 * MM      # single column

INK = '#222222'
GREY = '#8a8a8a'
LIGHT = '#d9d9d9'
# muted palette; every series also carries its own marker/linestyle so the figures read in greyscale
C = {'blue': '#2b5ca8', 'red': '#c8553d', 'teal': '#2a9d8f', 'gold': '#c49a1c', 'purple': '#7a5ca8', 'grey': '#6b6b6b'}
SERIES = [(C['blue'], 'o', '-'), (C['red'], 's', '-'), (C['teal'], '^', '-'), (C['gold'], 'D', '-'),
          (C['purple'], 'v', '-'), (C['grey'], 'P', '-')]

rcParams.update({
    'font.family': 'Liberation Sans',
    'font.size': 7.5,
    'axes.titlesize': 7.5,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'axes.linewidth': 0.6,
    'axes.edgecolor': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.2,
    'lines.markersize': 3.6,
    'legend.frameon': False,
    'savefig.dpi': 600,
    'pdf.fonttype': 42,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Liberation Sans',
    'mathtext.it': 'Liberation Sans:italic',
    'mathtext.bf': 'Liberation Sans:bold',
})


def panel_label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8.5, fontweight='bold', va='bottom', ha='left', color=INK)


def direct_label(ax, x, y, s, color, dx=3, dy=0, ha='left', va='center', fs=7):
    ax.annotate(s, (x, y), xytext=(dx, dy), textcoords='offset points', color=color, fontsize=fs, ha=ha, va=va)


def save(fig, name):
    fig.savefig(f'{FIGDIR}/{name}.png', dpi=600, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'{FIGDIR}/{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
