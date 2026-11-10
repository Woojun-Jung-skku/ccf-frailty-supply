"""figstyle — one style for every figure of a paper (from the RESS manuscript, Oct 2026).

    import figstyle as fs
    fs.apply(base=8)                          # base = axis-label size measured from the target journal
    fig, ax = plt.subplots(figsize=(fs.DOUBLE * fs.MM, 64 * fs.MM))
    ...
    fs.save(fig, "out/Fig3")                  # PDF (fonts embedded) + PNG 600 dpi, at the exact drawn size
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

MM = 1 / 25.4                                   # inches per mm
DOUBLE, SINGLE = 180, 90                        # figure widths (mm)
PAL = ["#2E5FA3", "#D4553F", "#2A9D8F", "#B8860B", "#7B5EA7", "#5B8C3E", "#E69F00", "#56B4E9", "#CC79A7"]
BLUE, RED, TEAL, GOLD, PURPLE, GREEN, ORANGE, SKY, PINK = PAL
INK, GREY, LIGHT = "#1A1A1A", "#5C636C", "#8B9199"      # text, second line, glyph axes and connectors
HAIR, DARK = "#C9CED4", "#262B31"                       # hairlines and cell grids, the one shared structure
REF = dict(color=LIGHT, lw=0.7, ls=(0, (3.0, 1.6)), zorder=0.5)   # reference/limit lines: thin grey dashes, under data
RAMP = {2: [BLUE, RED], 3: [BLUE, TEAL, RED], 4: [BLUE, TEAL, GOLD, RED], 5: [PURPLE, BLUE, TEAL, GOLD, RED]}
MARKS = ["o", "s", "^", "D", "v", "p"]
# one colour and marker per method in every figure: name -> (colour, marker, filled). Fill in once per paper.
# RESS: proposed model red circle, complete pooling blue triangle, no pooling teal square, Poisson gold open circle,
# control charts purple, machine learning green.
METHOD = {"proposed": (RED, "o", True), "pooled": (BLUE, "^", True), "unpooled": (TEAL, "s", True),
          "poisson": (GOLD, "o", False), "chart": (PURPLE, "D", False), "ml": (GREEN, "v", True)}
FS_H, FS_T, FS_S, FS_G = 8.0, 7.4, 6.8, 6.5     # schematics: header, bold title, grey line, glyph label (pt)
AP = r"\alpha\prime"                           # alpha-prime in mathtext; a plain ' is set as a raised superscript


def tint(c, f):
    """mix c with white; f = share of white."""
    r, g, b = mcolors.to_rgb(c)
    return mcolors.to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def shade(c, f):
    """mix c with black; f = share of black."""
    r, g, b = mcolors.to_rgb(c)
    return mcolors.to_hex((r * (1 - f), g * (1 - f), b * (1 - f)))


def apply(base=8.0, fonts=("Liberation Sans", "Arial", "Helvetica", "DejaVu Sans")):
    """rcParams for the whole paper. base = axis-label size (pt); ticks and legends base - 0.5."""
    have = {f.name for f in fm.fontManager.ttflist}
    f = next((x for x in fonts if x in have), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [f] + [x for x in fonts if x != f],
        "mathtext.fontset": "custom", "mathtext.rm": f, "mathtext.it": f + ":italic", "mathtext.bf": f + ":bold",
        "mathtext.sf": f, "mathtext.cal": f + ":italic", "mathtext.fallback": "stixsans",
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base, "xtick.labelsize": base - 0.5,
        "ytick.labelsize": base - 0.5, "legend.fontsize": base - 0.5, "legend.frameon": False,
        "legend.handlelength": 2.2, "legend.handletextpad": 0.5, "legend.columnspacing": 1.4,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45, "xtick.major.size": 2.8, "ytick.major.size": 2.8, "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6, "xtick.major.pad": 2.2, "ytick.major.pad": 2.2,
        "lines.linewidth": 1.1, "lines.markersize": 4.0, "lines.markeredgewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.edgecolor": INK,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "figure.dpi": 150,
    })


def use_korean(font_dir="fonts"):
    """Korean labels. Linux: if only the Noto CJK .ttc is installed, extract its KR faces once; Mac/Windows fall back
    to Apple SD Gothic Neo / Malgun Gothic."""
    import glob
    import os
    os.makedirs(font_dir, exist_ok=True)
    for w in ("Regular", "Bold"):
        out = os.path.join(font_dir, f"NotoSansCJKkr-{w}.ttf")
        src = glob.glob(f"/usr/share/fonts/**/NotoSansCJK-{w}.ttc", recursive=True)
        if not os.path.exists(out) and src:
            from fontTools.ttLib import TTCollection
            for face in TTCollection(src[0]).fonts:
                if "KR" in (face["name"].getDebugName(4) or ""):
                    face.save(out)
                    break
        if os.path.exists(out):
            fm.fontManager.addfont(out)
    plt.rcParams["font.sans-serif"] = (["Noto Sans CJK KR", "Apple SD Gothic Neo", "Malgun Gothic"]
                                       + list(plt.rcParams["font.sans-serif"]))
    plt.rcParams["axes.unicode_minus"] = False


def series(ax, x, y, k, n, label=None, marks=8, logx=True, mrange=None, ymin=None, **kw):
    """curve k of n ordered levels: ramp colour and its own open marker at evenly spaced (log-)x positions, staggered
    between curves; markers only inside mrange (x interval) and above ymin, i.e. off where all curves coincide."""
    c = RAMP[n][k]
    x, y = np.asarray(x, float), np.asarray(y, float)
    u = np.log(x) if logx else x
    lo, hi = (u[0], u[-1]) if mrange is None else ((np.log(mrange[0]), np.log(mrange[1])) if logx else mrange)
    step = (hi - lo) / (marks + 1)
    targets = lo + step * (np.arange(1, marks + 1) + (k - (n - 1) / 2) / n * 0.9)
    idx = sorted({int(np.argmin(np.abs(u - t))) for t in targets})
    if ymin is not None:
        idx = [i for i in idx if y[i] >= ymin]
    ax.plot(x, y, color=c, marker=MARKS[k], markevery=idx, mfc="white", mec=c, mew=0.9, ms=3.8, label=label, **kw)
    return c


def method_style(name, ms=4.6):
    c, m, filled = METHOD[name]
    return dict(color=c, marker=m, ms=ms, mfc=c if filled else "white", mec="white" if filled else c,
                mew=0.5 if filled else 0.9)


def method_point(ax, x, y, name, **kw):
    ax.plot(x, y, ls="none", **{**method_style(name), **kw})


def method_handle(name, label, line=False):
    return Line2D([], [], ls="-" if line else "none", label=label, **method_style(name))


def panel(ax, letter, fmt="({})", dx=-30, dy=5):
    """panel letter, only when a figure has two or more panels; fmt as the target journal prints it (RESS: (a))."""
    ax.annotate(fmt.format(letter), xy=(0, 1), xycoords="axes fraction", xytext=(dx, dy), textcoords="offset points",
                fontsize=plt.rcParams["font.size"] + 1, fontweight="bold", ha="left", va="bottom",
                annotation_clip=False)


def save(fig, path, dpi=600):
    """exact drawn size (never bbox_inches='tight'), so every figure keeps one type scale."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png", dpi=dpi)
    plt.close(fig)


# ------------------------------------------------------------------ schematics: small exact glyphs on a mm canvas
def canvas(w, h):
    """schematic canvas; data units = mm, origin bottom left."""
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def txt(ax, x, y, s, fs=FS_S, c=GREY, w="normal", ha="left", va="center", **kw):
    return ax.text(x, y, s, fontsize=fs, color=c, weight=w, ha=ha, va=va, **kw)


def label(ax, x, y, title, sub=None, ha="left"):
    """the only text an element gets: one bold title line and one grey line, centred on y."""
    txt(ax, x, y + (2.3 if sub else 0), title, fs=FS_T, c=INK, w="bold", ha=ha)
    if sub:
        txt(ax, x, y - 2.4, sub, fs=FS_S, c=GREY, ha=ha)


def hline(ax, x1, x2, y, c=HAIR, lw=0.5):
    ax.plot([x1, x2], [y, y], color=c, lw=lw, solid_capstyle="butt")


def arrow(ax, x1, y1, x2, y2, c=LIGHT, lw=0.6, both=False, ms=5.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>" if both else "-|>", mutation_scale=ms,
                                 lw=lw, color=c, shrinkA=0, shrinkB=0))


def dot(ax, x, y, r, c, z=4):
    ax.add_patch(Circle((x, y), r, fc=c, ec="none", zorder=z))


class Box:
    """a mini plot inside a schematic: maps data (x, y) onto the mm rectangle (x0, y0, w, h); log axes optional."""

    def __init__(self, x0, y0, w, h, xl=(0, 1), yl=(0, 1), logx=False, logy=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xl, self.yl, self.logx, self.logy = xl, yl, logx, logy

    def _f(self, v, lo, hi, log):
        v = np.asarray(v, float)
        if log:
            v, lo, hi = np.log10(v), np.log10(lo), np.log10(hi)
        return (v - lo) / (hi - lo)

    def X(self, v):
        return self.x0 + self._f(v, *self.xl, self.logx) * self.w

    def Y(self, v):
        return self.y0 + self._f(v, *self.yl, self.logy) * self.h

    def axes(self, ax, xlabel=None, ylabel=None):
        """L-shaped 0.5-pt grey axes with arrow ends; labels at the ends."""
        for end in ((self.x0 + self.w + 1.2, self.y0), (self.x0, self.y0 + self.h + 1.2)):
            ax.add_patch(FancyArrowPatch((self.x0, self.y0), end, arrowstyle="-|>", mutation_scale=3.6, lw=0.5,
                                         color=LIGHT, shrinkA=0, shrinkB=0, zorder=2))
        if xlabel:
            txt(ax, self.x0 + self.w + 1.2, self.y0 - 2.4, xlabel, fs=FS_G, ha="right")
        if ylabel:
            txt(ax, self.x0 - 1.6, self.y0 + self.h + 0.6, ylabel, fs=FS_G, ha="right")

    def line(self, ax, x, y, **kw):
        """data line clipped to the y range; pass color=..., or **REF for a reference line."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = (y >= min(self.yl)) & (y <= max(self.yl))
        ax.plot(self.X(x[ok]), self.Y(y[ok]), **{"lw": 1.0, "solid_capstyle": "round", "zorder": 3, **kw})


def raster(ax, x0, y0, ncol, nrow, events, c, pitch=1.6):
    """item (rows) x period (columns) grid; events = {(row, col): count}. Cells with an event are tinted, one dot
    per event. Use one cell size (pitch, mm) in every figure."""
    for i in range(nrow):
        for j in range(ncol):
            n = events.get((i, j), 0)
            x, y = x0 + j * pitch, y0 + (nrow - 1 - i) * pitch
            ax.add_patch(Rectangle((x, y), pitch, pitch, fc=tint(c, 0.72) if n else "white", ec=HAIR, lw=0.3,
                                   zorder=2))
            q = pitch / 1.6
            offs = {1: [(0, 0)], 2: [(-0.38 * q, 0), (0.38 * q, 0)],
                    3: [(-0.4 * q, -0.3 * q), (0.4 * q, -0.3 * q), (0, 0.38 * q)]}
            for dx, dy in offs.get(min(n, 3), []):
                dot(ax, x + pitch / 2 + dx, y + pitch / 2 + dy, min(0.26, 0.17 * pitch), c, z=3)
    ax.add_patch(Rectangle((x0, y0), ncol * pitch, nrow * pitch, fc="none", ec=LIGHT, lw=0.5, zorder=3))


# ------------------------------------------------------------------ checks
def margins(png, dpi=600):
    """drawn size and white margins (mm) of a saved PNG: same widths across figures, edges of 1 mm or more."""
    from PIL import Image
    a = np.asarray(Image.open(png).convert("L"))
    r, c = np.where((a < 235).any(1))[0], np.where((a < 235).any(0))[0]
    px = dpi / 25.4
    mm = lambda v, d=2: round(float(v) / px, d)
    return dict(w=mm(a.shape[1], 1), h=mm(a.shape[0], 1), L=mm(c.min()), R=mm(a.shape[1] - 1 - c.max()),
                T=mm(r.min()), B=mm(a.shape[0] - 1 - r.max()))


def sheet(pngs, out, width=1600, gap=24):
    """stack images into one sheet to look at together (journal survey, old vs new, all figures of a paper)."""
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in pngs]
    ims = [im.resize((width, int(width * im.height / im.width))) for im in ims]
    s = Image.new("RGB", (width, sum(im.height + gap for im in ims)), "white")
    y = 0
    for im in ims:
        s.paste(im, (0, y))
        y += im.height + gap
    s.save(out)
