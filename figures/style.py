"""Plot style of the notes' figures, the space-time panel they share, and where they are written (out/figures)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 100, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "image.cmap": "RdBu_r", "mathtext.fontset": "cm"})

FIGURES = Path("out/figures")
DATA = Path("out/data")
FIGURES.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)


def spacetime(ax, F, x, t, medium, vmax=1.0, title=None):
    """Draw F(t, x) as a space-time diagram, time upwards, with the slab edges dashed."""
    ax.imshow(F, origin="lower", extent=[x[0], x[-1], t[0], t[-1]], vmin=-vmax, vmax=vmax, aspect="auto",
              interpolation="bilinear")
    for e in medium.edges:
        ax.axvline(e, color="k", lw=0.5, ls="--")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(t[0], t[-1])
    ax.set_xlabel("$x$")
    if title:
        ax.set_title(title)
