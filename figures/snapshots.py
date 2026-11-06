"""Figure B: snapshots of E_inc, E_mat and their sum, thick slab."""
import numpy as np

from coupled_sheets import Medium, Pulse, fields, response
from style import FIGURES, plt

med = Medium(n_sheets=60)
p = Pulse()
R = response(med, p)
x = np.linspace(-5, 50, 2201)
times = [7.0, 12.0, 18.0, 30.0]
F = fields(R, x, times)
a, b = med.edges
fig, axs = plt.subplots(len(times), 1, figsize=(7.5, 7), sharex=True)
for i, (ax, ti) in enumerate(zip(axs, times)):
    ax.axvspan(a, b, color="0.92", zorder=0)
    ax.plot(x, F["Ei"][i], color="0.45", lw=1.0, ls="--", label=r"$E_\mathrm{inc}$ (vacuum, $c$)")
    ax.plot(x, F["Em"][i], color="tab:orange", lw=1.0, label=r"$E_\mathrm{mat}$ (sheets)")
    ax.plot(x, F["E"][i], color="k", lw=1.4, label=r"$E = E_\mathrm{inc} + E_\mathrm{mat}$")
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, 0, 1])
    ax.text(0.01, 0.92, f"$t = {ti:g}$", transform=ax.transAxes, va="top")
axs[0].legend(loc="lower left", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0, 1.0))
axs[-1].set_xlabel("$x$")
fig.tight_layout(h_pad=0.3)
fig.savefig(FIGURES / "figB_snapshots.png")
