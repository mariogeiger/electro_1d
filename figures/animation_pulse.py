"""Animation (GIF): E_inc, E_mat and E vs x over time, thick slab. Reads the data of spacetime.py."""
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from coupled_sheets import Medium
from style import DATA, FIGURES, plt

Z = np.load(DATA / "spacetime_thick.npz")
x, t = Z["x"], Z["t"]
med = Medium(n_sheets=60)
fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
a, b = med.edges
ax.axvspan(a, b, color="0.92", zorder=0)
li, = ax.plot([], [], color="0.45", ls="--", lw=1.0, label=r"$E_\mathrm{inc}$ (vacuum, $c$)")
lm, = ax.plot([], [], color="tab:orange", lw=1.0, label=r"$E_\mathrm{mat}$ (sheets)")
le, = ax.plot([], [], color="k", lw=1.5, label=r"$E = E_\mathrm{inc} + E_\mathrm{mat}$")
txt = ax.text(0.01, 0.95, "", transform=ax.transAxes, va="top")
ax.set_xlim(x[0], x[-1]); ax.set_ylim(-1.2, 1.2); ax.set_xlabel("$x$")
ax.legend(loc="lower left", fontsize=7, ncol=3, frameon=False)
fig.tight_layout()
frames = range(0, len(t), 4)


def draw(i):
    li.set_data(x, Z["Ei"][i]); lm.set_data(x, Z["Em"][i]); le.set_data(x, Z["E"][i])
    txt.set_text(f"t = {t[i]:.1f}")
    return li, lm, le, txt


FuncAnimation(fig, draw, frames=frames, blit=True).save(FIGURES / "animation_pulse.gif", writer=PillowWriter(fps=20))
