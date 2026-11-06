"""Figure H: the notebook's grid solver (collocated and staggered) against the exact coupled-sheet solution.

Reads out/data/compare.npz, written by validation/compare.py."""
import numpy as np

from style import DATA, FIGURES, plt

Z = np.load(DATA / "compare.npz")
x, ts, ex = Z["x"], Z["t"], Z["exact"]
col, yee, yee1 = Z["grid_col_0.001"], Z["grid_yee_0.001"], Z["grid_yee_1.0"]
fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axs[0]
i = np.argmin(np.abs(ts - 14.0))
ax.plot(x, col[i], color="tab:red", lw=0.5, label="collocated grid (notebook)")
ax.plot(x, yee[i], color="tab:blue", lw=1.0, label="staggered (Yee) grid")
ax.plot(x, ex[i], "k:", lw=1.3, label="exact coupled sheets")
ax.axvspan(10, 15, color="0.92", zorder=0)
ax.set_xlim(-8, 25)
ax.set_xlabel("$x$")
ax.set_title(r"$E_y$ at $t = 14$, amplitude $10^{-3}$ (linear regime), rescaled")
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax = axs[1]
err = lambda A: np.abs(A - ex).max(1)
ax.semilogy(ts, err(col), color="tab:red", label="collocated, $A=10^{-3}$")
ax.semilogy(ts, err(yee), color="tab:blue", label="Yee, $A=10^{-3}$")
ax.semilogy(ts, err(yee1), color="tab:blue", ls="--", label="Yee, $A=1$ (nonlinear $v\\times B$)")
ax.set_xlabel("$t$"); ax.set_ylabel(r"$\max_x |E_\mathrm{grid} - E_\mathrm{exact}|$")
ax.set_ylim(1e-4, 2)
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("deviation from the exact solution")
fig.tight_layout()
fig.savefig(FIGURES / "figH_validation.png")
