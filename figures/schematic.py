"""Figure S: schematic of the 1D model (sheets of charge on springs, incident pulse, radiated fields)."""
import numpy as np
from matplotlib.patches import Polygon

from style import FIGURES, plt

fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(-0.5, 15.5)
ax.set_ylim(-2.3, 3.0)
ax.axis("off")
# x axis
ax.annotate("", xy=(15.3, -1.9), xytext=(-0.3, -1.9), arrowprops=dict(arrowstyle="->", lw=0.8))
ax.text(15.3, -2.2, "$x$", ha="right")
# incident pulse
xs = np.linspace(0, 4.4, 400)
env = np.exp(-(((xs - 2.2) / 1.0) ** 2))
ax.plot(xs, 0.2 + 1.1 * env * np.sin(6 * (xs - 2.2)), color="tab:blue", lw=1.3)
ax.annotate("", xy=(4.9, 1.75), xytext=(3.6, 1.75), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.2))
ax.text(2.2, 1.95, r"incident pulse $E_\mathrm{inc}(x - ct)$" "\n" r"($E_y$, $B_z$; uniform in $y$, $z$)", ha="center",
        color="tab:blue", fontsize=8.5)
# sheets in perspective
x_sheets = np.linspace(6.0, 10.5, 9)
for i, x0 in enumerate(x_sheets):
    poly = np.array([[x0, -1.3], [x0 + 0.45, -0.75], [x0 + 0.45, 1.75], [x0, 1.2]])
    ax.add_patch(Polygon(poly, closed=True, facecolor="0.85", edgecolor="0.35", lw=0.6, alpha=0.9))
    # spring below each sheet
    ys = np.linspace(-1.75, -1.3, 30)
    ax.plot(x0 + 0.06 * np.sin(np.linspace(0, 8 * np.pi, 30)), ys, color="0.3", lw=0.6)
    ax.plot([x0 - 0.12, x0 + 0.12], [-1.75, -1.75], color="0.3", lw=1.0)
# emphasised sheet k
xk = x_sheets[4]
ax.annotate("", xy=(xk + 0.22, 1.05), xytext=(xk + 0.22, 0.25), arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.1))
ax.text(xk + 0.45, 0.62, r"$y_k(t)$", color="tab:red", fontsize=9)
ax.annotate("", xy=(xk - 1.9, 2.35), xytext=(xk + 0.1, 2.35), arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.1))
ax.annotate("", xy=(xk + 2.4, 2.35), xytext=(xk + 0.4, 2.35), arrowprops=dict(arrowstyle="->", color="tab:orange", lw=1.1))
ax.text(xk + 0.25, 2.55, r"$E_k(x,t) = -\frac{1}{2}\,K_k\!\left(t - |x - x_k|/c\right)$, $K_k = q\,\dot y_k$", ha="center",
        color="tab:orange", fontsize=9)
ax.text(8.4, -2.25, r"sheets $x_k$: charge $q$, mass $m$ per area, spring $\omega_0$; spacing $d$", ha="center", fontsize=8.5)
ax.text(13.6, 0.55, r"$E = E_\mathrm{inc} + \sum_k E_k$" "\n" r"every term travels at $c$", ha="center",
        fontsize=10, bbox=dict(boxstyle="round", fc="w", ec="0.5", lw=0.6))
fig.savefig(FIGURES / "figS_schematic.png")
