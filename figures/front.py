"""Figure E: causality. The front of the pulse is never slowed: nothing exists outside the light cone of the front."""
import numpy as np

from coupled_sheets import Medium, Response
from style import DATA, FIGURES, plt

med = Medium(n_sheets=60)
# (a) log |E| in space-time for Mario's pulse (front: a jump of ~1e-3 of the truncated Gaussian)
F = np.load(DATA / "spacetime_thick.npz")
x, t = F["x"], F["t"]
fig = plt.figure(figsize=(10.5, 4.6))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1.3, 1])
ax = fig.add_subplot(gs[:, 0])
im = ax.imshow(np.log10(np.abs(F["E"]) + 1e-12), origin="lower", extent=[x[0], x[-1], t[0], t[-1]], vmin=-6, vmax=0,
               aspect="auto", cmap="magma", interpolation="bilinear")
for e in med.edges:
    ax.axvline(e, color="w", lw=0.5, ls="--")
ax.plot([10, 50], [0, 40], color="c", lw=0.8, ls=":")
ax.set_xlabel("$x$"); ax.set_ylabel("$t$")
ax.set_title(r"$\log_{10}|E|$, Mario's pulse, thick slab; dotted: $x = x_\mathrm{front} + ct$")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

# (b) a sine switched on in 0.3 time units (Sommerfeld's problem), seen behind the slab
dt, nt = 0.005, 32768
tt = np.arange(nt) * dt
w, ramp = 3.0, 0.05
f = np.sin(w * tt + np.pi / 2) * np.where(tt < ramp, np.sin(0.5 * np.pi * tt / ramp) ** 2, 1.0) * (tt >= 0)
f *= np.where(tt > 40, np.exp(-((tt - 40) / 5) ** 2), 1.0)  # smooth switch-off, long after
R = Response(f, dt, med.x0 - 1.0, med.xs, med.eps, med.w0)
xp = med.edges[1] + 2.0
Em, _ = R.matter_field([xp])
Ei = R.incident_field([xp])[0]
tr = tt - (xp - (med.x0 - 1.0))
m = (tr > -1) & (tr < 22)
ax = fig.add_subplot(gs[0, 1])
ax.plot(tr[m], Ei[m] + 2.6, color="0.45", lw=0.9)
ax.text(21.5, 3.9, "vacuum (no slab)", ha="right", fontsize=8, color="0.3")
ax.plot(tr[m], (Ei + Em[0])[m], color="k", lw=0.9)
ax.text(21.5, 1.3, "behind the slab", ha="right", fontsize=8)
tg = (med.ng(w) - 1) * med.length
for a_ in [ax]:
    a_.axvline(0, color="tab:blue", lw=0.8, ls=":")
    a_.axvline(tg, color="tab:orange", lw=0.8, ls=":")
ax.set_ylim(-1.6, 4.2); ax.set_yticks([]); ax.tick_params(labelbottom=False)
ax.set_title(r"sine switched on at $t=0$ ($\omega=3$), slab $L = 10.3$")
ax2 = fig.add_subplot(gs[1, 1], sharex=ax)
ax2.semilogy(tr[m], np.abs(Ei + Em[0])[m] + 1e-16, color="k", lw=0.6)
ax2.axvline(0, color="tab:blue", lw=0.8, ls=":")
ax2.axvline(tg, color="tab:orange", lw=0.8, ls=":")
ax2.text(0.15, 3e-6, r"front $t = x/c$", color="tab:blue", fontsize=8)
ax2.text(tg + 0.15, 3e-6, r"$t = x/c + (n_g-1)L/c$", color="tab:orange", fontsize=8)
ax2.set_ylim(1e-7, 3)
ax2.set_xlim(-1, 22)
ax2.set_ylabel(r"$|E|$ behind the slab")
ax2.set_xlabel(r"retarded time $t - x/c$")
fig.tight_layout()
fig.savefig(FIGURES / "figE_front.png")
print("jump at front transmitted/incident:", (Ei + Em[0])[np.argmin(np.abs(tr - 0.31))], Ei[np.argmin(np.abs(tr - 0.31))])
