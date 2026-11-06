"""Animation until the slab is quiet: E_inc, E_mat, E; a x20 zoom; energy left in the slab (log scale)."""
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from coupled_sheets import Medium, Pulse, fields, response
from style import DATA, FIGURES, plt

T_END, DT_FRAME, ZOOM = 300.0, 0.25, 20
med = Medium(n_sheets=60)
p = Pulse()
R = response(med, p, dt=0.02, nt=2**15)
x = np.linspace(-10, 50, 1201)
dx = x[1] - x[0]
tf = np.arange(0, T_END + 1e-9, DT_FRAME)
F = fields(R, x, tf)
a, b = med.edges
inside = (x >= a) & (x <= b)
U0 = (p.at0(x) ** 2).sum() * dx
U_field = 0.5 * (F["E"][:, inside] ** 2 + F["B"][:, inside] ** 2).sum(1) * dx / U0
K, P = R.current(), R.polarization()
keep = R.t <= T_END
U_mech = (0.5 * (K**2 + med.w0**2 * P**2) / med.eps).sum(0)[keep] / U0
t_mech = R.t[keep]
U_mech_f = np.interp(tf, t_mech, U_mech)
U_slab = U_field + U_mech_f
np.savez_compressed(DATA / "ringdown.npz", x=x, t=tf, U_field=U_field, U_mech=U_mech_f)

fig, axs = plt.subplots(3, 1, figsize=(9, 7), dpi=120, gridspec_kw=dict(height_ratios=[1, 1, 1.1]))
for ax in axs[:2]:
    ax.axvspan(a, b, color="0.92", zorder=0)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-1.2, 1.2)
li, = axs[0].plot([], [], color="0.45", ls="--", lw=1.0, label=r"$E_\mathrm{inc}$ (vacuum, $c$)")
lm, = axs[0].plot([], [], color="tab:orange", lw=1.0, label=r"$E_\mathrm{mat}$ (sheets)")
le, = axs[0].plot([], [], color="k", lw=1.5, label=r"$E = E_\mathrm{inc} + E_\mathrm{mat}$")
axs[0].legend(loc="lower left", fontsize=7, ncol=3, frameon=False)
title = axs[0].set_title("")
lz, = axs[1].plot([], [], color="k", lw=1.0)
axs[1].text(0.01, 0.92, rf"zoom: ${ZOOM}\times E$", transform=axs[1].transAxes, va="top", fontsize=8)
axs[1].set_xlabel("$x$")
ax = axs[2]
ax.semilogy(tf, U_slab, color="k", lw=1.2, label="total in the slab")
ax.semilogy(tf, U_mech_f, color="tab:orange", lw=0.9, label="oscillators")
ax.semilogy(tf, U_field, color="tab:green", lw=0.9, label="field inside the slab")
tt = np.linspace(60, T_END, 50)
ax.semilogy(tt, 1.7e-3 * (tt / 100) ** -2, color="tab:blue", ls=":", lw=1.2, label=r"$\propto t^{-2}$ (band-edge slow light)")
ax.set_ylim(1e-5, 1.5)
ax.set_xlim(0, T_END)
ax.set_xlabel("$t$")
ax.set_ylabel("energy / initial energy")
ax.legend(fontsize=7, frameon=False, loc="upper right")
cursor = ax.axvline(0, color="tab:red", lw=0.8)
dot, = ax.plot([], [], "o", color="tab:red", ms=4)
fig.tight_layout()


def draw(i):
    li.set_data(x, F["Ei"][i])
    lm.set_data(x, F["Em"][i])
    le.set_data(x, F["E"][i])
    lz.set_data(x, np.clip(ZOOM * F["E"][i], -1.19, 1.19))
    title.set_text(f"t = {tf[i]:6.2f}      energy left in the slab: {U_slab[i]:.1e} of the initial pulse")
    cursor.set_xdata([tf[i], tf[i]])
    dot.set_data([tf[i]], [U_slab[i]])
    return li, lm, le, lz, title, cursor, dot


anim = FuncAnimation(fig, draw, frames=len(tf), blit=False)
anim.save(FIGURES / "animation_ringdown.mp4", writer=FFMpegWriter(fps=30, codec="libx264", bitrate=2500,
                                                     extra_args=["-pix_fmt", "yuv420p"]))
for tq in [40, 60, 100, 200, 300]:
    i = int(round(tq / DT_FRAME))
    print(f"t={tq}: slab energy {U_slab[i]:.2e}, max|E| in window {np.abs(F['E'][i]).max():.3f}")
for tq in [60, 120, 200]:
    draw(int(round(tq / DT_FRAME)))
    fig.savefig(FIGURES / f"animation_ringdown_t{tq}.png", dpi=110)
