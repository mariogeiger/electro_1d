"""2D (x, y) + time: an s-polarized pulsed beam hits the thick slab at 45 degrees. Figures and animation.

Measures the beam axes (incident, reflected, transmitted) and the angle inside the slab against Snell's law.
"""
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from coupled_sheets import Beam, Medium
from style import FIGURES, plt

med = Medium(n_sheets=60, x0=0.0)
beam = Beam()
W, THETA, T_C, X_REF = beam.w_c, beam.theta, beam.t_c, beam.x_ref
t, y = beam.t, beam.y
f = beam.trace()
print("incident trace: max |f| for t < 5:", np.abs(f[t < 5]).max(), " for t > 120:", np.abs(f[t > 120]).max())
R = beam.response(med)
x = np.arange(-22.0, 32.0001, 0.2)
tf = np.arange(0.0, 75.0001, 0.25)
Ei, Em = R.fields(x, tf)

# geometric optics of the beam centre: Snell (phase matching) and v_g inside
n, ng = med.n(W).real, med.ng(W)
th_t = np.arcsin(np.sin(THETA) / n)
xa, xb = med.edges
ta = T_C + (xa - X_REF) / np.cos(THETA)
tb = ta + ng * (xb - xa) / np.cos(th_t)
print(f"n = {n:.4f}, n_g = {ng:.4f}, Snell theta_t = {np.rad2deg(th_t):.2f} deg, centre enters at t = {ta:.1f}, exits at t = {tb:.1f}")

# measured geometry: for each x, energy of one beam integrated over a window around its passage
# (+-6 around the geometric-optics time): centroid y(x) and mean time t(x), then straight-line fits


def axis(F, sel, t_pass):
    xs_, yc, tm = x[sel], [], []
    for j in np.flatnonzero(sel):
        g = np.abs(tf - t_pass(x[j])) < 6
        u = F[g, j, :].astype(np.float64) ** 2
        yc.append((u.sum(0) * y).sum() / u.sum())
        tm.append((u.sum(1) * tf[g]).sum() / u.sum())
    return np.polyfit(xs_, yc, 1), np.polyfit(xs_, tm, 1)


E = Ei + Em
before, after = (x > -9) & (x < xa - 2), (x > xb + 2) & (x < 20)
(inc_s, inc_y), inc_t = axis(Ei, before, lambda xx: T_C + (xx - X_REF) / np.cos(THETA))
(ref_s, _), _ = axis(Em, before, lambda xx: ta + (xa - xx) / np.cos(THETA))  # in front, E_mat = reflected beam
(tra_s, tra_y), tra_t = axis(E, after, lambda xx: tb + (xx - xb) / np.cos(THETA))
y_in, y_out = np.polyval((inc_s, inc_y), xa), np.polyval((tra_s, tra_y), xb)
t_in, t_out = np.polyval(inc_t, xa), np.polyval(tra_t, xb)
th_meas = np.arctan((y_out - y_in) / (xb - xa))
v_meas = (xb - xa) / np.cos(th_meas) / (t_out - t_in)
deg = np.rad2deg
print(f"incident {deg(np.arctan(inc_s)):.2f} deg (45), reflected {180 + deg(np.arctan(ref_s)):.2f} deg (135), "
      f"transmitted {deg(np.arctan(tra_s)):.2f} deg (45)")
print(f"inside, from the lateral shift {y_out - y_in:.3f}: theta_t = {deg(th_meas):.2f} deg (Snell {deg(th_t):.2f}); "
      f"crossing time {t_out - t_in:.2f} -> group speed {v_meas:.4f} (c/n_g = {1 / ng:.4f})")
th_i = np.arctan(inc_s)
print(f"Snell from the measured incidence {deg(th_i):.2f} deg: {deg(np.arcsin(np.sin(th_i) / n)):.2f} deg")

# geometric optics drawn with the measured incidence: straight line at c, Snell path, mirror
th_snell = np.arcsin(np.sin(th_i) / n)
y_exit = y_in + (xb - xa) * np.tan(th_snell)
ext = [x[0], x[-1], y[0], y[-1]]


def overlay(ax):
    for e in med.edges:
        ax.axvline(e, color="k", lw=0.5, ls="--")
    ci, si = np.cos(th_i), np.sin(th_i)
    ax.plot([xa - 30 * ci, xa + 40 * ci], [y_in - 30 * si, y_in + 40 * si], color="0.3", lw=0.6, ls=":")
    ax.plot([xa - 30 * ci, xa, xb, xb + 40 * ci], [y_in - 30 * si, y_in, y_exit, y_exit + 40 * si],
            color="k", lw=0.7, alpha=0.7)
    ax.plot([xa, xa - 30 * ci], [y_in, y_in + 30 * si], color="k", lw=0.7, alpha=0.4)
    ax.set_xlim(-20, 30); ax.set_ylim(-22, 38); ax.set_aspect("equal"); ax.set_xlabel("$x$")


def panel(ax, F, title):
    ax.imshow(F.T, origin="lower", extent=ext, vmin=-0.8, vmax=0.8, cmap="RdBu_r", interpolation="bilinear")
    overlay(ax); ax.set_title(title)


TITLES = [r"total $E_z = E_\mathrm{inc} + E_\mathrm{mat}$", r"incident $E_\mathrm{inc}$ (straight on, at $c$)",
          r"radiated by the sheets $E_\mathrm{mat}$"]
i = int(round(0.5 * (ta + tb) / 0.25))
fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.6), sharey=True)
for ax, F_, title in zip(axs, [E[i], Ei[i], Em[i]], TITLES):
    panel(ax, F_, title)
axs[0].set_ylabel("$y$")
fig.suptitle(f"$t = {tf[i]:.1f}$, s-polarized pulsed beam, incidence {deg(th_i):.1f}$^\\circ$. Lines: straight path at $c$ "
             f"(dotted), Snell path ({deg(th_snell):.1f}$^\\circ$ inside), mirror reflection", fontsize=9)
fig.tight_layout(); fig.savefig(FIGURES / "fig2d_decomposition.png")

times = [15.0, 30.0, 45.0, 62.0]
fig, axs = plt.subplots(1, 4, figsize=(13, 4.4), sharey=True)
for ax, ti in zip(axs, times):
    panel(ax, E[int(round(ti / 0.25))], f"$E_z$, $t = {ti:g}$")
axs[0].set_ylabel("$y$")
fig.tight_layout(); fig.savefig(FIGURES / "fig2d_snapshots.png")

fig, axs = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True, dpi=150)
ims = []
for ax, F_, title in zip(axs, [E[0], Ei[0], Em[0]], TITLES):
    ims.append(ax.imshow(F_.T, origin="lower", extent=ext, vmin=-0.8, vmax=0.8, cmap="RdBu_r",
                         interpolation="bilinear"))
    overlay(ax); ax.set_title(title)
axs[0].set_ylabel("$y$")
clock = fig.suptitle("")
fig.tight_layout()


def draw(k):
    for im, F_ in zip(ims, [E[k], Ei[k], Em[k]]):
        im.set_data(F_.T)
    clock.set_text(f"t = {tf[k]:5.2f}")
    return ims


writer = FFMpegWriter(fps=24, codec="libx264", bitrate=3000, extra_args=["-pix_fmt", "yuv420p"])
FuncAnimation(fig, draw, frames=len(tf)).save(FIGURES / "animation_beam_2d.mp4", writer=writer, dpi=150)
