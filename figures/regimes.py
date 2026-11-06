"""Figure F: dispersion regimes. Space-time diagrams at four carrier frequencies + n(w), n_g(w) measured vs theory."""
import numpy as np

from coupled_sheets import MARIO, Pulse, Response, fields, lorentz_index, response
from style import FIGURES, plt, spacetime

med = MARIO
ws = [2.0, 4.0, 5.6, 8.0]
x = np.linspace(-15, 45, 801)
t = np.arange(0, 40.0001, 0.08)
fig = plt.figure(figsize=(11, 6.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.9])
for i, w in enumerate(ws):
    p = Pulse(center=2.5, a=3.0, k=w)
    R = response(med, p)
    F = fields(R, x, t)
    ax = fig.add_subplot(gs[0, i])
    spacetime(ax, F["E"], x, t, med, title=rf"$\omega = {w:g}$ ($\omega/\omega_0 = {w / med.w0:.2f}$)")
    ax.plot([2.5, 42.5], [0, 40], color="0.25", lw=0.6, ls=":")
    if i == 0:
        ax.set_ylabel("$t$")
    else:
        ax.set_yticklabels([])

# measured n and n_g from narrow-band pulses through the 30-sheet slab
dt, nt = 0.01, 16384
tt = np.arange(nt) * dt
xp = med.edges[1] + 1.0
meas = []
wm = np.concatenate([np.linspace(0.6, 4.6, 21), np.linspace(6.6, 12, 12)])
for w in wm:
    p = Pulse(center=0.0, a=10.0, k=w)
    f = p.at0(p.front - tt)
    R = Response(f, dt, p.front, med.xs, med.eps, med.w0)
    Em, _ = R.matter_field([xp])
    e = Em[0] + R.incident_field([xp])[0]
    e0 = R.incident_field([xp])[0]
    tr = tt - (xp - p.front)
    m = (tr > -2) & (tr < 110)
    gd = np.sum(tr[m] * e[m] ** 2) / np.sum(e[m] ** 2) - np.sum(tr[m] * e0[m] ** 2) / np.sum(e0[m] ** 2)
    ph = np.angle(np.sum((e * np.exp(-1j * w * tr))[m]) / np.sum((e0 * np.exp(-1j * w * tr))[m]))
    meas.append((w, gd, ph))
meas = np.array(meas)
L = med.length
ng_meas = 1 + meas[:, 1] / L
# unwrap the phase from the closest branch to the theory (only the 2 pi ambiguity is resolved by theory)
phase_th = -meas[:, 0] * (med.n(meas[:, 0]).real - 1) * L
ph = meas[:, 2] + 2 * np.pi * np.round((phase_th - meas[:, 2]) / (2 * np.pi))
n_meas = 1 - ph / (meas[:, 0] * L)

ax = fig.add_subplot(gs[1, :])
wg = np.linspace(0.05, 12, 3000)
nB = med.n(wg)
ax.axvspan(med.w0, np.sqrt(med.w0**2 + med.wp**2), color="0.9")
ax.text(5.58, 4.6, "stop band\n(total reflection)", ha="center", fontsize=8)
ax.plot(wg, nB.real, color="k", lw=1.2, label=r"$n(\omega)$ (periodic stack, exact)")
ax.plot(wg, med.n(wg, model=lorentz_index).real, color="k", lw=0.7, ls="--", label=r"$n(\omega)$ continuum Lorentz")
ok = np.abs(nB.imag) < 1e-9
ng = np.where(ok, med.ng(wg), np.nan)
ax.plot(wg, ng, color="tab:orange", lw=1.2, label=r"$n_g(\omega)$")
ax.plot(meas[:, 0], n_meas, "s", color="k", ms=3.5, label="measured: carrier phase")
ax.plot(meas[:, 0], ng_meas, "o", color="tab:orange", ms=4, label="measured: envelope delay")
ax.axhline(1, color="0.5", lw=0.5)
for w in ws:
    ax.axvline(w, color="tab:blue", lw=0.5, ls=":")
ax.set_ylim(0, 5.2)
ax.set_xlim(0, 12)
ax.set_xlabel(r"$\omega$  (here $\omega_0 = 5$, $\omega_p = 3.59$)")
ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
fig.tight_layout()
fig.savefig(FIGURES / "figF_regimes.png")
for w, a, b in zip(meas[:, 0], n_meas, ng_meas):
    print(f"w={w:.2f} n_meas={a:.4f} n_th={med.n(w).real:.4f} ng_meas={b:.4f} ng_th={med.ng(w):.4f}")
