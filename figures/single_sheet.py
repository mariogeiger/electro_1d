"""Figure C: one sheet, time domain. Its field is -tau dE/dt: forward it delays the pulse, backward it is the echo."""
import numpy as np

from coupled_sheets import Response
from style import FIGURES, plt

eps, w0 = 7.5, 5.0  # one strong sheet (3.4 of Mario's pairs merged), tau = eps/(2 w0^2) = 0.15
tau = eps / (2 * w0**2)
dt, nt = 0.002, 16384
t = np.arange(nt) * dt
a, t0 = 1.0, 4.0
f = np.exp(-(((t - t0) / a) ** 2)) * (np.abs(t - t0) < 3.5 * a)
R = Response(f, dt, 0.0, [1.0], eps, w0)
xp = np.array([3.0, -1.0])
Em, _ = R.matter_field(xp)
Ei = R.incident_field(xp)
sel = t < 12
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
# behind the sheet: tau_r = t - (x - x_ref) is the retarded time
tr = t - 3.0
ax = axs[0]
inc = Ei[0]
dinc = np.gradient(inc, dt)
ax.plot(tr[sel], inc[sel], color="0.45", ls="--", lw=1.2, label=r"$E_\mathrm{inc}$")
ax.plot(tr[sel], Em[0][sel], color="tab:orange", lw=1.4, label=r"$E_\mathrm{sheet}$")
tot = inc + Em[0]
ax.plot(tr[sel], -tau * np.gradient(tot, dt)[sel], color="tab:orange", ls=":", lw=1.4,
        label=r"$-\tau\,\partial_t E$ (field felt by the sheet)")
ax.plot(tr[sel], (inc + Em[0])[sel], color="k", lw=1.4, label=r"$E_\mathrm{inc} + E_\mathrm{sheet}$")
ax.plot(tr[sel] + tau, inc[sel], color="tab:blue", ls=":", lw=1.2, label=r"$E_\mathrm{inc}(t - \tau)$")
ax.set_xlim(0.5, 7.5)
ax.set_xlabel(r"retarded time $t - x/c$")
ax.set_title(r"behind the sheet ($x > x_1$): transmitted pulse")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax = axs[1]
tl = t - 2.0  # in front of the sheet (x = -1): the echo arrives at t - (1 + 1)
ax.plot(tl[sel], Em[1][sel], color="tab:orange", lw=1.4, label=r"$E_\mathrm{sheet}$ = echo")
ax.plot(tl[sel], -tau * np.gradient(tot, dt)[sel], color="tab:orange", ls=":", lw=1.4, label=r"$-\tau\,\partial_t E$")
ax.set_xlim(0.5, 7.5)
ax.set_xlabel(r"retarded time $t - (2x_1 - x)/c$")
ax.set_title(r"in front of the sheet ($x < x_1$): reflected pulse")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(FIGURES / "figC_single_sheet.png")
pk = lambda y: tr[np.argmax(y)]
print("tau", tau, "peak delay of transmitted:", pk(inc + Em[0]) - pk(inc))
