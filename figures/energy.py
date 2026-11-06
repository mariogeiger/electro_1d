"""Figure G: where is the energy? Field energy (inside / outside the slab) and oscillator energy vs time."""
import numpy as np

from coupled_sheets import Medium, Pulse, fields, response
from style import FIGURES, plt

med = Medium(n_sheets=60)
p = Pulse()
R = response(med, p)
x = np.linspace(-45, 65, 4401)
dx = x[1] - x[0]
t = np.arange(0, 40.0001, 0.1)
F = fields(R, x, t)
u = 0.5 * (F["E"] ** 2 + F["B"] ** 2)  # (nt, nx)
a, b = med.edges
ins = (x >= a) & (x <= b)
it = np.round(t / R.dt).astype(int)
K = R.current()[:, it]
P = R.polarization()[:, it]
U_mech = (0.5 * (K**2 + med.w0**2 * P**2) / med.eps).sum(0)
U_in = u[:, ins].sum(1) * dx
U_left = u[:, x < a].sum(1) * dx
U_right = u[:, x > b].sum(1) * dx
U0 = (p.at0(x) ** 2).sum() * dx
tot = U_in + U_left + U_right + U_mech
print("energy conservation: max |total/U0 - 1| =", np.abs(tot / U0 - 1).max())
n, ng = med.n(p.k).real, med.ng(p.k)
frac_th = 1 - (1 + n**2) / (2 * n * ng)
frac = U_mech / (U_mech + U_in + 1e-30)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
ax = axs[0]
ax.plot(t, U_left / U0, color="tab:blue", label="field, left of slab (incident + reflected)")
ax.plot(t, U_in / U0, color="tab:green", label="field inside the slab")
ax.plot(t, U_mech / U0, color="tab:orange", label="oscillators (kinetic + spring)")
ax.plot(t, U_right / U0, color="tab:purple", label="field, right of slab (transmitted)")
ax.plot(t, tot / U0, color="k", ls="--", lw=0.8, label="total")
ax.set_xlabel("$t$"); ax.set_ylabel("energy / initial energy")
ax.legend(fontsize=7, frameon=False)
ax.set_title("energy bookkeeping (thick slab, Mario's pulse)")
ax = axs[1]
ok = (U_mech + U_in) > 0.5 * (U_mech + U_in).max()
ax.plot(t[ok], frac[ok], color="tab:orange", label="simulation")
ax.axhline(frac_th, color="k", lw=0.8, ls="--", label=rf"$1 - \frac{{1+n^2}}{{2 n n_g}} = {frac_th:.3f}$")
ax.set_xlabel("$t$"); ax.set_ylabel("share of the energy in the slab held by matter")
ax.set_ylim(0, 0.6)
ax.legend(fontsize=7.5, frameon=False)
ax.set_title("while the pulse is inside")
fig.tight_layout()
fig.savefig(FIGURES / "figG_energy.png")
print("frac plateau", np.median(frac[ok]), "theory", frac_th)
