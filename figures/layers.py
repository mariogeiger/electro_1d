"""Figure D: layer by layer. Transmitted pulse after N sheets, in retarded time; delays grow like N."""
import numpy as np
from scipy.signal import hilbert

from coupled_sheets import MARIO, Medium, Pulse, Response
from style import FIGURES, plt

dt, nt = 0.01, 16384
t = np.arange(nt) * dt
NMAX = 64


def transmitted(pulse, Nk):
    """Field at a probe beyond the largest stack, vs retarded time t - (x_probe - front)."""
    f = pulse.at0(pulse.front - t)
    med = Medium(n_sheets=max(Nk, 1))
    xp = med.x0 + NMAX * med.d + 1.0
    tr = t - (xp - pulse.front)
    if Nk == 0:
        return tr, np.interp(tr, t, f, left=0, right=0)
    R = Response(f, dt, pulse.front, med.xs, med.eps, med.w0)
    Em, _ = R.matter_field([xp])
    return tr, Em[0] + R.incident_field([xp])[0]


# waterfall with Mario's pulse
p = Pulse()
Ns = [0, 1, 2, 4, 8, 16, 32, 64]
fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]
for i, Nk in enumerate(Ns):
    tr, e = transmitted(p, Nk)
    m = (tr > -3) & (tr < 30)
    off = -2.0 * i
    ax.plot(tr[m], e[m] + off, color="k" if Nk else "0.45", lw=0.8)
    ax.plot(tr[m], np.abs(hilbert(e[m])) + off, color="tab:orange", lw=0.7)
    ax.text(-0.5, off, f"$N={Nk}$", ha="right", va="center", fontsize=8)
ax.axvline(0, color="tab:blue", lw=0.8, ls=":")
ax.set_xlim(-2.8, 24)
ax.set_yticks([])
ax.set_xlabel(r"retarded time $t - x/c$ at the probe (front of the incident pulse at $0$)")
ax.set_title(r"Mario's pulse after $N$ sheets (spacing $d$); orange: envelope")

# delays with a narrow-band pulse, every N
pn = Pulse(a=8.0)
w = pn.k
tr0, e0 = transmitted(pn, 0)
m0 = (tr0 > -2) & (tr0 < 70)
z0 = np.sum((e0 * np.exp(-1j * w * tr0))[m0])
gd, ph = [], []
Nall = np.arange(0, NMAX + 1)
for Nk in Nall:
    tr, e = transmitted(pn, Nk)
    m = (tr > -2) & (tr < 70)
    gd.append(np.sum(tr[m] * e[m] ** 2) / np.sum(e[m] ** 2))
    ph.append(np.angle(np.sum((e * np.exp(-1j * w * tr))[m]) / z0))
gd = np.array(gd) - gd[0]
pd = -np.unwrap(np.array(ph)) / w
ax = axs[1]
nb, ngb = MARIO.n(w).real, MARIO.ng(w)
ax.plot(Nall[::4], gd[::4], "o", color="tab:orange", ms=4, label="envelope delay (energy centroid)")
ax.plot(Nall[::4], pd[::4], "s", color="k", ms=3.5, label=r"carrier phase delay $-\arg T/\omega$")
ax.plot(Nall, (ngb - 1) * Nall * MARIO.d, color="tab:orange", lw=0.8, label=f"$(n_g-1)Nd$, $n_g={ngb:.3f}$")
ax.plot(Nall, (nb - 1) * Nall * MARIO.d, color="k", lw=0.8, label=f"$(n-1)Nd$, $n={nb:.3f}$")
ax.set_xlabel("number of sheets $N$")
ax.set_ylabel(r"delay (units of $1/c$ times length)")
ax.legend(fontsize=7.5, frameon=False)
ax.set_title(r"narrow-band pulse ($\omega = 3$): group vs phase delay")
fig.tight_layout()
fig.savefig(FIGURES / "figD_layers.png")
print("group delay per sheet (fit)", np.polyfit(Nall[8:], gd[8:], 1)[0], "theory", (ngb - 1) * MARIO.d)
print("phase delay per sheet (fit)", np.polyfit(Nall[8:], pd[8:], 1)[0], "theory", (nb - 1) * MARIO.d)
