"""Figure A: space-time diagrams E = E_inc + E_mat (the notebook's slab and a thicker one); saves their data."""
import numpy as np

from coupled_sheets import MARIO, Medium, Pulse, centre_path, fields, response
from style import DATA, FIGURES, plt, spacetime

for tag, med, xlim, tmax in [("mario", MARIO, (-5, 40), 30.0), ("thick", Medium(n_sheets=60), (-10, 50), 40.0)]:
    p = Pulse()
    R = response(med, p)
    x = np.linspace(*xlim, 1001)
    t = np.arange(0, tmax + 1e-9, 0.05)
    F = fields(R, x, t)
    np.savez_compressed(DATA / f"spacetime_{tag}.npz", x=x, t=t, xs=med.xs, edges=med.edges, **F)
    print(tag, f"n={med.n(p.k).real:.4f} ng={med.ng(p.k):.4f} L={med.length:.3f} delay={(med.ng(p.k)-1)*med.length:.2f}")
    xc, tc = centre_path(med, p, p.k, tmax)
    fig, axs = plt.subplots(1, 3, figsize=(11, 4.4), sharey=True)
    titles = [r"total field $E = E_\mathrm{inc} + E_\mathrm{mat}$",
              r"incident field $E_\mathrm{inc}(x,t) = E_0(x - ct)$",
              r"field radiated by matter $E_\mathrm{mat} = -\frac{1}{2}\sum_k K_k(t - |x - x_k|/c)$"]
    for ax, key, title in zip(axs, ["E", "Ei", "Em"], titles):
        spacetime(ax, F[key], x, t, med, title=title)
        ax.plot([p.center, p.center + tmax], [0, tmax], color="0.25", lw=0.7, ls=":")
        ax.plot(xc, tc, color="k", lw=0.8, alpha=0.7)
    axs[0].set_ylabel("$t$")
    fig.tight_layout()
    fig.savefig(FIGURES / f"figA_spacetime_{tag}.png")
