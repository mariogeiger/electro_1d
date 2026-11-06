"""Dark 1920x1080 figures for the video, from the data of the notes' figures (out/data, see figures/).

Writes out/video/figures/*.png, and the axes boxes (figure fractions [left, bottom, width, height]) to
out/video/figures/boxes.json so that the scenes can point at panels.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, GREY, ORANGE, GOLD = "#0e1116", "#9aa0a6", "#ff9f40", "#ffd28a"
plt.style.use("dark_background")
plt.rcParams.update({"font.size": 20, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                     "axes.edgecolor": "#5f6368", "axes.labelcolor": "white"})
D = Path("out/data")
OUT = Path("out/video/figures")
OUT.mkdir(parents=True, exist_ok=True)
boxes = {}


def figure(name):
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
    boxes[name] = []
    return fig


def axes(fig, name, rect, **kw):
    boxes[name].append(rect)
    return fig.add_axes(rect, **kw)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=100)
    plt.close(fig)


def spacetime_image(ax, F, x, t, vmax=0.55):
    ax.imshow(F, origin="lower", extent=[x[0], x[-1], t[0], t[-1]], vmin=-vmax, vmax=vmax, cmap="berlin",
              aspect="auto", interpolation="bilinear")


# --- space-time triptych (thick slab)
S = np.load(D / "spacetime_thick.npz")
x, t = S["x"], S["t"]
a, b = S["edges"]
fig = figure("spacetime")
titles = [("champ total", "white"), ("onde incidente seule", GREY), ("rayonnement des charges", ORANGE)]
for k, (key, (title, color)) in enumerate(zip(["E", "Ei", "Em"], titles)):
    ax = axes(fig, "spacetime", [0.05 + k * 0.32, 0.12, 0.28, 0.72])
    spacetime_image(ax, S[key], x, t)
    for e in (a, b):
        ax.axvline(e, color="#5f6368", lw=1.2, ls="--")
    ax.plot([5, 45], [0, 40], color=GREY, lw=1.0, ls=":")
    ax.set_title(title, color=color, fontsize=26, pad=14)
    ax.set_xlabel("position  x")
    if k == 0:
        ax.set_ylabel("temps  t")
    else:
        ax.set_yticklabels([])
fig.text(0.5, 0.93, "Diagramme espace-temps : le temps monte, l'espace va vers la droite", ha="center", fontsize=26)
save(fig, "spacetime")

# --- front: log |E| in space-time, and a switched-on sine behind the slab
fig = figure("front")
ax = axes(fig, "front", [0.05, 0.12, 0.36, 0.72])
im = ax.imshow(np.log10(np.abs(S["E"]) + 1e-12), origin="lower", extent=[x[0], x[-1], t[0], t[-1]], vmin=-6,
               vmax=0, aspect="auto", cmap="magma", interpolation="bilinear")
ax.plot([10, 50], [0, 40], color="#7fd3ff", lw=1.5, ls=":")
for e in (a, b):
    ax.axvline(e, color="white", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("position  x"); ax.set_ylabel("temps  t")
ax.set_title("intensité (échelle log) : rien au-delà du front", fontsize=22, pad=14)
ax.text(36, 20, "front à c", color="#7fd3ff", fontsize=22, rotation=38)
Fr = np.load(D / "front.npz")
ax = axes(fig, "front", [0.48, 0.52, 0.48, 0.32])
ax.plot(Fr["tr"], Fr["vac"], color=GREY, lw=1.6)
ax.set_xlim(-1, 22); ax.set_ylim(-1.3, 1.3); ax.set_yticks([]); ax.set_xticklabels([])
ax.set_title("sinusoïde allumée à t = 0 : sans la lame", color=GREY, fontsize=22, pad=10)
ax2 = axes(fig, "front", [0.48, 0.12, 0.48, 0.32])
ax2.plot(Fr["tr"], Fr["slab"], color="white", lw=1.6)
for ax_ in (ax, ax2):
    ax_.axvline(0, color="#7fd3ff", lw=1.5, ls=":")
ax2.axvline(float(Fr["t_group"]), color=ORANGE, lw=1.5, ls=":")
ax2.set_xlim(-1, 22); ax2.set_ylim(-1.3, 1.3); ax2.set_yticks([])
ax2.set_xlabel("temps, compté depuis l'arrivée du front")
ax2.set_title("derrière la lame", fontsize=22, pad=10)
ax2.text(0.3, 1.0, "le front arrive à c", color="#7fd3ff", fontsize=19)
ax2.text(float(Fr["t_group"]) + 0.3, -1.15, "le gros du signal, en retard", color=ORANGE, fontsize=19)
save(fig, "front")

# --- delays after N sheets: envelope vs crests
L_ = np.load(D / "layers.npz")
N, d = L_["N"], float(L_["d"])
fig = figure("delays")
ax = axes(fig, "delays", [0.12, 0.15, 0.80, 0.68])
ax.plot(N[::4], L_["group"][::4], "o", color=ORANGE, ms=11, label="enveloppe (vitesse de groupe)")
ax.plot(N[::4], L_["phase"][::4], "s", color="white", ms=9, label="crêtes (vitesse de phase)")
ax.plot(N, (float(L_["ng"]) - 1) * N * d, color=ORANGE, lw=2)
ax.plot(N, (float(L_["n"]) - 1) * N * d, color="white", lw=2)
ax.set_xlabel("nombre de feuillets traversés"); ax.set_ylabel("retard")
ax.legend(frameon=False, fontsize=22, loc="upper left")
ax.text(64, (float(L_["ng"]) - 1) * 64 * d + 0.3, f"$n_g = {float(L_['ng']):.2f}$", color=ORANGE, ha="right",
        fontsize=24)
ax.text(64, (float(L_["n"]) - 1) * 64 * d + 0.3, f"$n = {float(L_['n']):.2f}$", color="white", ha="right",
        fontsize=24)
ax.set_title("Mesuré (symboles) contre théorie (traits)", fontsize=24, pad=14)
save(fig, "delays")

# --- regimes: four carriers + dispersion
R = np.load(D / "regimes.npz")
fig = figure("colors")
labels = ["sous la résonance", "près de la résonance", "bande interdite : miroir", "au-dessus : n < 1"]
xr, tr_ = R["x"], R["t"]
for k in range(4):
    ax = axes(fig, "colors", [0.04 + k * 0.24, 0.50, 0.21, 0.38])
    spacetime_image(ax, R["E"][k], xr, tr_)
    for e in R["edges"]:
        ax.axvline(e, color="#5f6368", lw=1.0, ls="--")
    ax.plot([2.5, 42.5], [0, 40], color=GREY, lw=0.8, ls=":")
    ax.set_title(f"ω = {R['ws'][k]:g}  —  {labels[k]}", fontsize=19, pad=10)
    ax.set_xticks([]); ax.set_yticks([])
ax = axes(fig, "colors", [0.07, 0.08, 0.88, 0.34])
w0, wp = float(R["w0"]), float(R["wp"])
ax.axvspan(w0, np.sqrt(w0**2 + wp**2), color="#3c4043")
ax.plot(R["wg"], R["n"], color="white", lw=2.2, label="indice de phase n")
ax.plot(R["wg"], R["ng"], color=ORANGE, lw=2.2, label="indice de groupe $n_g$")
ax.plot(R["wm"], R["n_meas"], "s", color="white", ms=7)
ax.plot(R["wm"], R["ng_meas"], "o", color=ORANGE, ms=8)
ax.axhline(1, color=GREY, lw=1, ls=":")
for w in R["ws"]:
    ax.axvline(w, color="#7fd3ff", lw=1, ls=":")
ax.set_xlim(0, 12); ax.set_ylim(0, 5)
ax.set_xlabel("fréquence ω   (ressorts : ω₀ = 5)")
ax.legend(frameon=False, fontsize=19, loc="upper right")
ax.text(5.58, 4.3, "miroir", ha="center", fontsize=19, color=GOLD)
save(fig, "colors")

# --- slow light: energy left in the slab
A = np.load(D / "ringdown.npz")
ta, U = A["t"], A["U_field"] + A["U_mech"]
fig = figure("slowlight")
ax = axes(fig, "slowlight", [0.12, 0.15, 0.80, 0.68])
ax.semilogy(ta, U, color="white", lw=2.5, label="énergie restant dans la lame")
tt = np.linspace(60, 300, 50)
ax.semilogy(tt, 1.7e-3 * (tt / 100) ** -2, color="#7fd3ff", lw=2.5, ls=":", label="pente 1/t²")
ax.set_ylim(1e-5, 2); ax.set_xlim(0, 300)
ax.set_xlabel("temps"); ax.set_ylabel("fraction de l'énergie de l'impulsion")
ax.legend(frameon=False, fontsize=24)
ax.set_title("Après l'impulsion, la lame continue de vibrer", fontsize=24, pad=14)
save(fig, "slowlight")

(OUT / "boxes.json").write_text(json.dumps(boxes, indent=1))
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
