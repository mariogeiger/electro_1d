"""Video clip of the oblique beam (dark theme, 1920x1080, 30 fps), timed to the 'refraction' narration.

Needs the narration timings (tts.py) and the measured beam geometry (figures/beam_2d.py); writes
out/video/clips/refraction.mp4.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

from coupled_sheets import Beam, Medium  # noqa: E402
from narration import timing  # noqa: E402

AUDIO = Path("out/video/audio")
TIMINGS = json.loads((AUDIO / "timings.json").read_text())
lead, speech, total = timing({k: v["speech"] for k, v in TIMINGS.items()})["refraction"]
FPS = 30
nframes = int(round(total * FPS))

med = Medium(n_sheets=60, x0=0.0)
beam = Beam()
R = beam.response(med)
y = beam.y
x = np.arange(-18.0, 30.0001, 0.15)
keep = (y >= -20) & (y <= 34)
tf = np.linspace(2.0, 74.0, nframes)
Ei, Em = R.fields(x, tf)
Ei, Em = Ei[:, :, keep], Em[:, :, keep]
yk = y[keep]
G = np.load("out/data/beam_2d_geometry.npz")
xa, xb, y_in, y_exit, th_i, th_t, th_measured = (float(G[k]) for k in
                                                 ["xa", "xb", "y_in", "y_exit", "th_i", "th_t", "th_measured"])

plt.style.use("dark_background")
plt.rcParams.update({"font.size": 20, "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "figure.facecolor": "#0e1116", "axes.facecolor": "#0e1116", "savefig.facecolor": "#0e1116"})
fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
titles = ["champ total  $E = E_\\mathrm{inc} + E_\\mathrm{mat}$", "onde incidente seule (à $c$)",
          "rayonnement des charges $E_\\mathrm{mat}$"]
colors = ["white", "#9aa0a6", "#ff9f40"]
ims = []
w_ax, h_ax, bot = 0.30, 0.66, 0.17
for k in range(3):
    ax = fig.add_axes([0.035 + k * 0.322, bot, w_ax, h_ax])
    ims.append(ax.imshow(np.zeros((len(yk), len(x))), origin="lower", extent=[x[0], x[-1], yk[0], yk[-1]],
                         vmin=-0.55, vmax=0.55, cmap="berlin", interpolation="bilinear", aspect="equal"))
    ax.add_patch(plt.Rectangle((xa, yk[0]), xb - xa, yk[-1] - yk[0], fill=False, ec="#5f6368", lw=1.2, ls="--"))
    ci, si = np.cos(th_i), np.sin(th_i)
    ax.plot([xa - 30 * ci, xa + 40 * ci], [y_in - 30 * si, y_in + 40 * si], color="#9aa0a6", lw=1.0, ls=":")
    ax.plot([xa - 30 * ci, xa, xb, xb + 40 * ci], [y_in - 30 * si, y_in, y_exit, y_exit + 40 * si], color="white",
            lw=1.0, alpha=0.55)
    ax.plot([xa, xa - 30 * ci], [y_in, y_in + 30 * si], color="white", lw=1.0, alpha=0.35)
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(yk[0], yk[-1])
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color("#3c4043")
    ax.set_title(titles[k], color=colors[k], fontsize=22, pad=12)
fig.text(0.5, 0.93, "Un faisceau oblique sur la lame (60 feuillets)", ha="center", fontsize=28, color="white")
degrees = "{:.2f}°".format
snell = fig.text(0.5, 0.075, f"loi de Snell-Descartes :  angle dans la lame  {degrees(np.rad2deg(th_measured))} mesuré"
                 f"  —  {degrees(np.rad2deg(th_t))} prédit".replace(".", ","),
                 ha="center", fontsize=24, color="#ffd28a", alpha=0.0)
legend = fig.text(0.5, 0.025, "pointillé : trajet en ligne droite à c      trait : trajet réfracté et réfléchi",
                  ha="center", fontsize=17, color="#9aa0a6")
t_snell = lead + TIMINGS["refraction"]["sentences"][2][0]
writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=8000, extra_args=["-pix_fmt", "yuv420p"])
clip = Path("out/video/clips/refraction.mp4")
clip.parent.mkdir(parents=True, exist_ok=True)
with writer.saving(fig, str(clip), dpi=100):
    for i in range(nframes):
        tv = i / FPS
        for im, F in zip(ims, [Ei[i] + Em[i], Ei[i], Em[i]]):
            im.set_data(F.T)
        snell.set_alpha(float(np.clip((tv - t_snell) / 0.6, 0, 1)))
        writer.grab_frame()
print("frames", nframes, "duration", total)
