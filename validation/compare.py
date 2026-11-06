"""Run the notebook's grid solver (collocated and Yee) and the exact solver on the same pulse and slab.

Saves out/data/compare.npz, E_y at every saved time for each solver, for figures/validation.py, and prints the
largest deviation of each grid run from the exact solution at a few times.
"""
import time
from pathlib import Path

import numpy as np

from coupled_sheets import MARIO, Pulse, fields, response
from run_grid import run

x_min, x_max, nx = -30.0, 70.0, 10240
t_end, nt = 30.0, 301
grid = {}
for yee, amp, atol in [(False, 1.0, 1e-6), (True, 1.0, 1e-6), (False, 1e-3, 1e-9), (True, 1e-3, 1e-9)]:
    t0 = time.time()
    x, ts, Ey, stats = run(Pulse(amp=amp).at0, MARIO.xs, x_min, x_max, nx, t_end, nt, q=1.0,
                           m=2 / MARIO.eps, omega=MARIO.w0, yee=yee, atol=atol)
    print(f"yee={yee} amp={amp}: {int(stats['num_steps'])} steps, {time.time() - t0:.1f} s")
    grid[f"grid_{'yee' if yee else 'col'}_{amp}"] = Ey / amp
exact = fields(response(MARIO, Pulse()), x, ts)["E"]
for name, Ey in grid.items():
    for ti in [4.0, 9.0, 14.0, 24.0]:
        it = int(round(ti / (t_end / (nt - 1))))
        print(f"{name} t={ti}: max |grid - exact| = {np.abs(Ey[it] - exact[it]).max():.2e}")
out = Path("out/data")
out.mkdir(parents=True, exist_ok=True)
np.savez(out / "compare.npz", x=x, t=ts, exact=exact, **grid)
