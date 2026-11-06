"""Run the notebook's grid Maxwell solver, collocated or staggered (Yee), for a pulse hitting a slab of +-q pairs."""
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from grid_collocated import Charge, Simulator1D, World
from grid_yee import YeeSimulator1D, YeeWorld


def evolve(sim, state, t_end, nt, rtol, atol, max_steps=64 * 1024):
    """Simulator1D.evolve with explicit tolerances (same Tsit5 + PID controller)."""
    @eqx.filter_jit
    def go(state):
        return diffrax.diffeqsolve(diffrax.ODETerm(lambda t, y, _: y.dot(sim)), diffrax.Tsit5(), t0=0.0, t1=t_end,
                                   dt0=t_end / nt, y0=state, saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, nt)),
                                   stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol), max_steps=max_steps)
    return go(state)


def run(wave_fn, anchors, x_min, x_max, nx, t_end, nt, q=1.0, m=0.9, omega=5.0, yee=False, rtol=1e-6, atol=1e-6):
    sim = (YeeSimulator1D if yee else Simulator1D)(nx, x_min=x_min, x_max=x_max)
    xg = np.asarray(sim.x_grid)
    wave = jnp.asarray(wave_fn(xg))
    waveB = jnp.asarray(wave_fn(xg + 0.5 * float(sim.delta))) if yee else wave
    nc = 2 * len(anchors)

    def init_charge(qq, x0):
        return Charge(m=m, q=qq, pos=jnp.array([x0, 0.0, 0.0]), vel=jnp.zeros(3),
                      omega=jnp.array(omega), anchor=jnp.array([x0, 0.0, 0.0]))

    state = (YeeWorld if yee else World)(
        charges=jax.vmap(init_charge)(q * (-1.0) ** jnp.arange(nc), jnp.repeat(jnp.asarray(anchors), 2)),
        Ey=wave, Ez=jnp.zeros_like(wave), By=jnp.zeros_like(wave), Bz=waveB)
    sol = evolve(sim, state, t_end, nt, rtol, atol)
    return np.asarray(sim.x_grid), np.asarray(sol.ts), np.asarray(sol.ys.Ey), sol.stats
