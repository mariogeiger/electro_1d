"""The notebook's grid Maxwell solver before the Yee fix, E and B collocated on the same nodes.

A verbatim copy of the model cells of notebook.ipynb at commit 0bb4070, kept as the reference that
figures/validation.py compares with the staggered grid and the exact solution.
"""
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=3, suppress=True)

class State(eqx.Module):
    def dot(self, simulator: "Simulator1D") -> "State":
        raise NotImplementedError


class Simulator1D(eqx.Module):
    x_grid: jax.Array  # space grid of shape (nx,)

    def __init__(self, nx: int, x_min: float, x_max: float):
        # Define the space grid
        self.x_grid = jnp.linspace(x_min, x_max, nx)

    @property
    def delta(self) -> float:
        return self.x_grid[1] - self.x_grid[0]

    def at(self, field: jax.Array, x: jax.Array) -> jax.Array:
        assert field.shape == self.x_grid.shape
        assert x.shape == ()
        return jnp.interp(x, self.x_grid, field)

    def dx(self, field: jax.Array) -> jax.Array:
        assert field.shape == self.x_grid.shape

        # periodic boundary conditions
        right = jnp.roll(field, -1)
        left = jnp.roll(field, 1)

        # right = jnp.append(field[1:], 0.0)
        # left = jnp.append(0.0, field[:-1])

        # right = jnp.append(field[1:], field[-2])
        # left = jnp.append(field[1], field[:-1])

        return (right - left) / (2 * self.delta)

    def dirac(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x)
        assert x.shape == ()

        dist = self.x_grid - x
        return (
            jnp.where(jnp.abs(dist) < self.delta, 1 - jnp.abs(dist) / self.delta, 0.0)
            / self.delta
        )

    @eqx.filter_jit
    def evolve(self, state: State, t_end: float, nt: int) -> diffrax.Solution:
        def f(t, y: State, _) -> State:
            return y.dot(self)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(f),
            diffrax.Tsit5(),
            t0=0.0,
            t1=t_end,
            dt0=t_end / nt,
            y0=state,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, nt)),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=32 * 1024,
        )
        return solution

# I tried to separate the physics (state and equation of motion) from the simulation (grid and time stepping).


class Charge(eqx.Module):
    # rho(t, x, y, z) = rho(t, x) = q dirac(x - pos[0])
    # J(t, x, y, z) = J(t, x) = q vel dirac(x - pos[0])
    pos: jax.Array
    vel: jax.Array
    # Electro-Magnetic properties
    m: jax.Array  # mass - relating the EM force to the acceleration
    q: (
        jax.Array
    )  # charge - relating (i) the EM field to the EM force and (ii) the current
    # The (YZ plane of) charge is hold by a spring
    anchor: jax.Array  # anchor position of the spring
    omega: jax.Array  # angular frequency of the spring constant


class World(eqx.Module):
    # E(t, x, y, z) = E(t, x)
    # B(t, x, y, z) = B(t, x)
    charges: Charge  # vmapped over the charges

    # Ex only depends on the charges and a global shift (set to 0)
    Ey: jax.Array
    Ez: jax.Array
    # Bx is a constant (set to 0)
    By: jax.Array
    Bz: jax.Array

    def E_at(self, x: float, sim: "Simulator1D") -> jax.Array:
        Ex = (
            0.5
            * jax.vmap(lambda ch: ch.q * jnp.sign(x - ch.pos[0]))(self.charges).sum()
        )
        return jnp.array([Ex, sim.at(self.Ey, x), sim.at(self.Ez, x)])

    def B_at(self, x: float, sim: "Simulator1D") -> jax.Array:
        return jnp.array([0.0, sim.at(self.By, x), sim.at(self.Bz, x)])

    def J(self, sim: "Simulator1D") -> jax.Array:
        return jax.vmap(lambda ch: ch.q * ch.vel * sim.dirac(ch.pos[0])[:, None])(
            self.charges
        ).sum(axis=0)  # (nx, 3)

    def dot(self, sim: "Simulator1D") -> "World":
        def dot_charge(charge: Charge) -> Charge:
            return Charge(
                m=0.0,
                q=0.0,
                pos=charge.vel,
                vel=charge.q
                * (
                    self.E_at(charge.pos[0], sim)
                    + jnp.cross(charge.vel, self.B_at(charge.pos[0], sim))
                )
                / charge.m
                - charge.omega**2 * (charge.pos - charge.anchor),
                anchor=jnp.zeros_like(charge.anchor),
                omega=jnp.zeros_like(charge.omega),
            )

        return World(
            charges=jax.vmap(dot_charge)(self.charges),
            Ey=-sim.dx(self.Bz) - self.J(sim)[:, 1],
            Ez=sim.dx(self.By) - self.J(sim)[:, 2],
            By=sim.dx(self.Ez),
            Bz=-sim.dx(self.Ey),
        )
