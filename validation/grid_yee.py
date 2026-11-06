"""The notebook's grid solver on a staggered Yee grid: E_y, E_z on nodes x_i, B_y, B_z on x_i + delta/2.

Only the spatial discretization of grid_collocated.py changes; the physics (World.dot) is the same equations.
This is the fix that the notebook adopted in commit d45f7c7.
"""
import equinox as eqx
import jax
import jax.numpy as jnp

from grid_collocated import Charge, Simulator1D

jax.config.update("jax_enable_x64", True)


class YeeSimulator1D(Simulator1D):
    def at_half(self, field, x):
        return jnp.interp(x, self.x_grid + 0.5 * self.delta, field, period=self.x_grid[-1] - self.x_grid[0] + self.delta)

    def at(self, field, x):
        return jnp.interp(x, self.x_grid, field, period=self.x_grid[-1] - self.x_grid[0] + self.delta)

    def dx_to_half(self, field):  # nodes -> half nodes (periodic)
        return (jnp.roll(field, -1) - field) / self.delta

    def dx_to_node(self, field):  # half nodes -> nodes (periodic)
        return (field - jnp.roll(field, 1)) / self.delta


class YeeWorld(eqx.Module):
    charges: Charge
    Ey: jax.Array  # nodes
    Ez: jax.Array  # nodes
    By: jax.Array  # half nodes
    Bz: jax.Array  # half nodes

    def E_at(self, x, sim):
        Ex = 0.5 * jax.vmap(lambda ch: ch.q * jnp.sign(x - ch.pos[0]))(self.charges).sum()
        return jnp.array([Ex, sim.at(self.Ey, x), sim.at(self.Ez, x)])

    def B_at(self, x, sim):
        return jnp.array([0.0, sim.at_half(self.By, x), sim.at_half(self.Bz, x)])

    def J(self, sim):
        return jax.vmap(lambda ch: ch.q * ch.vel * sim.dirac(ch.pos[0])[:, None])(self.charges).sum(axis=0)

    def dot(self, sim):
        def dot_charge(charge):
            return Charge(
                m=0.0, q=0.0, pos=charge.vel,
                vel=charge.q * (self.E_at(charge.pos[0], sim) + jnp.cross(charge.vel, self.B_at(charge.pos[0], sim))) / charge.m
                - charge.omega**2 * (charge.pos - charge.anchor),
                anchor=jnp.zeros_like(charge.anchor), omega=jnp.zeros_like(charge.omega))

        J = self.J(sim)
        return YeeWorld(
            charges=jax.vmap(dot_charge)(self.charges),
            Ey=-sim.dx_to_node(self.Bz) - J[:, 1],
            Ez=sim.dx_to_node(self.By) - J[:, 2],
            By=sim.dx_to_half(self.Ez),
            Bz=-sim.dx_to_half(self.Ey))
