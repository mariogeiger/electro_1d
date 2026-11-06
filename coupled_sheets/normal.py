"""Exact linear response of a stack of charged, spring-held sheets to a normally incident pulse.

Units: c = eps0 = 1.  A sheet at x_k carrying a transverse surface current K_k(t) radiates
    E_y(x, t) = -K_k(t - |x - x_k|) / 2,   B_z(x, t) = sign(x - x_k) E_y(x, t)
(exact 1D Green function; the same expression as Feynman's field of a plane of oscillators).
Each sheet obeys  m y'' = q E_y(x_k) - m w0^2 y  with K = q y'; only eps = q^2/m matters.
In the frequency domain (numpy convention f(t) = sum F exp(i s t), s = w - i eta, so that a delay
tau is exp(-i s tau) and d/dt is i s) the sheet currents solve
    [ -i (w0^2 - s^2)/(s eps_j) delta_jk + exp(-i s |x_j - x_k|)/2 ] K_k = E_inc(x_j, s).
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


class Response:
    """Sheet currents K_k(s) on the complex frequencies s, for an incident field f(t - (x - x_ref))."""

    def __init__(self, f_t, dt, x_ref, xs, eps, w0, decay=25.0):
        f_t = np.asarray(f_t, float)
        self.nt = len(f_t)
        self.dt = dt
        self.t = np.arange(self.nt) * dt
        self.eta = decay / (self.nt * dt)  # exp(-eta T) = e^-25 suppresses wrap-around
        self.s = 2 * np.pi * np.fft.rfftfreq(self.nt, dt) - 1j * self.eta
        self.F = np.fft.rfft(f_t * np.exp(-self.eta * self.t))
        self.x_ref = x_ref
        self.xs = np.asarray(xs, float)
        self.eps = np.broadcast_to(np.asarray(eps, float), self.xs.shape)
        self.w0 = np.broadcast_to(np.asarray(w0, float), self.xs.shape)
        self.f_t = f_t
        self.K = self._solve()  # (nk, nf)

    def _solve(self, max_elements=2**24):
        """Solve the N x N system at every frequency, in frequency batches bounded by max_elements."""
        xs = jnp.asarray(self.xs)
        dist = jnp.abs(xs[:, None] - xs[None, :])
        w0, eps, x_ref = jnp.asarray(self.w0), jnp.asarray(self.eps), self.x_ref

        def at(s, F):
            inv_chi = -1j * (w0**2 - s**2) / (s * eps)
            M = 0.5 * jnp.exp(-1j * s * dist) + jnp.diag(inv_chi)
            return jnp.linalg.solve(M, F * jnp.exp(-1j * s * (xs - x_ref)))

        batch = max(1, max_elements // dist.size)
        K = jax.lax.map(lambda a: at(*a), (jnp.asarray(self.s), jnp.asarray(self.F)), batch_size=batch)
        return np.asarray(K.T)

    def to_time(self, X):
        """X(s) (..., nf) -> real time signal (..., nt)."""
        return np.fft.irfft(X, n=self.nt) * np.exp(self.eta * self.t)

    def current(self):
        return self.to_time(self.K)

    def polarization(self):
        """P_k = q y_k, the time integral of K_k."""
        return self.to_time(self.K / (1j * self.s))

    def matter_field(self, x, it=slice(None), max_elements=2**24):
        """E_y and B_z radiated by all sheets at positions x and time indices it: (nx, nt_kept) each."""
        K = jnp.asarray(self.K)
        s = jnp.asarray(self.s)
        xs = jnp.asarray(self.xs)
        growth = np.exp(self.eta * self.t)

        @jax.jit
        def chunk_fields(xc):
            d = xc[:, None] - xs[None, :]  # (nc, nk)
            ph = jnp.exp(-1j * s[None, None, :] * jnp.abs(d)[:, :, None]) * K[None]  # (nc, nk, nf)
            E = -0.5 * ph.sum(1)
            B = -0.5 * (jnp.sign(d)[:, :, None] * ph).sum(1)
            return jnp.fft.irfft(E, n=self.nt), jnp.fft.irfft(B, n=self.nt)

        x = np.asarray(x, float)
        chunk = max(1, max_elements // K.size)
        E, B = [], []
        for i in range(0, len(x), chunk):
            e, b = chunk_fields(jnp.asarray(x[i : i + chunk]))
            E.append((np.asarray(e) * growth)[:, it])
            B.append((np.asarray(b) * growth)[:, it])
        return np.concatenate(E), np.concatenate(B)

    def incident_field(self, x, it=slice(None)):
        """E_y = B_z of the incident pulse at positions x and time indices it, by exact translation of f."""
        tau = self.t[it][None, :] - (np.asarray(x)[:, None] - self.x_ref)
        return np.interp(tau, self.t, self.f_t, left=0.0, right=0.0)

