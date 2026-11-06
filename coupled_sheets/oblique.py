"""Exact linear response of a stack of charged sheets to an s-polarized pulsed beam in the (x, y) plane.

Units c = eps0 = 1. Fields are uniform along z and E = E_z(x, y, t) z_hat, parallel to the sheets x = x_k.
Every point of a sheet oscillates along z on its own spring:
    z_k''(y, t) = (q/m) E_z(x_k, y, t) - w0^2 z_k(y, t),   surface current K_k = q z_k'.
Fourier in t (numpy convention f = sum F exp(i s t), s = w - i eta) and in y (exp(i k_y y)). The stack is
invariant along y, so each k_y evolves alone: k_y is conserved, which is Snell's law. With
k_x = sqrt(s^2 - k_y^2), Im k_x <= 0, a sheet radiates E_z = -(s / (2 k_x)) K_k exp(-i k_x |x - x_k|), and
    [ -i (w0^2 - s^2)/(s eps) delta_jk + (s / (2 k_x)) exp(-i k_x |x_j - x_k|) ] K_k = E_inc(x_j).
At k_y = 0 this is the normal-incidence problem of normal.py.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


class Response2D:
    """Sheet currents K_k(s, k_y) for an incident field f(t, y) given on the plane x = x_ref (zero before t = 0)."""

    def __init__(self, f_ty, dt, dy, x_ref, xs, eps, w0, w_max, decay=25.0, max_elements=2**24):
        f_ty = np.asarray(f_ty, float)
        self.nt, self.ny = f_ty.shape
        self.dt, self.dy = dt, dy
        self.eta = decay / (self.nt * dt)  # exp(-eta T) = e^-25 suppresses wrap-around in time
        t = np.arange(self.nt) * dt
        w = 2 * np.pi * np.fft.rfftfreq(self.nt, dt)
        keep = w <= w_max  # the incident spectrum is negligible above w_max: so are all responses
        self.s = w[keep] - 1j * self.eta  # (nw,)
        self.ky = 2 * np.pi * np.fft.fftfreq(self.ny, dy)  # (ny,)
        # f must be the trace of a free field coming from the left (see pulsed_beam): causal, and its
        # spectrum lies inside the light cone |k_y| < w. No cut is applied here: a sharp cut would make
        # the drive non-causal, and exp(eta t) would amplify the resulting tails at late times.
        propagating = np.abs(self.ky)[None, :] < w[keep][:, None]
        self.F = np.fft.fft(np.fft.rfft(f_ty * np.exp(-self.eta * t)[:, None], axis=0)[keep], axis=1)
        # outgoing branch, Im k_x <= 0 (and Re k_x >= 0 for w >= 0)
        self.kx = -1j * np.sqrt(-(self.s[:, None] ** 2 - self.ky[None, :] ** 2))  # (nw, ny)
        # The free incident field needs no causality: at real frequencies its translation in x is exact
        # in both directions (with s = w - i eta, propagating it backwards would amplify by exp(eta |dx|)).
        self.w = w[keep]
        self.F_free = np.where(propagating, np.fft.fft(np.fft.rfft(f_ty, axis=0)[keep], axis=1), 0.0)
        self.kx_free = np.sqrt(np.maximum(self.w[:, None] ** 2 - self.ky[None, :] ** 2, 0.0))
        self.x_ref = x_ref
        self.xs = np.asarray(xs, float)
        self.eps = np.broadcast_to(np.asarray(eps, float), self.xs.shape)
        self.w0 = np.broadcast_to(np.asarray(w0, float), self.xs.shape)
        self.max_elements = max_elements
        self.K = self._solve()  # (nk, nw, ny)

    def _solve(self):
        xs = jnp.asarray(self.xs)
        dist = jnp.abs(xs[:, None] - xs[None, :])
        w0, eps, x_ref = jnp.asarray(self.w0), jnp.asarray(self.eps), self.x_ref

        def at(s, kx, F):
            M = (s / (2 * kx)) * jnp.exp(-1j * kx * dist) + jnp.diag(-1j * (w0**2 - s**2) / (s * eps))
            return jnp.linalg.solve(M, F * jnp.exp(-1j * kx * (xs - x_ref)))

        nw, ny = self.kx.shape
        args = (jnp.asarray(np.repeat(self.s, ny)), jnp.asarray(self.kx.ravel()), jnp.asarray(self.F.ravel()))
        K = jax.lax.map(lambda a: at(*a), args, batch_size=max(1, self.max_elements // dist.size))
        return np.asarray(K.T.reshape(len(self.xs), nw, ny))

    def fields(self, x, t, chunk=16):
        """Incident and matter E_z at positions x, times t, on the y grid: two arrays (nt, nx, ny), float32."""
        s, kx = jnp.asarray(self.s), jnp.asarray(self.kx)
        K, xs = jnp.asarray(self.K), jnp.asarray(self.xs)
        t = np.asarray(t, float)
        # time synthesis of numpy's irfft at arbitrary t, restricted to the kept bins (none is the Nyquist bin)
        weight = np.where(np.arange(len(self.s)) == 0, 1.0, 2.0)
        synth = jnp.asarray(weight[None, :] * np.exp(1j * np.outer(t, self.w)) / self.nt)  # (nt, nw)
        growth = jnp.asarray(np.exp(self.eta * t))[:, None]
        F_free, kx_free = jnp.asarray(self.F_free), jnp.asarray(self.kx_free)

        def to_space_time(X):  # (nw, ny) spectrum at one x -> (nt, ny)
            return (synth @ jnp.fft.ifft(X, axis=1)).real

        @jax.jit
        def at_x(xc):
            def add_sheet(acc, k):
                return acc + K[k] * jnp.exp(-1j * kx * jnp.abs(xc - xs[k])), None

            Em, _ = jax.lax.scan(add_sheet, jnp.zeros_like(kx), jnp.arange(len(self.xs)))
            Em = -(s[:, None] / (2 * kx)) * Em
            Ei = F_free * jnp.exp(-1j * kx_free * (xc - x_ref))
            return to_space_time(Ei), to_space_time(Em) * growth

        x_ref = self.x_ref
        Ei, Em = [], []
        x = np.asarray(x, float)
        for i in range(0, len(x), chunk):
            ei, em = jax.vmap(at_x)(jnp.asarray(x[i : i + chunk]))
            Ei.append(np.asarray(ei, np.float32))
            Em.append(np.asarray(em, np.float32))
        return np.concatenate(Ei, 0).transpose(1, 0, 2), np.concatenate(Em, 0).transpose(1, 0, 2)


def pulsed_beam(t, y, w_c, dw, theta, ds, t_c):
    """Trace on a plane x = const of a free, right-moving, s-polarized pulsed beam.

    Angular spectrum F(w, k_y) = exp(-((w - w_c)/dw)^2) exp(-((k_y/w + sin theta)/ds)^2) (1 - (k_y/w)^2)^2
    exp(-i w t_c): every component lies inside the light cone and vanishes smoothly at its edge, so the
    field is a sum of propagating plane waves only. Direction (cos theta, sin theta), centre crossing
    the plane at y = 0 at t = t_c, duration ~ 2/dw, direction-sine spread ds. Normalized to peak 1.
    """
    nt, ny = len(t), len(y)
    dt, dy = t[1] - t[0], y[1] - y[0]
    w = 2 * np.pi * np.fft.rfftfreq(nt, dt)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(ny, dy)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        sine = np.where(w > 0, ky / w, 2.0)  # numpy convention: k_y = -w sin(direction)
    inside = np.abs(sine) < 1
    F = np.where(inside, np.exp(-(((w - w_c) / dw) ** 2) - ((sine + np.sin(theta)) / ds) ** 2)
                 * (1 - np.where(inside, sine, 0) ** 2) ** 2 * np.exp(-1j * w * t_c), 0.0)
    f = np.fft.irfft(np.fft.ifft(F * np.exp(1j * ky * y[0]), axis=1), n=nt, axis=0)
    return f / np.abs(f).max()
