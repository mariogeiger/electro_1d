"""The exact solvers against an independent method: the transfer matrix of the sheet stack.

At each of a solver's complex frequencies s = w - i eta (and, in 2D, each transverse wavenumber k_y), a plane wave
crossing the stack is transmitted and reflected by amounts that 2x2 transfer matrices give in closed form. The
fields the solvers build from their sheet currents must agree with them to rounding error.
"""
import numpy as np

from coupled_sheets import Response, Response2D, pulsed_beam

RNG = np.random.default_rng(0)
XS = np.sort(RNG.uniform(0.0, 5.0, 12))
EPS = RNG.uniform(0.5, 3.0, 12)
W0 = RNG.uniform(3.0, 7.0, 12)


def transfer(s, kx, xs, eps, w0):
    """Transmission t and reflection r of the stack, per frequency, for a unit plane wave arriving at xs[0].

    E = exp(-i kx (x - x_1)) + r exp(i kx (x - x_1)) before the first sheet, t exp(-i kx (x - x_N)) after the last.
    State (E, dE/dx): free space propagates it by the harmonic-oscillator matrix; a sheet, whose current is
    K = i s eps E / (w0^2 - s^2), makes dE/dx jump by i s K.
    """
    one, zero = np.ones_like(kx), np.zeros_like(kx)
    a, b, c, d = one, zero, zero, one
    for k in range(len(xs)):
        if k:
            h = xs[k] - xs[k - 1]
            cs, sn = np.cos(kx * h), np.sin(kx * h)
            a, b, c, d = cs * a + sn / kx * c, cs * b + sn / kx * d, -kx * sn * a + cs * c, -kx * sn * b + cs * d
        g = -(s**2) * eps[k] / (w0[k] ** 2 - s**2)
        c, d = c + g * a, d + g * b
    u = -1j * kx
    r = (u * a + u**2 * b - c - u * d) / (c - u * d - u * a + u**2 * b)
    return a + b * u + r * (a - b * u), r


def test_normal_incidence():
    dt, nt, x_ref = 0.05, 1024, -1.0
    t = np.arange(nt) * dt
    R = Response(np.exp(-(((t - 3.0) / 0.3) ** 2)), dt, x_ref, XS, EPS, W0)
    s, F, K = R.s, R.F, R.K
    x1, xN, xp, xq = XS[0], XS[-1], XS[-1] + 2.0, XS[0] - 3.0
    t_tm, r_tm = transfer(s, s, XS, EPS, W0)
    arriving = F * np.exp(-1j * s * (x1 - x_ref))
    transmitted = F * np.exp(-1j * s * (xp - x_ref)) - 0.5 * (K * np.exp(-1j * s * (xp - XS[:, None]))).sum(0)
    reflected = -0.5 * (K * np.exp(-1j * s * (XS[:, None] - xq))).sum(0)
    scale = np.abs(F).max()
    assert np.abs(transmitted - t_tm * arriving * np.exp(-1j * s * (xp - xN))).max() < 1e-10 * scale
    assert np.abs(reflected - r_tm * arriving * np.exp(-1j * s * (x1 - xq))).max() < 1e-10 * scale


def test_oblique_incidence():
    dt, nt, dy, ny, x_ref = 0.05, 512, 0.5, 64, -1.0
    t, y = np.arange(nt) * dt, (np.arange(ny) - ny // 2) * dy
    f = pulsed_beam(t, y, w_c=3.0, dw=1.0, theta=np.deg2rad(40.0), ds=0.2, t_c=8.0)
    R = Response2D(f, dt, dy, x_ref, XS, EPS, W0, w_max=9.0)
    s, kx, F, K = R.s[:, None], R.kx, R.F, R.K
    x1, xN, xp, xq = XS[0], XS[-1], XS[-1] + 2.0, XS[0] - 3.0
    t_tm, r_tm = transfer(s, kx, XS, EPS, W0)
    arriving = F * np.exp(-1j * kx * (x1 - x_ref))
    radiated = -(s / (2 * kx))
    transmitted = F * np.exp(-1j * kx * (xp - x_ref)) + radiated * (
        K * np.exp(-1j * kx * (xp - XS[:, None, None]))).sum(0)
    reflected = radiated * (K * np.exp(-1j * kx * (XS[:, None, None] - xq))).sum(0)
    scale = np.abs(F).max()
    assert np.abs(transmitted - t_tm * arriving * np.exp(-1j * kx * (xp - xN))).max() < 1e-10 * scale
    assert np.abs(reflected - r_tm * arriving * np.exp(-1j * kx * (x1 - xq))).max() < 1e-10 * scale
