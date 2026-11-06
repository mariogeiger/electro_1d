"""The slab, pulses and beam of the study, and the exact response of one to the other.

Medium: identical sheets spaced by d (by default the +-1 charge pairs of the notebook). Pulse: the notebook's
truncated Gaussian wave packet. response() and fields() run normal.Response on them. Beam: the oblique pulsed
beam, run by oblique.Response2D.
"""
from dataclasses import dataclass

import numpy as np

from .dispersion import bloch_index, group_index
from .normal import Response
from .oblique import Response2D, pulsed_beam


@dataclass
class Medium:
    """n_sheets identical sheets (eps = q^2/m, natural frequency w0) spaced by d, the first at x0."""
    n_sheets: int = 30
    d: float = 5 / 29
    eps: float = 2 / 0.9  # a +-1 pair of mass 0.9 each, as in Mario's notebook
    w0: float = 5.0
    x0: float = 10.0

    @property
    def xs(self):
        return self.x0 + self.d * np.arange(self.n_sheets)

    @property
    def edges(self):  # each sheet stands for a cell of thickness d
        return self.x0 - self.d / 2, self.x0 + self.d * (self.n_sheets - 0.5)

    @property
    def length(self):
        return self.n_sheets * self.d

    @property
    def wp(self):
        return np.sqrt(self.eps / self.d)

    def n(self, w, model=bloch_index):
        return model(w, self.eps, self.d, self.w0)

    def ng(self, w, model=bloch_index):
        return group_index(lambda ww: model(ww, self.eps, self.d, self.w0), w)


MARIO = Medium()


@dataclass
class Pulse:
    """At t = 0: E_y = B_z = exp(-((x - c)/a)^2) sin(k (x - c)) on |x - c| < cut a, moving to +x."""
    center: float = 5.0
    a: float = 2.0
    k: float = 3.0
    cut: float = 2.5
    amp: float = 1.0

    @property
    def front(self):
        return self.center + self.cut * self.a

    def at0(self, x):
        u = (x - self.center) / self.a
        return np.where(np.abs(u) < self.cut, self.amp * np.exp(-(u**2)) * np.sin(self.k * (x - self.center)), 0.0)


def response(medium, pulse, dt=0.01, nt=8192):
    """Exact linear response; incident f(t) = w(front - t) at x_ref = front, so E_inc(x, t) = w(x - t)."""
    t = np.arange(nt) * dt
    return Response(pulse.at0(pulse.front - t), dt, pulse.front, medium.xs, medium.eps, medium.w0)


def fields(R, x, t):
    """Total, incident and matter E_y (and B_z) on the (t, x) grid, arrays of shape (nt, nx)."""
    it = np.round(np.asarray(t) / R.dt).astype(int)
    Em, Bm = R.matter_field(x, it)
    Ei = R.incident_field(x, it)
    return dict(E=(Ei + Em).T, Ei=Ei.T, Em=Em.T, B=(Ei + Bm).T, Bm=Bm.T)


def centre_path(medium, pulse, w, t_max):
    """Envelope centre: at c in vacuum, at v_g = c/n_g in the slab (group-velocity picture)."""
    a, b = medium.edges
    t_in = a - pulse.center
    t_out = t_in + medium.ng(w) * (b - a)
    return np.array([pulse.center, a, b, b + (t_max - t_out)]), np.array([0, t_in, t_out, t_max])



@dataclass
class Beam:
    """The s-polarized pulsed beam of pulsed_beam(), sampled in time and y on the plane x = x_ref."""
    w_c: float = 3.0
    dw: float = 1.0
    theta_deg: float = 45.0
    ds: float = 0.12
    t_c: float = 25.0
    x_ref: float = -1.0
    dt: float = 0.04
    nt: int = 4096
    dy: float = 0.25
    ny: int = 512
    y0: float = -50.0

    @property
    def theta(self):
        return np.deg2rad(self.theta_deg)

    @property
    def t(self):
        return np.arange(self.nt) * self.dt

    @property
    def y(self):
        return self.y0 + self.dy * np.arange(self.ny)

    def trace(self):
        """f(t, y) on the plane x = x_ref, shape (nt, ny)."""
        return pulsed_beam(self.t, self.y, w_c=self.w_c, dw=self.dw, theta=self.theta, ds=self.ds, t_c=self.t_c)

    def response(self, medium, w_max=9.0):
        """Exact response of the medium; frequencies above w_max, where the beam has no spectrum, are dropped."""
        return Response2D(self.trace(), self.dt, self.dy, self.x_ref, medium.xs, medium.eps, medium.w0, w_max=w_max)
