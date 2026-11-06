"""Refractive and group indices of a periodic stack of identical sheets (eps = q^2/m, natural frequency w0, spacing d).

Units c = eps0 = 1. These are infinite-medium quantities, the yardsticks for what the exact solvers measure.
"""

import numpy as np


def lorentz_index(w, eps, d, w0):
    """Continuum Lorentz index n(w) = sqrt(1 + wp^2/(w0^2 - w^2)), wp^2 = eps/d."""
    return np.sqrt(1 + (eps / d) / (w0**2 - w**2) + 0j)


def bloch_index(w, eps, d, w0):
    """Exact index of the infinite periodic stack: cos(k d) = cos(w d) - sigma sin(w d)."""
    sigma = w * eps / (2 * (w0**2 - w**2))
    c = np.cos(w * d) - sigma * np.sin(w * d)
    return np.arccos(c + 0j) / (w * d)


def group_index(n_of_w, w, h=1e-5):
    """n_g = d(w n)/dw of any index function, by a central difference of step h."""
    return np.real(n_of_w(w) + w * (n_of_w(w + h) - n_of_w(w - h)) / (2 * h))
