"""Cosmologie quantique à boucles effective : rebond à la place du Big Bang.

Équations effectives (FLRW plat, unités de Planck ħ = c = G = 1) :

    H² = (8π/3) ρ (1 - ρ/ρ_c),   Ḣ = -4π(ρ + P)(1 - 2ρ/ρ_c),   φ̈ + 3Hφ̇ + V'(φ) = 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .constants import RHO_CRIT


@dataclass
class Trajectory:
    t: np.ndarray
    a: np.ndarray        # facteur d'échelle normalisé a(0)=1 au rebond
    H: np.ndarray
    rho: np.ndarray
    phi: np.ndarray


def _rhs(potential, dpotential, rho_c):
    def f(_, y):
        lna, H, phi, dphi = y
        rho = 0.5 * dphi**2 + potential(phi)
        dH = -4 * np.pi * dphi**2 * (1 - 2 * rho / rho_c)
        return [H, dH, dphi, -3 * H * dphi - dpotential(phi)]
    return f


def bounce(mass: float = 0.0, phi_b: float = 0.0, t_max: float = 5.0, n: int = 4000,
           rho_c: float = RHO_CRIT, sign: int = +1) -> Trajectory:
    """Intègre la solution passant par le rebond (H=0, ρ=ρ_c) à t=0, de -t_max à +t_max.

    ``mass`` : masse du champ scalaire (0 → champ sans masse, solution analytique).
    ``sign`` : sens de φ̇ au rebond.
    """
    V = lambda p: 0.5 * mass**2 * p**2
    dV = lambda p: mass**2 * p
    dphi_b = sign * np.sqrt(2.0 * (rho_c - V(phi_b)))
    y0 = [0.0, 0.0, phi_b, dphi_b]
    f = _rhs(V, dV, rho_c)
    t_eval = np.linspace(0, t_max, n)
    kw = dict(rtol=1e-11, atol=1e-13, method="DOP853")
    fwd = solve_ivp(f, (0, t_max), y0, t_eval=t_eval, **kw)
    # renversement du temps : (H, φ̇) -> (-H, -φ̇) ; la dynamique est réversible
    bwd = solve_ivp(lambda t, y: [-v for v in f(t, [y[0], y[1], y[2], y[3]])],
                    (0, t_max), y0, t_eval=t_eval, **kw)
    t = np.concatenate([-bwd.t[::-1][:-1], fwd.t])
    Y = np.concatenate([bwd.y[:, ::-1][:, :-1], fwd.y], axis=1)
    lna, H, phi, dphi = Y
    rho = 0.5 * dphi**2 + V(phi)
    return Trajectory(t, np.exp(lna), H, rho, phi)


def massless_analytic(t: np.ndarray, rho_c: float = RHO_CRIT):
    """Solution exacte pour le champ sans masse : (a, ρ) avec a(0)=1."""
    x = 1 + 24 * np.pi * rho_c * t**2
    return x ** (1 / 6), rho_c / x


def classical_massless(t: np.ndarray, rho_c: float = RHO_CRIT):
    """Branche classique (RG) qui coïncide avec la solution quantique à grand |t| :
    a ∝ |t|^{1/3}, ρ = 1/(24π t²) — singulière en t = 0."""
    t = np.where(t == 0, np.nan, t)
    return (24 * np.pi * rho_c * t**2) ** (1 / 6), 1 / (24 * np.pi * t**2)


def efolds(traj: Trajectory) -> float:
    return float(np.log(traj.a[-1] / traj.a[traj.a.argmin()]))
