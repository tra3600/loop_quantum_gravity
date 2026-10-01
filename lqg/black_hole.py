"""Entropie des trous noirs en LQG : comptage des états de ponctuation de l'horizon."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .constants import GAMMA_IMMIRZI


def _two_js(jmax: float):
    return np.arange(1, int(2 * jmax) + 1)  # 2j = 1, 2, ...


def meissner_sum(gamma: float, jmax: float = 400) -> float:
    """Σ_j (2j+1) exp(-2πγ√(j(j+1))) — vaut 1 pour S = A/4."""
    j = _two_js(jmax) / 2
    return float(np.sum((2 * j + 1) * np.exp(-2 * np.pi * gamma * np.sqrt(j * (j + 1)))))


def solve_gamma(jmax: float = 400) -> float:
    """γ pour lequel la pente asymptotique de S(A) vaut 1/4 (ponctures discernables).

    Donne γ ≈ 0.2741 (Domagała-Lewandowski). Le comptage de Meissner, qui traite
    autrement les états de projection, donne γ ≈ 0.2375 (valeur par défaut de
    ``constants.GAMMA_IMMIRZI``, reprise de la littérature et non recalculée ici).
    """
    return brentq(lambda g: meissner_sum(g, jmax) - 1.0, 0.05, 1.0, xtol=1e-14)


def count_microstates(area_max: float, gamma: float = GAMMA_IMMIRZI, delta: float = 0.02):
    """Nombre d'états N(A) de l'horizon d'aire ≤ A (ℓ_P²), contrainte Σ m = 0.

    Programmation dynamique sur des cases d'aire de largeur ``delta`` : on compte
    les suites ordonnées de ponctures (j, m) d'aire totale ≤ A et de projection
    totale nulle. Retourne (aires, N cumulé).
    """
    nb = int(area_max / delta) + 1
    a_j = lambda t: 8 * np.pi * gamma * np.sqrt(t / 2 * (t / 2 + 1))
    tjs = [t for t in range(1, 4000) if a_j(t) <= area_max]
    tmax = tjs[-1]
    m2max = int(0.45 * area_max / (8 * np.pi * gamma) * 4) + tmax + 4  # borne sur |Σ 2m|
    pad = tmax + 4
    width = 2 * m2max + 1 + 2 * pad
    zero = m2max + pad
    N = np.zeros((nb, width))
    N[0, zero] = 1.0
    widths = {t: int(round(a_j(t) / delta)) for t in tjs}
    for b in range(1, nb):
        acc = np.zeros(width)
        for t in tjs:
            w = widths[t]
            if w > b:
                break
            row = N[b - w]
            c2 = row.copy()  # somme cumulée de pas 2 (même parité de 2M)
            for s in (0, 1):
                c2[s::2] = np.cumsum(row[s::2])
            # res[M] = Σ_{k=-t,-t+2..t} row[M-k] = c2[M+t] - c2[M-t-2]
            hi = np.zeros(width); lo = np.zeros(width)
            hi[: width - t] = c2[t:]
            lo[t + 2 :] = c2[: width - t - 2]
            acc += hi - lo
        N[b] = acc
    cumulative = np.cumsum(N[:, zero])
    return np.arange(nb) * delta, cumulative
