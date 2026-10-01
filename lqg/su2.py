"""Outils SU(2) : matrices de spin, décomposition de Clebsch-Gordan, symboles 6j.

Convention : les spins sont manipulés sous forme *doublée* (``two_j`` entier)
pour éviter toute erreur d'arrondi sur les demi-entiers.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np
import scipy.sparse as sp


def spin_matrices(two_j: int):
    """Retourne (Jx, Jy, Jz) creux de dimension 2j+1, base m = j, j-1, ..., -j."""
    d = two_j + 1
    j = two_j / 2.0
    m = j - np.arange(d)
    jz = sp.diags(m).astype(complex)
    # J+ |m> = sqrt(j(j+1) - m(m+1)) |m+1>  ; |m+1> est l'indice précédent
    coef = np.sqrt(np.maximum(j * (j + 1) - m[1:] * (m[1:] + 1), 0.0))
    jp = sp.diags(coef, 1).astype(complex)
    jm = jp.getH().tocsr()
    jx = (jp + jm) / 2
    jy = (jp - jm) / (2j)
    return jx.tocsr(), jy.tocsr(), jz.tocsr()


def triangle_ok(a2: int, b2: int, c2: int) -> bool:
    """Inégalité triangulaire + parité (spins doublés)."""
    return abs(a2 - b2) <= c2 <= a2 + b2 and (a2 + b2 + c2) % 2 == 0


def decompose(two_js) -> dict[int, int]:
    """Multiplicités {2J: n} de ⊗_i V_{j_i} (produit tensoriel itéré)."""
    mult = {0: 1}
    for t in two_js:
        new: dict[int, int] = {}
        for J2, n in mult.items():
            for K2 in range(abs(J2 - t), J2 + t + 1, 2):
                new[K2] = new.get(K2, 0) + n
        mult = new
    return mult


def intertwiner_dim(two_js) -> int:
    """Dimension de l'espace des intertwineurs invariants (multiplicité de J=0)."""
    return decompose(two_js).get(0, 0)


def _fact(n2: int) -> int:
    return math.factorial(n2)


@lru_cache(maxsize=None)
def _delta2(a2: int, b2: int, c2: int) -> Fraction:
    """Δ(abc)² = (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!."""
    return Fraction(
        _fact((a2 + b2 - c2) // 2) * _fact((a2 - b2 + c2) // 2) * _fact((-a2 + b2 + c2) // 2),
        _fact((a2 + b2 + c2) // 2 + 1),
    )


@lru_cache(maxsize=None)
def wigner_6j(j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> float:
    """Symbole 6j {j1 j2 j3; j4 j5 j6} (arguments doublés), formule de Racah exacte."""
    triads = [(j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3)]
    if not all(triangle_ok(*t) for t in triads):
        return 0.0
    a = [sum(t) // 2 for t in triads]
    b = [(j1 + j2 + j4 + j5) // 2, (j2 + j3 + j5 + j6) // 2, (j3 + j1 + j6 + j4) // 2]
    total = Fraction(0)
    for t in range(max(a), min(b) + 1):
        den = 1
        for x in a:
            den *= math.factorial(t - x)
        for y in b:
            den *= math.factorial(y - t)
        total += Fraction((-1) ** t * math.factorial(t + 1), den)
    prefactor = 1.0
    for tr in triads:
        prefactor *= float(_delta2(*tr))
    return float(total) * math.sqrt(prefactor)
