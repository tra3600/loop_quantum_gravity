"""Spectres géométriques de la LQG : aire, volume, espaces d'intertwineurs."""
from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from .constants import GAMMA_IMMIRZI
from .su2 import intertwiner_dim, spin_matrices


# ----------------------------------------------------------------- aire ----
def area_eigenvalue(j: float, gamma: float = GAMMA_IMMIRZI) -> float:
    """A_j = 8π γ ℓ_P² √(j(j+1))."""
    return 8.0 * np.pi * gamma * np.sqrt(j * (j + 1.0))


def area_gap(gamma: float = GAMMA_IMMIRZI) -> float:
    """Plus petite aire non nulle (j = 1/2)."""
    return area_eigenvalue(0.5, gamma)


# ------------------------------------------------------------- volume ----
def _embed(ops, k, dims):
    """Plonge l'opérateur ``ops`` (sur la jambe k) dans le produit tensoriel."""
    mats = [sp.identity(d, format="csr", dtype=complex) for d in dims]
    mats[k] = ops
    out = mats[0]
    for m in mats[1:]:
        out = sp.kron(out, m, format="csr")
    return out


class _Node:
    """Espace invariant d'un nœud de spins ``two_js`` + opérateurs de flux."""

    def __init__(self, two_js):
        self.two_js = tuple(two_js)
        dims = [t + 1 for t in self.two_js]
        self.dims = dims
        self.J = [spin_matrices(t) for t in self.two_js]
        full = [[_embed(self.J[i][a], i, dims) for a in range(3)] for i in range(len(dims))]
        self.full = full
        # sous-espace M_tot = 0 (conservé par tout scalaire)
        grids = np.indices(dims).reshape(len(dims), -1)
        two_m = sum(t - 2 * g for t, g in zip(self.two_js, grids))
        idx = np.flatnonzero(two_m == 0)
        self.idx = idx
        tot = [sum(full[i][a] for i in range(len(dims))) for a in range(3)]
        j2 = sum((t @ t) for t in tot).tocsr()[idx][:, idx].toarray()
        w, v = np.linalg.eigh((j2 + j2.conj().T) / 2)
        self.basis = v[:, w < 1e-8]  # base orthonormée des intertwineurs
        assert self.basis.shape[1] == intertwiner_dim(self.two_js)

    @property
    def dim(self) -> int:
        return self.basis.shape[1]

    def q_triple(self, i, j, k) -> np.ndarray:
        """Q_ijk = J_i · (J_j × J_k) restreint aux intertwineurs."""
        F = self.full
        x, y, z = 0, 1, 2
        q = (
            F[i][x] @ (F[j][y] @ F[k][z] - F[j][z] @ F[k][y])
            + F[i][y] @ (F[j][z] @ F[k][x] - F[j][x] @ F[k][z])
            + F[i][z] @ (F[j][x] @ F[k][y] - F[j][y] @ F[k][x])
        )
        qr = q.tocsr()[self.idx][:, self.idx]
        m = self.basis.conj().T @ (qr @ self.basis)
        return (m + m.conj().T) / 2


@lru_cache(maxsize=512)
def _node(two_js) -> _Node:
    return _Node(two_js)


def _sqrt_abs(m: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(m)
    return (v * np.sqrt(np.abs(w))) @ v.conj().T


#: Triplets orthogonaux d'un nœud cubique (jambes +x,-x,+y,-y,+z,-z).
_CUBE_TRIPLES = [(a, b, c) for a in (0, 1) for b in (2, 3) for c in (4, 5)]


def volume_operator(two_js, kind: str = "auto", gamma: float = GAMMA_IMMIRZI) -> np.ndarray:
    """Matrice de l'opérateur volume d'un nœud, en unités ℓ_P³.

    * ``tetra`` (4 jambes) : V = (√2/3)(8πγ)^{3/2} √|J1·(J2×J3)|  (Bianchi-Haggard).
    * ``cube`` (6 jambes ordonnées +x,-x,+y,-y,+z,-z) : moyenne sur les 8 triplets
      orthogonaux, normalisée pour reproduire L³ pour un cube classique.
    * ``generic`` : approximation moyenne des √|Q_ijk| sur tous les triplets
      (normalisation du tétraèdre) ; valable qualitativement.
    """
    node = _node(tuple(two_js))
    n = len(two_js)
    if kind == "auto":
        kind = {4: "tetra", 6: "cube"}.get(n, "generic")
    pref = (8.0 * np.pi * gamma) ** 1.5
    if node.dim == 0:
        return np.zeros((0, 0))
    if kind == "tetra":
        if n != 4:
            raise ValueError("kind='tetra' exige 4 jambes")
        return (np.sqrt(2) / 3) * pref * _sqrt_abs(node.q_triple(0, 1, 2))
    if kind == "cube":
        if n != 6:
            raise ValueError("kind='cube' exige 6 jambes")
        acc = sum(_sqrt_abs(node.q_triple(*t)) for t in _CUBE_TRIPLES)
        return pref * acc / len(_CUBE_TRIPLES)
    triples = list(itertools.combinations(range(n), 3))
    if not triples:
        return np.zeros((node.dim, node.dim))
    acc = sum(_sqrt_abs(node.q_triple(*t)) for t in triples)
    return (np.sqrt(2) / 3) * pref * acc / len(triples)


def volume_spectrum(two_js, kind: str = "auto", gamma: float = GAMMA_IMMIRZI) -> np.ndarray:
    """Valeurs propres (triées) du volume du nœud, en ℓ_P³."""
    m = volume_operator(two_js, kind, gamma)
    return np.sort(np.linalg.eigvalsh(m)) if m.size else np.array([])


def mean_node_volume(two_js, kind: str = "auto", gamma: float = GAMMA_IMMIRZI) -> float:
    """Volume moyen dans l'état d'intertwineur maximalement mélangé (tr V / dim)."""
    m = volume_operator(two_js, kind, gamma)
    return float(np.trace(m).real / m.shape[0]) if m.size else 0.0
