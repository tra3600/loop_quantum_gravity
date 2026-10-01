"""Mousses de spin : amplitude de Ponzano-Regge (SU(2), gravité 3D) et asymptotique du 6j."""
from __future__ import annotations

import numpy as np

from .su2 import wigner_6j

# arêtes (i,j) des 4 sommets ; le 6j {j1 j2 j3; j4 j5 j6} a pour opposées (1,4),(2,5),(3,6)
_EDGES = [(0, 1), (1, 2), (0, 2), (2, 3), (0, 3), (1, 3)]


def tetra_geometry(lengths):
    """Volume et angles dihédraux extérieurs d'un tétraèdre euclidien d'arêtes ``lengths``
    ordonnées comme (j1..j6). Retourne (V, θ[6]) ou (None, None) si non réalisable."""
    L = dict(zip(_EDGES, lengths))
    d = lambda a, b: L[(min(a, b), max(a, b))]
    G = np.array([[(d(0, a) ** 2 + d(0, b) ** 2 - (0 if a == b else d(a, b) ** 2)) / 2
                   for b in (1, 2, 3)] for a in (1, 2, 3)])
    det = np.linalg.det(G)
    if det <= 0:
        return None, None
    X = np.vstack([np.zeros(3), np.linalg.cholesky(G)])  # sommets
    V = np.sqrt(det) / 6
    cen = X.mean(axis=0)
    normals = {}
    for k in range(4):  # face opposée au sommet k
        a, b, c = [X[i] for i in range(4) if i != k]
        n = np.cross(b - a, c - a)
        n /= np.linalg.norm(n)
        if np.dot(n, a - cen) < 0:
            n = -n  # normale sortante
        normals[k] = n
    thetas = []
    for (a, b) in _EDGES:
        c, e = [i for i in range(4) if i not in (a, b)]
        # faces contenant l'arête (a,b) : opposées à c et e
        # angle dihédral extérieur = angle entre normales sortantes
        thetas.append(np.arccos(np.clip(np.dot(normals[c], normals[e]), -1, 1)))
    return V, np.array(thetas)


def ponzano_regge_asymptotic(two_js) -> float | None:
    """{6j} ≈ cos(Σ (j+½) θ + π/4) / √(12πV) pour de grands spins."""
    V, th = tetra_geometry([t / 2 + 0.5 for t in two_js])
    if V is None:
        return None
    return float(np.cos(np.sum((np.array(two_js) / 2 + 0.5) * th) + np.pi / 4) / np.sqrt(12 * np.pi * V))


def exact_vs_asymptotic(scale_range=range(5, 41, 1), shape=(1, 1, 1, 1, 1, 1)):
    """Compare le 6j exact au Ponzano-Regge pour la famille de spins λ·shape."""
    out = []
    for s in scale_range:
        tj = [int(2 * s * x) for x in shape]
        out.append((s, wigner_6j(*tj), ponzano_regge_asymptotic(tj)))
    return out
