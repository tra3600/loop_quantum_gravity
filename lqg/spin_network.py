"""Réseaux de spin : graphe orienté, spins sur les arêtes, intertwineurs aux nœuds."""
from __future__ import annotations

import math

import networkx as nx
import numpy as np

from .constants import GAMMA_IMMIRZI
from .spectra import area_eigenvalue, mean_node_volume
from .su2 import intertwiner_dim

#: ordre des jambes d'un nœud cubique
_DIRS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


class SpinNetwork:
    """Réseau de spin sur un graphe plongé (positions 3D facultatives).

    ``graph`` : ``networkx.Graph`` ; chaque arête porte ``two_j`` (spin doublé).
    """

    def __init__(self, graph: nx.Graph, gamma: float = GAMMA_IMMIRZI):
        self.g = graph
        self.gamma = gamma
        for _, _, d in self.g.edges(data=True):
            d.setdefault("two_j", 0)

    # --- constructeurs ------------------------------------------------
    @classmethod
    def cubic(cls, n: int, two_j: int = 2, gamma: float = GAMMA_IMMIRZI) -> "SpinNetwork":
        """Réseau cubique n×n×n, tous les spins égaux à ``two_j``/2."""
        g = nx.grid_graph(dim=[n, n, n])
        g = nx.Graph(g)
        for node in g.nodes:
            g.nodes[node]["pos"] = tuple(float(c) for c in node)
        for _, _, d in g.edges(data=True):
            d["two_j"] = two_j
        return cls(g, gamma)

    # --- propriétés élémentaires -----------------------------------------
    def spin(self, e) -> float:
        return self.g.edges[e]["two_j"] / 2

    def node_legs(self, v) -> list[int]:
        """Spins doublés des jambes de v, ordonnés selon ``_DIRS`` si cubique."""
        pos = self.g.nodes[v].get("pos")
        if pos is not None and self.g.degree[v] <= 6 and all(
            "pos" in self.g.nodes[u] for u in self.g[v]
        ):
            legs = [0] * 6
            ok = True
            for u in self.g[v]:
                d = tuple(int(round(a - b)) for a, b in zip(self.g.nodes[u]["pos"], pos))
                if d not in _DIRS:
                    ok = False
                    break
                legs[_DIRS.index(d)] = self.g.edges[v, u]["two_j"]
            if ok:
                return legs
        return [self.g.edges[v, u]["two_j"] for u in self.g[v]]

    def is_admissible(self, v=None) -> bool:
        """Invariance de jauge : l'espace d'intertwineurs de chaque nœud est non vide."""
        nodes = [v] if v is not None else list(self.g.nodes)
        return all(intertwiner_dim([t for t in self.node_legs(n) if t]) > 0 or
                   not any(self.node_legs(n)) for n in nodes)

    # --- observables ---------------------------------------------------
    def total_area(self) -> float:
        """Somme des aires de toutes les plaquettes duales aux arêtes (ℓ_P²)."""
        return sum(area_eigenvalue(d["two_j"] / 2, self.gamma) for *_, d in self.g.edges(data=True))

    def surface_area(self, edges) -> float:
        """Aire d'une surface coupant les arêtes ``edges`` : 8πγ Σ √(j(j+1))."""
        return sum(area_eigenvalue(self.spin(e), self.gamma) for e in edges)

    def node_volume(self, v) -> float:
        legs = self.node_legs(v)
        if not any(legs):
            return 0.0
        if len(legs) == 6:
            return mean_node_volume(tuple(legs), "cube", self.gamma)
        legs = [t for t in legs if t]
        return mean_node_volume(tuple(legs), "auto", self.gamma) if len(legs) >= 3 else 0.0

    def total_volume(self) -> float:
        return sum(self.node_volume(v) for v in self.g.nodes)

    def hilbert_dim(self) -> int:
        """Dimension de l'espace des intertwineurs (produit sur les nœuds)."""
        out = 1
        for v in self.g.nodes:
            legs = [t for t in self.node_legs(v) if t]
            out *= intertwiner_dim(legs) if legs else 1
        return out

    # --- échantillonnage Monte-Carlo --------------------------------------
    def sample(self, steps: int = 2000, beta: float = 0.05, max_two_j: int = 2,
               seed: int | None = None) -> "SpinNetwork":
        """Metropolis sur les spins avec poids exp(-β·A_tot), en préservant l'admissibilité.

        Génère un ensemble de géométries quantiques « typiques » à aire totale
        fluctuante ; β grand favorise les petits spins (géométrie « froide »).
        """
        rng = np.random.default_rng(seed)
        edges = list(self.g.edges)
        for _ in range(steps):
            e = edges[rng.integers(len(edges))]
            old = self.g.edges[e]["two_j"]
            new = old + 2 * int(rng.choice((-1, 1)))  # pas de ±1 en j : conserve la parité
            if new < 0 or new > max_two_j:
                continue
            self.g.edges[e]["two_j"] = new
            dA = area_eigenvalue(new / 2, self.gamma) - area_eigenvalue(old / 2, self.gamma)
            ok = all(self.is_admissible(v) for v in e)
            if not ok or (dA > 0 and rng.random() > math.exp(-beta * dA)):
                self.g.edges[e]["two_j"] = old
        return self
