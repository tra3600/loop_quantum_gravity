"""lqg — briques de base de la gravité quantique à boucles (spectres, réseaux de spin,
mousses de spin, entropie des trous noirs, cosmologie à rebond)."""
from .constants import GAMMA_IMMIRZI, RHO_CRIT  # noqa: F401
from .spin_network import SpinNetwork  # noqa: F401

__all__ = ["GAMMA_IMMIRZI", "RHO_CRIT", "SpinNetwork"]
