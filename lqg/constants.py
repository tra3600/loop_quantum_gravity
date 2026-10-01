"""Constantes et conventions (unités de Planck : ħ = c = G = 1)."""
import math

#: Paramètre de Barbero-Immirzi fixé par l'entropie de Bekenstein-Hawking
#: (valeur de Meissner ; voir ``black_hole.solve_gamma`` pour le comptage discernable).
GAMMA_IMMIRZI = 0.23753295796592

#: Densité critique de la cosmologie quantique à boucles, en unités de Planck.
RHO_CRIT = math.sqrt(3.0) / (32.0 * math.pi**2 * GAMMA_IMMIRZI**3)  # ≈ 0.41

L_PLANCK = 1.0
