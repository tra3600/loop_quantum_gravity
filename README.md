# loop_quantum_gravity

Boîte à outils Python pour **simuler et visualiser des briques réelles de la gravité quantique à boucles (LQG)**.
Remplace l'ancienne grille 3D aléatoire (purement illustrative) par des calculs vérifiés contre des résultats connus.

## Installation et utilisation

```bash
pip install -r requirements.txt
python loop_quantum_gravity.py            # tout exécuter, figures dans ./output
python loop_quantum_gravity.py cosmology  # spectra | network | blackhole | cosmology | foam
python -m pytest                          # tests
```

## Contenu

| Module | Ce qu'il calcule | Vérification |
|---|---|---|
| `lqg/su2.py` | matrices de spin, multiplicités de Clebsch-Gordan, dimension des intertwineurs, symboles 6j exacts (Racah, arithmétique rationnelle) | {1 1 1;1 1 1} = 1/6, dimensions d'intertwineurs |
| `lqg/spectra.py` | spectre d'aire `A_j = 8πγ ℓ_P² √(j(j+1))`, opérateur volume d'un nœud (tétraèdre quantique, nœud cubique) par diagonalisation exacte de `J₁·(J₂×J₃)` sur l'espace invariant | `|Q| = √3/4` pour 4 spins ½ |
| `lqg/spin_network.py` | réseaux de spin (graphe + spins + intertwineurs), admissibilité (invariance de jauge), aire, volume, dimension de l'espace de Hilbert, échantillonnage Monte-Carlo préservant la jauge | tests d'admissibilité |
| `lqg/spin_foam.py` | amplitude du tétraèdre (6j) et asymptotique de Ponzano-Regge (géométrie de Regge) | accord exact/asymptotique dès j ≈ 20 |
| `lqg/black_hole.py` | entropie d'un trou noir : comptage dynamique des états de l'horizon (`Σm = 0`), γ tel que `S = A/4` | pente 0,251 ≈ 1/4 |
| `lqg/cosmology.py` | cosmologie quantique à boucles effective : rebond à `ρ_c ≈ 0,41 ρ_P` au lieu du Big Bang, champ massif ou non | égal à la solution analytique à 1e-11 |

## Résultats

- Gap d'aire : `A_{1/2} = 5,17 ℓ_P²` (γ = 0,2375).
- Le comptage à ponctures discernables donne γ ≈ 0,2741 (Domagała-Lewandowski) ; la valeur 0,2375 par défaut est celle de Meissner, reprise de la littérature et non recalculée ici.
- Cosmologie : `ρ ≤ ρ_c`, `H = 0` au rebond, retour à la relativité générale pour `ρ ≪ ρ_c`.

## Limites (honnêteté scientifique)

- L'opérateur volume des nœuds 6-valents et d'ordre supérieur est une approximation normalisée sur le cube classique ; seul le cas 4-valent suit la formule standard de Bianchi-Haggard.
- Pas de mousse de spin 4D complète (EPRL) : seule l'amplitude du tétraèdre SU(2) de Ponzano-Regge est implémentée.
- La cosmologie est la dynamique **effective** (champ moyen), pas la dynamique quantique complète.
