"""Interface en ligne de commande : ``python loop_quantum_gravity.py [commande]``."""
from __future__ import annotations

import argparse
import os

from . import black_hole, cosmology, spectra, visualize
from .constants import GAMMA_IMMIRZI, RHO_CRIT
from .spin_network import SpinNetwork


def cmd_spectra(out):
    print(f"γ = {GAMMA_IMMIRZI:.5f}   gap d'aire = {spectra.area_gap():.3f} ℓ_P²")
    for t in (1, 2, 3):
        print(f"  tétraèdre j={t/2}: V ∈", spectra.volume_spectrum((t,) * 4).round(3))
    print("  ->", visualize.plot_spectra(os.path.join(out, "spectres.png")))


def cmd_network(out, n=4, steps=4000, seed=1):
    net = SpinNetwork.cubic(n, two_j=2).sample(steps=steps, beta=0.01, max_two_j=4, seed=seed)
    print(f"réseau cubique {n}³ : A = {net.total_area():.1f} ℓ², V = {net.total_volume():.1f} ℓ³, "
          f"admissible = {net.is_admissible()}, dim = {net.hilbert_dim()}")
    print("  ->", visualize.plot_spin_network(net, os.path.join(out, "reseau_de_spin.png")))


def cmd_blackhole(out):
    print(f"γ tel que S=A/4 (comptage discernable) : {black_hole.solve_gamma():.5f}")
    print("  ->", visualize.plot_black_hole(os.path.join(out, "trou_noir.png")))


def cmd_cosmology(out):
    tr = cosmology.bounce(0.0, t_max=3.0, n=1500)
    print(f"ρ_c = {RHO_CRIT:.4f} ρ_P ; ρ_max = {tr.rho.max():.4f} ; a_min = {tr.a.min():.3f}")
    print("  ->", visualize.plot_bounce(os.path.join(out, "rebond.png")))


def cmd_foam(out):
    print("  ->", visualize.plot_ponzano_regge(os.path.join(out, "ponzano_regge.png")))


COMMANDS = {"spectra": cmd_spectra, "network": cmd_network, "blackhole": cmd_blackhole,
            "cosmology": cmd_cosmology, "foam": cmd_foam}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", nargs="?", default="all", choices=["all", *COMMANDS])
    p.add_argument("-o", "--output", default="output", help="dossier des figures")
    a = p.parse_args(argv)
    for name, fn in COMMANDS.items():
        if a.command in ("all", name):
            print(f"[{name}]")
            fn(a.output)
