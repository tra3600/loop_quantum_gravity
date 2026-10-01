"""Figures matplotlib (backend Agg si aucun écran n'est disponible)."""
from __future__ import annotations

import os

import matplotlib

if not os.environ.get("DISPLAY") and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from . import black_hole, cosmology, spectra, spin_foam
from .constants import GAMMA_IMMIRZI, RHO_CRIT


def plot_spectra(path=None):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    tj = np.arange(1, 13)
    ax[0].plot(tj / 2, [spectra.area_eigenvalue(t / 2) for t in tj], "o-")
    ax[0].axhline(spectra.area_gap(), ls=":", c="gray", label="gap d'aire (j=1/2)")
    ax[0].set(xlabel="spin j", ylabel=r"$A_j\ [\ell_P^2]$", title="Spectre d'aire")
    ax[0].legend()
    for t in (1, 2, 3, 4):
        v = spectra.volume_spectrum((t,) * 4)
        ax[1].plot([t / 2] * len(v), v, "_", ms=18, mew=2)
    ax[1].set(xlabel="spin j (nœud 4-valent monochromatique)", ylabel=r"$V\ [\ell_P^3]$",
              title="Spectre de volume (tétraèdre quantique)")
    fig.tight_layout()
    return _finish(fig, path)


def plot_spin_network(net, path=None):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    g = net.g
    js = np.array([d["two_j"] / 2 for *_, d in g.edges(data=True)])
    cmap, vmax = plt.cm.viridis, max(js.max(), 1)
    for (u, v, d) in g.edges(data=True):
        if d["two_j"] == 0:
            continue
        p, q = np.array(g.nodes[u]["pos"]), np.array(g.nodes[v]["pos"])
        ax.plot(*zip(p, q), color=cmap(d["two_j"] / 2 / vmax), lw=0.8 + 1.6 * d["two_j"] / 2)
    pos = np.array([g.nodes[v]["pos"] for v in g.nodes])
    vol = np.array([net.node_volume(v) for v in g.nodes])
    sc = ax.scatter(*pos.T, c=vol, s=20 + 160 * vol / max(vol.max(), 1e-9), cmap="plasma")
    fig.colorbar(sc, ax=ax, shrink=0.6, label=r"volume du nœud $[\ell_P^3]$")
    ax.set_title(f"Réseau de spin : A = {net.total_area():.0f} ℓ²,  V = {net.total_volume():.0f} ℓ³")
    return _finish(fig, path)


def plot_black_hole(path=None, area_max=120.0):
    g = black_hole.solve_gamma()
    A, N = black_hole.count_microstates(area_max, g, delta=0.05)
    S = np.log(N[1:])
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(A[1:], S, label=f"comptage LQG (γ={g:.4f})")
    ax.plot(A[1:], A[1:] / 4, "--", label="Bekenstein-Hawking A/4")
    ax.set(xlabel=r"$A\ [\ell_P^2]$", ylabel="S = ln N", title="Entropie d'un trou noir")
    ax.legend()
    return _finish(fig, path)


def plot_bounce(path=None):
    tr = cosmology.bounce(0.0, t_max=3.0, n=1500)
    ac, rc = cosmology.classical_massless(tr.t)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].plot(tr.t, tr.a, label="LQC"); ax[0].plot(tr.t, ac, "--", label="RG classique")
    ax[0].set(xlabel="t [t_P]", ylabel="a(t)", title="Facteur d'échelle", ylim=(0, 2.5)); ax[0].legend()
    ax[1].plot(tr.t, tr.rho / RHO_CRIT, label="LQC")
    ax[1].plot(tr.t, np.minimum(rc / RHO_CRIT, 10), "--", label="RG classique")
    ax[1].set(xlabel="t [t_P]", ylabel=r"$\rho/\rho_c$", title="Densité d'énergie (borne ρ_c)", ylim=(0, 3))
    ax[1].legend()
    ax[2].plot(tr.t, tr.H); ax[2].set(xlabel="t [t_P]", ylabel="H", title="Taux de Hubble (H=0 au rebond)")
    fig.tight_layout()
    return _finish(fig, path)


def plot_ponzano_regge(path=None):
    data = spin_foam.exact_vs_asymptotic(range(4, 41), (1, 1, 1, 1.1, 1.2, 1.3))
    s, ex, asy = zip(*[(a, b, c) for a, b, c in data if c is not None])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, ex, "o", label="6j exact"); ax.plot(s, asy, "-", label="Ponzano-Regge")
    ax.set(xlabel="échelle λ des spins", title="Amplitude du tétraèdre quantique (λ → ∞)")
    ax.legend()
    return _finish(fig, path)


def _finish(fig, path):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return path
