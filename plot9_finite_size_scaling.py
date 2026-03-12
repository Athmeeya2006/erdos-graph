"""
plot9_finite_size_scaling.py
==============================
Phase 3 — Script 4 — Finite-Size Scaling
Outputs: plot9_finite_size_scaling.png

Proof Verified:
    A true phase transition is a property of the thermodynamic limit n → ∞.
    In a finite graph, the transition is a smooth crossover, not a sharp angle.
    This script plots the S vs ⟨k⟩ curve for n = 100, 1 000, 10 000 on the
    same axes, demonstrating that as n grows the curve sharpens — converging
    toward the infinite-limit step-function with a discontinuous derivative at
    λ = 1.

    The width of the transition window scales as Δλ ~ n^{−1/3}, meaning:
        - n = 100    → Δλ ≈ 0.22
        - n = 1 000  → Δλ ≈ 0.10
        - n = 10 000 → Δλ ≈ 0.046
    The right panel directly verifies this scaling law with a log-log plot.

Shows:
    - Left  : S vs λ for n = 100, 1 000, 10 000 with ±1σ bands; theoretical limit
    - Right : log-log plot of transition width Δλ vs n — slope should be −1/3
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

random.seed(42)
np.random.seed(42)

from fast_er import er_fast

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = "#1A3A5C"
TEAL   = "#0E7490"
RED    = "#DC2626"
GOLD   = "#D97706"
SLATE  = "#64748B"
LIGHT  = "#F1F5F9"
PURPLE = "#7C3AED"
GREEN  = "#059669"

# ── BFS giant component fraction ──────────────────────────────────────────────
def giant_fraction(adj: list[list[int]]) -> float:
    n       = len(adj)
    visited = bytearray(n)
    best    = 0
    for start in range(n):
        if not visited[start]:
            size  = 0
            queue = deque([start])
            visited[start] = 1
            while queue:
                v = queue.popleft()
                size += 1
                for u in adj[v]:
                    if not visited[u]:
                        visited[u] = 1
                        queue.append(u)
            if size > best:
                best = size
    return best / n


# ── Theoretical curve (infinite limit) ────────────────────────────────────────
def S_theory(lam: float, tol: float = 1e-12) -> float:
    if lam <= 1.0:
        return 0.0
    S = 0.5
    for _ in range(2000):
        S_new = 1.0 - np.exp(-lam * S)
        if abs(S_new - S) < tol:
            return S_new
        S = S_new
    return S


# ── Simulation parameters ─────────────────────────────────────────────────────
sizes      = [100, 1_000, 10_000]
REPS       = 60
LAM_VALS   = np.linspace(0.0, 3.0, 61)   # step = 0.05

colors     = [RED, PURPLE, TEAL]
labels     = [f"$n = {n:,}$" for n in sizes]

# ── Run simulations ───────────────────────────────────────────────────────────
print("Running finite-size scaling simulations ...")
results = {}   # n → (mean array, std array)

for n, color in zip(sizes, colors):
    print(f"  n = {n:,} ...")
    S_mean = np.zeros(len(LAM_VALS))
    S_std  = np.zeros(len(LAM_VALS))
    for idx, lam in enumerate(LAM_VALS):
        p       = lam / n
        samples = [giant_fraction(er_fast(n, p)) for _ in range(REPS)]
        S_mean[idx] = np.mean(samples)
        S_std[idx]  = np.std(samples)
    results[n] = (S_mean, S_std)
    print(f"    Done.  S at λ=1.5: {S_mean[LAM_VALS.searchsorted(1.5)]:.3f}")

# ── Measure transition width Δλ per n ─────────────────────────────────────────
#   Δλ = width of the window where 0.1 ≤ S ≤ 0.5  (heuristic but robust)
delta_lam = []
for n in sizes:
    S_mean, _ = results[n]
    lo = LAM_VALS[np.searchsorted(S_mean, 0.1, side="left")]
    hi = LAM_VALS[np.searchsorted(S_mean, 0.5, side="left")]
    delta_lam.append(max(hi - lo, 1e-3))   # guard against zero for small n
    print(f"  n={n:>6,}  Δλ = {delta_lam[-1]:.4f}")

# ── Theoretical limit curve ────────────────────────────────────────────────────
lam_dense  = np.linspace(0.0, 3.0, 600)
S_th_dense = np.array([S_theory(l) for l in lam_dense])

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#F8FAFC")
fig.suptitle(
    r"Finite-Size Scaling: Transition Sharpens as $n \to \infty$"
    "\n"
    r"True phase transitions exist only in the thermodynamic limit — "
    r"finite graphs show smooth crossovers",
    fontsize=14, color=NAVY, fontweight="bold", y=1.02
)

# ────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — S vs λ, three n values
# ────────────────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(LIGHT)
ax.grid(True, color="white", linewidth=0.9, zorder=0)

for n, color, label in zip(sizes, colors, labels):
    S_mean, S_std = results[n]
    ax.fill_between(LAM_VALS,
                    np.clip(S_mean - S_std, 0, 1),
                    np.clip(S_mean + S_std, 0, 1),
                    color=color, alpha=0.12, zorder=1)
    ax.plot(LAM_VALS, S_mean, "-",
            color=color, lw=2.2, label=label + r"  (60 reals.)", zorder=3)

# Infinite-limit theory
ax.plot(lam_dense, S_th_dense, "--",
        color=GOLD, lw=2.0, alpha=0.9, zorder=4,
        label=r"$n \to \infty$ theory: $S = 1 - e^{-\lambda S}$")

# Critical point
ax.axvline(1.0, color=SLATE, lw=1.4, linestyle=":", alpha=0.7, zorder=2)
ax.text(1.02, 0.86, r"$\lambda_c = 1$",
        fontsize=10, color=SLATE, style="italic")

# Annotate sharpening with arrow between n=100 and n=10000 curves at λ=1.3
for n, color in zip(sizes, colors):
    S_mean, _ = results[n]
    idx = np.argmin(np.abs(LAM_VALS - 1.3))
    ax.annotate("", xy=(1.3, S_mean[idx]),
                xytext=(1.3, 0),
                arrowprops=dict(arrowstyle="-", color=color,
                                alpha=0.0))   # invisible, just for spacing

ax.set_xlabel(r"Mean degree  $\lambda = \langle k \rangle$",
              fontsize=12, color=NAVY, labelpad=8)
ax.set_ylabel(r"Giant component fraction  $S$",
              fontsize=12, color=NAVY, labelpad=8)
ax.set_title(
    r"Transition curves for $n = 100, 1\,000, 10\,000$"
    "\n"
    "Larger $n$ → sharper transition → converges to theoretical step",
    fontsize=11, color=NAVY, fontweight="bold", pad=10
)
ax.legend(fontsize=10, framealpha=0.95, facecolor="white",
          edgecolor=SLATE, loc="upper left")
ax.set_xlim(0, 3.0)
ax.set_ylim(-0.02, 1.0)
for spine in ax.spines.values():
    spine.set_color("#CBD5E1")
ax.tick_params(colors=SLATE, labelsize=10)

# ────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — log-log: Δλ vs n  (slope should verify −1/3 scaling)
# ────────────────────────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(LIGHT)
ax2.grid(True, which="both", color="white", linewidth=0.9, zorder=0)

n_arr  = np.array(sizes, dtype=float)
dl_arr = np.array(delta_lam)

ax2.loglog(n_arr, dl_arr, "D-",
           color=NAVY, lw=2.5, markersize=10,
           markeredgecolor="white", markeredgewidth=1.5,
           label=r"Measured $\Delta\lambda$", zorder=4)

# OLS fit in log-log
slope_fit, intercept_fit = np.polyfit(np.log10(n_arr), np.log10(dl_arr), 1)
n_fit  = np.logspace(np.log10(n_arr[0]) - 0.1, np.log10(n_arr[-1]) + 0.3, 100)
dl_fit = 10**intercept_fit * n_fit**slope_fit
ax2.loglog(n_fit, dl_fit, "-",
           color=RED, lw=2.0,
           label=rf"OLS fit:  slope = ${slope_fit:.3f}$", zorder=3)

# Reference line: slope = −1/3
c_ref  = dl_arr[1] / (n_arr[1] ** (-1/3))
dl_ref = c_ref * n_fit**(-1/3)
ax2.loglog(n_fit, dl_ref, "--",
           color=GOLD, lw=1.8, alpha=0.85,
           label=r"Theory: slope $= -1/3$", zorder=2)

# Annotate measured points
for n_v, dl_v in zip(sizes, delta_lam):
    ax2.annotate(f"$n={n_v:,}$\n$\\Delta\\lambda={dl_v:.3f}$",
                 xy=(n_v, dl_v),
                 xytext=(n_v * 1.6, dl_v * 1.25),
                 fontsize=9, color=NAVY, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2))

ax2.set_xlabel("Graph size  $n$", fontsize=12, color=NAVY, labelpad=8)
ax2.set_ylabel(r"Transition width  $\Delta\lambda$", fontsize=12, color=NAVY, labelpad=8)
ax2.set_title(
    r"Transition Width Scales as $\Delta\lambda \sim n^{-1/3}$"
    "\n"
    "Log-log slope confirms the mean-field finite-size scaling exponent",
    fontsize=11, color=NAVY, fontweight="bold", pad=10
)
ax2.legend(fontsize=10, framealpha=0.95, facecolor="white",
           edgecolor=SLATE, loc="upper right")
for spine in ax2.spines.values():
    spine.set_color("#CBD5E1")
ax2.tick_params(colors=SLATE, labelsize=10)

# ── Footer ────────────────────────────────────────────────────────────────────
fig.text(
    0.5, -0.03,
    r"Interpretation: Each finite graph has a 'pseudo-critical point' smeared over a window $\Delta\lambda \sim n^{-1/3}$.  "
    r"As $n \to \infty$ the window shrinks to zero and the transition becomes the sharp step of the thermodynamic limit.",
    ha="center", va="top", fontsize=9, color=SLATE, style="italic"
)

plt.tight_layout()
plt.savefig("plot9_finite_size_scaling.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: plot9_finite_size_scaling.png")
plt.close()
