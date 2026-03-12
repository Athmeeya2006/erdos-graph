"""
plot3_degree_distribution.py
=============================
Outputs: plot3_degree_distribution.png

Shows:
    - Both algorithms produce IDENTICAL degree distributions
    - Both match the theoretical Poisson(lambda) curve
    - Side-by-side histograms with Poisson overlay
    - Runs for three different lambda values
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
from naive_er import er_naive
from fast_er  import er_fast

random.seed(42)
np.random.seed(42)

NAVY  = "#1A3A5C"
TEAL  = "#0E7490"
RED   = "#DC2626"
GOLD  = "#D97706"
SLATE = "#64748B"
LIGHT = "#F1F5F9"

# Three lambda values to demonstrate — critical point is lambda = 1
# Subcritical : lambda < 1  (p < 1/n)
# Critical    : lambda = 1  (p = 1/n)
# Supercritical: lambda > 1  (p > 1/n)
configs = [
    (1000, 0.0005, "Subcritical  λ = 0.4995  < 1"),
    (1000, 0.001,  "Critical     λ = 0.999   ≈ 1"),
    (1000, 0.005,  "Supercritical λ = 4.995  > 1"),
]
SAMPLES = 300   # graph samples per cell

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor("#F8FAFC")
fig.suptitle("Both Algorithms Produce Identical Poisson(λ) Degree Distributions\n"
             f"({SAMPLES} samples × n=1000 nodes each)",
             fontsize=14, color=NAVY, fontweight="bold", y=1.01)

for col, (n, p, label) in enumerate(configs):
    lam  = (n - 1) * p
    # For subcritical lambda < 1, lam*3 would give fewer than 3 points.
    # Use at least 8 k values so the Poisson curve is always visible.
    k_max   = max(int(lam * 4) + 1, 8)
    k_range   = np.arange(0, k_max)
    poi_pmf   = poisson.pmf(k_range, lam)

    print(f"Collecting {SAMPLES} samples for {label}...")
    naive_degs, fast_degs = [], []

    for _ in range(SAMPLES):
        naive_degs.extend(len(nb) for nb in er_naive(n, p))
        fast_degs.extend(len(nb)  for nb in er_fast(n,  p))

    for row, (degrees, algo_name, color) in enumerate([
        (naive_degs, "Naïve  O(n²)",             RED),
        (fast_degs,  "Batagelj-Brandes  O(n+M)", TEAL),
    ]):
        ax = axes[row][col]
        ax.set_facecolor(LIGHT)
        ax.grid(axis="y", color="white", linewidth=0.9, zorder=0)

        max_deg = max(degrees)
        bins    = np.arange(-0.5, max_deg + 1.5, 1)

        ax.hist(degrees, bins=bins, density=True,
                color=color, alpha=0.60, edgecolor="white", linewidth=0.4,
                label="Empirical", zorder=2)

        ax.plot(k_range, poi_pmf, "o-",
                color=NAVY, lw=2, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8,
                label=f"Poisson(λ={lam:.1f})", zorder=4)

        # Mean line
        mean_deg = np.mean(degrees)
        ax.axvline(mean_deg, color=GOLD, lw=1.8, linestyle="--", alpha=0.85, zorder=3)
        ax.text(mean_deg + 0.3, ax.get_ylim()[1] * 0.88,
                f"mean={mean_deg:.2f}\n(λ={lam:.2f})",
                fontsize=8, color=GOLD, va="top", fontweight="bold")

        ax.set_xlabel("Degree  k", fontsize=10, color=NAVY)
        ax.set_xlim(-0.5, max(lam * 3.0, 8))

        if col == 0:
            ax.set_ylabel(f"{algo_name}\n\nP(degree = k)",
                          fontsize=9, color=NAVY, labelpad=8)
        else:
            ax.set_ylabel("P(degree = k)", fontsize=10, color=NAVY)

        if row == 0:
            ax.set_title(label, fontsize=11, color=NAVY,
                         fontweight="bold", pad=8)

        ax.legend(fontsize=8, framealpha=0.9, facecolor="white", edgecolor=SLATE)

        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
        ax.tick_params(colors=SLATE, labelsize=9)

plt.tight_layout()
plt.savefig("plot3_degree_distribution.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: plot3_degree_distribution.png")
plt.close()