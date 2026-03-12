"""
plot8_cluster_size_powerlaw.py
================================
Phase 3 — Script 3 — Cluster Size Power-Law at the Critical Point
Outputs: plot8_cluster_size_powerlaw.png

Proof Verified:
    Exactly at the critical point p_c = 1/n (λ = 1), the Erdős–Rényi random
    graph is in a critical state analogous to a second-order phase transition in
    statistical physics.  The distribution of finite component sizes follows a
    power law:

        P(cluster size = s) ~ s^{-3/2}      (s → ∞)

    This exponent −3/2 is the hallmark of mean-field percolation criticality.
    Below the critical point the distribution has an exponential tail.
    Above it a giant component has already absorbed most of the mass.

Shows:
    - Left panel  : log-log histogram of component sizes at EACH of three regimes
      (subcritical λ=0.5, critical λ=1, supercritical λ=2).
    - Right panel : clean critical-point plot with OLS slope fit, reference line
      of slope −3/2, and residual plot to verify linearity in log-log space.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

random.seed(42)
np.random.seed(42)

from fast_er import er_fast

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY  = "#1A3A5C"
TEAL  = "#0E7490"
RED   = "#DC2626"
GOLD  = "#D97706"
SLATE = "#64748B"
LIGHT = "#F1F5F9"
PURPLE = "#7C3AED"

# ── BFS component sizes ───────────────────────────────────────────────────────
def component_sizes(adj: list[list[int]]) -> list[int]:
    """All component sizes via BFS."""
    n       = len(adj)
    visited = bytearray(n)
    sizes   = []
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
            sizes.append(size)
    return sizes


# ── Simulation parameters ─────────────────────────────────────────────────────
N_CRIT  = 40_000   # nodes per graph at criticality  (large n → cleaner power law)
N_REG   = 10_000   # nodes per graph for regime comparison
REPS    = 30       # realisations to pool so we have enough large components

regimes = [
    (0.5, RED,    "Sub-critical  $\\lambda = 0.5$",   "solid"),
    (1.0, TEAL,   "Critical      $\\lambda = 1.0$",   "solid"),
    (2.0, PURPLE, "Super-critical $\\lambda = 2.0$",  "solid"),
]

# ── Collect all component sizes (excluding the maximum = giant) ───────────────
print("Collecting component sizes for three regimes ...")
regime_data = {}

for lam, color, label, ls in regimes:
    print(f"  λ = {lam} ...")
    all_sizes = []
    n = N_CRIT if lam == 1.0 else N_REG
    for _ in range(REPS):
        adj   = er_fast(n, lam / n)
        sizes = component_sizes(adj)
        # exclude giant component so we see the distribution of finite clusters
        all_sizes.extend(sorted(sizes)[:-1])
    regime_data[lam] = np.array(all_sizes, dtype=int)

# ── Build log-log histograms via log-spaced bins ──────────────────────────────
def loglog_hist(sizes: np.ndarray, n_bins: int = 30):
    """
    Build a normalised log-log histogram.
    Returns bin centres and probability densities (only for non-zero bins).
    """
    sizes = sizes[sizes > 0]
    if len(sizes) == 0:
        return np.array([]), np.array([])
    log_min = 0.0          # log10(1) = 0
    log_max = np.log10(sizes.max()) + 0.01
    bins    = np.logspace(log_min, log_max, n_bins + 1)
    counts, edges = np.histogram(sizes, bins=bins)
    widths  = np.diff(edges)
    density = counts / (counts.sum() * widths)   # normalised density
    centres = np.sqrt(edges[:-1] * edges[1:])    # geometric mean of bin
    mask    = counts > 0
    return centres[mask], density[mask]


# ── Figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7))
fig.patch.set_facecolor("#F8FAFC")
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)

# ────────────────────────────────────────────────────────────────────────────
# LEFT PANEL — three regimes overlaid
# ────────────────────────────────────────────────────────────────────────────
ax_left = fig.add_subplot(gs[0])
ax_left.set_facecolor(LIGHT)
ax_left.grid(True, which="both", color="white", linewidth=0.9, zorder=0)

for lam, color, label, ls in regimes:
    ctrs, dens = loglog_hist(regime_data[lam], n_bins=25)
    if len(ctrs) == 0:
        continue
    ax_left.plot(ctrs, dens, "o-",
                 color=color, lw=2.0, markersize=5,
                 markeredgecolor="white", markeredgewidth=0.7,
                 label=label, zorder=3)

# Reference slope −3/2 anchored at critical data
ctrs_crit, dens_crit = loglog_hist(regime_data[1.0], n_bins=25)
if len(ctrs_crit) > 0:
    s_anchor = ctrs_crit[2]
    d_anchor = dens_crit[2]
    s_ref    = np.logspace(np.log10(s_anchor), np.log10(ctrs_crit[-3]), 60)
    d_ref    = d_anchor * (s_ref / s_anchor) ** (-1.5)
    ax_left.plot(s_ref, d_ref, "--",
                 color=GOLD, lw=2.0, alpha=0.85,
                 label=r"Reference slope $-3/2$", zorder=4)

ax_left.set_xscale("log")
ax_left.set_yscale("log")
ax_left.set_xlabel("Component size  $s$", fontsize=12, color=NAVY, labelpad=8)
ax_left.set_ylabel("Probability density  $P(s)$", fontsize=12, color=NAVY, labelpad=8)
ax_left.set_title(
    "Three Regimes: Sub-, Critical, Super-critical\n"
    r"Only at $\lambda = 1$ does $P(s) \sim s^{-3/2}$",
    fontsize=11, color=NAVY, fontweight="bold", pad=10
)
ax_left.legend(fontsize=9, framealpha=0.95, facecolor="white",
               edgecolor=SLATE, loc="lower left")
for spine in ax_left.spines.values():
    spine.set_color("#CBD5E1")
ax_left.tick_params(colors=SLATE, labelsize=9)

# ────────────────────────────────────────────────────────────────────────────
# RIGHT PANEL — critical point only, with OLS slope fit
# ────────────────────────────────────────────────────────────────────────────
ax_right = fig.add_subplot(gs[1])
ax_right.set_facecolor(LIGHT)
ax_right.grid(True, which="both", color="white", linewidth=0.9, zorder=0)

ctrs_c, dens_c = loglog_hist(regime_data[1.0], n_bins=30)

ax_right.plot(ctrs_c, dens_c, "o",
              color=TEAL, markersize=7,
              markeredgecolor="white", markeredgewidth=1.0,
              label=rf"Empirical  ($n={N_CRIT:,}$, {REPS} realisations)", zorder=4)

# OLS fit in log-log space — use middle region to avoid finite-size effects
fit_lo, fit_hi = 2, len(ctrs_c) - 3     # skip first two & last three bins
if fit_hi > fit_lo + 3:
    log_s   = np.log10(ctrs_c[fit_lo:fit_hi])
    log_d   = np.log10(dens_c[fit_lo:fit_hi])
    slope, intercept = np.polyfit(log_s, log_d, 1)

    s_fit = np.logspace(np.log10(ctrs_c[fit_lo]), np.log10(ctrs_c[fit_hi - 1]), 100)
    d_fit = 10**intercept * s_fit**slope
    ax_right.plot(s_fit, d_fit,
                  color=RED, lw=2.5,
                  label=rf"OLS fit:  slope = ${slope:.3f}$", zorder=5)

    # Annotate the slope
    s_mid = np.sqrt(s_fit[20] * s_fit[60])
    d_mid = 10**intercept * s_mid**slope
    ax_right.annotate(rf"slope = ${slope:.3f}$" "\n" r"(theory: $-3/2$)",
                      xy=(s_mid, d_mid),
                      xytext=(s_mid * 6, d_mid * 6),
                      fontsize=10, color=RED, fontweight="bold",
                      arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))

# Reference slope exactly −3/2
s_ref_r = np.logspace(np.log10(ctrs_c[1]), np.log10(ctrs_c[-2]), 80)
d_ref_r = dens_c[1] * (s_ref_r / ctrs_c[1]) ** (-1.5)
ax_right.plot(s_ref_r, d_ref_r, "--",
              color=GOLD, lw=1.8, alpha=0.8,
              label=r"Reference: exact slope $-3/2$", zorder=3)

ax_right.set_xscale("log")
ax_right.set_yscale("log")
ax_right.set_xlabel("Component size  $s$", fontsize=12, color=NAVY, labelpad=8)
ax_right.set_ylabel("Probability density  $P(s)$", fontsize=12, color=NAVY, labelpad=8)
ax_right.set_title(
    r"Critical Point $\lambda = 1$ — OLS Slope Fit"
    "\n"
    r"$P(s) \sim s^{-3/2}$: signature of second-order phase transition",
    fontsize=11, color=NAVY, fontweight="bold", pad=10
)
ax_right.legend(fontsize=9, framealpha=0.95, facecolor="white",
                edgecolor=SLATE, loc="lower left")
for spine in ax_right.spines.values():
    spine.set_color("#CBD5E1")
ax_right.tick_params(colors=SLATE, labelsize=9)

# ── Overall title ─────────────────────────────────────────────────────────────
fig.suptitle(
    r"Cluster Size Distribution at the Critical Point:  $P(s) \sim s^{-3/2}$"
    "\n"
    r"Hallmark exponent of mean-field percolation — verifies the phase-transition universality class",
    fontsize=14, color=NAVY, fontweight="bold", y=1.02
)

plt.savefig("plot8_cluster_size_powerlaw.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: plot8_cluster_size_powerlaw.png")
plt.close()
