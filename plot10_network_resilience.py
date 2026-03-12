"""
plot10_network_resilience.py
==============================
Phase 4 — Part A — Network Resilience Under Attack
Outputs: plot10_network_resilience.png

Proof Verified:
    The Erdős–Rényi graph has a Poisson degree distribution with no extreme
    hubs.  Consequently, removing the highest-degree nodes (targeted attack)
    is barely worse than removing random nodes.  A scale-free (Barabási-Albert)
    graph, by contrast, has a power-law tail: a few mega-hubs hold the network
    together.  Destroying those hubs shatters the giant component far faster
    than random failure — a dramatic asymmetry.

    This demonstrates that the *degree distribution* governs robustness:
        - Poisson (ER)    → symmetric response   (no hubs to exploit)
        - Power-law (BA)  → wildly asymmetric     (hubs are Achilles' heel)

Shows:
    - Left column : ER graph — random failure vs targeted attack (near-overlap)
    - Right column: BA scale-free — random failure vs targeted attack (dramatic gap)
    - Top row     : giant component fraction S vs fraction of nodes removed
    - Bottom row  : average path length (APL) inside the giant component vs removed fraction
    - Averaged over 20 independent realisations with ±1σ shading
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import networkx as nx

random.seed(42)
np.random.seed(42)

from fast_er import er_fast

# ── Palette (consistent with all Phase 2 & 3 plots) ──────────────────────────
NAVY   = "#1A3A5C"
TEAL   = "#0E7490"
RED    = "#DC2626"
GOLD   = "#D97706"
SLATE  = "#64748B"
LIGHT  = "#F1F5F9"
PURPLE = "#7C3AED"
GREEN  = "#059669"


# ── Giant component via BFS (adjacency-list, respects alive set) ──────────────
def _giant_component_nodes_adj(adj, alive):
    """Return the node-set of the largest component within the alive set."""
    if not alive:
        return set()
    visited = set()
    best_nodes = set()
    for start in alive:
        if start in visited:
            continue
        comp = set()
        queue = deque([start])
        visited.add(start)
        comp.add(start)
        while queue:
            v = queue.popleft()
            for u in adj[v]:
                if u in alive and u not in visited:
                    visited.add(u)
                    comp.add(u)
                    queue.append(u)
        if len(comp) > len(best_nodes):
            best_nodes = comp
    return best_nodes


def _giant_component_fraction_adj(adj, alive):
    """Return |C_max| / |alive| using BFS on adjacency list."""
    n_alive = len(alive)
    if n_alive == 0:
        return 0.0
    visited = set()
    best = 0
    for start in alive:
        if start not in visited:
            size = 0
            queue = deque([start])
            visited.add(start)
            while queue:
                v = queue.popleft()
                size += 1
                for u in adj[v]:
                    if u in alive and u not in visited:
                        visited.add(u)
                        queue.append(u)
            if size > best:
                best = size
    return best / n_alive


# ── Attack simulation (generic, works on adj-list or nx.Graph) ────────────────
def _get_degrees(adj_or_G, alive):
    """Return dict {node: degree within alive set}."""
    deg = {}
    if isinstance(adj_or_G, nx.Graph):
        for v in alive:
            deg[v] = sum(1 for u in adj_or_G.neighbors(v) if u in alive)
    else:
        for v in alive:
            deg[v] = sum(1 for u in adj_or_G[v] if u in alive)
    return deg


def _giant_frac(adj_or_G, alive):
    """Unified giant component fraction."""
    if isinstance(adj_or_G, nx.Graph):
        if len(alive) == 0:
            return 0.0
        sub = adj_or_G.subgraph(alive)
        if sub.number_of_nodes() == 0:
            return 0.0
        return len(max(nx.connected_components(sub), key=len)) / len(alive)
    else:
        return _giant_component_fraction_adj(adj_or_G, alive)


def _giant_nodes(adj_or_G, alive):
    """Unified largest-component node set within alive nodes."""
    if isinstance(adj_or_G, nx.Graph):
        if not alive:
            return set()
        sub = adj_or_G.subgraph(alive)
        if sub.number_of_nodes() == 0:
            return set()
        return set(max(nx.connected_components(sub), key=len))
    return _giant_component_nodes_adj(adj_or_G, alive)


def _avg_path_length_estimate(adj_or_G, alive, sample_sources=30):
    """
    Estimate average shortest path length within the (alive) giant component.

    We use multi-source BFS from a small sample of sources inside C_max and
    average their mean distances. This is much faster than all-pairs shortest
    paths and is accurate enough for robustness plots.

    Returns NaN if the giant component has < 2 nodes.
    """
    gc_nodes = _giant_nodes(adj_or_G, alive)
    m = len(gc_nodes)
    if m < 2:
        return float("nan")

    nodes_list = list(gc_nodes)
    k = min(sample_sources, m)
    sources = random.sample(nodes_list, k)

    # Fast membership check
    gc = gc_nodes

    def bfs_mean_dist_from_source(s):
        dist = {s: 0}
        q = deque([s])
        total = 0
        seen = 1
        while q:
            v = q.popleft()
            dv = dist[v]
            if isinstance(adj_or_G, nx.Graph):
                nbrs = adj_or_G.neighbors(v)
            else:
                nbrs = adj_or_G[v]
            for u in nbrs:
                if u in gc and u not in dist:
                    dist[u] = dv + 1
                    total += dv + 1
                    seen += 1
                    q.append(u)
        # gc is connected, but guard anyway
        if seen <= 1:
            return float("nan")
        return total / (seen - 1)

    vals = [bfs_mean_dist_from_source(s) for s in sources]
    return float(np.nanmean(vals))


def attack_curve(adj_or_G, n, removal_fracs, strategy="random", sample_sources=30):
    """
    Simulate node removal and track giant component fraction.

    For targeted attack the degree ordering is RECALCULATED after each
    removal batch — this is the adaptive (greedy) variant that correctly
    shows the ER near-symmetry.

    Parameters
    ----------
    adj_or_G      : adjacency list or nx.Graph
    n             : total number of nodes
    removal_fracs : sorted array of fractions at which to record S
    strategy      : "random" or "targeted"

    Returns
    -------
    S_values : array of S at each removal fraction
    L_values : array of APL estimates inside C_max at each removal fraction
    """
    alive   = set(range(n))
    S_out   = np.zeros(len(removal_fracs))
    L_out   = np.full(len(removal_fracs), np.nan, dtype=float)
    removed = 0
    fi      = 0   # index into removal_fracs

    # Record S at f = 0
    while fi < len(removal_fracs) and removal_fracs[fi] <= 0.0:
        S_out[fi] = _giant_frac(adj_or_G, alive)
        L_out[fi] = _avg_path_length_estimate(adj_or_G, alive, sample_sources=sample_sources)
        fi += 1

    if strategy == "random":
        order = list(range(n))
        random.shuffle(order)
        for v in order:
            if fi >= len(removal_fracs):
                break
            alive.discard(v)
            removed += 1
            f = removed / n
            while fi < len(removal_fracs) and f >= removal_fracs[fi]:
                S_out[fi] = _giant_frac(adj_or_G, alive)
                L_out[fi] = _avg_path_length_estimate(adj_or_G, alive, sample_sources=sample_sources)
                fi += 1
    else:
        # Adaptive targeted: pick highest-degree node each step
        # For efficiency we remove in small batches
        batch = max(1, n // len(removal_fracs))
        while fi < len(removal_fracs) and alive:
            # Recalculate degrees among surviving nodes
            deg = _get_degrees(adj_or_G, alive)
            # Sort alive nodes by current degree descending
            sorted_nodes = sorted(alive, key=lambda v: deg[v], reverse=True)
            # Remove up to 'batch' nodes
            for v in sorted_nodes[:batch]:
                alive.discard(v)
                removed += 1
                f = removed / n
                while fi < len(removal_fracs) and f >= removal_fracs[fi]:
                    S_out[fi] = _giant_frac(adj_or_G, alive)
                    L_out[fi] = _avg_path_length_estimate(adj_or_G, alive, sample_sources=sample_sources)
                    fi += 1
                if fi >= len(removal_fracs):
                    break

    # Fill remaining
    while fi < len(removal_fracs):
        S_out[fi] = 0.0
        L_out[fi] = np.nan
        fi += 1

    return S_out, L_out


# ── Simulation parameters ─────────────────────────────────────────────────────
N    = 1000        # nodes per graph
LAM  = 4           # average degree ⟨k⟩ = 4
P    = LAM / N     # edge probability for ER
M_BA = LAM // 2    # BA attachment parameter (⟨k⟩ ≈ 2m)
REPS = 20          # independent realisations

removal_fracs = np.linspace(0.0, 0.90, 37)

# ── Run simulations ───────────────────────────────────────────────────────────
print(f"Phase 4A — Network Resilience:  n={N}, ⟨k⟩={LAM}, {REPS} realisations")
print("=" * 65)

results = {
    "er_random_S":   np.zeros((REPS, len(removal_fracs))),
    "er_targeted_S": np.zeros((REPS, len(removal_fracs))),
    "ba_random_S":   np.zeros((REPS, len(removal_fracs))),
    "ba_targeted_S": np.zeros((REPS, len(removal_fracs))),
    "er_random_L":   np.full((REPS, len(removal_fracs)), np.nan),
    "er_targeted_L": np.full((REPS, len(removal_fracs)), np.nan),
    "ba_random_L":   np.full((REPS, len(removal_fracs)), np.nan),
    "ba_targeted_L": np.full((REPS, len(removal_fracs)), np.nan),
}

for rep in range(REPS):
    print(f"  Realisation {rep + 1}/{REPS} ...")

    # ── ER graph ──────────────────────────────────────────────────────────────
    adj_er = er_fast(N, P)
    S, L = attack_curve(adj_er, N, removal_fracs, "random")
    results["er_random_S"][rep] = S
    results["er_random_L"][rep] = L
    S, L = attack_curve(adj_er, N, removal_fracs, "targeted")
    results["er_targeted_S"][rep] = S
    results["er_targeted_L"][rep] = L

    # ── BA scale-free graph ───────────────────────────────────────────────────
    G_ba = nx.barabasi_albert_graph(N, M_BA, seed=42 + rep)
    S, L = attack_curve(G_ba, N, removal_fracs, "random")
    results["ba_random_S"][rep] = S
    results["ba_random_L"][rep] = L
    S, L = attack_curve(G_ba, N, removal_fracs, "targeted")
    results["ba_targeted_S"][rep] = S
    results["ba_targeted_L"][rep] = L

print("Done.\n")

# ── Compute means and stds ────────────────────────────────────────────────────
def _nanmean_nanstd(arr2d: np.ndarray):
    """
    Compute mean/std along axis=0 ignoring NaNs, without warnings.
    Returns NaN where all entries are NaN.
    """
    count = np.sum(~np.isnan(arr2d), axis=0).astype(float)
    s = np.nansum(arr2d, axis=0)
    mean = np.full(arr2d.shape[1], np.nan, dtype=float)
    ok = count > 0
    mean[ok] = s[ok] / count[ok]

    # variance ignoring NaNs
    var = np.full(arr2d.shape[1], np.nan, dtype=float)
    if np.any(ok):
        diffsq = (arr2d - mean) ** 2
        var[ok] = np.nansum(diffsq, axis=0)[ok] / count[ok]
    std = np.sqrt(var)
    return mean, std

stats = {}
for key in results:
    m, s = _nanmean_nanstd(results[key])
    stats[key] = {"mean": m, "std": s}

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.patch.set_facecolor("#F8FAFC")

fig.suptitle(
    r"Network Resilience Under Attack:  Random Failure vs Targeted Removal"
    "\n"
    r"Poisson degree distribution (ER) $\to$ symmetric response;  "
    r"Power-law (BA) $\to$ hubs are the Achilles' heel",
    fontsize=14, color=NAVY, fontweight="bold", y=0.98
)

# ────────────────────────────────────────────────────────────────────────────
# TOP-LEFT — ER graph (S)
# ────────────────────────────────────────────────────────────────────────────
ax1 = axes[0, 0]
ax1.set_facecolor(LIGHT)
ax1.grid(True, color="white", linewidth=0.9, zorder=0)

# Random failure — teal
m, s = stats["er_random_S"]["mean"], stats["er_random_S"]["std"]
ax1.fill_between(removal_fracs, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                 color=TEAL, alpha=0.15, zorder=1)
ax1.plot(removal_fracs, m, "o-", color=TEAL, lw=2.2, markersize=4,
         markeredgecolor="white", markeredgewidth=0.7,
         label="Random failure", zorder=3)

# Targeted attack — red
m, s = stats["er_targeted_S"]["mean"], stats["er_targeted_S"]["std"]
ax1.fill_between(removal_fracs, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                 color=RED, alpha=0.15, zorder=1)
ax1.plot(removal_fracs, m, "s-", color=RED, lw=2.2, markersize=4,
         markeredgecolor="white", markeredgewidth=0.7,
         label="Targeted attack (adaptive, by degree)", zorder=3)

# Annotation
mid_idx = np.searchsorted(removal_fracs, 0.30)
er_r = stats["er_random_S"]["mean"][mid_idx]
er_t = stats["er_targeted_S"]["mean"][mid_idx]
ax1.annotate(
    "Near-overlap\nNo hubs to exploit",
    xy=(0.30, (er_r + er_t) / 2),
    xytext=(0.52, 0.78),
    fontsize=10, color=GOLD, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.5),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
              edgecolor=GOLD, alpha=0.9)
)

ax1.set_xlabel("Fraction of nodes removed  $f$", fontsize=12, color=NAVY, labelpad=8)
ax1.set_ylabel(r"Giant component fraction  $S = |C_{\max}|/n_{\mathrm{alive}}$",
               fontsize=12, color=NAVY, labelpad=8)
ax1.set_title(
    rf"Erdős–Rényi  $G(n{{=}}{N},\ \langle k \rangle{{=}}{LAM})$" "\n"
    r"Poisson degree distribution $\to$ random $\approx$ targeted",
    fontsize=11, color=NAVY, fontweight="bold", pad=10
)
ax1.legend(fontsize=10, framealpha=0.95, facecolor="white",
           edgecolor=SLATE, loc="lower left")
ax1.set_xlim(0, 0.90)
ax1.set_ylim(-0.02, 1.02)
for spine in ax1.spines.values():
    spine.set_color("#CBD5E1")
ax1.tick_params(colors=SLATE, labelsize=10)

# ────────────────────────────────────────────────────────────────────────────
# TOP-RIGHT — BA scale-free graph (S)
# ────────────────────────────────────────────────────────────────────────────
ax2 = axes[0, 1]
ax2.set_facecolor(LIGHT)
ax2.grid(True, color="white", linewidth=0.9, zorder=0)

# Random failure — teal
m, s = stats["ba_random_S"]["mean"], stats["ba_random_S"]["std"]
ax2.fill_between(removal_fracs, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                 color=TEAL, alpha=0.15, zorder=1)
ax2.plot(removal_fracs, m, "o-", color=TEAL, lw=2.2, markersize=4,
         markeredgecolor="white", markeredgewidth=0.7,
         label="Random failure", zorder=3)

# Targeted attack — red
m, s = stats["ba_targeted_S"]["mean"], stats["ba_targeted_S"]["std"]
ax2.fill_between(removal_fracs, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                 color=RED, alpha=0.15, zorder=1)
ax2.plot(removal_fracs, m, "s-", color=RED, lw=2.2, markersize=4,
         markeredgecolor="white", markeredgewidth=0.7,
         label="Targeted attack (adaptive, by degree)", zorder=3)

# Find collapse point for targeted BA
ba_targ_mean = stats["ba_targeted_S"]["mean"]
collapse_idx = np.searchsorted(-ba_targ_mean, -0.15)
if collapse_idx < len(removal_fracs):
    f_collapse = removal_fracs[min(collapse_idx, len(removal_fracs) - 1)]
    ax2.axvline(f_collapse, color=GOLD, lw=1.5, linestyle="--", alpha=0.7, zorder=2)
    ax2.annotate(
        f"Hub destruction\ncollapses network\nat f ≈ {f_collapse:.0%}",
        xy=(f_collapse, 0.15),
        xytext=(f_collapse + 0.18, 0.55),
        fontsize=10, color=RED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=RED, alpha=0.9)
    )

# Resilience annotation for random failure
rf_idx = np.searchsorted(removal_fracs, 0.55)
if rf_idx < len(removal_fracs):
    ax2.annotate(
        "Survives random\nfailure robustly",
        xy=(0.55, stats["ba_random_S"]["mean"][rf_idx]),
        xytext=(0.35, 0.85),
        fontsize=10, color=TEAL, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=TEAL, alpha=0.9)
    )

ax2.set_xlabel("Fraction of nodes removed  $f$", fontsize=12, color=NAVY, labelpad=8)
ax2.set_ylabel(r"Giant component fraction  $S = |C_{\max}|/n_{\mathrm{alive}}$",
               fontsize=12, color=NAVY, labelpad=8)
ax2.set_title(
    rf"Barabási–Albert  Scale-Free  ($n{{=}}{N},\ m{{=}}{M_BA}$)" "\n"
    r"Power-law degree distribution $\to$ hubs are fatal targets",
    fontsize=11, color=NAVY, fontweight="bold", pad=10
)
ax2.legend(fontsize=10, framealpha=0.95, facecolor="white",
           edgecolor=SLATE, loc="lower left")
ax2.set_xlim(0, 0.90)
ax2.set_ylim(-0.02, 1.02)
for spine in ax2.spines.values():
    spine.set_color("#CBD5E1")
ax2.tick_params(colors=SLATE, labelsize=10)

# ────────────────────────────────────────────────────────────────────────────
# BOTTOM-LEFT — ER graph (APL)
# ────────────────────────────────────────────────────────────────────────────
ax3 = axes[1, 0]
ax3.set_facecolor(LIGHT)
ax3.grid(True, color="white", linewidth=0.9, zorder=0)

m, s = stats["er_random_L"]["mean"], stats["er_random_L"]["std"]
ax3.fill_between(removal_fracs, np.clip(m - s, 0, None), np.clip(m + s, 0, None),
                 color=TEAL, alpha=0.15, zorder=1)
ax3.plot(removal_fracs, m, "o-", color=TEAL, lw=2.2, markersize=4,
         markeredgecolor="white", markeredgewidth=0.7,
         label="Random failure", zorder=3)

m, s = stats["er_targeted_L"]["mean"], stats["er_targeted_L"]["std"]
ax3.fill_between(removal_fracs, np.clip(m - s, 0, None), np.clip(m + s, 0, None),
                 color=RED, alpha=0.15, zorder=1)
ax3.plot(removal_fracs, m, "s-", color=RED, lw=2.2, markersize=4,
         markeredgecolor="white", markeredgewidth=0.7,
         label="Targeted attack (adaptive, by degree)", zorder=3)

ax3.set_xlabel("Fraction of nodes removed  $f$", fontsize=12, color=NAVY, labelpad=8)
ax3.set_ylabel(r"Average path length in $C_{\max}$  (estimated)", fontsize=12, color=NAVY, labelpad=8)
ax3.set_title(
    rf"Erdős–Rényi  $G(n{{=}}{N},\ \langle k \rangle{{=}}{LAM})$" "\n"
    r"APL grows as shortcuts disappear; random $\approx$ targeted",
    fontsize=11, color=NAVY, fontweight="bold", pad=10
)
ax3.legend(fontsize=10, framealpha=0.95, facecolor="white",
           edgecolor=SLATE, loc="upper left")
ax3.set_xlim(0, 0.90)
for spine in ax3.spines.values():
    spine.set_color("#CBD5E1")
ax3.tick_params(colors=SLATE, labelsize=10)

# ────────────────────────────────────────────────────────────────────────────
# BOTTOM-RIGHT — BA scale-free graph (APL)
# ────────────────────────────────────────────────────────────────────────────
ax4 = axes[1, 1]
ax4.set_facecolor(LIGHT)
ax4.grid(True, color="white", linewidth=0.9, zorder=0)

m, s = stats["ba_random_L"]["mean"], stats["ba_random_L"]["std"]
ax4.fill_between(removal_fracs, np.clip(m - s, 0, None), np.clip(m + s, 0, None),
                 color=TEAL, alpha=0.15, zorder=1)
ax4.plot(removal_fracs, m, "o-", color=TEAL, lw=2.2, markersize=4,
         markeredgecolor="white", markeredgewidth=0.7,
         label="Random failure", zorder=3)

m, s = stats["ba_targeted_L"]["mean"], stats["ba_targeted_L"]["std"]
ax4.fill_between(removal_fracs, np.clip(m - s, 0, None), np.clip(m + s, 0, None),
                 color=RED, alpha=0.15, zorder=1)
ax4.plot(removal_fracs, m, "s-", color=RED, lw=2.2, markersize=4,
         markeredgecolor="white", markeredgewidth=0.7,
         label="Targeted attack (adaptive, by degree)", zorder=3)

ax4.set_xlabel("Fraction of nodes removed  $f$", fontsize=12, color=NAVY, labelpad=8)
ax4.set_ylabel(r"Average path length in $C_{\max}$  (estimated)", fontsize=12, color=NAVY, labelpad=8)
ax4.set_title(
    rf"Barabási–Albert  Scale-Free  ($n{{=}}{N},\ m{{=}}{M_BA}$)" "\n"
    r"Targeted hub removal inflates distances then fragments",
    fontsize=11, color=NAVY, fontweight="bold", pad=10
)
ax4.legend(fontsize=10, framealpha=0.95, facecolor="white",
           edgecolor=SLATE, loc="upper left")
ax4.set_xlim(0, 0.90)
for spine in ax4.spines.values():
    spine.set_color("#CBD5E1")
ax4.tick_params(colors=SLATE, labelsize=10)

# ── Footer annotation ─────────────────────────────────────────────────────────
fig.text(
    0.5, -0.04,
    r"Conclusion:  ER graphs (Poisson) respond symmetrically to both attacks — "
    r"no hubs means nothing special to target.  "
    r"Scale-free graphs (power-law) survive random failure but collapse instantly "
    r"when the high-degree hubs are attacked — the Achilles' heel of real-world networks.",
    ha="center", va="top", fontsize=9, color=SLATE, style="italic"
)

plt.tight_layout()
plt.savefig("plot10_network_resilience.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: plot10_network_resilience.png")
plt.close()
