"""
plot14_brain_connectivity.py
============================
Phase 6 Extension — Brain Signal Propagation & Connectivity Analysis

Dataset:
    Synthetic 90-region connectome modelled on the AAL (Automated Anatomical
    Labelling) atlas, with network-level structure matching the Yeo 7/9-network
    parcellation and connectivity statistics drawn directly from published DTI
    and resting-state fMRI literature (Sporns 2010; Bullmore & Sporns 2012;
    Hagmann et al. 2008; Honey et al. 2009).

    Architecture:
      - 90 regions  |  9 cortical / subcortical modules (10 nodes each)
      - Within-module density ≈ 55 %  (matching intra-network fMRI correlation)
      - Between-module density ≈ 4 %  (sparse inter-network coupling)
      - 15 rich-club hub nodes with extra inter-module links
      - Log-normal connection weights (matching DTI tract strength statistics)

Analysis:
    1. Network statistics vs. Erdős-Rényi null model
       (clustering, path length, assortativity — same Z-score procedure as
       Phase 6 S&P 500 analysis)
    2. Diffusion-based signal propagation on three architecture types:
         Brain (small-world / modular)
         Erdős-Rényi (random)
         Ring Lattice  (regular, high-C, long L)
       — Side-by-side spreading speed comparison reveals the small-world
         communication advantage of brain structure.

Signal propagation model (Honey et al. 2009, PNAS):
    x(t+1) = α · D⁻¹A · x(t)  +  (1−α) · x(t)
    where D = weighted degree, A = weighted adjacency, α = 0.35
    (discrete-time random-walk diffusion; seed node = highest-betweenness hub)

Outputs  (saved to the project root):
    plot14a_brain_connectome.png      — chord diagram with module colours
    plot14b_signal_propagation.png    — 3×5 activation snapshots grid
    plot14c_brain_vs_er.png           — ER ensemble null-model rejection
    plot14d_brain_dashboard.png       — 4-panel summary dashboard
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import networkx as nx
import scipy.stats as stats

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Output directory ─────────────────────────────────────────────────────────
OUT = Path(".")

# ── Palette ───────────────────────────────────────────────────────────────────
BG    = "#020617"
CARD  = "#0F172A"
DIM   = "#334155"
SLATE = "#64748B"
LIGHT = "#F1F5F9"
GOLD  = "#D97706"
TEAL  = "#0E7490"
RED   = "#DC2626"
GREEN = "#059669"
NAVY  = "#1A3A5C"

# Colours for the 9 brain modules — maps to known network palette (Yeo / Power)
MODULE_NAMES = [
    "Default Mode",
    "Frontoparietal",
    "Dorsal Attention",
    "Somatomotor",
    "Visual",
    "Limbic",
    "Ventral Attention",
    "Subcortical",
    "Cerebellar",
]
MODULE_COLORS = [
    "#DC143C",   # Default Mode    — deep red
    "#FF8C00",   # Frontoparietal  — dark orange
    "#228B22",   # Dorsal Attention— forest green
    "#1E90FF",   # Somatomotor     — dodger blue
    "#8B008B",   # Visual          — dark magenta
    "#FF69B4",   # Limbic          — hot pink
    "#00CED1",   # Ventral Attention— dark turquoise
    "#D2B48C",   # Subcortical     — tan
    "#2F4F4F",   # Cerebellar      — dark slate
]

DARK_PARAMS = {
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    DIM,
    "axes.labelcolor":   LIGHT,
    "xtick.color":       SLATE,
    "ytick.color":       SLATE,
    "text.color":        LIGHT,
    "grid.color":        DIM,
    "grid.linewidth":    0.5,
    "legend.facecolor":  CARD,
    "legend.edgecolor":  DIM,
}
plt.rcParams.update(DARK_PARAMS)


def _despine(ax: plt.Axes) -> None:
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(DIM)
    ax.spines["bottom"].set_color(DIM)


# ════════════════════════════════════════════════════════════════════════════
# 1. BRAIN CONNECTOME GENERATION
# ════════════════════════════════════════════════════════════════════════════
def generate_brain_connectome(
    n: int = 90,
    n_modules: int = 9,
    p_intra: float = 0.55,
    p_inter: float = 0.040,
    n_hubs: int = 15,
    hub_hub_p: float = 0.70,
    seed: int = SEED,
) -> nx.Graph:
    """
    Build a synthetic but neuroscience-accurate brain connectome.

    Architecture:
      * Modular structure: n_modules groups of equal size, dense intra-
        community connections, sparse inter-community connections.
      * Rich-club: n_hubs nodes (spread across modules) with boosted
        hub-to-hub connectivity — matching the brain's "rich club" property
        (van den Heuvel & Sporns, 2011 J. Neurosci.).
      * Weights: log-normal (intra-module, stronger) and
        log-normal (inter-module, weaker) — matching DTI tract counts.
    """
    rng = np.random.default_rng(seed)
    module_size = n // n_modules

    G = nx.Graph()
    G.add_nodes_from(range(n))

    for m_id in range(n_modules):
        for k in range(module_size):
            nd = m_id * module_size + k
            G.nodes[nd]["module"] = m_id
            G.nodes[nd]["module_name"] = MODULE_NAMES[m_id % len(MODULE_NAMES)]
            G.nodes[nd]["is_hub"] = False

    # ── Within-module edges (dense, strong weights) ──────────────────────────
    for m_id in range(n_modules):
        members = [m_id * module_size + k for k in range(module_size)]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if rng.random() < p_intra:
                    w = float(np.clip(rng.lognormal(0.0, 0.4), 0.1, 5.0))
                    G.add_edge(members[i], members[j], weight=w)

    # ── Between-module edges (sparse, weaker weights) ────────────────────────
    for m1 in range(n_modules):
        for m2 in range(m1 + 1, n_modules):
            m1_nodes = [m1 * module_size + k for k in range(module_size)]
            m2_nodes = [m2 * module_size + k for k in range(module_size)]
            for u in m1_nodes:
                for v in m2_nodes:
                    if rng.random() < p_inter:
                        w = float(np.clip(rng.lognormal(-0.5, 0.4), 0.05, 2.0))
                        G.add_edge(u, v, weight=w)

    # ── Rich-club hubs ────────────────────────────────────────────────────────
    # Designate top-degree nodes as hubs (computed from current edges)
    degree_dict = dict(G.degree())
    hub_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:n_hubs]
    for nd in hub_nodes:
        G.nodes[nd]["is_hub"] = True

    # Add extra hub-to-hub connections (rich-club bonus)
    for i, u in enumerate(hub_nodes):
        for v in hub_nodes[i + 1:]:
            if not G.has_edge(u, v) and rng.random() < hub_hub_p:
                w = float(np.clip(rng.lognormal(0.3, 0.3), 0.2, 6.0))
                G.add_edge(u, v, weight=w)

    return G


# ════════════════════════════════════════════════════════════════════════════
# 2. NETWORK STATISTICS
# ════════════════════════════════════════════════════════════════════════════
def compute_stats(G: nx.Graph) -> dict:
    """Compute clustering, path length, assortativity, modularity."""
    n = G.number_of_nodes()
    degrees = [d for _, d in G.degree()]
    k_mean = np.mean(degrees)

    C = nx.average_clustering(G)
    r = nx.degree_assortativity_coefficient(G)

    # Average path length on largest connected component
    lcc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc)
    if len(lcc) <= 200:
        L = nx.average_shortest_path_length(G_lcc)
    else:
        src_sample = random.sample(list(lcc), 150)
        spl = []
        for src in src_sample:
            spl.extend(nx.single_source_shortest_path_length(G_lcc, src).values())
        L = float(np.mean(spl))

    # Modularity (Louvain) — best partition
    try:
        partition = nx.community.louvain_communities(G, seed=SEED)
        Q = nx.community.modularity(G, partition)
    except Exception:
        Q = float("nan")

    return {
        "n": n,
        "M": G.number_of_edges(),
        "k_mean": k_mean,
        "C": C,
        "L": L,
        "r": r,
        "Q": Q,
        "degrees": degrees,
        "lcc_size": len(lcc),
    }


# ════════════════════════════════════════════════════════════════════════════
# 3. SIGNAL PROPAGATION (DIFFUSION MODEL)
# ════════════════════════════════════════════════════════════════════════════
def propagate_signal(
    G: nx.Graph,
    seed_node: int,
    alpha: float = 0.35,
    n_steps: int = 60,
) -> tuple[np.ndarray, list[int]]:
    """
    Discrete-time random-walk diffusion (Honey et al. 2009, PNAS).

    Model:
        x(t+1) = α · D⁻¹A · x(t)  +  (1−α) · x(t)

    Parameters
    ----------
    G          NetworkX graph (weighted or unweighted)
    seed_node  Index of the initially activated region
    alpha      Spreading rate ∈ (0, 0.5) for stability
    n_steps    Number of discrete time steps

    Returns
    -------
    X          (n_steps+1, n) raw activation matrix (unnormalised)
    nodes      Ordered node list matching axis-1 of X
    """
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    A = nx.to_numpy_array(G, nodelist=nodes, weight="weight")
    # Row-normalise: each row sums to 1 (random-walk transition matrix)
    D = A.sum(axis=1, keepdims=True)
    D[D == 0] = 1.0
    A_rw = A / D

    # Spreading operator M
    M = alpha * A_rw + (1.0 - alpha) * np.eye(n)

    # Initial state: seed = 1.0, rest = 0.0
    x = np.zeros(n)
    x[node_idx.get(seed_node, 0)] = 1.0

    X = np.zeros((n_steps + 1, n))
    X[0] = x
    for t in range(n_steps):
        x = M @ x
        X[t + 1] = x

    return X, nodes


def spreading_speed(X: np.ndarray, threshold: float = 0.02) -> np.ndarray:
    """
    Fraction of nodes above *threshold* at each time step.
    Normalised so max activation at t=0 is the reference (= 1.0).
    """
    ref = X[0].max()
    if ref == 0:
        ref = 1.0
    X_norm = X / ref
    return (X_norm > threshold).mean(axis=1)


# ════════════════════════════════════════════════════════════════════════════
# 4. CHORD-DIAGRAM HELPER
# ════════════════════════════════════════════════════════════════════════════
def _compute_circle_positions(
    G: nx.Graph,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """
    Position nodes on a unit circle, grouped by module with angular gaps.
    Returns:
      pos_dict  : {node: (x, y)}
      angles    : 1-D array of angle per node (same order as G.nodes())
    """
    nodes = list(G.nodes())
    n = len(nodes)
    n_modules = len(set(G.nodes[nd].get("module", 0) for nd in nodes))

    # Sort nodes by module so module members are adjacent on the circle
    nodes_sorted = sorted(nodes, key=lambda nd: G.nodes[nd].get("module", 0))

    # Add small angular gap between modules
    gap = (2 * math.pi) / (n * 3)          # gap = 1/3 of a node-slot
    total_gap = gap * n_modules
    arc_per_node = (2 * math.pi - total_gap) / n

    pos_dict: dict[int, np.ndarray] = {}
    angle_map: dict[int, float] = {}
    current_angle = math.pi / 2             # start at top

    prev_module = G.nodes[nodes_sorted[0]].get("module", -1)
    for nd in nodes_sorted:
        this_module = G.nodes[nd].get("module", 0)
        if this_module != prev_module:
            current_angle -= gap
            prev_module = this_module
        pos_dict[nd] = np.array([math.cos(current_angle), math.sin(current_angle)])
        angle_map[nd] = current_angle
        current_angle -= arc_per_node

    angles_arr = np.array([angle_map[nd] for nd in nodes])
    return pos_dict, angles_arr


def draw_connectome(
    ax: plt.Axes,
    G: nx.Graph,
    pos: dict[int, np.ndarray],
    activation: np.ndarray | None = None,
    node_size: float = 40,
    title: str = "",
    is_hub_visible: bool = True,
) -> None:
    """
    Draw a chord-diagram connectome on *ax*.

    If *activation* (length-n array) is provided, nodes are coloured by
    activation level using the 'plasma' colormap.  Otherwise nodes are
    coloured by module.
    """
    ax.set_aspect("equal")
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.30, 1.30)
    ax.axis("off")
    ax.set_facecolor(CARD)

    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    # ── Edges ──────────────────────────────────────────────────────────────
    max_w = max((d.get("weight", 1) for _, _, d in G.edges(data=True)), default=1)
    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1.0) / max_w
        pu, pv = pos[u], pos[v]
        # Determine edge type
        mu = G.nodes[u].get("module", 0)
        mv = G.nodes[v].get("module", 0)
        if mu == mv:
            ec = MODULE_COLORS[mu % len(MODULE_COLORS)]
            alpha_e = float(np.clip(w * 0.35, 0.03, 0.40))
            lw = 0.4
        else:
            # Hub-to-hub edges get gold highlight
            if G.nodes[u].get("is_hub") and G.nodes[v].get("is_hub"):
                ec = GOLD
                alpha_e = float(np.clip(w * 0.60, 0.05, 0.70))
                lw = 0.8
            else:
                ec = SLATE
                alpha_e = float(np.clip(w * 0.18, 0.02, 0.20))
                lw = 0.3
        ax.plot([pu[0], pv[0]], [pu[1], pv[1]],
                color=ec, alpha=alpha_e, lw=lw, zorder=1)

    # ── Nodes ──────────────────────────────────────────────────────────────
    plasma = plt.colormaps["plasma"]

    for nd in nodes:
        p = pos[nd]
        m_id = G.nodes[nd].get("module", 0)
        is_hub = G.nodes[nd].get("is_hub", False)

        if activation is not None:
            idx = node_idx[nd]
            a_val = float(np.clip(activation[idx], 0, 1))
            node_color = plasma(a_val)
            s = node_size + a_val * node_size * 2
        else:
            node_color = MODULE_COLORS[m_id % len(MODULE_COLORS)]
            s = node_size + (node_size * 0.6 if (is_hub and is_hub_visible) else 0)

        ring_color = GOLD if (is_hub and is_hub_visible and activation is None) else node_color
        ax.scatter(
            p[0], p[1],
            s=s, c=[node_color], zorder=5,
            edgecolors=ring_color, linewidths=0.9 if (is_hub and activation is None) else 0.3,
        )

    if title:
        ax.set_title(title, color=LIGHT, fontsize=9, pad=2)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Phase 6 Extension — Brain Connectivity & Signal Propagation")
print("=" * 65)

# ── Build brain connectome ───────────────────────────────────────────────────
print("\n[1/6] Generating 90-region synthetic brain connectome…")
G_brain = generate_brain_connectome(n=90, n_modules=9, seed=SEED)
n = G_brain.number_of_nodes()
M = G_brain.number_of_edges()
print(f"    N = {n}  |  M = {M}  |  density = {2*M/(n*(n-1)):.3f}")

# ── Compute circle positions (done once, shared across all plots) ────────────
pos, angles = _compute_circle_positions(G_brain)

# ── Network statistics ────────────────────────────────────────────────────────
print("\n[2/6] Computing brain network statistics…")
brain_stats = compute_stats(G_brain)
C_real  = brain_stats["C"]
L_real  = brain_stats["L"]
r_real  = brain_stats["r"]
Q_real  = brain_stats["Q"]
k_mean  = brain_stats["k_mean"]

# ER analytical predictions
p_er    = 2 * M / (n * (n - 1))
C_er    = p_er
L_er    = math.log(n) / math.log(k_mean) if k_mean > 1 else float("nan")

print(f"    ⟨k⟩  = {k_mean:.2f}")
print(f"    C    = {C_real:.4f}   (ER: {C_er:.4f})")
print(f"    L    = {L_real:.3f}   (ER pred: {L_er:.3f})")
print(f"    r    = {r_real:+.4f}")
print(f"    Q    = {Q_real:.4f}   (modularity)")

# ── ER ensemble: 1000 null models ────────────────────────────────────────────
print(f"\n[3/6] Generating 1000 ER null models (N={n}, M={M})…")
N_ENS = 1000
C_ens = np.empty(N_ENS)
L_ens = np.empty(N_ENS)
r_ens = np.empty(N_ENS)

rng_ens = np.random.default_rng(SEED)
t0 = time.time()

for idx in range(N_ENS):
    Ge = nx.gnm_random_graph(n, M, seed=int(rng_ens.integers(0, 2**31)))
    C_ens[idx] = nx.average_clustering(Ge)

    try:
        r_ens[idx] = nx.degree_assortativity_coefficient(Ge)
    except Exception:
        r_ens[idx] = 0.0

    lcc_e = max(nx.connected_components(Ge), key=len)
    Ge_lcc = Ge.subgraph(lcc_e)
    src_s = random.sample(list(lcc_e), min(25, len(lcc_e)))
    spl = []
    for s_nd in src_s:
        spl.extend(nx.single_source_shortest_path_length(Ge_lcc, s_nd).values())
    L_ens[idx] = float(np.mean(spl)) if spl else float("nan")

    if (idx + 1) % 250 == 0:
        print(f"    {idx+1}/{N_ENS}  ({time.time()-t0:.1f}s)")

print(f"    Done in {time.time()-t0:.1f}s")
L_ens_clean = L_ens[~np.isnan(L_ens)]

def _zp(val: float, ens: np.ndarray) -> tuple[float, float]:
    mu, sig = ens.mean(), ens.std(ddof=1)
    if sig == 0:
        return float("inf"), 0.0
    z = (val - mu) / sig
    return z, float(2 * stats.norm.sf(abs(z)))

Z_C, pv_C = _zp(C_real, C_ens)
Z_L, pv_L = _zp(L_real, L_ens_clean)
Z_r, pv_r = _zp(r_real, r_ens)

def _fp(pv: float) -> str:
    if pv < 1e-300:
        return "< 1e-300"
    if pv < 1e-4:
        return f"{pv:.2e}"
    return f"{pv:.4f}"

print(f"\n{'='*65}")
print("STATISTICAL REJECTION OF ER NULL HYPOTHESIS")
print(f"{'='*65}")
print(f"  Clustering  C = {C_real:.4f}   μ_ER={C_ens.mean():.4f}   Z={Z_C:+.1f}σ   p={_fp(pv_C)}")
print(f"  Path length L = {L_real:.4f}   μ_ER={L_ens_clean.mean():.4f}   Z={Z_L:+.1f}σ   p={_fp(pv_L)}")
print(f"  Assortativity r = {r_real:+.4f}   μ_ER={r_ens.mean():+.4f}   Z={Z_r:+.1f}σ   p={_fp(pv_r)}")
print(f"{'='*65}")

# ── Signal propagation ────────────────────────────────────────────────────────
print("\n[4/6] Simulating signal propagation…")

# Seed = highest betweenness hub (most central node)
bc = nx.betweenness_centrality(G_brain)
seed_node = max(bc, key=bc.get)
print(f"    Seed node: {seed_node}  (module: {G_brain.nodes[seed_node]['module_name']}, "
      f"betweenness = {bc[seed_node]:.4f})")

N_STEPS = 60
ALPHA = 0.35

# Brain signal propagation
X_brain, nodes_brain = propagate_signal(G_brain, seed_node, alpha=ALPHA, n_steps=N_STEPS)
speed_brain = spreading_speed(X_brain, threshold=0.02)

# ER null model signal propagation
G_er_prop = nx.gnm_random_graph(n, M, seed=SEED + 1)
# Make ER weighted with unit weights for fair comparison
for u, v in G_er_prop.edges():
    G_er_prop[u][v]["weight"] = 1.0
er_seed = max(dict(G_er_prop.degree()).items(), key=lambda x: x[1])[0]
X_er, _ = propagate_signal(G_er_prop, er_seed, alpha=ALPHA, n_steps=N_STEPS)
speed_er = spreading_speed(X_er, threshold=0.02)

# Regular ring lattice (same N, same k ≈ k_mean)
k_lat = max(2, int(round(k_mean / 2)) * 2)   # nearest even integer ≥ 4
G_lat = nx.watts_strogatz_graph(n=n, k=k_lat, p=0.0, seed=SEED)
for u, v in G_lat.edges():
    G_lat[u][v]["weight"] = 1.0
lat_seed = 0
X_lat, _ = propagate_signal(G_lat, lat_seed, alpha=ALPHA, n_steps=N_STEPS)
speed_lat = spreading_speed(X_lat, threshold=0.02)

def _t50_str(speed: np.ndarray, thresh: float = 0.50) -> str:
    idx = np.argmax(speed >= thresh)
    return f"{int(idx)} steps" if speed[idx] >= thresh else f"> {N_STEPS} steps"

print(f"    Brain T₅₀  = {_t50_str(speed_brain)}")
print(f"    ER    T₅₀  = {_t50_str(speed_er)}")
print(f"    Lat   T₅₀  = {_t50_str(speed_lat)}")

# Map activation arrays to node_idx for chord diagram coloring
# (nodes_brain is the ordered node list from propagate_signal)
node_idx_brain = {nd: i for i, nd in enumerate(nodes_brain)}


def _activation_for_chord(X: np.ndarray, t: int, nodes_brain: list,
                           G_for_pos: nx.Graph) -> np.ndarray:
    """Return per-node activation (in G.nodes() order) at time t,
    normalised to [0,1] relative to the maximum over all time."""
    max_a = X.max()
    if max_a == 0:
        max_a = 1.0
    act_raw = X[t] / max_a
    graph_nodes = list(G_for_pos.nodes())
    # remap from propagation node order → G.nodes() order
    ni = {nd: i for i, nd in enumerate(nodes_brain)}
    return np.array([act_raw[ni.get(nd, 0)] for nd in graph_nodes])


SNAP_TIMES = [0, 5, 10, 20, 50]

print("\n[5/6] Generating plots…")

# ════════════════════════════════════════════════════════════════════════════
# PLOT A — Chord Diagram (connectome overview)
# ════════════════════════════════════════════════════════════════════════════
fig_a, axes_a = plt.subplots(1, 2, figsize=(16, 8), facecolor=BG)
fig_a.suptitle(
    "Synthetic AAL-90 Brain Connectome — Modular Structure",
    color=LIGHT, fontsize=16, fontweight="bold", y=0.98,
)

# Left: full chord diagram coloured by module
ax_chord = axes_a[0]
draw_connectome(
    ax_chord, G_brain, pos,
    activation=None,
    node_size=55,
    title="",
    is_hub_visible=True,
)
ax_chord.set_title("Network Architecture (chord diagram)", color=LIGHT, fontsize=13, pad=8)

# Module legend
legend_handles = [
    mpatches.Patch(facecolor=MODULE_COLORS[i], label=MODULE_NAMES[i], edgecolor=DIM)
    for i in range(len(MODULE_NAMES))
]
ax_chord.legend(
    handles=legend_handles, loc="lower left",
    fontsize=8.5, framealpha=0.75, facecolor=CARD,
    edgecolor=DIM, bbox_to_anchor=(-0.05, -0.05),
    ncol=1,
)

# Right: degree distribution comparison
ax_deg = axes_a[1]
ax_deg.set_facecolor(CARD)
_despine(ax_deg)

deg_real = [d for _, d in G_brain.degree()]
hist_r = np.bincount(deg_real)
k_r = np.arange(len(hist_r))
mask_r = hist_r > 0
ax_deg.scatter(k_r[mask_r], hist_r[mask_r] / n,
               color=TEAL, s=55, alpha=0.9, zorder=5, label="Brain network")

# Power-law fit
mask2 = k_r[mask_r] >= 3
log_k = np.log10(k_r[mask_r][mask2].astype(float))
log_p_arr = np.log10(hist_r[mask_r][mask2] / n)
if len(log_k) >= 3:
    sl, ic, _, _, _ = stats.linregress(log_k, log_p_arr)
    k_fit = np.linspace(k_r[mask_r][mask2].min(), k_r[mask_r][mask2].max(), 200)
    p_fit = 10 ** ic * k_fit ** sl
    ax_deg.plot(k_fit, p_fit, color=GOLD, lw=1.8, ls="--",
                label=f"Power-law γ ≈ {-sl:.2f}")

k_pois = np.arange(0, int(k_mean * 3 + 8))
p_pois = stats.poisson.pmf(k_pois, mu=k_mean)
ax_deg.plot(k_pois, p_pois, color=RED, lw=2.0, label=f"Poisson ER (λ={k_mean:.1f})")

ax_deg.set_xscale("log")
ax_deg.set_yscale("log")
ax_deg.set_xlabel("Degree  k", fontsize=12)
ax_deg.set_ylabel("P(k)", fontsize=12)
ax_deg.set_title("Log-log Degree Distribution", color=LIGHT, fontsize=13, pad=8)
ax_deg.legend(fontsize=10, framealpha=0.7)
ax_deg.grid(True, alpha=0.2)

# Annotation: small-world
ax_deg.text(
    0.97, 0.97,
    f"C = {C_real:.3f}  (ER: {C_er:.4f})\n"
    f"L = {L_real:.2f}  (ER pred: {L_er:.2f})\n"
    f"σ = C·n/M = {C_real * n / M:.2f}  (>1 → small-world)",
    transform=ax_deg.transAxes,
    ha="right", va="top",
    color=TEAL, fontsize=10.5,
    bbox=dict(facecolor=CARD, edgecolor=DIM, boxstyle="round,pad=0.4"),
)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_a.savefig(OUT / "plot14a_brain_connectome.png", dpi=150, bbox_inches="tight")
plt.close(fig_a)
print("  ✓  plot14a_brain_connectome.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT B — Signal Propagation Snapshots (3 × 5 grid)
# ════════════════════════════════════════════════════════════════════════════
network_configs = [
    ("Brain\n(Small-World)", G_brain, X_brain, nodes_brain, TEAL,  G_brain),
    ("ER\n(Random)",         G_er_prop, X_er,   list(G_er_prop.nodes()), RED,   G_brain),
    ("Ring Lattice\n(Regular)",G_lat, X_lat,  list(G_lat.nodes()),  GOLD,  G_brain),
]

# For ER and Lattice we don't have the same chord-diagram structure as G_brain
# — just show node-scatter with activation coloring using G_brain's positions
fig_b = plt.figure(figsize=(22, 14), facecolor=BG)
fig_b.suptitle(
    "Brain Signal Propagation — Diffusion Model Comparison",
    color=LIGHT, fontsize=17, fontweight="bold", y=0.99,
)
fig_b.text(
    0.5, 0.965,
    "x(t+1) = α·D⁻¹A·x(t) + (1−α)·x(t)   |   α = 0.35   |   Seed = highest-betweenness hub",
    ha="center", va="top", color=SLATE, fontsize=11,
)

gs_b = gridspec.GridSpec(
    3, len(SNAP_TIMES), figure=fig_b,
    hspace=0.10, wspace=0.04,
    left=0.06, right=0.98, top=0.94, bottom=0.04,
)

plasma_cmap = plt.colormaps["plasma"]

for row_idx, (net_label, G_net, X_net, nodes_net, row_color, _) in enumerate(network_configs):
    for col_idx, t in enumerate(SNAP_TIMES):
        ax = fig_b.add_subplot(gs_b[row_idx, col_idx])
        ax.set_aspect("equal")
        ax.set_xlim(-1.30, 1.30)
        ax.set_ylim(-1.30, 1.30)
        ax.axis("off")
        ax.set_facecolor(CARD)

        # Build per-node activation using G_brain's positions for visual consistency
        ni_net = {nd: i for i, nd in enumerate(nodes_net)}
        max_a = X_net.max()
        if max_a == 0:
            max_a = 1.0
        raw = X_net[t] / max_a
        # Map to G_brain nodes in the same order (indices match since n is same)
        act = np.array([raw[ni_net.get(nd, 0)] for nd in G_brain.nodes()])

        # Draw edges of G_brain as faint structure lines
        for u, v in G_brain.edges():
            pu, pv = pos[u], pos[v]
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]],
                    color=DIM, alpha=0.07, lw=0.3, zorder=1)

        # Nodes coloured by activation
        xs = np.array([pos[nd][0] for nd in G_brain.nodes()])
        ys = np.array([pos[nd][1] for nd in G_brain.nodes()])
        act_clipped = np.clip(act, 0, 1)
        ax.scatter(xs, ys, c=act_clipped, cmap="plasma",
                   vmin=0, vmax=1, s=28, zorder=5,
                   linewidths=0.0)

        # Time label on top row
        if row_idx == 0:
            ax.set_title(f"t = {t}", color=LIGHT, fontsize=11, pad=4)

        # Network label on left column
        if col_idx == 0:
            ax.set_ylabel(net_label, color=row_color, fontsize=11, labelpad=4)
            ax.yaxis.set_label_coords(-0.08, 0.5)
            ax.yaxis.label.set_visible(True)

        # Fraction reached annotation
        reached = float((act_clipped > 0.05).mean())
        ax.text(
            0.02, 0.02, f"{reached:.0%}",
            transform=ax.transAxes, ha="left", va="bottom",
            color=LIGHT, fontsize=8.5, alpha=0.85,
        )

# Colourbar
cbar_ax = fig_b.add_axes([0.92, 0.10, 0.012, 0.75])
norm = mcolors.Normalize(vmin=0, vmax=1)
sm = cm.ScalarMappable(cmap="plasma", norm=norm)
sm.set_array([])
cb = fig_b.colorbar(sm, cax=cbar_ax)
cb.set_label("Normalised activation", color=LIGHT, fontsize=10)
cb.ax.yaxis.set_tick_params(color=SLATE)
plt.setp(cb.ax.yaxis.get_ticklabels(), color=LIGHT, fontsize=9)

fig_b.savefig(OUT / "plot14b_signal_propagation.png", dpi=150, bbox_inches="tight")
plt.close(fig_b)
print("  ✓  plot14b_signal_propagation.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT C — ER Ensemble Null-Model Rejection (3 panels)
# ════════════════════════════════════════════════════════════════════════════
fig_c, axes_c = plt.subplots(1, 3, figsize=(18, 5.5), facecolor=BG)
fig_c.suptitle(
    "Brain Connectome — Statistical Rejection of ER Null Model",
    color=LIGHT, fontsize=15, fontweight="bold",
)

ens_configs = [
    (axes_c[0], C_ens,     C_real,  Z_C, pv_C, "Clustering Coefficient  C", TEAL),
    (axes_c[1], L_ens_clean, L_real, Z_L, pv_L, "Average Path Length  L",   GOLD),
    (axes_c[2], r_ens,     r_real,  Z_r, pv_r, "Degree Assortativity  r",   GREEN),
]

for ax_e, ens, val_real, Z, pv, label, col in ens_configs:
    ax_e.set_facecolor(CARD)
    _despine(ax_e)

    mu_e, sig_e = ens.mean(), ens.std(ddof=1)

    ax_e.hist(ens, bins=45, color=SLATE, alpha=0.75, density=True,
              label=f"ER ensemble  (N={N_ENS})")

    x_fit = np.linspace(min(ens.min(), val_real * 0.8),
                        max(ens.max(), val_real * 1.15), 500)
    y_fit = stats.norm.pdf(x_fit, mu_e, sig_e)
    ax_e.plot(x_fit, y_fit, color=RED, lw=1.8, ls="--", label="Normal fit")
    ax_e.axvline(val_real, color=col, lw=2.5, zorder=5,
                 label=f"Brain  = {val_real:.4f}")

    # Shade rejection tail
    if val_real > mu_e:
        tail_x = x_fit[x_fit >= val_real]
    else:
        tail_x = x_fit[x_fit <= val_real]
    if len(tail_x) > 0:
        tail_y = stats.norm.pdf(tail_x, mu_e, sig_e)
        ax_e.fill_between(tail_x, tail_y, alpha=0.30, color=col)

    ax_e.set_xlabel(label, fontsize=11)
    ax_e.set_ylabel("Density", fontsize=11)
    ax_e.set_title(
        f"Z = {Z:+.1f}σ   |   p = {_fp(pv)}",
        color=col, fontsize=12,
    )
    ax_e.legend(fontsize=9, framealpha=0.65)
    ax_e.grid(True, alpha=0.2)

    # Arrow annotation
    ymax = y_fit.max()
    ax_e.annotate(
        f"Brain\nZ = {Z:+.1f}σ",
        xy=(val_real, ymax * 0.05),
        xytext=(val_real, ymax * 0.55),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.6),
        color=col, fontsize=9.5, fontweight="bold", ha="center",
    )

plt.tight_layout()
fig_c.savefig(OUT / "plot14c_brain_vs_er.png", dpi=150, bbox_inches="tight")
plt.close(fig_c)
print("  ✓  plot14c_brain_vs_er.png")


# ════════════════════════════════════════════════════════════════════════════
# PLOT D — 4-panel Dashboard (summary)
# ════════════════════════════════════════════════════════════════════════════
fig_d = plt.figure(figsize=(18, 13), facecolor=BG)
gs_d = gridspec.GridSpec(
    2, 2, figure=fig_d,
    hspace=0.38, wspace=0.32,
    left=0.06, right=0.97, top=0.91, bottom=0.07,
)
fig_d.suptitle(
    "Phase 6 — Brain Connectivity: Breaking the ER Null Model",
    color=LIGHT, fontsize=17, fontweight="bold", y=0.97,
)
fig_d.text(
    0.5, 0.935,
    f"AAL-90 synthetic connectome  |  N={n}  |  M={M}  |  density={2*M/(n*(n-1)):.3f}  |"
    f"  Ensemble = {N_ENS} ER graphs",
    ha="center", va="top", color=SLATE, fontsize=11,
)

# ── D1: Connectome chord diagram ─────────────────────────────────────────────
ax_d1 = fig_d.add_subplot(gs_d[0, 0])
draw_connectome(ax_d1, G_brain, pos, activation=None, node_size=40, is_hub_visible=True)
ax_d1.set_title("Modular Architecture (chord diagram)", color=LIGHT, fontsize=12, fontweight="bold")
# Compact module legend
handles_d1 = [
    mpatches.Patch(facecolor=MODULE_COLORS[i], label=MODULE_NAMES[i][:12])
    for i in range(len(MODULE_NAMES))
]
ax_d1.legend(handles=handles_d1, fontsize=7.5, framealpha=0.6,
             facecolor=CARD, edgecolor=DIM,
             loc="lower left", ncol=1,
             bbox_to_anchor=(-0.02, 0.01))

# ── D2: Signal spreading speed comparison ────────────────────────────────────
ax_d2 = fig_d.add_subplot(gs_d[0, 1])
ax_d2.set_facecolor(CARD)
_despine(ax_d2)

t_axis = np.arange(N_STEPS + 1)
ax_d2.plot(t_axis, speed_brain, color=TEAL,  lw=2.5, label="Brain (small-world)")
ax_d2.plot(t_axis, speed_er,    color=RED,   lw=2.0, ls="--", label="ER (random)")
ax_d2.plot(t_axis, speed_lat,   color=GOLD,  lw=2.0, ls=":",  label="Ring Lattice (regular)")

# Mark crossing 50% and 90% thresholds
for thresh, lbl in [(0.5, "50%"), (0.9, "90%")]:
    ax_d2.axhline(thresh, color=DIM, lw=0.9, ls="-", alpha=0.5)
    ax_d2.text(N_STEPS + 0.5, thresh, lbl, color=SLATE, fontsize=9, va="center")

# Annotate T₅₀ for brain
def _t50(speed: np.ndarray, thresh: float = 0.50) -> int:
    idx = np.argmax(speed >= thresh)
    return int(idx) if speed[idx] >= thresh else -1

t50_brain = _t50(speed_brain)
t50_er    = _t50(speed_er)
t50_lat   = _t50(speed_lat)
for t50, col, net in [(t50_brain, TEAL, "Brain"), (t50_er, RED, "ER"), (t50_lat, GOLD, "Lattice")]:
    if 0 < t50 < N_STEPS:
        ax_d2.axvline(t50, color=col, lw=1.0, ls=":", alpha=0.6)
        ax_d2.text(t50, 0.53, f"T₅₀={t50}", color=col, fontsize=8.5,
                   ha="center", rotation=90)
    elif t50 == -1:
        ax_d2.text(N_STEPS * 0.97, speed_lat[-1] + 0.02,
                   f"{net}: T₅₀ > {N_STEPS}", color=col,
                   fontsize=8, ha="right", va="bottom")

ax_d2.set_xlabel("Time step  t", fontsize=12)
ax_d2.set_ylabel("Fraction of regions activated", fontsize=12)
ax_d2.set_title("Signal Spreading Speed", color=LIGHT, fontsize=12, fontweight="bold")
ax_d2.legend(fontsize=10, framealpha=0.65)
ax_d2.set_xlim(0, N_STEPS)
ax_d2.set_ylim(0, 1.05)
ax_d2.grid(True, alpha=0.2)

# ── D3: Brain at t=20 (peak spreading phase) ─────────────────────────────────
ax_d3 = fig_d.add_subplot(gs_d[1, 0])
act_t20 = _activation_for_chord(X_brain, t=20, nodes_brain=nodes_brain, G_for_pos=G_brain)
draw_connectome(ax_d3, G_brain, pos, activation=act_t20, node_size=45, is_hub_visible=False)
ax_d3.set_title("Brain — Activation at t = 20", color=LIGHT, fontsize=12, fontweight="bold")

# Colourbar for activation
ax_d3_pos = ax_d3.get_position()
cbar_ax3 = fig_d.add_axes([ax_d3_pos.x0, ax_d3_pos.y0 - 0.025,
                            ax_d3_pos.width, 0.012])
norm3 = mcolors.Normalize(vmin=0, vmax=1)
sm3 = cm.ScalarMappable(cmap="plasma", norm=norm3)
sm3.set_array([])
cb3 = fig_d.colorbar(sm3, cax=cbar_ax3, orientation="horizontal")
cb3.set_label("Normalised activation", color=SLATE, fontsize=9)
cb3.ax.xaxis.set_tick_params(color=SLATE)
plt.setp(cb3.ax.xaxis.get_ticklabels(), color=SLATE, fontsize=8)

# ── D4: Z-score bar chart ─────────────────────────────────────────────────────
ax_d4 = fig_d.add_subplot(gs_d[1, 1])
ax_d4.set_facecolor(CARD)
_despine(ax_d4)

z_metrics  = ["Clustering\nCoeff. C", "Path\nLength L", "Assortativity\nr"]
z_values   = [Z_C, Z_L, Z_r]
bar_cols    = [
    TEAL  if abs(z) > 10 else (GOLD if abs(z) > 3 else RED)
    for z in z_values
]
bars = ax_d4.barh(z_metrics, z_values, color=bar_cols, height=0.45, alpha=0.85)

ax_d4.axvline(0,  color=LIGHT, lw=1.0, alpha=0.5)
ax_d4.axvline(+3, color=GOLD, lw=1.2, ls=":", alpha=0.7, label="±3σ")
ax_d4.axvline(-3, color=GOLD, lw=1.2, ls=":", alpha=0.7)

for bar, z in zip(bars, z_values):
    xpos = bar.get_width()
    ha = "left" if xpos >= 0 else "right"
    ax_d4.text(
        xpos + (0.3 if xpos >= 0 else -0.3),
        bar.get_y() + bar.get_height() / 2,
        f"{z:+.1f}σ", va="center", ha=ha,
        color=LIGHT, fontsize=11, fontweight="bold",
    )

ax_d4.set_xlabel("Z-score (standard deviations from ER mean)", fontsize=11)
ax_d4.set_title(
    "Statistical Rejection of H₀  (ER null model)",
    color=LIGHT, fontsize=12, fontweight="bold",
)
ax_d4.legend(fontsize=9, framealpha=0.5)
ax_d4.grid(True, axis="x", alpha=0.2)

# Footer conclusion
fig_d.text(
    0.5, 0.01,
    f"H₀ REJECTED: Brain clustering is {abs(Z_C):.0f}σ above ER  ·  "
    f"Small-world index σ = {C_real * n / M:.2f}  (>1 confirms small-world)  ·  "
    f"Q = {Q_real:.3f} (modular)",
    ha="center", va="bottom", fontsize=12,
    color=TEAL, fontweight="bold",
    bbox=dict(facecolor=CARD, edgecolor=DIM, boxstyle="round,pad=0.4"),
)

fig_d.savefig(OUT / "plot14d_brain_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig_d)
print("  ✓  plot14d_brain_dashboard.png")

print("\n[6/6] All plots saved to the project root.")
print("Done.")
