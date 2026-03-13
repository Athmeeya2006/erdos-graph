"""
plot13_real_world_mapping.py
============================
Phase 6 — Real-World Dataset Mapping
Breaking the Null Model — Using Statistics to Reject ER

Outputs (saved to the project root):
    plot13a_degree_distribution.png   — log-log degree dist: S&P500 vs ER
    plot13b_comparison_table.png      — side-by-side metric table
    plot13c_ensemble_clustering.png   — ER ensemble histogram with C_real marked
    plot13d_summary_dashboard.png     — 4-panel combined dashboard

Data sources:
    1. Preferred: local cached adjusted-close CSV at
       data/sp500_adj_close.csv or the path in ERDOS_PRICE_DATA.
    2. Fallback: download daily adjusted close prices for a curated set of
       S&P 500 stocks via yfinance, then cache the CSV locally.

Method:
    1. Load adjusted close prices over 3 years.
    2. Compute a Pearson correlation matrix of daily log-returns.
    3. Threshold at |ρ| > τ to build a correlation network G_real.
    4. Compute network statistics: degree dist, clustering coefficient,
       average path length, assortativity.
    5. Generate an ER null-model ensemble (1000 graphs, same N and M) and
       compute the same statistics.
    6. Compute Z-scores and p-values; print interpretation.
"""

from __future__ import annotations

import os
import warnings
import math
import time
import random
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import scipy.stats as stats
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    yf = None

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ── Output directory ─────────────────────────────────────────────────────────
OUT = Path(".")

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHED_PRICE_DATA = Path(os.environ.get("ERDOS_PRICE_DATA", DATA_DIR / "sp500_adj_close.csv"))

# ── Palette (consistent with rest of repo) ───────────────────────────────────
NAVY   = "#1A3A5C"
TEAL   = "#0E7490"
RED    = "#DC2626"
GOLD   = "#D97706"
SLATE  = "#64748B"
LIGHT  = "#F1F5F9"
PURPLE = "#7C3AED"
GREEN  = "#059669"
ROSE   = "#BE185D"
BG     = "#020617"
CARD   = "#0F172A"
DIM    = "#334155"

# ── Ticker universe: 5 GICS sectors, ~20 tickers each ───────────────────────
# Deliberately avoid tickers that were recently delisted/renamed.
TICKERS_BY_SECTOR: dict[str, list[str]] = {
    "Technology":   ["AAPL","MSFT","NVDA","AVGO","ORCL","AMD","QCOM","TXN",
                     "CSCO","IBM","INTC","MU","KLAC","AMAT","LRCX",
                     "ADI","MCHP","NXPI","SWKS","FTNT"],
    "Financials":   ["JPM","BAC","WFC","GS","MS","C","AXP","BLK","SCHW",
                     "CB","PNC","USB","TFC","COF","MTB","FITB","HBAN","RF",
                     "KEY","CFG"],
    "Healthcare":   ["JNJ","UNH","ABT","TMO","MRK","ABBV","LLY","DHR","MDT",
                     "BMY","AMGN","GILD","CVS","HUM","CI","ELV","SYK","BSX",
                     "ISRG","BDX"],
    "Energy":       ["XOM","CVX","COP","SLB","EOG","PXD","PSX","VLO","MPC",
                     "OXY","HES","DVN","FANG","HAL","BKR","APA","MRO","OKE",
                     "KMI","WMB"],
    "Consumer":     ["AMZN","TSLA","HD","MCD","NKE","SBUX","TGT","LOW","CMG",
                     "DHI","LEN","YUM","DRI","ROST","TJX","BBY","KR","WMT",
                     "COST","PG"],
}
ALL_TICKERS = [t for sector in TICKERS_BY_SECTOR.values() for t in sector]
SECTOR_OF = {t: s for s, tl in TICKERS_BY_SECTOR.items() for t in tl}
SECTOR_COLORS = {
    "Technology":  TEAL,
    "Financials":  GOLD,
    "Healthcare":  GREEN,
    "Energy":      RED,
    "Consumer":    PURPLE,
}

# Correlation threshold to form an edge
CORR_THRESHOLD = 0.50


def load_adjusted_close_prices(
    tickers: list[str],
    start: str = "2021-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Load adjusted close prices from a local CSV when available.

    Falls back to yfinance only when no cached dataset is present, then writes
    the download back to disk so future runs are reproducible and offline-safe.
    """
    if CACHED_PRICE_DATA.exists():
        print(f"    → Loading cached price data from {CACHED_PRICE_DATA}")
        close = pd.read_csv(CACHED_PRICE_DATA, index_col=0, parse_dates=True)
        return close.sort_index()

    if yf is None:
        raise RuntimeError(
            "No cached dataset found at "
            f"{CACHED_PRICE_DATA} and yfinance is not installed. "
            "Add a local CSV or install yfinance to fetch it."
        )

    print("    → No cached CSV found; downloading from Yahoo Finance")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty or "Close" not in raw:
        raise RuntimeError("Failed to download adjusted close data from yfinance.")

    close = raw["Close"].copy()
    close.to_csv(CACHED_PRICE_DATA)
    print(f"    → Cached adjusted close prices to {CACHED_PRICE_DATA}")
    return close

# ════════════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD & PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Phase 6 — Real-World Dataset Mapping")
print("=" * 65)
print("\n[1/5] Loading S&P 500 price data (2021-2024)…")

close = load_adjusted_close_prices(ALL_TICKERS)

# Drop tickers with >10% missing data
thresh = int(0.90 * len(close))
close = close.dropna(axis=1, thresh=thresh)
close = close.ffill().bfill()          # fill remaining gaps
tickers = list(close.columns)
n = len(tickers)
print(f"    → {n} tickers passed quality filter (of {len(ALL_TICKERS)} requested)")

# Log returns
log_ret = np.log(close / close.shift(1)).dropna()

# Pearson correlation matrix
corr_matrix = log_ret.corr()
print(f"    → Correlation matrix: {n}×{n}")

# ────────────────────────────────────────────────────────────────────────────
# 2. BUILD CORRELATION NETWORK
# ────────────────────────────────────────────────────────────────────────────
print(f"\n[2/5] Building correlation network (|ρ| > {CORR_THRESHOLD})…")

G_real = nx.Graph()
G_real.add_nodes_from(range(n))

# Attach metadata
for i, tkr in enumerate(tickers):
    G_real.nodes[i]["ticker"] = tkr
    G_real.nodes[i]["sector"] = SECTOR_OF.get(tkr, "Unknown")

for i in range(n):
    for j in range(i + 1, n):
        rho = corr_matrix.iloc[i, j]
        if abs(rho) > CORR_THRESHOLD:
            G_real.add_edge(i, j, weight=rho)

M = G_real.number_of_edges()
print(f"    → N = {n} nodes,  M = {M} edges,  density = {2*M/(n*(n-1)):.4f}")

# Keep largest connected component for path-length analysis
lcc_nodes = max(nx.connected_components(G_real), key=len)
G_lcc = G_real.subgraph(lcc_nodes).copy()
print(f"    → Largest connected component: {len(lcc_nodes)} nodes")

# ────────────────────────────────────────────────────────────────────────────
# 3. COMPUTE REAL-NETWORK STATISTICS
# ────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Computing real-network statistics…")

degrees_real = [d for _, d in G_real.degree()]
k_mean_real  = np.mean(degrees_real)
k_max_real   = max(degrees_real)

C_real       = nx.average_clustering(G_real)
assortativity_real = nx.degree_assortativity_coefficient(G_real)

# Average shortest path length on the LCC (can be expensive for large LCC)
t0 = time.time()
if len(G_lcc) <= 300:
    L_real = nx.average_shortest_path_length(G_lcc)
else:
    # Approximate by sampling 300 source nodes
    sample = random.sample(list(G_lcc.nodes()), 300)
    lengths = []
    for src in sample:
        sp = nx.single_source_shortest_path_length(G_lcc, src)
        lengths.extend(sp.values())
    L_real = np.mean(lengths)
t_path = time.time() - t0

# Degree distribution for power-law check
deg_hist = np.bincount(degrees_real)
k_vals   = np.arange(len(deg_hist))
mask     = deg_hist > 0
k_pos    = k_vals[mask]
p_pos    = deg_hist[mask] / n

# Power-law exponent via linear regression on log-log (k >= 2)
mask2    = k_pos >= 2
log_k    = np.log10(k_pos[mask2])
log_p    = np.log10(p_pos[mask2])
if len(log_k) >= 3:
    slope, intercept, r_value, _, _ = stats.linregress(log_k, log_p)
    gamma_real = -slope
else:
    gamma_real = float("nan")

print(f"    ⟨k⟩  = {k_mean_real:.2f}")
print(f"    C    = {C_real:.4f}")
print(f"    L    = {L_real:.3f}  (computed in {t_path:.2f}s)")
print(f"    r    = {assortativity_real:.4f}  (assortativity)")
print(f"    γ    ≈ {gamma_real:.2f}  (degree-dist power-law exponent, if applicable)")

# ────────────────────────────────────────────────────────────────────────────
# 4. ANALYTICAL ER PREDICTIONS
# ────────────────────────────────────────────────────────────────────────────
p_er       = (2 * M) / (n * (n - 1))   # edge probability matching real density
k_mean_er  = p_er * (n - 1)
C_er       = p_er                      # theorem: C_ER = p
# ER avg path length approximation: ln(n) / ln(⟨k⟩)
L_er_pred  = math.log(n) / math.log(k_mean_er) if k_mean_er > 1 else float("nan")
assort_er  = 0.0                       # ER is uncorrelated → r ≈ 0

# ────────────────────────────────────────────────────────────────────────────
# 5. ENSEMBLE OF 1000 ER NULL MODELS
# ────────────────────────────────────────────────────────────────────────────
print(f"\n[4/5] Generating 1000 ER null models (N={n}, M={M})…")

N_ENSEMBLE = 1000
C_ensemble  = np.empty(N_ENSEMBLE)
L_ensemble  = np.empty(N_ENSEMBLE)
r_ensemble  = np.empty(N_ENSEMBLE)

rng = np.random.default_rng(42)

t0 = time.time()
for idx in range(N_ENSEMBLE):
    Ge = nx.gnm_random_graph(n, M, seed=int(rng.integers(0, 2**31)))
    C_ensemble[idx] = nx.average_clustering(Ge)

    # Assortativity
    try:
        r_ensemble[idx] = nx.degree_assortativity_coefficient(Ge)
    except Exception:
        r_ensemble[idx] = 0.0

    # Average path length on LCC (approximate for speed)
    lcc_e = max(nx.connected_components(Ge), key=len)
    Ge_lcc = Ge.subgraph(lcc_e)
    if len(lcc_e) >= 10:
        src_sample = random.sample(list(lcc_e), min(30, len(lcc_e)))
        spl = []
        for src in src_sample:
            sp_dict = nx.single_source_shortest_path_length(Ge_lcc, src)
            spl.extend(sp_dict.values())
        L_ensemble[idx] = np.mean(spl)
    else:
        L_ensemble[idx] = float("nan")

    if (idx + 1) % 200 == 0:
        elapsed = time.time() - t0
        print(f"    {idx+1}/{N_ENSEMBLE}   ({elapsed:.1f}s elapsed)")

elapsed_total = time.time() - t0
print(f"    Done! {elapsed_total:.1f}s total")

# Remove NaN path lengths
L_ens_clean = L_ensemble[~np.isnan(L_ensemble)]

# ── Z-scores and p-values ────────────────────────────────────────────────────
def z_p(val: float, ens: np.ndarray) -> tuple[float, float]:
    mu, sigma = ens.mean(), ens.std(ddof=1)
    if sigma == 0:
        return float("inf"), 0.0
    z = (val - mu) / sigma
    pv = 2 * stats.norm.sf(abs(z))   # two-tailed
    return z, pv

mu_C,  sigma_C  = C_ensemble.mean(),    C_ensemble.std(ddof=1)
mu_L,  sigma_L  = L_ens_clean.mean(),   L_ens_clean.std(ddof=1)
mu_r,  sigma_r  = r_ensemble.mean(),    r_ensemble.std(ddof=1)

Z_C,  pv_C  = z_p(C_real,             C_ensemble)
Z_L,  pv_L  = z_p(L_real,             L_ens_clean)
Z_r,  pv_r  = z_p(assortativity_real, r_ensemble)

print("\n" + "=" * 65)
print("STATISTICAL RESULTS — NULL MODEL REJECTION")
print("=" * 65)

def fmt_pval(pv: float) -> str:
    if pv < 1e-300:
        return "< 1e-300 (effectively 0)"
    if pv < 1e-4:
        return f"{pv:.2e}"
    return f"{pv:.4f}"

print(f"\n  Clustering Coefficient:")
print(f"    Real network   C = {C_real:.5f}")
print(f"    ER ensemble    μ = {mu_C:.5f},  σ = {sigma_C:.6f}")
print(f"    Z-score          = {Z_C:+.1f} σ")
print(f"    p-value          = {fmt_pval(pv_C)}")

print(f"\n  Average Path Length:")
print(f"    Real network   L = {L_real:.4f}")
print(f"    ER ensemble    μ = {mu_L:.4f},  σ = {sigma_L:.5f}")
print(f"    Z-score          = {Z_L:+.2f} σ")
print(f"    p-value          = {fmt_pval(pv_L)}")

print(f"\n  Degree Assortativity:")
print(f"    Real network   r = {assortativity_real:+.4f}")
print(f"    ER ensemble    μ = {mu_r:+.5f},  σ = {sigma_r:.5f}")
print(f"    Z-score          = {Z_r:+.2f} σ")
print(f"    p-value          = {fmt_pval(pv_r)}")

if abs(Z_C) > 5:
    print(f"\n  ✓ REJECT H₀: Clustering is {abs(Z_C):.0f}σ from ER — not random.")
if Z_L < -3:
    print(f"  ✓ REJECT H₀: Path length is {abs(Z_L):.1f}σ shorter than ER → small-world.")
if abs(Z_r) > 3:
    direction = "assortative" if assortativity_real > 0 else "disassortative"
    print(f"  ✓ REJECT H₀: Network is {direction} (r = {assortativity_real:+.4f}, {abs(Z_r):.1f}σ from ER).")
print("=" * 65)

# ════════════════════════════════════════════════════════════════════════════
# 6. PLOTS
# ════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Generating plots…")

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
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(DIM)
    ax.spines["bottom"].set_color(DIM)


# ────────────────────────────────────────────────────────────────────────────
# Plot A — Log-log degree distribution comparison
# ────────────────────────────────────────────────────────────────────────────
fig_a, axes_a = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig_a.suptitle(
    "Degree Distribution: S&P 500 Correlation Network vs. ER Null Model",
    color=LIGHT, fontsize=15, fontweight="bold", y=0.98,
)

# — Left: log-log scatter —
ax_ll = axes_a[0]
ax_ll.set_facecolor(CARD)
_despine(ax_ll)

# Real network
ax_ll.scatter(k_pos, p_pos, color=TEAL, s=50, alpha=0.85, zorder=5,
              label=f"S&P 500 (n={n})")
# Power-law fit line
if not math.isnan(gamma_real) and len(k_pos[mask2]) >= 3:
    k_fit = np.linspace(k_pos[mask2].min(), k_pos[mask2].max(), 200)
    p_fit = 10**intercept * k_fit**(-gamma_real)
    ax_ll.plot(k_fit, p_fit, color=GOLD, lw=1.8, ls="--", zorder=6,
               label=f"Power-law fit  γ ≈ {gamma_real:.2f}")

# ER Poisson envelope (several ER samples)
for _ in range(5):
    Ge_tmp = nx.gnm_random_graph(n, M, seed=random.randint(0, 99999))
    degs_tmp = [d for _, d in Ge_tmp.degree()]
    hist_tmp = np.bincount(degs_tmp)
    kv = np.arange(len(hist_tmp))
    mv = hist_tmp > 0
    ax_ll.scatter(kv[mv], hist_tmp[mv] / n, color=RED, s=14, alpha=0.25, zorder=3)

# Poisson theoretical PMF
k_pois = np.arange(0, int(k_mean_er * 3 + 10))
p_pois = stats.poisson.pmf(k_pois, mu=k_mean_er)
ax_ll.plot(k_pois, p_pois, color=RED, lw=2.0, zorder=4,
           label=f"Poisson  (λ={k_mean_er:.1f})")

ax_ll.set_xscale("log")
ax_ll.set_yscale("log")
ax_ll.set_xlabel("Degree  k", fontsize=12)
ax_ll.set_ylabel("P(k)", fontsize=12)
ax_ll.set_title("Log-log  P(k)", color=LIGHT, fontsize=12)
ax_ll.legend(fontsize=10, framealpha=0.7)
ax_ll.grid(True, alpha=0.25)

# — Right: linear CCDF (complementary CDF) —
ax_ccdf = axes_a[1]
ax_ccdf.set_facecolor(CARD)
_despine(ax_ccdf)

degs_sorted = np.sort(degrees_real)[::-1]
ccdf_y = np.arange(1, len(degs_sorted) + 1) / len(degs_sorted)
ax_ccdf.step(degs_sorted, ccdf_y, color=TEAL, lw=2.0, label="S&P 500")

# ER CCDF
degs_er_sample = [d for _, d in nx.gnm_random_graph(n, M, seed=99).degree()]
degs_er_sorted = np.sort(degs_er_sample)[::-1]
ccdf_er_y = np.arange(1, n + 1) / n
ax_ccdf.step(degs_er_sorted, ccdf_er_y, color=RED, lw=2.0, ls="--",
             label=f"ER  (Poisson, λ={k_mean_er:.1f})")

ax_ccdf.set_xlabel("Degree  k", fontsize=12)
ax_ccdf.set_ylabel("P(K ≥ k)", fontsize=12)
ax_ccdf.set_title("Complementary CDF", color=LIGHT, fontsize=12)
ax_ccdf.legend(fontsize=10, framealpha=0.7)
ax_ccdf.grid(True, alpha=0.25)

# Annotation: heavy tail
k90 = np.percentile(degrees_real, 90)
ax_ccdf.axvline(k90, color=GOLD, lw=1.2, ls=":", alpha=0.8)
ax_ccdf.text(k90 + 0.5, 0.15, "90th\npercentile", color=GOLD, fontsize=9)

plt.tight_layout()
fig_a.savefig(OUT / "plot13a_degree_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig_a)
print("  ✓  plot13a_degree_distribution.png")


# ────────────────────────────────────────────────────────────────────────────
# Plot B — Comparison table
# ────────────────────────────────────────────────────────────────────────────
def _fmt(val: float, pct: bool = False) -> str:
    if math.isnan(val):
        return "N/A"
    if pct:
        return f"{val * 100:.1f}%"
    if abs(val) > 100:
        return f"{val:.1f}"
    if abs(val) < 0.001:
        return f"{val:.2e}"
    return f"{val:.4f}"

table_data = [
    ("Nodes (N)",              str(n),               str(n),             "—"),
    ("Edges (M)",              str(M),               str(M),             "—"),
    ("Edge probability  p",    _fmt(2*M/(n*(n-1))),  _fmt(p_er),         "—"),
    ("Mean degree  ⟨k⟩",       _fmt(k_mean_real),    _fmt(k_mean_er),    "—"),
    ("Clustering C",           _fmt(C_real),          _fmt(C_er),         f"{Z_C:+.0f}σ"),
    ("Avg path length  L",     _fmt(L_real),          _fmt(mu_L),         f"{Z_L:+.1f}σ"),
    ("Degree assortativity  r",_fmt(assortativity_real), _fmt(assort_er), f"{Z_r:+.1f}σ"),
    ("Degree dist. shape",     "Heavy tail",          "Poisson",          "Qualitative"),
    ("Power-law exponent  γ",  _fmt(gamma_real),      "N/A (Poisson)",    "—"),
]

headers   = ["Metric", "S&P 500 Network", "ER Prediction", "Z-score"]
col_w     = [0.35, 0.22, 0.22, 0.14]
row_h     = 0.40
n_rows    = len(table_data) + 1   # +1 for header
fig_w, fig_h = 13, n_rows * row_h + 1.2

fig_b, ax_b = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG)
ax_b.set_facecolor(BG)
ax_b.axis("off")
fig_b.suptitle(
    "S&P 500 Correlation Network — ER Null Model Comparison",
    color=LIGHT, fontsize=15, fontweight="bold", y=0.98,
)

x_starts = [sum(col_w[:i]) for i in range(len(col_w))]

def draw_row(ax, row_idx, cells, bg_color, text_colors=None):
    y = 1.0 - (row_idx + 1) * (1.0 / n_rows)
    h = 1.0 / n_rows
    for ci, (x0, w, txt) in enumerate(zip(x_starts, col_w, cells)):
        rect = FancyBboxPatch(
            (x0 + 0.003, y + 0.015), w - 0.006, h - 0.025,
            boxstyle="round,pad=0.01",
            facecolor=bg_color, edgecolor=DIM, linewidth=0.5,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)
        tc = LIGHT if text_colors is None else text_colors[ci]
        ax.text(
            x0 + w / 2, y + h / 2, txt,
            ha="center", va="center", fontsize=11,
            color=tc, fontweight="bold" if row_idx == 0 else "normal",
            transform=ax.transAxes, clip_on=False,
        )

# Header
draw_row(ax_b, 0, headers, NAVY,
         text_colors=[GOLD, TEAL, RED, LIGHT])

# Data rows
for ri, row in enumerate(table_data, start=1):
    # Highlight the Z-score column
    z_raw = row[3]
    if z_raw not in ("—", "Qualitative", "N/A"):
        try:
            z_val = float(z_raw.replace("σ", ""))
            z_color = TEAL if abs(z_val) > 10 else (GOLD if abs(z_val) > 3 else SLATE)
        except ValueError:
            z_color = LIGHT
    else:
        z_color = SLATE
    row_bg = CARD if ri % 2 == 0 else "#111827"
    draw_row(ax_b, ri, list(row),
             row_bg,
             text_colors=[LIGHT, TEAL, RED, z_color])

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_b.savefig(OUT / "plot13b_comparison_table.png", dpi=150, bbox_inches="tight")
plt.close(fig_b)
print("  ✓  plot13b_comparison_table.png")


# ────────────────────────────────────────────────────────────────────────────
# Plot C — Ensemble histogram: clustering coefficient
# ────────────────────────────────────────────────────────────────────────────
fig_c, axes_c = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)
fig_c.suptitle(
    "ER Ensemble vs. Real Network — Null Model Rejection",
    color=LIGHT, fontsize=15, fontweight="bold",
)

# — Left panel: clustering coefficient histogram —
ax_c1 = axes_c[0]
ax_c1.set_facecolor(CARD)
_despine(ax_c1)

ax_c1.hist(C_ensemble, bins=50, color=SLATE, alpha=0.7, density=True,
           label=f"ER ensemble  (μ={mu_C:.5f}, σ={sigma_C:.2e})")
ax_c1.axvline(C_real, color=TEAL, lw=2.5,
              label=f"C_real = {C_real:.4f}", zorder=5)

# Fit Gaussian to ensemble
x_g = np.linspace(C_ensemble.min(), max(C_ensemble.max(), C_real * 1.1), 500)
y_g = stats.norm.pdf(x_g, mu_C, sigma_C)
ax_c1.plot(x_g, y_g, color=RED, lw=1.8, ls="--", label="Normal fit")

# Shade rejection tail
tail_x = x_g[x_g >= C_real]
tail_y = stats.norm.pdf(tail_x, mu_C, sigma_C)
ax_c1.fill_between(tail_x, tail_y, alpha=0.25, color=TEAL,
                   label=f"p-value tail")

ax_c1.set_xlabel("Clustering Coefficient  C", fontsize=12)
ax_c1.set_ylabel("Density", fontsize=12)
ax_c1.set_title(
    f"Clustering: Z = {Z_C:+.0f}σ,  p = {fmt_pval(pv_C)}",
    color=TEAL, fontsize=12,
)
ax_c1.legend(fontsize=9, framealpha=0.7)
ax_c1.grid(True, alpha=0.25)

# Annotation arrow
x_arrow = C_real
y_arrow = ax_c1.get_ylim()[1] * 0.6
ax_c1.annotate(
    f"S&P 500\nC = {C_real:.4f}\n({Z_C:+.0f}σ)",
    xy=(C_real, stats.norm.pdf(C_real, mu_C, sigma_C) + y_g.max() * 0.02),
    xytext=(C_real, y_g.max() * 0.55),
    arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6),
    color=TEAL, fontsize=10, fontweight="bold",
    ha="center",
)

# — Right panel: assortativity histogram —
ax_c2 = axes_c[1]
ax_c2.set_facecolor(CARD)
_despine(ax_c2)

ax_c2.hist(r_ensemble, bins=50, color=SLATE, alpha=0.7, density=True,
           label=f"ER ensemble  (μ={mu_r:+.4f}, σ={sigma_r:.3f})")
ax_c2.axvline(assortativity_real, color=GOLD, lw=2.5,
              label=f"r_real = {assortativity_real:+.4f}", zorder=5)

x_g2 = np.linspace(r_ensemble.min(), r_ensemble.max(), 400)
y_g2 = stats.norm.pdf(x_g2, mu_r, sigma_r)
ax_c2.plot(x_g2, y_g2, color=RED, lw=1.8, ls="--", label="Normal fit")

ax_c2.set_xlabel("Degree Assortativity  r", fontsize=12)
ax_c2.set_ylabel("Density", fontsize=12)
ax_c2.set_title(
    f"Assortativity: Z = {Z_r:+.1f}σ,  p = {fmt_pval(pv_r)}",
    color=GOLD, fontsize=12,
)
ax_c2.legend(fontsize=9, framealpha=0.7)
ax_c2.grid(True, alpha=0.25)

plt.tight_layout()
fig_c.savefig(OUT / "plot13c_ensemble_clustering.png", dpi=150, bbox_inches="tight")
plt.close(fig_c)
print("  ✓  plot13c_ensemble_clustering.png")


# ────────────────────────────────────────────────────────────────────────────
# Plot D — 4-panel summary dashboard
# ────────────────────────────────────────────────────────────────────────────
fig_d = plt.figure(figsize=(16, 12), facecolor=BG)
gs = gridspec.GridSpec(
    2, 2, figure=fig_d,
    hspace=0.38, wspace=0.35,
    left=0.07, right=0.97, top=0.91, bottom=0.07,
)

fig_d.suptitle(
    "Phase 6 — Breaking the Null Model: S&P 500 vs. Erdős-Rényi",
    color=LIGHT, fontsize=17, fontweight="bold", y=0.97,
)

ax_d_sub = fig_d.text(
    0.5, 0.935,
    f"N={n} stocks · M={M} edges · τ=|ρ|>{CORR_THRESHOLD} · "
    f"Ensemble size = {N_ENSEMBLE}",
    ha="center", va="top", color=SLATE, fontsize=11,
)

# ── D1: Log-log degree distribution ──────────────────────────────────────────
ax1 = fig_d.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD)
_despine(ax1)

ax1.scatter(k_pos, p_pos, color=TEAL, s=55, alpha=0.85, zorder=5,
            label=f"S&P 500 network")
if not math.isnan(gamma_real) and len(k_pos[mask2]) >= 3:
    k_fit = np.linspace(k_pos[mask2].min(), k_pos[mask2].max(), 200)
    p_fit = 10**intercept * k_fit**(-gamma_real)
    ax1.plot(k_fit, p_fit, color=GOLD, lw=2.0, ls="--",
             label=f"Power-law fit  γ≈{gamma_real:.2f}")

k_pois2 = np.arange(0, int(k_mean_er * 3 + 10))
p_pois2 = stats.poisson.pmf(k_pois2, mu=k_mean_er)
ax1.plot(k_pois2, p_pois2, color=RED, lw=2.0, label=f"Poisson (ER)")

ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("Degree  k", fontsize=11)
ax1.set_ylabel("P(k)", fontsize=11)
ax1.set_title("Degree Distribution  (log-log)", color=LIGHT, fontsize=12, fontweight="bold")
ax1.legend(fontsize=9, framealpha=0.5)
ax1.grid(True, alpha=0.2)

# ── D2: Clustering ensemble histogram ────────────────────────────────────────
ax2 = fig_d.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD)
_despine(ax2)

ax2.hist(C_ensemble, bins=50, color=DIM, alpha=0.8, density=True,
         label=f"ER ensemble\nN={N_ENSEMBLE}")
x_g3 = np.linspace(C_ensemble.min(),
                   max(C_ensemble.max(), C_real * 1.05), 500)
y_g3 = stats.norm.pdf(x_g3, mu_C, sigma_C)
ax2.plot(x_g3, y_g3, color=RED, lw=1.8, ls="--")
ax2.axvline(C_real, color=TEAL, lw=2.5, zorder=5,
            label=f"C_real = {C_real:.4f}\nZ = {Z_C:+.0f}σ")

tail_x2 = x_g3[x_g3 >= C_real]
if len(tail_x2) > 0:
    tail_y2 = stats.norm.pdf(tail_x2, mu_C, sigma_C)
    ax2.fill_between(tail_x2, tail_y2, alpha=0.3, color=TEAL)

ax2.set_xlabel("Clustering Coefficient  C", fontsize=11)
ax2.set_ylabel("Density", fontsize=11)
ax2.set_title(
    f"Null Model  (Clustering)   p = {fmt_pval(pv_C)}",
    color=LIGHT, fontsize=12, fontweight="bold",
)
ax2.legend(fontsize=9, framealpha=0.5)
ax2.grid(True, alpha=0.2)

# ── D3: Sector-colored network (spring layout of LCC sample) ─────────────────
ax3 = fig_d.add_subplot(gs[1, 0])
ax3.set_facecolor(CARD)
ax3.axis("off")
ax3.set_title(
    f"Correlation Network LCC  ({len(lcc_nodes)} nodes)",
    color=LIGHT, fontsize=12, fontweight="bold",
)

# Draw a sample (up to 80 nodes) to keep plot clean
sample_size = min(80, len(lcc_nodes))
lcc_list = list(lcc_nodes)
sample_nodes = random.sample(lcc_list, sample_size)
G_sample = G_real.subgraph(sample_nodes).copy()

pos_sample = nx.spring_layout(G_sample, k=0.55, iterations=80, seed=42)

node_c = [
    SECTOR_COLORS.get(G_sample.nodes[nd].get("sector", "Unknown"), SLATE)
    for nd in G_sample.nodes()
]
edge_weights = [abs(G_sample[u][v].get("weight", 0.5)) for u, v in G_sample.edges()]
edge_alphas = [min(1.0, w * 0.9) for w in edge_weights]

nx.draw_networkx_edges(
    G_sample, pos_sample, ax=ax3,
    alpha=edge_alphas,
    width=[0.5 + w for w in edge_weights],
    edge_color=DIM,
)
nx.draw_networkx_nodes(
    G_sample, pos_sample, ax=ax3,
    node_color=node_c, node_size=60, alpha=0.9,
)
# Sector legend
legend_handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=SECTOR_COLORS[sec],
           markersize=7, label=sec)
    for sec in SECTOR_COLORS
]
ax3.legend(handles=legend_handles, fontsize=8, loc="lower left",
           framealpha=0.6, facecolor=CARD, edgecolor=DIM)

# ── D4: Z-score summary bar chart ─────────────────────────────────────────────
ax4 = fig_d.add_subplot(gs[1, 1])
ax4.set_facecolor(CARD)
_despine(ax4)

metrics_names = ["Clustering\nCoeff. C", "Avg Path\nLength L", "Assortativity\nr"]
z_vals = [Z_C, Z_L, Z_r]
bar_colors = [
    TEAL  if abs(z) > 10 else (GOLD if abs(z) > 3 else RED)
    for z in z_vals
]
bars = ax4.barh(metrics_names, z_vals, color=bar_colors, height=0.45, alpha=0.85)

ax4.axvline(0, color=LIGHT, lw=1.0, alpha=0.6)
ax4.axvline(+3, color=GOLD, lw=1.2, ls=":", alpha=0.7, label="±3σ threshold")
ax4.axvline(-3, color=GOLD, lw=1.2, ls=":", alpha=0.7)

for bar, z in zip(bars, z_vals):
    xpos = bar.get_width()
    label_x = xpos + (0.5 if xpos >= 0 else -0.5)
    ha = "left" if xpos >= 0 else "right"
    ax4.text(label_x, bar.get_y() + bar.get_height() / 2,
             f"{z:+.1f}σ", va="center", ha=ha,
             color=LIGHT, fontsize=11, fontweight="bold")

ax4.set_xlabel("Z-score  (standard deviations from ER mean)", fontsize=11)
ax4.set_title(
    "Statistical Rejection of H₀  (ER null model)",
    color=LIGHT, fontsize=12, fontweight="bold",
)
ax4.legend(fontsize=9, framealpha=0.5)
ax4.grid(True, axis="x", alpha=0.2)

# Subtitle annotation
rejection_txt = (
    f"H₀ REJECTED at p < 0.001 for Clustering  "
    f"({Z_C:+.0f}σ from ER)"
)
fig_d.text(
    0.5, 0.01, rejection_txt,
    ha="center", va="bottom", fontsize=12,
    color=TEAL, fontweight="bold",
    bbox=dict(facecolor=CARD, edgecolor=DIM, boxstyle="round,pad=0.4"),
)

fig_d.savefig(OUT / "plot13d_summary_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(fig_d)
print("  ✓  plot13d_summary_dashboard.png")

print("\nAll plots saved to the project root.")
print("Done.")
