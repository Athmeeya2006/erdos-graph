"""
plot1_runtime_benchmark.py
==========================
Outputs: plot1_runtime_benchmark.png

Shows:
    - Wall-clock time vs n for both algorithms on log-log axes
    - Reference slope lines  (slope=2 for naive, slope=1 for fast)
    - The divergence makes the O(n²) vs O(n+M) gap visually undeniable
"""

import random, math, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from naive_er import er_naive
from fast_er  import er_fast

random.seed(42)
np.random.seed(42)

# ── Config ────────────────────────────────────────────────────────────────────
NAVY  = "#1A3A5C"
TEAL  = "#0E7490"
RED   = "#DC2626"
GOLD  = "#D97706"
SLATE = "#64748B"
LIGHT = "#F1F5F9"

# n values where we run BOTH algorithms
n_both = [100, 200, 400, 700, 1000, 1500, 2000, 3000, 4500, 7000, 10000]

# n values where we run ONLY the fast algorithm  (naive is too slow)
n_fast_only = [20000, 40000, 80000, 150000, 300000, 600000, 1000000]

REPEATS = 5   # median over this many runs per data point
LAM     = 5   # average degree kept constant  → p = LAM / n

# ── Timing helper ────────────────────────────────────────────────────────────
def measure(func, n, p, reps=REPEATS):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        func(n, p)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

# ── Collect data ─────────────────────────────────────────────────────────────
print("Benchmarking naive vs fast (both)...")
t_naive, t_fast_shared = [], []

for n in n_both:
    p = LAM / n
    tn = measure(er_naive, n, p)
    tf = measure(er_fast,  n, p)
    t_naive.append(tn)
    t_fast_shared.append(tf)
    print(f"  n={n:>6,}  naive={tn:.5f}s  fast={tf:.5f}s  speedup={tn/tf:.1f}x")

print("\nBenchmarking fast only (large n)...")
t_fast_large = []

for n in n_fast_only:
    p = LAM / n
    tf = measure(er_fast, n, p, reps=3)
    t_fast_large.append(tf)
    print(f"  n={n:>8,}  fast={tf:.5f}s")

# ── Combine fast data ─────────────────────────────────────────────────────────
n_fast_all = n_both + n_fast_only
t_fast_all = t_fast_shared + t_fast_large

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#F8FAFC")
ax.set_facecolor(LIGHT)
ax.grid(True, which="both", color="white", linewidth=0.9, linestyle="-", zorder=0)

# Naive — red
ax.loglog(n_both, t_naive, "o-",
          color=RED, lw=2.5, markersize=8,
          markeredgecolor="white", markeredgewidth=1.5,
          label="Naive  O(n²)", zorder=4)

# Fast — full range — teal
ax.loglog(n_fast_all, t_fast_all, "s-",
          color=TEAL, lw=2.5, markersize=7,
          markeredgecolor="white", markeredgewidth=1.5,
          label="Batagelj-Brandes  O(n + M)", zorder=4)

# ── Reference slope lines ────────────────────────────────────────────────────
xr = np.array([min(n_both), max(n_both)], dtype=float)

# O(n²) — calibrated to first naive point
c2 = t_naive[0] / xr[0]**2
ax.loglog(xr, c2 * xr**2, "--", color=RED,  alpha=0.45, lw=1.6,
          label=r"Reference slope = 2  $(n^2)$")

# O(n) — calibrated to first fast point
c1 = t_fast_shared[0] / xr[0]
ax.loglog(xr, c1 * xr,    "--", color=TEAL, alpha=0.45, lw=1.6,
          label=r"Reference slope = 1  $(n)$")

# ── Annotate the divergence gap at n = 10,000 ────────────────────────────────
idx_10k = n_both.index(10000)
y_naive = t_naive[idx_10k]
y_fast  = t_fast_shared[idx_10k]

ax.annotate("", xy=(10000, y_fast), xytext=(10000, y_naive),
            arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.8))
ax.text(10000 * 1.15, math.sqrt(y_naive * y_fast),
        f"{y_naive/y_fast:.0f}×\nspeedup",
        color=GOLD, fontsize=11, fontweight="bold", va="center")

# ── Extrapolation note for n = 100K ─────────────────────────────────────────
n_100k = 100_000
t_naive_extrap = c2 * n_100k**2
ax.scatter([n_100k], [t_naive_extrap], marker="x", s=120,
           color=RED, zorder=5, linewidths=2.5)
ax.text(n_100k * 1.08, t_naive_extrap * 1.3,
        f"n=100K naive\n≈ {t_naive_extrap/60:.0f} min\n(extrapolated)",
        color=RED, fontsize=9, va="bottom")

# ── Axis formatting ───────────────────────────────────────────────────────────
ax.set_xlabel("Number of vertices  $n$", fontsize=13, color=NAVY, labelpad=10)
ax.set_ylabel("Wall-clock time (seconds)", fontsize=13, color=NAVY, labelpad=10)
ax.set_title("Runtime: Naïve O(n²) vs Batagelj-Brandes O(n + M)\n"
             r"$p = \lambda/n$ with $\lambda = 5$ (sparse regime, constant average degree)",
             fontsize=13, color=NAVY, fontweight="bold", pad=15)

ax.legend(fontsize=11, framealpha=0.95, facecolor="white",
          edgecolor=SLATE, loc="upper left")

for spine in ax.spines.values():
    spine.set_color("#CBD5E1")
ax.tick_params(colors=SLATE, labelsize=10)

plt.tight_layout()
plt.savefig("plot1_runtime_benchmark.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("\nSaved: plot1_runtime_benchmark.png")
plt.close()