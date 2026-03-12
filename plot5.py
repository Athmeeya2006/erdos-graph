"""
plot5_fast_scaling.py
======================
Outputs: plot5_fast_scaling.png

Shows:
    - The Batagelj-Brandes algorithm scaling to n = 1,000,000
    - Linear time growth confirms O(n) behaviour
    - Inset: the naive algorithm would need hours for these same n values
"""

import random, time
import numpy as np
import matplotlib.pyplot as plt
from fast_er import er_fast

random.seed(42)

NAVY  = "#1A3A5C"
TEAL  = "#0E7490"
RED   = "#DC2626"
GOLD  = "#D97706"
SLATE = "#64748B"
LIGHT = "#F1F5F9"

n_values = [
    1_000, 2_000, 5_000, 10_000, 20_000, 50_000,
    100_000, 200_000, 400_000, 700_000, 1_000_000
]
LAM = 5

print("Benchmarking fast algorithm for large n...")
times, edge_counts = [], []

for n in n_values:
    p = LAM / n
    t0  = time.perf_counter()
    adj = er_fast(n, p)
    t   = time.perf_counter() - t0
    M   = sum(len(nb) for nb in adj) // 2
    times.append(t)
    edge_counts.append(M)
    print(f"  n={n:>9,}  M={M:>8,}  time={t:.4f}s")

# O(n) reference
n_arr  = np.array(n_values, dtype=float)
t_arr  = np.array(times)
# fit linear: t = c * n
c_fit  = np.mean(t_arr / n_arr)
t_theory = c_fit * n_arr

# Extrapolate naive using the actual measured time at n=10,000
# t_naive ~ c * n^2  =>  c = t_measured / n^2
# We run a quick single measurement here to get the true constant
import time as _time
from naive_er import er_naive as _er_naive
_n_cal = 3000
_p_cal = LAM / _n_cal
_t0 = _time.perf_counter()
_er_naive(_n_cal, _p_cal)
_t_cal = _time.perf_counter() - _t0
c_naive_fit = _t_cal / (_n_cal ** 2)
print(f"Naive calibration: {_t_cal:.4f}s at n={_n_cal}  →  c = {c_naive_fit:.2e}")
t_naive_extrap = c_naive_fit * n_arr**2

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor("#F8FAFC")

# Left: actual timing
ax = axes[0]
ax.set_facecolor(LIGHT)
ax.grid(True, color="white", linewidth=0.9, zorder=0)

ax.plot(n_values, times, "s-",
        color=TEAL, lw=2.5, markersize=8,
        markeredgecolor="white", markeredgewidth=1.5,
        label="Measured time", zorder=4)

ax.plot(n_arr, t_theory, "--",
        color=GOLD, lw=1.8, alpha=0.7,
        label=r"$O(n)$ fit  ($t = cn$)", zorder=3)

# Annotate last point
ax.annotate(f"n = 1,000,000\n{times[-1]:.2f} s",
            xy=(n_values[-1], times[-1]),
            xytext=(-130, -30), textcoords="offset points",
            fontsize=10, color=TEAL, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))

ax.set_xlabel("Number of vertices  $n$", fontsize=12, color=NAVY, labelpad=8)
ax.set_ylabel("Wall-clock time (seconds)", fontsize=12, color=NAVY, labelpad=8)
ax.set_title("Batagelj-Brandes Scales to  n = 1,000,000\n"
             "Linear time growth confirms O(n + M) complexity",
             fontsize=12, color=NAVY, fontweight="bold", pad=12)
ax.legend(fontsize=11, framealpha=0.95, facecolor="white", edgecolor=SLATE)
ax.set_xlim(0, max(n_values) * 1.05)
ax.set_ylim(0, max(times) * 1.2)

for spine in ax.spines.values():
    spine.set_color("#CBD5E1")
ax.tick_params(colors=SLATE, labelsize=9)

# Right: naive vs fast extrapolated comparison
ax2 = axes[1]
ax2.set_facecolor(LIGHT)
ax2.grid(True, which="both", color="white", linewidth=0.9, zorder=0)

ax2.loglog(n_arr, t_arr, "s-",
           color=TEAL, lw=2.5, markersize=7,
           markeredgecolor="white", markeredgewidth=1.5,
           label="Fast  O(n+M)  — measured", zorder=4)

ax2.loglog(n_arr, t_naive_extrap, "x--",
           color=RED, lw=2, markersize=9, markeredgewidth=2,
           label="Naive  O(n²)  — extrapolated", zorder=3)

# Shade the "infeasible" region
ax2.fill_between(n_arr, 60, t_naive_extrap,
                 where=(t_naive_extrap > 60),
                 color=RED, alpha=0.08, label="Naive > 1 minute (infeasible)")

ax2.axhline(60, color=RED, lw=1.2, linestyle=":", alpha=0.6)
ax2.text(n_arr[2], 60 * 1.4, "1 minute threshold",
         fontsize=9, color=RED, alpha=0.8)

ax2.set_xlabel("Number of vertices  $n$", fontsize=12, color=NAVY, labelpad=8)
ax2.set_ylabel("Time (seconds)", fontsize=12, color=NAVY, labelpad=8)
ax2.set_title("Why the Naive Algorithm is Dead Above n = 50K\n"
              "Log-log with naive extrapolated (crosses 1 minute at n ≈ 30K)",
              fontsize=12, color=NAVY, fontweight="bold", pad=12)
ax2.legend(fontsize=10, framealpha=0.95, facecolor="white", edgecolor=SLATE)

for spine in ax2.spines.values():
    spine.set_color("#CBD5E1")
ax2.tick_params(colors=SLATE, labelsize=9)

fig.suptitle("Phase 2 — Batagelj-Brandes Handles Graphs the Naïve Algorithm Cannot",
             fontsize=14, color=NAVY, fontweight="bold", y=1.01)

plt.tight_layout()
plt.savefig("plot5_fast_scaling.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: plot5_fast_scaling.png")
plt.close()