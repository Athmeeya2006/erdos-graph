"""
plot2_speedup_ratio.py
======================
Outputs: plot2_speedup_ratio.png

Shows:
    - Speedup = t_naive / t_fast  plotted vs n on a log-linear scale
    - The speedup grows roughly as O(n) — confirming the asymptotic analysis
    - Annotated with exact speedup values at key n points
"""

import random, time
import numpy as np
import matplotlib.pyplot as plt
from naive_er import er_naive
from fast_er  import er_fast

random.seed(42)

NAVY  = "#1A3A5C"
TEAL  = "#0E7490"
RED   = "#DC2626"
GOLD  = "#D97706"
SLATE = "#64748B"
LIGHT = "#F1F5F9"

n_values = [100, 200, 400, 700, 1000, 1500, 2500, 4000, 6000, 8000, 10000]
LAM      = 5
REPEATS  = 5

def measure(func, n, p, reps=REPEATS):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        func(n, p)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))

print("Computing speedup ratios...")
speedups, t_naive_list, t_fast_list = [], [], []

for n in n_values:
    p  = LAM / n
    tn = measure(er_naive, n, p)
    tf = measure(er_fast,  n, p)
    speedups.append(tn / tf)
    t_naive_list.append(tn)
    t_fast_list.append(tf)
    print(f"  n={n:>6,}  speedup={tn/tf:>6.1f}x")

# ── Theory line: speedup should grow as ~ n / (lambda)
# Since naive ~ n², fast ~ n: speedup ~ n / lambda
n_arr   = np.array(n_values, dtype=float)
theory  = n_arr / (2 * LAM)   # the /2 is the constant from the exact ratio
# calibrate to match the actual data at the midpoint
mid_idx = len(n_values) // 2
scale   = speedups[mid_idx] / theory[mid_idx]
theory_scaled = theory * scale

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor("#F8FAFC")

# Left: speedup vs n
ax = axes[0]
ax.set_facecolor(LIGHT)
ax.grid(True, color="white", linewidth=0.9, zorder=0)

ax.plot(n_values, speedups, "o-",
        color=TEAL, lw=2.5, markersize=9,
        markeredgecolor="white", markeredgewidth=1.5,
        label="Measured speedup", zorder=4)

ax.plot(n_values, theory_scaled, "--",
        color=GOLD, lw=1.8, alpha=0.7,
        label=r"Theoretical $\propto n/\lambda$", zorder=3)

# Annotate key points
for n, sp in zip(n_values[::3], speedups[::3]):
    ax.annotate(f"{sp:.0f}×",
                xy=(n, sp), xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=9, color=TEAL, fontweight="bold")

ax.set_xlabel("Number of vertices  $n$", fontsize=12, color=NAVY, labelpad=8)
ax.set_ylabel("Speedup  (t_naive / t_fast)", fontsize=12, color=NAVY, labelpad=8)
ax.set_title("Speedup Ratio Grows Linearly with $n$\n"
             r"Speedup $\approx \frac{n}{2\lambda}$ because naive $\sim n^2$, fast $\sim n$",
             fontsize=12, color=NAVY, fontweight="bold", pad=12)
ax.legend(fontsize=10, framealpha=0.95, facecolor="white", edgecolor=SLATE)
ax.set_xlim(0, max(n_values) * 1.05)
ax.set_ylim(0, max(speedups) * 1.25)

for spine in ax.spines.values():
    spine.set_color("#CBD5E1")

# Right: absolute times side-by-side bar chart at selected n values
ax2 = axes[1]
ax2.set_facecolor(LIGHT)
ax2.grid(True, axis="y", color="white", linewidth=0.9, zorder=0)

selected_idx = [0, 3, 5, 7, 9, 10]   # indices into n_values
sel_n     = [n_values[i] for i in selected_idx]
sel_naive = [t_naive_list[i] for i in selected_idx]
sel_fast  = [t_fast_list[i]  for i in selected_idx]

x     = np.arange(len(sel_n))
width = 0.35

bars_naive = ax2.bar(x - width/2, sel_naive, width,
                     color=RED,  alpha=0.85, label="Naive O(n²)",
                     edgecolor="white", linewidth=0.8, zorder=3)
bars_fast  = ax2.bar(x + width/2, sel_fast,  width,
                     color=TEAL, alpha=0.85, label="Batagelj-Brandes O(n+M)",
                     edgecolor="white", linewidth=0.8, zorder=3)

# Label bars with their times
for bar in bars_naive:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.001,
             f"{h:.3f}s", ha="center", va="bottom", fontsize=7.5, color=RED)
for bar in bars_fast:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.001,
             f"{h:.4f}s", ha="center", va="bottom", fontsize=7.5, color=TEAL)

ax2.set_xticks(x)
ax2.set_xticklabels([f"{n:,}" for n in sel_n], fontsize=9, color=SLATE)
ax2.set_xlabel("Number of vertices  $n$", fontsize=12, color=NAVY, labelpad=8)
ax2.set_ylabel("Time (seconds)", fontsize=12, color=NAVY, labelpad=8)
ax2.set_title("Absolute Runtime Comparison\n(both algorithms, same n values)",
              fontsize=12, color=NAVY, fontweight="bold", pad=12)
ax2.legend(fontsize=10, framealpha=0.95, facecolor="white", edgecolor=SLATE)

for spine in ax2.spines.values():
    spine.set_color("#CBD5E1")

fig.suptitle("Speedup Analysis: Naïve vs Batagelj-Brandes",
             fontsize=14, color=NAVY, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig("plot2_speedup_ratio.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: plot2_speedup_ratio.png")
plt.close()