"""
plot4_iteration_count.py
=========================
Outputs: plot4_iteration_count.png

Shows:
    - Exact theoretical iteration counts for both algorithms
    - Naive: C(n,2) = n(n-1)/2  (all pairs, no randomness)
    - Fast:  E[M] = C(n,2)*p = lambda*n/2  (expected edges only)
    - Log-log scale exposes the factor-of-n gap clearly
    - Also shows a table of exact numbers at key n values
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

NAVY  = "#1A3A5C"
TEAL  = "#0E7490"
RED   = "#DC2626"
GOLD  = "#D97706"
SLATE = "#64748B"
LIGHT = "#F1F5F9"

LAM = 5  # constant average degree

n_vals = np.logspace(2, 7, 200)   # 100 to 10,000,000

naive_iters = n_vals * (n_vals - 1) / 2          # C(n,2)
fast_iters  = n_vals * (n_vals - 1) / 2 * (LAM / (n_vals - 1))  # E[M] = lambda*n/2

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor("#F8FAFC")

# ── Left: log-log plot ────────────────────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(LIGHT)
ax.grid(True, which="both", color="white", linewidth=0.9, zorder=0)

ax.loglog(n_vals, naive_iters, "-",
          color=RED, lw=2.8, label=r"Naive: $\binom{n}{2} = \frac{n(n-1)}{2}$ iterations",
          zorder=3)

ax.loglog(n_vals, fast_iters, "-",
          color=TEAL, lw=2.8, label=r"Batagelj-Brandes: $E[M] = \frac{\lambda n}{2}$ iterations",
          zorder=3)

# Vertical lines at key n values with gap annotations
for n_mark in [1_000, 10_000, 100_000, 1_000_000]:
    ni = n_mark * (n_mark - 1) / 2
    fi = LAM * n_mark / 2
    ratio = ni / fi

    ax.axvline(n_mark, color=SLATE, lw=0.8, linestyle=":", alpha=0.5)
    ax.scatter([n_mark], [ni], color=RED,  s=50, zorder=5)
    ax.scatter([n_mark], [fi], color=TEAL, s=50, zorder=5)

    ax.text(n_mark * 1.08, (ni * fi)**0.5,
            f"{ratio:.0f}×" if ratio < 1000 else f"{ratio/1000:.0f}K×",
            fontsize=8.5, color=GOLD, fontweight="bold", va="center")

ax.set_xlabel("Number of vertices  $n$", fontsize=12, color=NAVY, labelpad=8)
ax.set_ylabel("Number of iterations", fontsize=12, color=NAVY, labelpad=8)
ax.set_title("Theoretical Iteration Count\n"
             r"Gap $= \frac{n-1}{\lambda} \approx \frac{n}{\lambda}$ — grows linearly with $n$",
             fontsize=12, color=NAVY, fontweight="bold", pad=12)
ax.legend(fontsize=10, framealpha=0.95, facecolor="white", edgecolor=SLATE,
          loc="upper left")

ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x:.0e}".replace("e+0", "e").replace("e+", "e")))
for spine in ax.spines.values():
    spine.set_color("#CBD5E1")
ax.tick_params(colors=SLATE, labelsize=9)

# ── Right: table-style bar chart ─────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(LIGHT)
ax2.grid(True, axis="x", color="white", linewidth=0.9, zorder=0)

table_n = [1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000]
table_naive = [n*(n-1)//2 for n in table_n]
table_fast  = [int(LAM * n / 2) for n in table_n]

y_pos = np.arange(len(table_n))
height = 0.35

bars_n = ax2.barh(y_pos + height/2, table_naive, height,
                  color=RED,  alpha=0.8, label="Naive  O(n²)",
                  edgecolor="white", linewidth=0.8, zorder=3)
bars_f = ax2.barh(y_pos - height/2, table_fast, height,
                  color=TEAL, alpha=0.8, label="Batagelj-Brandes  O(n+M)",
                  edgecolor="white", linewidth=0.8, zorder=3)

# Label each bar
for bar, val in zip(bars_n, table_naive):
    w = bar.get_width()
    label = f"{val:,}" if val < 1e6 else f"{val/1e9:.2f}B" if val > 1e9 else f"{val/1e6:.1f}M"
    ax2.text(w * 1.02, bar.get_y() + bar.get_height()/2,
             label, va="center", fontsize=8, color=RED)

for bar, val in zip(bars_f, table_fast):
    w = bar.get_width()
    label = f"{val:,}" if val < 1e6 else f"{val/1e6:.1f}M"
    ax2.text(w * 1.02, bar.get_y() + bar.get_height()/2,
             label, va="center", fontsize=8, color=TEAL)

ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"n = {n:,}" for n in table_n], fontsize=10, color=SLATE)
ax2.set_xlabel("Iterations required", fontsize=12, color=NAVY, labelpad=8)
ax2.set_title(f"Exact Iteration Counts  (λ = {LAM})\nLog scale — red bars extend far off screen",
              fontsize=12, color=NAVY, fontweight="bold", pad=12)
ax2.set_xscale("log")
ax2.legend(fontsize=10, framealpha=0.95, facecolor="white", edgecolor=SLATE)

for spine in ax2.spines.values():
    spine.set_color("#CBD5E1")
ax2.tick_params(colors=SLATE, labelsize=9)

fig.suptitle(r"Iteration Count: $O(n^2)$ vs $O(n+M)$ — Theory",
             fontsize=14, color=NAVY, fontweight="bold", y=1.01)

plt.tight_layout()
plt.savefig("plot4_iteration_count.png",
            dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved: plot4_iteration_count.png")
plt.close()