"""
naive_er.py
===========
Naive O(n²) Erdos-Renyi G(n, p) Generator

Algorithm:
    For every pair (i, j) with i < j  — that is C(n,2) = n(n-1)/2 pairs —
    flip an independent Bernoulli(p) coin and add the edge if heads.

Complexity:
    Time  : O(n²)   — visits every possible pair regardless of p
    Space : O(n + M) — adjacency list stores only actual edges
"""

import random
import math
import time


# ─────────────────────────────────────────────────────────────────────────────
# CORE ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

def er_naive(n: int, p: float) -> list[list[int]]:
    """
    Generate G(n, p) using the naive O(n²) algorithm.

    Parameters
    ----------
    n : int   — number of vertices, labeled 0 … n-1
    p : float — edge probability in [0, 1]

    Returns
    -------
    adj : list[list[int]]
        Adjacency list. adj[v] = list of all neighbours of vertex v.
    """
    # Step 1: initialise empty adjacency list — O(n)
    adj = [[] for _ in range(n)]

    # Step 2: iterate over ALL C(n,2) = n(n-1)/2 unordered pairs — O(n²)
    for i in range(n):
        for j in range(i + 1, n):      # j > i ensures each pair visited once
            if random.random() < p:    # Bernoulli(p) coin flip
                adj[i].append(j)
                adj[j].append(i)

    return adj


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_edges(adj):
    """Count edges. Each edge stored twice in adj list."""
    return sum(len(nb) for nb in adj) // 2


def average_degree(adj):
    n = len(adj)
    return sum(len(nb) for nb in adj) / n if n > 0 else 0


# ─────────────────────────────────────────────────────────────────────────────
# DEMO — run directly to see it work
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    print("=" * 55)
    print("  NAIVE O(n²) ER GRAPH GENERATOR — DEMO")
    print("=" * 55)

    test_cases = [
        (100,  0.05),
        (500,  0.01),
        (1000, 0.005),
        (3000, 0.002),
        (6000, 0.001),
    ]

    print(f"\n{'n':>6}  {'p':>8}  {'λ=np':>6}  {'edges':>8}  {'avg_deg':>8}  {'time(s)':>9}")
    print("-" * 57)

    for n, p in test_cases:
        t0  = time.perf_counter()
        adj = er_naive(n, p)
        t   = time.perf_counter() - t0

        lam = (n - 1) * p
        M   = count_edges(adj)
        deg = average_degree(adj)

        print(f"{n:>6}  {p:>8.4f}  {lam:>6.2f}  {M:>8}  {deg:>8.4f}  {t:>9.5f}")

    print()
    print("Note: for n = 100,000 this would take ~minutes.")
    print("That is the O(n²) bottleneck this phase demonstrates.")