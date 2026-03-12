"""
fast_er.py
==========
Batagelj-Brandes O(n + M) Erdos-Renyi G(n, p) Generator

Reference:
    Batagelj, V. & Brandes, U. (2005).
    "Efficient Generation of Large Random Networks."
    Physical Review E, 71(3), 036113.

Key Mathematical Idea
---------------------
Label every possible edge with an integer index  e = 0, 1, ..., C(n,2)-1

    Index  e  corresponds to the vertex pair  (row i, col j)  where:
        i = floor( (1 + sqrt(1 + 8e)) / 2 )
        j = e - C(i, 2)  =  e - i*(i-1)//2

Instead of visiting every index and flipping a coin, ask:
"How many indices do I SKIP before the next edge that IS included?"

Since each edge is included independently with probability p, the number
of consecutive failures before the next success is Geometric(p):

    P(skip = k) = (1-p)^k * p   for k = 0, 1, 2, ...

Sampling a Geometric(p) random variable in O(1):
    Given U ~ Uniform(0,1):
        skip = floor( log(U) / log(1-p) )

This gives the index of the next included edge directly, with no wasted
iterations over edges that are not included.

Complexity:
    Time  : O(n + M)  — one iteration per included edge, O(n) initialisation
    Space : O(n + M)
"""

import random
import math
import time


# ─────────────────────────────────────────────────────────────────────────────
# INDEX → PAIR CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def _index_to_pair(e: int) -> tuple[int, int]:
    """
    Convert flat edge index e to vertex pair (i, j) with i > j.

    Enumeration (lower-triangular, row-major):
        e=0  -> (1, 0)
        e=1  -> (2, 0)
        e=2  -> (2, 1)
        e=3  -> (3, 0)
        e=4  -> (3, 1)
        e=5  -> (3, 2)
        ...

    Formula:
        i = floor( (1 + sqrt(1 + 8e)) / 2 )
        j = e - i*(i-1)//2

    Derivation:
        Row i starts at flat index C(i,2) = i*(i-1)//2.
        Find the largest i such that i*(i-1)//2 <= e.
        Solve the quadratic: i ~ (1 + sqrt(1+8e)) / 2.
    """
    i = int(math.floor((1.0 + math.sqrt(1.0 + 8.0 * e)) / 2.0))
    j = e - i * (i - 1) // 2
    return i, j


# ─────────────────────────────────────────────────────────────────────────────
# CORE ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

def er_fast(n: int, p: float) -> list[list[int]]:
    """
    Generate G(n, p) using the Batagelj-Brandes O(n + M) algorithm.

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

    # Edge cases
    if p <= 0.0:
        return adj
    if p >= 1.0:
        for i in range(n):
            for j in range(i + 1, n):
                adj[i].append(j)
                adj[j].append(i)
        return adj

    max_edges = n * (n - 1) // 2          # C(n, 2) — total possible edges
    log1mp    = math.log(1.0 - p)         # precompute  log(1-p)  once

    # Step 2: geometric skip loop — O(M) iterations total
    e = -1   # current edge index; start just before index 0

    while True:
        # Sample skip ~ Geometric(p) in O(1)
        # U ~ Uniform(0,1)
        # skip = floor( log(U) / log(1-p) )
        # 1.0 - random.random() gives (0.0, 1.0] — log is never called on 0
        U    = 1.0 - random.random()
        skip = int(math.floor(math.log(U) / log1mp))

        e += skip + 1      # jump to next included edge

        if e >= max_edges:
            break          # past the last possible edge

        # Step 3: decode flat index e to vertex pair (i, j)
        i, j = _index_to_pair(e)

        adj[i].append(j)
        adj[j].append(i)

    return adj


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def count_edges(adj):
    return sum(len(nb) for nb in adj) // 2


def average_degree(adj):
    n = len(adj)
    return sum(len(nb) for nb in adj) / n if n > 0 else 0


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    print("=" * 65)
    print("  BATAGELJ-BRANDES O(n+M) ER GRAPH GENERATOR — DEMO")
    print("=" * 65)

    test_cases = [
        (100,     0.05),
        (1_000,   0.005),
        (10_000,  0.0005),
        (100_000, 0.00005),
        (500_000, 0.00001),
    ]

    print(f"\n{'n':>8}  {'p':>10}  {'λ=np':>6}  {'edges':>10}  {'avg_deg':>8}  {'time(s)':>9}")
    print("-" * 65)

    for n, p in test_cases:
        t0  = time.perf_counter()
        adj = er_fast(n, p)
        t   = time.perf_counter() - t0

        lam = (n - 1) * p
        M   = count_edges(adj)
        deg = average_degree(adj)

        print(f"{n:>8,}  {p:>10.6f}  {lam:>6.2f}  {M:>10,}  {deg:>8.4f}  {t:>9.5f}")

    print()
    print("n = 500,000 generated in seconds.")
    print("Naive would need ~125 billion coin flips for that.")