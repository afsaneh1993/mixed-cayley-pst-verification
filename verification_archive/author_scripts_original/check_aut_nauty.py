"""Automorphism orders with nauty (pynauty) for Remark S1.2 of the supplementary note:
|Aut Cay(Dic_n, S)| = n 2^(n+1) for n = 3..12 and |Aut Cay(SD_{2^k}, S)| = 2^7, 2^12, 2^21, 2^38 for k = 4..7.
A mixed graph is encoded as a digraph (undirected edge = two opposite arcs, arc = one arc), so its
automorphisms are exactly the digraph automorphisms.  Requires: pip install pynauty.
Usage: python3 check_aut_nauty.py            (exit status 0 iff all checks pass)"""
import sys
import numpy as np
from pynauty import Graph, autgrp
from common import *

def aut_nauty(G, S):
    H = hermitian(G, S)
    N = len(H)
    adj = {i: [j for j in range(N) if abs(H[i, j] - 1) < 1e-9 or abs(H[i, j] - 1j) < 1e-9] for i in range(N)}
    g = Graph(N, directed=True, adjacency_dict=adj)
    _, g1, g2, _, _ = autgrp(g)          # |Aut| = g1 * 10**g2
    return int(round(g1 * 10 ** g2))

fails = []
def check(name, cond):
    print(("PASS  " if cond else "FAIL  ") + name, flush=True)
    if not cond: fails.append(name)

for n in range(3, 13):
    G = dicyclic(n); c = ((n,), 0)
    S = [((0,), 1), ((n,), 1), ((1,), 1), c]
    a = aut_nauty(G, S)
    check(f"Dic_{n}: |Aut| = n 2^(n+1) = {n * 2 ** (n + 1)} (nauty: {a})", a == n * 2 ** (n + 1))
for k, e in ((4, 7), (5, 12), (6, 21), (7, 38)):
    G = semidihedral(k); c = ((2 ** (k - 2),), 0)
    S = [((0,), 1), ((2 ** (k - 2),), 1), ((1,), 1), c]
    a = aut_nauty(G, S)
    check(f"SD_{2 ** k}: |Aut| = 2^{e} (nauty: {a} = 2^{a.bit_length() - 1 if a & (a - 1) == 0 else 'not a power of 2'})", a == 2 ** e)
print("\nALL CHECKS PASSED" if not fails else f"\n{len(fails)} CHECK(S) FAILED: {fails}")
sys.exit(1 if fails else 0)
