"""Counts connected weight-4 sets S' (k=4, |S'|=10, K^2=16 I) on Dic_n with c=y.

  python enumerate_dicn_weight4.py --structured 5 6 7 8 9
  python enumerate_dicn_weight4.py --unrestricted 3 4 5        (all subsets of G\\{1,c})
  python enumerate_dicn_weight4.py --unrestricted 6            (about 2 minutes)

--unrestricted tests every subset with exactly 16 nonzero entries per row, using only
the exact group-ring identity; it assumes nothing from the manuscript.
--structured enumerates sets of the two types allowed by Theorem gd:fourier(b):
  Type I  (P empty): R_u (2 y-pairs in xA), T, R_a;  Type II (R_u empty): P (2 +-pairs in A), T, R_a.
Fourier conditions are used only as a FILTER; every survivor is re-verified exactly.
"""
import sys
import numpy as np
from itertools import combinations, product
from common import *


def unrestricted(n):
    G = dicyclic(n)
    c = ((n,), 0)
    cand = [e for e in G.els if e != G.one and e != c]
    sols = 0
    for mask in range(1 << len(cand)):
        S = [cand[i] for i in range(len(cand)) if mask >> i & 1]
        Ss = set(S)
        if len(Ss | {G.inv(s) for s in S}) != 16:
            continue
        if square_is_scalar(G, S, 4) and generates(G, S + [c]):
            sols += 1
    return sols


def structured(n):
    G = dicyclic(n)
    m, c = 2 * n, ((n,), 0)
    J = np.arange(m)
    ch = np.exp(2j * np.pi * np.outer(J, np.arange(m)) / m)
    ev, od = J % 2 == 0, J % 2 == 1
    apairs = [(a, m - a) for a in range(1, n)]
    ypairs = [(r, (r + n) % m) for r in range(n)]
    Tall = []
    for sz in range(7):
        for comb in combinations(range(len(apairs)), sz):
            for sg in product((0, 1), repeat=sz):
                Tall.append(tuple(apairs[i][s] for i, s in zip(comb, sg)))
    sT = [ch[:, list(T)].imag.sum(axis=1) if T else np.zeros(m) for T in Tall]
    cand = []
    for Rp in combinations(range(n), 2):                       # Type I
        Ru = [r for p in Rp for r in ypairs[p]]
        SRu = np.abs(ch[:, Ru].sum(axis=1))
        rem = [p for p in range(n) if p not in Rp]
        for T, s in zip(Tall, sT):
            if not np.allclose(4 * s[ev] ** 2 + SRu[ev] ** 2, 16, atol=1e-7):
                continue
            need = 6 - len(T)
            if need < 0 or need > len(rem):
                continue
            for comb in combinations(rem, need):
                for sg in product((0, 1), repeat=need):
                    Ra = [ypairs[p][q] for p, q in zip(comb, sg)]
                    SRa = np.abs(ch[:, Ra].sum(axis=1)) if Ra else np.zeros(m)
                    if np.allclose(4 * s[od] ** 2 + 4 * SRa[od] ** 2, 16, atol=1e-7):
                        cand.append(("I", [((r,), 1) for r in Ru] + [((t,), 0) for t in T]
                                     + [((r,), 1) for r in Ra]))
    for comb2 in combinations(range(len(apairs)), 2):          # Type II
        P = [x for i in comb2 for x in apairs[i]]
        p = ch[:, P].sum(axis=1).real
        for T, s in zip(Tall, sT):
            if any(t in P or (m - t) % m in P for t in T):
                continue
            if not (np.allclose(p * s, 0, atol=1e-7) and np.allclose((p[ev] + 2 * s[ev]) ** 2, 16, atol=1e-7)):
                continue
            need = 6 - len(T)
            if need < 0 or need > n:
                continue
            for comb in combinations(range(n), need):
                for sg in product((0, 1), repeat=need):
                    Ra = [ypairs[pp][q] for pp, q in zip(comb, sg)]
                    SRa = np.abs(ch[:, Ra].sum(axis=1)) if Ra else np.zeros(m)
                    if (np.allclose(p[od] * SRa[od], 0, atol=1e-7)
                            and np.allclose((p[od] + 2 * s[od]) ** 2 + 4 * SRa[od] ** 2, 16, atol=1e-7)):
                        cand.append(("II", [((a,), 0) for a in P] + [((t,), 0) for t in T]
                                     + [((r,), 1) for r in Ra]))
    good = [(t, S) for t, S in cand if square_is_scalar(G, S, 4) and generates(G, S + [c])]
    spurious = len(cand) - sum(1 for t, S in cand if square_is_scalar(G, S, 4))
    return len(good), sum(t == "I" for t, _ in good), sum(t == "II" for t, _ in good), spurious


if __name__ == "__main__":
    mode = sys.argv[1]
    for n in map(int, sys.argv[2:]):
        if mode == "--unrestricted":
            print(f"Dic_{n}: connected weight-4 sets (all subsets tested) = {unrestricted(n)}", flush=True)
        else:
            tot, t1, t2, sp = structured(n)
            print(f"Dic_{n}: connected weight-4 sets = {tot}  (Type I: {t1}, Type II: {t2}; "
                  f"filter false positives removed by exact check: {sp})", flush=True)
