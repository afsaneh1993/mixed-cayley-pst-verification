"""Cross-check of Table tab:ab by a second method.
For every isomorphism class of connected non-normal PST graphs on G (G of order 12 or 16), the class is declared an abelian Cayley graph
by t7_ab16_exact.py when its nauty certificate equals that of a mixed Cayley graph of an abelian group of the same order.
Here the same question is decided by Aut: the graph is a Cayley graph of an abelian group iff Aut contains an ABELIAN subgroup acting REGULARLY.
All of Aut is generated from the nauty generators (permutations of the vertex set) and a depth-first search looks for such a subgroup.
Agreement of the two methods is reported class by class."""
import sys, time
import numpy as np
import pynauty
from collections import Counter
from lib import *
from t5_tabnn import groups
from t7_ab16_exact import abelian_certs, pst_targets_exact

def aut_gens(H):
    n = H.shape[0]
    adj = {i: [] for i in range(n)}
    for g in range(n):
        for h in range(n):
            if abs(H[g, h] - 1) < 1e-9 or abs(H[g, h] - 1j) < 1e-9:
                adj[g].append(h)
    gr = pynauty.Graph(n, directed=True, adjacency_dict=adj)
    gens, g1, g2, orb, norb = pynauty.autgrp(gr)
    return [tuple(p) for p in gens], int(round(g1 * 10 ** g2))

def closure(gens, n, cap=200000):
    idt = tuple(range(n)); seen = {idt}; frontier = [idt]
    while frontier:
        nf = []
        for p in frontier:
            for g in gens:
                q = tuple(g[p[i]] for i in range(n))      # g o p
                if q not in seen:
                    seen.add(q); nf.append(q)
                    if len(seen) > cap: return None
        frontier = nf
    return seen

def compose(a, b):  # a o b
    return tuple(a[b[i]] for i in range(len(a)))

def has_abelian_regular(elements, n):
    idt = tuple(range(n))
    ffree = [p for p in elements if p != idt and all(p[i] != i for i in range(n))]
    ffset = set(ffree)
    def commute(a, b): return compose(a, b) == compose(b, a)
    def generate(R, g):
        new = set(R); cur = set(R); p = g
        # <R, g> for commuting g: union of g^k R
        pw = g; mult = []
        while True:
            mult.append(pw)
            pw = compose(g, pw)
            if pw in R or pw == idt:
                break
            if len(mult) > n: return None
        out = set()
        for m in mult + [idt]:
            for r in R: out.add(compose(m, r))
        return out
    def dfs(R):
        if len(R) == n: return True
        orbit0 = {r[0] for r in R}
        for g in ffree:
            if g in R or g[0] in orbit0: continue
            if not all(commute(g, r) for r in R): continue
            newR = generate(R, g)
            if newR is None: continue
            if any((p != idt and p not in ffset) for p in newR): continue    # semiregular
            if len({p[0] for p in newR}) != len(newR): continue
            if len(newR) > n or n % len(newR): continue
            if dfs(frozenset(newR)): return True
        return False
    return dfs(frozenset([idt]))

def run(order, names, cap=200000):
    t0 = time.time(); certs = abelian_certs(order)
    print("order", order, ": abelian certificates:", len(certs), flush=True)
    for name in names:
        mods, f, y = groups()[name]; G = Grp(mods, f, y); n = G.n
        nonid = [s for s in range(n) if s != G.one]; e1 = np.zeros(n, dtype=complex); e1[G.one] = 1
        conj = np.array([[G.mt[G.mt[G.inv[g], s], g] for s in range(n)] for g in range(n)])
        reps = {}
        for mask in range(1 << (n - 1)):
            S = [nonid[i] for i in range(n - 1) if mask >> i & 1]; Sset = set(S)
            if not any(G.inv[s] in Sset for s in S): continue
            H = G.H(S); T = pst_targets_exact(G, S, H, e1)
            if not T: continue
            if all(set(conj[g, s] for s in S) == Sset for g in range(n)): continue
            if len(gen_subgroup(G, S)) != n: continue
            c = digraph_cert(H)
            if c not in reps: reps[c] = H
        agree = disagree = skipped = 0; by = Counter()
        for c, H in reps.items():
            gens, order_aut = aut_gens(H)
            els = closure(gens, n, cap) if order_aut <= cap else None
            if els is None: skipped += 1; continue
            found = has_abelian_regular(els, n)
            declared = c in certs
            if found == declared: agree += 1
            else: disagree += 1; print("  DISAGREEMENT", name, "Aut", order_aut, "found", found, "declared", declared)
            by[(declared, found)] += 1
        print(f"{name}: classes {len(reps)}; abelian-Cayley by certificate: {sum(1 for c in reps if c in certs)}; by regular abelian subgroup in Aut: "
              f"{by[(True, True)] + by[(False, True)]}; agreement {agree}, disagreement {disagree}, skipped (|Aut| too large) {skipped}  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    run(int(sys.argv[1]), sys.argv[2:])
