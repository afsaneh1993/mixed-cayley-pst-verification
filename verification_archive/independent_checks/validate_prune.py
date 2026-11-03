"""Validation of the pruning used in search_noncyclic_k4.py: on Dic(Z6 x Z2, y), all four y and all three c in A[2]\\{0},
the pruned search (u = 4 undirected, 6 arcs) and the enumeration that uses ONLY the row-count condition (wenum_exact.py)
must produce exactly the same family of sets S' (with c not in S').  Both are exact (integer arithmetic)."""
import sys
from lib import *
from wenum_exact import enum_weighing_exact, to_set
from search_noncyclic_k4 import GROUPS, build, enumerate_sets, all_y


def pruned_sets(G):
    found, _ = enumerate_sets(G, 4)
    return {frozenset(int(s) for s in range(G.n) if Wr[s] == 1 or Wi[s] == 1) for Wr, Wi in found}


def main(name):
    mods = GROUPS[name]
    zero = tuple(0 for _ in mods)
    allok = True
    for y in all_y(name):
        G = build(mods, y)
        A2 = [a for a in G.Aels if G.add(a, a) == zero and a != zero]
        P = pruned_sets(G)
        for c in A2:
            ci = G.idx[(c, 0)]
            sols, pairs, tot = enum_weighing_exact(G, ci, 4)
            U = {frozenset(to_set(s, pairs)) for s in sols}
            Pc = {S for S in P if ci not in S}
            ok = (U == Pc)
            allok &= ok
            print(f"{name} y={y} c={c}: unpruned candidates {tot}, sets {len(U)}; pruned-search sets avoiding c {len(Pc)}; identical: {ok}", flush=True)
    print("PRUNING VALIDATED" if allok else "PRUNING MISMATCH")


if __name__ == "__main__":
    main(sys.argv[1])
