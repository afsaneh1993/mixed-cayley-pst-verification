"""Reproduces the rows of Table tab:nn (and, with --abelian, the abelian-Cayley status of
Table tab:ab) for the non-abelian groups with an abelian subgroup of index 2 and order <= 16.

  python verify_nn_tables.py S3 D4 Q8 D5 D6 Dic3 D7
  python verify_nn_tables.py D8 Q16 SD16 M16 Z4sdZ4 Z2xD4 Z2xQ8 Z4oD4 Z4Z2sdZ2 --abelian   (about 15 min)

Per group it prints: number of sets S in G\\{1} with S cap S^-1 nonempty; sets with PST (from
vertex 1 to some other vertex, any time); non-normal sets with PST; connected non-normal sets
with PST; those that are undirected; isomorphism classes of the connected non-normal ones.
PST test: integral spectrum, then U(t) at t = pi*l/(2*gamma), l = 1,2,3 (Theorem of the paper
on periodicity/times); floating point with tolerance 1e-7.  --abelian additionally compares each
class with all mixed Cayley graphs of all abelian groups of the same order.
"""
import sys, math, collections
import numpy as np
from itertools import product
from common import *


def spec_group(mods, imgs, y):
    A = list(product(*[range(m) for m in mods]))
    add = lambda a, b: tuple((x + z) % m for x, z, m in zip(a, b, mods))
    f = {}
    for a in A:
        v = tuple(0 for _ in mods)
        for ai, g in zip(a, imgs):
            for _ in range(ai):
                v = add(v, g)
        f[a] = v
    return Group(mods, lambda a: f[a], y)


GROUPS = {
    "S3": lambda: cyclic_dihedral(3), "D4": lambda: cyclic_dihedral(4), "Q8": lambda: dicyclic(2),
    "D5": lambda: cyclic_dihedral(5), "D6": lambda: cyclic_dihedral(6), "Dic3": lambda: dicyclic(3),
    "D7": lambda: cyclic_dihedral(7), "D8": lambda: cyclic_dihedral(8), "Q16": lambda: dicyclic(4),
    "SD16": lambda: semidihedral(4), "M16": lambda: Group([8], lambda a: ((5 * a[0]) % 8,), (0,)),
    "Z4sdZ4": lambda: spec_group([4, 2], ((1, 1), (0, 1)), (0, 1)),   # Z4 x| Z4
    "Z2xD4": lambda: spec_group([4, 2], ((3, 0), (0, 1)), (0, 0)),
    "Z2xQ8": lambda: spec_group([4, 2], ((3, 0), (0, 1)), (2, 0)),
    "Z4oD4": lambda: spec_group([4, 2], ((1, 0), (2, 1)), (0, 0)),   # central product, centre Z4
    "Z4Z2sdZ2": lambda: spec_group([4, 2], ((1, 1), (0, 1)), (0, 0)),  # (Z4 x Z2) x| Z2
}
ABELIAN = {6: [[6]], 8: [[8], [4, 2], [2, 2, 2]], 10: [[10]], 12: [[12], [6, 2]],
           14: [[14]], 16: [[16], [8, 2], [4, 4], [4, 2, 2], [2, 2, 2, 2]]}


def pst(G, S):
    H = hermitian(G, S)
    w, V = np.linalg.eigh(H)
    if np.max(np.abs(w - np.round(w))) > 1e-7:
        return False
    Ss = set(S)
    zeta = sum(1 for s in S if G.inv(s) in Ss)
    d = 0
    for t in np.round(w).astype(int):
        d = math.gcd(d, abs(int(t) - zeta))
    if d == 0:
        return False
    for l in (1, 2, 3):
        t = math.pi * l / (2 * d)
        U = (V * np.exp(-1j * t * w)) @ V.conj().T
        col = np.abs(U[:, 0])
        col[0] = 0
        if col.max() > 1 - 1e-7:
            return True
    return False


def is_normal(G, S):
    Ss = set(S)
    return all(G.mul(G.mul(g, s), G.inv(g)) in Ss for g in G.els for s in S)


def sig(H):
    return tuple(np.round(np.linalg.eigvalsh(H), 5))


def key(M):
    N = len(M)
    return str(sorted((tuple(sorted(M[u])), tuple(sorted(M[v][u] for v in range(N)))) for u in range(N)))


def iso(M1, M2):
    N = len(M1)
    pr = lambda M, u: (sorted(M[u]), sorted(M[v][u] for v in range(N)))
    p1 = [pr(M1, u) for u in range(N)]
    p2 = [pr(M2, u) for u in range(N)]
    if sorted(map(str, p1)) != sorted(map(str, p2)):
        return False
    order, seen = [0], {0}
    for u in order:
        for v in range(N):
            if v not in seen and (M1[u][v] or M1[v][u]):
                seen.add(v)
                order.append(v)
    order += [v for v in range(N) if v not in seen]
    img, used = [-1] * N, [False] * N

    def rec(i):
        if i == N:
            return True
        u = order[i]
        for w in range(N):
            if used[w] or p1[u] != p2[w] or M1[u][u] != M2[w][w]:
                continue
            if all(M1[u][order[j]] == M2[w][img[order[j]]] and M1[order[j]][u] == M2[img[order[j]]][w]
                   for j in range(i)):
                img[u] = w
                used[w] = True
                if rec(i + 1):
                    return True
                used[w] = False
                img[u] = -1
        return False

    return rec(0)


def abelian_graphs(n, needed):
    out = {}
    for mods in ABELIAN[n]:
        B = list(product(*[range(m) for m in mods]))
        els = B[1:]
        idx = {b: i for i, b in enumerate(B)}
        add = lambda a, b: tuple((x + y) % m for x, y, m in zip(a, b, mods))
        neg = lambda a: tuple((-x) % m for x, m in zip(a, mods))
        for mask in range(1 << len(els)):
            S = [els[i] for i in range(len(els)) if mask >> i & 1]
            Ss = set(S)
            if not any(neg(s) in Ss for s in S):
                continue
            H = np.zeros((n, n), complex)
            for g in B:
                for s in S:
                    h = add(g, s)
                    if neg(s) in Ss:
                        H[idx[g], idx[h]] = 1
                    else:
                        H[idx[g], idx[h]] = 1j
                        H[idx[h], idx[g]] = -1j
            sg = sig(H)
            if sg in needed:
                M = colour_matrix(H)
                out.setdefault((sg, key(M)), []).append(M)
    return out


def row(name, G):
    cand = [e for e in G.els if e != G.one]
    tot = npst = nn = 0
    conn = []
    for mask in range(1 << len(cand)):
        S = [cand[i] for i in range(len(cand)) if mask >> i & 1]
        Ss = set(S)
        if not any(G.inv(s) in Ss for s in S):
            continue
        tot += 1
        if not pst(G, S):
            continue
        npst += 1
        if is_normal(G, S):
            continue
        nn += 1
        if generates(G, S):
            conn.append(S)
    und = sum(1 for S in conn if all(G.inv(s) in set(S) for s in S))
    classes = []
    for S in conn:
        H = hermitian(G, S)
        M = colour_matrix(H)
        sg = sig(H)
        for c in classes:
            if c[0] == sg and iso(c[1], M):
                break
        else:
            classes.append((sg, M))
    print(f"{name}: sets={tot} PST={npst} nonnormal-PST={nn} connected-nonnormal-PST={len(conn)} "
          f"undirected={und} iso-classes={len(classes)}", flush=True)
    return classes


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_ab = "--abelian" in sys.argv
    allcl = {}
    for name in args:
        G = GROUPS[name]()
        allcl[name] = (len(G.els), row(name, G))
    if want_ab:
        distinct = []
        for n in sorted({v[0] for v in allcl.values()}):
            names = [k for k, v in allcl.items() if v[0] == n]
            needed = {sg for k in names for sg, _ in allcl[k][1]}
            ab = abelian_graphs(n, needed)
            for k in names:
                bad = [(sg, M) for sg, M in allcl[k][1]
                       if not any(iso(M, A) for A in ab.get((sg, key(M)), []))]
                print(f"{k}: classes={len(allcl[k][1])} not-abelian-Cayley={len(bad)}", flush=True)
                for sg, M in bad:
                    if not any(sg == d[0] and iso(M, d[1]) for d in distinct if len(d[1]) == n):
                        distinct.append((sg, M))
            dn = [d for d in distinct if len(d[1]) == n]
            print(f"order {n}: distinct graphs not Cayley graphs of any abelian group = {len(dn)}; "
                  f"|Aut| distribution {sorted(collections.Counter(aut_order(d[1]) for d in dn).items())}", flush=True)
