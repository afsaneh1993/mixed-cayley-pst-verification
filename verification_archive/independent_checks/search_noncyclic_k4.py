"""Exact enumeration of ALL weighing sets of weight 4 on G = Dic(A,y) = A x_f Z_2  (f = -id, x^2 = y), A abelian, y in A[2].

A weighing set is S' subset G\\{1} with K^2 = 16 I, where K = sum_{s in S'} w_s rho(s) and
w_s = 1 on elements s with s^-1 also in S' (undirected), w_s = i on an arc s (s in S', s^-1 not in S'), w = -i on s^-1.
The condition does not involve c.  For each c in A[2]\\{0} with c not in S' the set is "connected" iff <S' u {c}> = G.

Pruning (consequences of K^2 = 16 I alone): the diagonal of K^2 says every row of K has 16 non-zero entries; K 1 = u 1 with
u = |S' cap S'^-1| (an arc contributes i - i = 0 to a row sum), so u^2 = 16, u = 4; the other 12 non-zero entries are s and s^-1 of 6 arcs.
Candidates: 4 undirected elements (j involutions + b inverse pairs, j + 2b = 4) and 6 arcs on 6 further inverse pairs, two orientations each.
Every candidate is tested EXACTLY: kappa*kappa = 16 delta_1 in Z[i][G] (int64 arrays, comparison with ==, no tolerance).
Type I: all undirected elements lie in xA; Type II: all lie in A; otherwise 'mixed'.

Usage:  python3 search_noncyclic_k4.py NAME [NAME ...]      NAME in Z6xZ2 Z8xZ2 Z4xZ4 Z4xZ2xZ2 Z4xZ2 Z8
        (every y in A[2] is run, including y = 0)
"""
import numpy as np, itertools, sys, time
from lib import *

GROUPS = {'Z6xZ2': (6, 2), 'Z8xZ2': (8, 2), 'Z4xZ4': (4, 4), 'Z4xZ2xZ2': (4, 2, 2), 'Z4xZ2': (4, 2), 'Z8': (8,)}


def build(mods, y):
    f = lambda a, mods=mods: tuple((-p) % m for p, m in zip(a, mods))
    return Grp(mods, f, y)


def enumerate_sets(G, k=4, chunk=250):
    n = G.n
    one = G.one
    pairs, invols, seen = [], [], set()
    for s in range(n):
        if s == one or s in seen:
            continue
        t = int(G.inv[s])
        seen.add(s)
        seen.add(t)
        if t == s:
            invols.append(s)
        else:
            pairs.append((s, t))
    npairs = len(pairs)
    TABr = np.zeros((npairs, 3, n), dtype=np.int64)
    TABi = np.zeros((npairs, 3, n), dtype=np.int64)
    for p, (s, t) in enumerate(pairs):
        TABi[p, 0, s] = 1
        TABi[p, 0, t] = -1          # arc s: weight i on s, -i on s^-1
        TABi[p, 1, t] = 1
        TABi[p, 1, s] = -1          # arc s^-1
        TABr[p, 2, s] = 1
        TABr[p, 2, t] = 1           # both s and s^-1 in S' (undirected)
    Q = np.array([[G.mt[G.inv[s], g] for g in range(n)] for s in range(n)])   # Q[s,g] = s^-1 g
    probe = [g for g in range(n) if g != one][:10]
    narcs, nund = 2 * k - 2, k
    orient = np.array(list(itertools.product((0, 1), repeat=narcs)), dtype=np.int64)
    cache, found, ncand = {}, [], 0

    def coeff(wr, wi, g):
        qg = Q[:, g]
        return (wr * wr[qg] - wi * wi[qg]).sum(), (wr * wi[qg] + wi * wr[qg]).sum()

    def test(Wr, Wi):
        ok = np.ones(Wr.shape[0], dtype=bool)
        for g in probe:                       # cheap necessary test on a few coefficients of kappa^2
            qg = Q[:, g]
            cr = (Wr * Wr[:, qg] - Wi * Wi[:, qg]).sum(axis=1)
            ci = (Wr * Wi[:, qg] + Wi * Wr[:, qg]).sum(axis=1)
            ok &= (cr == 0) & (ci == 0)
        res = []
        for b in np.where(ok)[0]:             # full exact test on the survivors
            wr, wi = Wr[b], Wi[b]
            good = True
            for g in range(n):
                cr, ci = coeff(wr, wi, g)
                if (g == one and (cr != k * k or ci != 0)) or (g != one and (cr != 0 or ci != 0)):
                    good = False
                    break
            if good:
                res.append(b)
        return res

    for j in range(0, min(nund, len(invols)) + 1):
        if (nund - j) % 2:
            continue
        bpairs = (nund - j) // 2
        for I in itertools.combinations(invols, j):
            for B in itertools.combinations(range(npairs), bpairs):
                R = [p for p in range(npairs) if p not in B]
                if len(R) < narcs:
                    continue
                key = tuple(R)
                if key not in cache:
                    cache[key] = np.array(list(itertools.combinations(R, narcs)), dtype=np.int64)
                C = cache[key]
                Wur = np.zeros(n, dtype=np.int64)
                for s in I:
                    Wur[s] = 1
                for p in B:
                    Wur += TABr[p, 2]
                for c0 in range(0, len(C), chunk):
                    Cc = C[c0:c0 + chunk]
                    nc = len(Cc)
                    P = np.broadcast_to(Cc[:, None, :], (nc, len(orient), narcs))
                    O = np.broadcast_to(orient[None, :, :], (nc, len(orient), narcs))
                    Wr = (TABr[P, O].sum(axis=2) + Wur).reshape(-1, n)
                    Wi = TABi[P, O].sum(axis=2).reshape(-1, n)
                    ncand += Wr.shape[0]
                    for b in test(Wr, Wi):
                        found.append((Wr[b].copy(), Wi[b].copy()))
    return found, ncand


def analyse(name, y, k=4):
    mods = GROUPS[name]
    G = build(mods, y)
    n = G.n
    zero = tuple(0 for _ in mods)
    A2 = [a for a in G.Aels if G.add(a, a) == zero and a != zero]
    t0 = time.time()
    found, ncand = enumerate_sets(G, k)
    sets = []
    for Wr, Wi in found:
        S = frozenset(int(s) for s in range(n) if Wr[s] == 1 or Wi[s] == 1)
        U = [s for s in S if int(G.inv[s]) in S]
        inA = lambda s: G.els[s][1] == 0
        typ = 'I' if all(not inA(s) for s in U) else ('II' if all(inA(s) for s in U) else 'mixed')
        sets.append((S, typ))
    conn, connpairs = 0, 0
    typecnt = {'I': 0, 'II': 0, 'mixed': 0}
    typeconn = {'I': 0, 'II': 0, 'mixed': 0}
    for S, typ in sets:
        typecnt[typ] += 1
        ok = False
        for c in A2:
            ci = G.idx[(c, 0)]
            if ci in S:
                continue
            if len(gen_subgroup(G, list(S) + [ci])) == n:
                connpairs += 1
                ok = True
        if ok:
            conn += 1
            typeconn[typ] += 1
    return dict(group=name, y=y, order=n, candidates=ncand, weighing_sets=len(sets), by_type=typecnt,
                connected_for_some_c=conn, connected_by_type=typeconn, pairs_set_c_connected=connpairs,
                seconds=round(time.time() - t0, 1))


def all_y(name):
    mods = GROUPS[name]
    zero = tuple(0 for _ in mods)
    A = list(itertools.product(*[range(m) for m in mods]))
    add = lambda a, b: tuple((p + q) % m for p, q, m in zip(a, b, mods))
    return [a for a in A if add(a, a) == zero]


if __name__ == "__main__":
    # arguments: NAME  (all y)   or   NAME:y1,y2,..  (one y)
    for arg in sys.argv[1:]:
        if ":" in arg:
            name, ys = arg.split(":")
            for y in [tuple(int(v) for v in ys.split(","))]:
                print(analyse(name, y), flush=True)
        else:
            for y in all_y(arg):
                print(analyse(arg, y), flush=True)
